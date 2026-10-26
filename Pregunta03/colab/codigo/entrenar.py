# -*- coding: utf-8 -*-
"""entrenar.py — Entrenamiento YOLOv8 que NO se repite si ya existe un entrenamiento terminado.

  - Si hay best.pt de una corrida terminada  -> se omite y se devuelve esa ruta.
  - Si hay una corrida cortada (last.pt)     -> se REANUDA donde quedó (resume=True).
  - Si no hay nada (o cfg.RETRAIN=True)      -> entrena desde los pesos preentrenados.
Los pesos se guardan en Drive (resultados/runs/<exp>/weights) y una copia en resultados/modelos/.
"""
import shutil, time
from pathlib import Path
import colab_comun as C

def entrenar(cfg, exp, log=print):
    from ultralytics import YOLO
    if exp not in C.EXPERIMENTOS:
        raise KeyError(f"Experimento desconocido '{exp}'. Opciones: {list(C.EXPERIMENTOS)}")
    spec, name = C.EXPERIMENTOS[exp], C.nombre_exp(cfg, exp)
    wdir = C.buscar_weights_dir(cfg, name)
    estado = C.estado_entrenamiento(wdir) if wdir else "nuevo"
    log(f"Experimento {name}: estado previo = {estado}" + ("  (RETRAIN=True)" if cfg.RETRAIN else ""))

    if cfg.RETRAIN and wdir is not None:                     # conservar la corrida anterior, no mezclarla
        viejo = wdir.parent.with_name(f"{name}_previo_{time.strftime('%Y%m%d_%H%M%S')}")
        shutil.move(str(wdir.parent), str(viejo)); log(f"  corrida anterior movida a {viejo.name}")
        wdir, estado = None, "nuevo"

    if estado == "terminado":
        best = wdir / "best.pt"
        log(f"  ENTRENAMIENTO YA REALIZADO -> se omite. Pesos: {best}")
        return _copia_estable(cfg, name, best)

    if estado == "incompleto":
        log(f"  reanudando desde {wdir / 'last.pt'} ...")
        model = YOLO(str(wdir / "last.pt"))
        model.train(resume=True)
    else:
        weights = Path(cfg.WEIGHTS)
        loc = cfg.CODIGO / weights.name                       # si guardaste yolov8s.pt junto al código, se usa; si no, se descarga
        model = YOLO(str(loc) if loc.exists() else str(weights))
        data = C.yaml_train(cfg, exp)
        model.train(data=str(data), imgsz=cfg.IMGSZ, epochs=cfg.EPOCHS, batch=cfg.BATCH, seed=cfg.SEED,
                    deterministic=True, project=str(cfg.RUNS), name=name, exist_ok=True,
                    patience=cfg.PATIENCE, cos_lr=True, device=cfg.DEVICE, workers=cfg.WORKERS,
                    fraction=cfg.TRAIN_FRACTION, plots=True, **C.AUG_PRESETS[spec["aug"]])
    best = Path(model.trainer.save_dir).resolve() / "weights" / "best.pt"
    log(f"  entrenamiento terminado. Pesos: {best}")
    return _copia_estable(cfg, name, best)

def _copia_estable(cfg, name, best):
    dst = cfg.MODELOS / f"{name}_best.pt"
    cfg.MODELOS.mkdir(parents=True, exist_ok=True)
    if not dst.exists() or dst.stat().st_mtime < Path(best).stat().st_mtime:
        shutil.copy2(best, dst)
    return Path(best)

def curvas(cfg, exp, log=print):
    """Curvas de entrenamiento (pérdidas y mAP por época) a partir de results.csv. Guarda curvas_<name>.png."""
    import pandas as pd
    import matplotlib; import matplotlib.pyplot as plt
    name = C.nombre_exp(cfg, exp)
    wdir = C.buscar_weights_dir(cfg, name)
    csv = wdir.parent / "results.csv" if wdir else None
    if csv is None or not csv.exists():
        log("No hay results.csv todavía."); return None
    r = pd.read_csv(csv); r.columns = [c.strip() for c in r.columns]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for c in ["train/box_loss", "train/cls_loss", "val/box_loss", "val/cls_loss"]:
        if c in r: ax[0].plot(r["epoch"], r[c], label=c)
    ax[0].set_xlabel("época"); ax[0].set_title("Pérdidas"); ax[0].legend(fontsize=7)
    for c in ["metrics/mAP50(B)", "metrics/mAP50-95(B)"]:
        if c in r: ax[1].plot(r["epoch"], r[c], label=c)
    ax[1].set_xlabel("época"); ax[1].set_title("mAP en val_cis"); ax[1].legend(fontsize=7)
    plt.tight_layout(); out = cfg.RES / f"curvas_{name}.png"; plt.savefig(out, dpi=150); plt.close(fig)
    log(f"Curvas guardadas en {out}")
    return out
