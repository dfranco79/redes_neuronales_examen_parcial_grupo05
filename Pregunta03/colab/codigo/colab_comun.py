# -*- coding: utf-8 -*-
"""colab_comun.py — Configuración, preparación de datos y utilidades comunes (Pregunta 3, YOLO sobre CCT-20, Colab).

Estructura esperada en Google Drive:
    MyDrive/CCT20_P3/yolo_data_cct20.zip     (generado en el PC con preparar_local.py)
    MyDrive/CCT20_P3/codigo/*.py             (estos módulos)
    MyDrive/CCT20_P3/resultados/             (se crea sola: runs/, tablas/, modelos/)
Los datos se descomprimen en el disco local de Colab (/content/cct): leer imágenes desde Drive es muy lento.
"""
import json, os, shutil, time, zipfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import yaml

SPLITS = ["train", "val_cis", "val_trans", "test_cis", "test_trans"]

# Configuraciones de la ablación del informe (A, B, C)
EXPERIMENTOS = {
    "A_base":       dict(aug="base",   oversample=1),
    "B_strong":     dict(aug="strong", oversample=1),
    "C_strong_os3": dict(aug="strong", oversample=3),   # sobremuestreo de especies raras (factor máx. 3)
}
AUG_PRESETS = {
    "base":   dict(mosaic=0.0, mixup=0.0, copy_paste=0.0, scale=0.3, hsv_h=0.015, hsv_s=0.4, hsv_v=0.3, fliplr=0.5),
    "strong": dict(mosaic=1.0, mixup=0.15, copy_paste=0.3, scale=0.6, hsv_h=0.03, hsv_s=0.7, hsv_v=0.5, fliplr=0.5, degrees=5.0),
}

# ----------------------------------------------------------------------------- configuración
def detectar_device():
    try:
        import torch
        return 0 if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"

def crear_config(**kw):
    """Crea la configuración. Cualquier parámetro se puede sobrescribir: crear_config(EPOCHS=30, BATCH=16)."""
    c = dict(DRIVE_BASE="/content/drive/MyDrive/CCT20_P3", ZIP_NAME="yolo_data_cct20.zip", LOCAL="/content/cct",
             WEIGHTS="yolov8s.pt", IMGSZ=640, EPOCHS=50, BATCH=32, PATIENCE=15, WORKERS=2, SEED=42,
             TRAIN_FRACTION=1.0, RETRAIN=False, KEEP_ZIP=False, DEVICE=None)
    c.update(kw)
    cfg = SimpleNamespace(**c)
    cfg.DRIVE_BASE, cfg.LOCAL = Path(cfg.DRIVE_BASE), Path(cfg.LOCAL)
    cfg.ZIP_DRIVE = cfg.DRIVE_BASE / cfg.ZIP_NAME
    cfg.CODIGO = cfg.DRIVE_BASE / "codigo"
    cfg.DATA = cfg.LOCAL / "yolo_data"
    cfg.OUT = cfg.DRIVE_BASE / "resultados"
    cfg.RUNS, cfg.RES, cfg.MODELOS = cfg.OUT / "runs", cfg.OUT / "tablas", cfg.OUT / "modelos"
    if cfg.DEVICE is None:
        cfg.DEVICE = detectar_device()
    for d in (cfg.RUNS, cfg.RES, cfg.MODELOS, cfg.LOCAL):
        d.mkdir(parents=True, exist_ok=True)
    if cfg.DEVICE == "cpu":
        print("AVISO: no hay GPU. En Colab: Entorno de ejecución > Cambiar tipo de entorno > GPU (T4). En CPU esto tardará horas.")
    return cfg

def nombre_exp(cfg, exp):
    """Nombre de la carpeta de la corrida, p. ej. 'yolov8s_B_strong'."""
    return f"{Path(cfg.WEIGHTS).stem}_{exp}"

def info_gpu():
    import subprocess
    try:
        print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=20).stdout.strip() or "(sin GPU)")
    except Exception as e:
        print("(sin GPU)", e)

def montar_drive():
    try:
        from google.colab import drive
        drive.mount("/content/drive")
    except ImportError:
        print("(no estás en Colab: se usan las rutas locales tal cual)")

# ----------------------------------------------------------------------------- datos
def _conteos(cfg):
    out = {}
    for s in SPLITS:
        di, dl = cfg.DATA / "images" / s, cfg.DATA / "labels" / s
        ni = sum(1 for f in di.glob("*.jpg") if "_dup" not in f.name) if di.exists() else 0
        nl = sum(1 for f in dl.glob("*.txt") if "_dup" not in f.name) if dl.exists() else 0
        out[s] = (ni, nl)
    return out

def verificar_datos(cfg, log=print):
    """Compara lo descomprimido con la Tabla 1 que viaja dentro del zip. Lanza error si no coincide."""
    t1 = pd.read_csv(cfg.DATA / "tabla1_dataset.csv", index_col=0)
    ok = True
    for s, (ni, nl) in _conteos(cfg).items():
        esp = int(t1.loc[s, "imagenes"]) if s in t1.index else -1
        marca = "OK" if (ni == nl == esp) else "REVISAR"
        ok &= marca == "OK"
        log(f"  {s:11s} imágenes={ni:6d} etiquetas={nl:6d} esperado={esp:6d}  {marca}")
    if not ok:
        raise RuntimeError("Los datos descomprimidos no coinciden con la Tabla 1. Borra /content/cct y vuelve a ejecutar la preparación.")
    return t1

def write_yamls(data_dir, classes):
    data_dir = Path(data_dir).resolve()
    for s in SPLITS:
        if (data_dir / "images" / s).exists():
            yaml.safe_dump({"path": str(data_dir), "train": "images/train", "val": f"images/{s}",
                            "names": {i: n for i, n in enumerate(classes)}},
                           open(data_dir / f"{s}.yaml", "w", encoding="utf-8"), allow_unicode=True)

def preparar_datos(cfg, forzar=False, log=print):
    """Copia el zip de Drive al disco local, lo descomprime, regenera los .yaml y verifica los conteos."""
    marca = cfg.DATA / ".extraido_ok"
    if marca.exists() and not forzar:
        log(f"Datos ya descomprimidos en {cfg.DATA} (se omite la copia y la extracción).")
    else:
        if not cfg.ZIP_DRIVE.exists():
            raise FileNotFoundError(f"No encuentro {cfg.ZIP_DRIVE}. Sube yolo_data_cct20.zip a esa carpeta de Drive.")
        if cfg.DATA.exists():
            shutil.rmtree(cfg.DATA)
        zloc = cfg.LOCAL / cfg.ZIP_NAME
        t0 = time.time()
        log(f"Copiando {cfg.ZIP_DRIVE.stat().st_size/2**30:.2f} GiB de Drive al disco local ...")
        shutil.copyfile(cfg.ZIP_DRIVE, zloc)
        log(f"  copiado en {(time.time()-t0)/60:.1f} min. Descomprimiendo ...")
        with zipfile.ZipFile(zloc) as z:
            nombres = z.namelist()
            for i, n in enumerate(nombres, 1):
                z.extract(n, cfg.LOCAL)
                if i % 20000 == 0:
                    log(f"  {i}/{len(nombres)} archivos")
        if not cfg.KEEP_ZIP:
            zloc.unlink()
        log(f"  listo en {(time.time()-t0)/60:.1f} min")
    classes = cargar_clases(cfg)
    write_yamls(cfg.DATA, classes)
    log("Verificación contra la Tabla 1:")
    t1 = verificar_datos(cfg, log)
    marca.write_text("ok")
    return t1

def cargar_clases(cfg):
    return json.load(open(cfg.DATA / "classes.json", encoding="utf-8"))

def cargar_meta(cfg):
    """meta.csv: una fila por imagen (id, location, seq_id, frame, dt, split, has_animal, ...)."""
    return pd.read_csv(cfg.DATA / "meta.csv", parse_dates=["dt"], dtype={"id": str, "location": str, "seq_id": str})

def conteo_train(cfg):
    """Imágenes de train por especie (para la Tabla 3 y la correlación frecuencia-AP)."""
    a = pd.read_csv(cfg.DATA / "anotaciones.csv", dtype={"image_id": str})
    return a[a.split == "train"].groupby("especie").image_id.nunique().sort_values(ascending=False)

# ----------------------------------------------------------------------------- sobremuestreo (config C)
def sobremuestrear(cfg, factor_max, log=print):
    """Crea images/train_os{N} y labels/train_os{N} con copias de las imágenes de especies raras (solo train).
    El .yaml de entrenamiento incluye ambas carpetas; así las configuraciones A/B no se ven afectadas."""
    classes = cargar_clases(cfg); cid = {n: i for i, n in enumerate(classes)}
    cnt = conteo_train(cfg); med = float(np.median(cnt.values))
    cf = {cid[e]: int(min(factor_max, max(1, round(med / n)))) for e, n in cnt.items()}
    dI, dL = cfg.DATA / "images" / f"train_os{factor_max}", cfg.DATA / "labels" / f"train_os{factor_max}"
    dI.mkdir(parents=True, exist_ok=True); dL.mkdir(parents=True, exist_ok=True)
    n_new = 0
    for lab in sorted((cfg.DATA / "labels" / "train").glob("*.txt")):
        if "_dup" in lab.name:
            continue
        cls = [int(l.split()[0]) for l in lab.read_text().splitlines() if l.strip()]
        f = max((cf.get(c, 1) for c in cls), default=1)
        for k in range(1, f):
            img_src = cfg.DATA / "images" / "train" / f"{lab.stem}.jpg"
            dst = dI / f"{lab.stem}_dup{k}.jpg"
            if not dst.exists():
                try: os.link(img_src, dst)
                except OSError: shutil.copyfile(img_src, dst)
                n_new += 1
            (dL / f"{lab.stem}_dup{k}.txt").write_text(lab.read_text())
    n_tot = len(list(dI.glob("*.jpg")))
    log(f"Sobremuestreo x{factor_max}: {n_tot} imágenes extra en images/train_os{factor_max} ({n_new} nuevas). Factores por clase: "
        f"{ {classes[c]: v for c, v in cf.items() if v > 1} }")
    return n_tot

def yaml_train(cfg, exp):
    """Escribe DATA/train_<exp>.yaml (train, y validación = val_cis) y devuelve su ruta."""
    spec = EXPERIMENTOS[exp]
    train = ["images/train"]
    if spec["oversample"] > 1:
        sobremuestrear(cfg, spec["oversample"])
        train.append(f"images/train_os{spec['oversample']}")
    classes = cargar_clases(cfg)
    p = cfg.DATA / f"train_{exp}.yaml"
    yaml.safe_dump({"path": str(cfg.DATA.resolve()), "train": train if len(train) > 1 else train[0],
                    "val": "images/val_cis", "names": {i: n for i, n in enumerate(classes)}},
                   open(p, "w", encoding="utf-8"), allow_unicode=True)
    return p

# ----------------------------------------------------------------------------- estado del entrenamiento
def _leer_ckpt(p):
    import torch
    return torch.load(str(p), map_location="cpu", weights_only=False)

def estado_entrenamiento(wdir):
    """'nuevo' (no hay pesos), 'incompleto' (last.pt con optimizador: se puede reanudar) o 'terminado'."""
    wdir = Path(wdir)
    last, best = wdir / "last.pt", wdir / "best.pt"
    if not last.exists():
        return "terminado" if best.exists() else "nuevo"
    try:
        ck = _leer_ckpt(last)
        # al terminar, Ultralytics 'limpia' el checkpoint (optimizer=None); si sigue ahí, el entrenamiento se cortó
        return "terminado" if ck.get("optimizer") is None else "incompleto"
    except Exception as e:
        print(f"(no pude leer {last.name}: {e}; lo trato como terminado si existe best.pt)")
        return "terminado" if best.exists() else "nuevo"

def buscar_weights_dir(cfg, name):
    """Carpeta weights/ de la corrida 'name' dentro de resultados/runs (busca también carpetas anidadas)."""
    c = [p for p in cfg.RUNS.glob(f"**/{name}/weights") if p.is_dir()]
    return sorted(c, key=lambda p: p.stat().st_mtime, reverse=True)[0] if c else None
