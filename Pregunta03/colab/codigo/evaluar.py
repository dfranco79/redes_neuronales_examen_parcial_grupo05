# -*- coding: utf-8 -*-
"""evaluar.py — Evaluación por partición (cis/trans) y por especie: Tablas 2 y 3 del informe."""
from pathlib import Path
import pandas as pd
import colab_comun as C

def evaluar(cfg, model, exp, splits=("val_cis", "val_trans", "test_cis", "test_trans"), log=print):
    """Devuelve (M, tab3, PC). Guarda tabla2_*, tabla3_* y ap_por_especie_* en resultados/tablas."""
    name = C.nombre_exp(cfg, exp)
    classes, cnt = C.cargar_clases(cfg), C.conteo_train(cfg)
    metrics, rows = {}, []
    for s in splits:
        if not (cfg.DATA / f"{s}.yaml").exists():
            continue
        m = model.val(data=str(cfg.DATA / f"{s}.yaml"), imgsz=cfg.IMGSZ, conf=0.001, iou=0.6, batch=cfg.BATCH,
                      device=cfg.DEVICE, workers=cfg.WORKERS, project=str(cfg.RUNS), name=f"{name}_eval_{s}",
                      exist_ok=True, plots=True, verbose=False)
        metrics[s] = {"mAP50": m.box.map50, "mAP50-95": m.box.map, "P": m.box.mp, "R": m.box.mr}
        for k, ci in enumerate(m.box.ap_class_index):
            rows.append({"split": s, "especie": classes[int(ci)], "AP50": m.box.ap50[k], "AP50-95": m.box.ap[k]})
    M = pd.DataFrame(metrics).T
    log("Tabla 2 — métricas por partición:\n" + str(M.round(4)))
    for k in ("val", "test"):
        if f"{k}_cis" in M.index and f"{k}_trans" in M.index:
            gap = (M.loc[f"{k}_cis", ["mAP50", "mAP50-95"]] - M.loc[f"{k}_trans", ["mAP50", "mAP50-95"]]).round(4).to_dict()
            log(f"brecha {k} (cis - trans): {gap}")
    M.to_csv(cfg.RES / f"tabla2_metricas_{name}.csv")

    PC = pd.DataFrame(rows)
    PC["imgs_train"] = PC.especie.map(cnt).fillna(0).astype(int)
    PC.to_csv(cfg.RES / f"ap_por_especie_{name}.csv", index=False)
    tab3 = PC[PC.split.isin(["test_cis", "test_trans"])].pivot(index="especie", columns="split", values="AP50")
    tab3["imgs_train"] = cnt.reindex(tab3.index).fillna(0).astype(int)
    tab3 = tab3.sort_values("imgs_train")
    log("Tabla 3 — AP50 por especie (ordenada por frecuencia en train):\n" + str(tab3.round(3)))
    tab3.to_csv(cfg.RES / f"tabla3_ap_por_especie_{name}.csv")
    sub = PC[PC.split == "test_cis"]
    if len(sub) > 2:
        log(f"Spearman (frecuencia vs AP50, test_cis): {sub.imgs_train.corr(sub.AP50, method='spearman'):.3f}")
    return M, tab3, PC

def comparar_experimentos(cfg, log=print):
    """Junta las Tablas 2 de todas las configuraciones evaluadas (A, B, C) en una sola tabla."""
    frames = {}
    for exp in C.EXPERIMENTOS:
        p = cfg.RES / f"tabla2_metricas_{C.nombre_exp(cfg, exp)}.csv"
        if p.exists():
            frames[exp] = pd.read_csv(p, index_col=0)
    if not frames:
        log("Aún no hay Tablas 2 guardadas."); return None
    T = pd.concat(frames, names=["config", "split"])
    log(str(T.round(4))); T.to_csv(cfg.RES / "comparacion_configuraciones.csv")
    return T
