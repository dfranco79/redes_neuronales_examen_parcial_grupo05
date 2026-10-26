# -*- coding: utf-8 -*-
"""robustez.py — Recall por tamaño de objeto y por un indicador APROXIMADO de oclusión (Tabla 4).

CCT no anota oclusión: se usa el contraste de la caja contra un anillo de fondo + contacto con el borde.
Es un proxy de camuflaje/oclusión parcial, no una medida directa (declararlo así en el informe).
"""
from pathlib import Path
import numpy as np, pandas as pd
import colab_comun as C

def iou(a, b):
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    u = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / u if u > 0 else 0.0

def read_gt(path, W, H):
    out = []
    if Path(path).exists():
        for ln in open(path):
            p = ln.split()
            if len(p) == 5:
                c, cx, cy, w, h = int(p[0]), *map(float, p[1:])
                out.append((c, [(cx-w/2)*W, (cy-h/2)*H, (cx+w/2)*W, (cy+h/2)*H]))
    return out

def occ_proxy(gray, box):
    H, W = gray.shape
    x1, y1, x2, y2 = [int(round(v)) for v in box]; x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
    if x2-x1 < 3 or y2-y1 < 3: return 0.0, 1
    inside = gray[y1:y2, x1:x2].astype("float32")
    pw, ph = int(.25*(x2-x1))+2, int(.25*(y2-y1))+2
    X1, Y1, X2, Y2 = max(0, x1-pw), max(0, y1-ph), min(W, x2+pw), min(H, y2+ph)
    ring = gray[Y1:Y2, X1:X2].astype("float32"); mk = np.ones_like(ring, bool); mk[y1-Y1:y2-Y1, x1-X1:x2-X1] = False
    rp = ring[mk]
    if rp.size == 0: return 0.0, 1
    return abs(inside.mean() - rp.mean()) / (rp.std() + 1e-6), int(x1 <= 1 or y1 <= 1 or x2 >= W-1 or y2 >= H-1)

def robustez(cfg, model, exp, split="test_trans", conf=0.25, lote=32, log=print):
    """Devuelve (tab4, R). Guarda tabla4_robustez_<name>.csv y detalle_robustez_<name>.csv."""
    import cv2
    name = C.nombre_exp(cfg, exp); classes = C.cargar_clases(cfg)
    files = [f for f in sorted((cfg.DATA / "images" / split).glob("*.jpg")) if "_dup" not in f.name]
    log(f"Robustez sobre {split}: {len(files)} imágenes (conf={conf}, IoU>=0.5 y misma clase = acierto)")
    recs = []
    for i in range(0, len(files), lote):
        b = files[i:i+lote]
        for f, pr in zip(b, model.predict([str(x) for x in b], imgsz=cfg.IMGSZ, conf=conf, iou=0.6,
                                          device=cfg.DEVICE, verbose=False)):
            H, W = pr.orig_shape
            gts = read_gt(cfg.DATA / "labels" / split / f"{f.stem}.txt", W, H)
            if not gts: continue
            gray = cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2GRAY)
            dets = sorted([(int(c), float(s), bx.tolist()) for c, s, bx in zip(pr.boxes.cls, pr.boxes.conf, pr.boxes.xyxy)],
                          key=lambda d: -d[1])
            used = set()
            for c, box in gts:
                best, bj = 0.0, -1
                for j, (dc, ds, db) in enumerate(dets):
                    if j in used or dc != c: continue
                    v = iou(box, db)
                    if v > best: best, bj = v, j
                hit = best >= 0.5
                if hit: used.add(bj)
                ct, tb = occ_proxy(gray, box)
                recs.append(dict(img=f.name, especie=classes[c], hit=int(hit),
                                 rel_area=(box[2]-box[0])*(box[3]-box[1])/(W*H), contraste=ct, borde=tb))
        if (i // lote) % 50 == 0:
            log(f"  {min(i+lote, len(files))}/{len(files)}")
    R = pd.DataFrame(recs)
    R["tamano"] = pd.cut(R.rel_area, [0, .01, .08, 1], labels=["pequeño", "mediano", "grande"])
    q1, q2 = R.contraste.quantile([1/3, 2/3])
    R["oclusion_proxy"] = np.where(R.borde == 1, "alto", np.where(R.contraste <= q1, "alto",
                                   np.where(R.contraste <= q2, "medio", "bajo")))
    tab4 = pd.concat({c: R.groupby(c, observed=True).hit.agg(n="size", recall="mean") for c in ["oclusion_proxy", "tamano"]})
    log("Tabla 4 — recall por tamaño y oclusión (proxy):\n" + str(tab4.round(3)))
    tab4.to_csv(cfg.RES / f"tabla4_robustez_{name}.csv"); R.to_csv(cfg.RES / f"detalle_robustez_{name}.csv", index=False)
    return tab4, R
