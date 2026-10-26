# -*- coding: utf-8 -*-
"""gradcam_cct.py — Grad-CAM sobre el backbone (capa 9, SPPF) y métricas de atención (Tabla 5).

Energía en caja = fracción del mapa normalizado dentro de las cajas reales.
Enriquecimiento = energía / fracción de área de las cajas (≈1: atención repartida como el área, sin preferencia por el animal).
"""
import random
from pathlib import Path
import numpy as np, pandas as pd
import colab_comun as C
from robustez import read_gt

def _letterbox(cv2, img, size):
    h, w = img.shape[:2]; r = size / max(h, w); nh, nw = int(round(h*r)), int(round(w*r))
    can = np.full((size, size, 3), 114, np.uint8); t, l = (size-nh)//2, (size-nw)//2
    can[t:t+nh, l:l+nw] = cv2.resize(img, (nw, nh)); return can, (t, l, nh, nw)

def gradcam_cct(cfg, best_path, exp, n_cam=100, n_vis=6, splits=("val_cis", "val_trans"), log=print):
    """Devuelve tab5 (promedio por partición). Guarda tabla5_gradcam_<name>.csv, imágenes en resultados/tablas/gradcam_<name>/
    y la galería gradcam_galeria_<name>.png (cis arriba, trans abajo)."""
    import cv2, torch
    from ultralytics import YOLO
    import matplotlib; import matplotlib.pyplot as plt
    name = C.nombre_exp(cfg, exp); classes = C.cargar_clases(cfg); IMGSZ = cfg.IMGSZ
    dev_s = "cpu" if cfg.DEVICE == "cpu" else f"cuda:{cfg.DEVICE}"
    net = YOLO(str(best_path)).model.to(dev_s).float().eval()      # instancia nueva (sin fusionar)
    for p_ in net.parameters(): p_.requires_grad_(True)
    LAYER = net.model[9]; dev = next(net.parameters()).device

    def gradcam(img):
        can, (t, l, nh, nw) = _letterbox(cv2, img, IMGSZ)
        x = torch.from_numpy(can[:, :, ::-1].copy()).permute(2, 0, 1).float().div(255)[None].to(dev).requires_grad_(True)
        st = {}; hk = LAYER.register_forward_hook(lambda m, i, o: st.__setitem__("a", o))
        with torch.enable_grad():
            out = net(x); hk.remove()
            pred = out[0] if isinstance(out, (tuple, list)) else out
            pred = pred[0] if pred.dim() == 3 else pred
            target = pred[4:, :].max(); act = st["a"]
            g = torch.autograd.grad(target, act)[0]
        cam = torch.relu((g.mean((2, 3), keepdim=True) * act).sum(1, keepdim=True))[0, 0].detach().cpu().numpy()
        cam = cv2.resize(cam, (IMGSZ, IMGSZ))[t:t+nh, l:l+nw]
        cam = cv2.resize(cam, (img.shape[1], img.shape[0])); return (cam - cam.min()) / (cam.max() - cam.min() + 1e-9)

    outd = cfg.RES / f"gradcam_{name}"; outd.mkdir(parents=True, exist_ok=True); grows = []
    for s in splits:
        fl = [f for f in sorted((cfg.DATA / "images" / s).glob("*.jpg")) if "_dup" not in f.name]
        random.Random(cfg.SEED).shuffle(fl); n = 0
        for f in fl:
            img = cv2.imread(str(f)); H, W = img.shape[:2]
            gts = read_gt(cfg.DATA / "labels" / s / f"{f.stem}.txt", W, H)
            if not gts: continue
            cam = gradcam(img); m = np.zeros((H, W), bool)
            for _, b in gts: m[max(0, int(b[1])):int(b[3]), max(0, int(b[0])):int(b[2])] = True
            e, fr = cam[m].sum() / (cam.sum() + 1e-9), m.mean()
            grows.append(dict(split=s, energia_en_caja=e, fraccion_area=fr, enriquecimiento=e / max(fr, 1e-6)))
            if n < n_vis:
                ov = cv2.addWeighted(img, .55, cv2.applyColorMap((cam*255).astype("uint8"), cv2.COLORMAP_JET), .45, 0)
                for _, b in gts: cv2.rectangle(ov, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (255, 255, 255), 3)
                cv2.imwrite(str(outd / f"{s}_{classes[gts[0][0]]}_{f.stem}.jpg"), ov)
            n += 1
            if n >= n_cam: break
        log(f"  Grad-CAM {s}: {n} imágenes")
    tab5 = pd.DataFrame(grows).groupby("split").mean()
    log("Tabla 5 — Grad-CAM:\n" + str(tab5.round(3))); tab5.to_csv(cfg.RES / f"tabla5_gradcam_{name}.csv")

    imgs = sorted(outd.glob("*.jpg"))
    sel = [f for f in imgs if f.name.startswith("val_cis")][:3] + [f for f in imgs if f.name.startswith("val_trans")][:3]
    if sel:
        fig, axs = plt.subplots(2, 3, figsize=(13, 6))
        for ax, f in zip(axs.ravel(), sel):
            ax.imshow(cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB)); ax.set_title(f.stem[:40], fontsize=8); ax.axis("off")
        plt.tight_layout(); plt.savefig(cfg.RES / f"gradcam_galeria_{name}.png", dpi=150); plt.close(fig)
    return tab5
