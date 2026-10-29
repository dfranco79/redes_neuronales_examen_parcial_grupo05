"""Métricas de la clase agua sobre los píxeles válidos.

- IoU agregado: TP / (TP+FP+FN) sumando los píxeles de todos los chips.
- IoU promedio por chip: media del IoU de cada chip (la métrica de los autores de Sen1Floods11); los chips sin agua
  real ni predicha (unión vacía) no entran al promedio.
- Omisión = FN/(TP+FN) = 1 - recall; comisión = FP/(TP+FP) = 1 - precisión.
"""
import numpy as np
import pandas as pd
import torch


def scores(tp, fp, fn, tn):
    p, r = tp / (tp + fp + 1e-9), tp / (tp + fn + 1e-9)
    return dict(iou=tp / (tp + fp + fn + 1e-9), f1=2 * p * r / (p + r + 1e-9), precision=p, recall=r,
                omission=fn / (tp + fn + 1e-9), commission=fp / (tp + fp + 1e-9),
                accuracy=(tp + tn) / max(tp + fp + fn + tn, 1),
                tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn))


@torch.no_grad()
def collect(model, loader, device):
    """Guarda las probabilidades de cada chip para poder probar varios umbrales sin volver a inferir."""
    model.eval()
    out = []
    for img, lbl, valid, cids in loader:
        prob = torch.sigmoid(model(img.to(device)).float()).squeeze(1).cpu().numpy()
        for j, cid in enumerate(cids):
            out.append(dict(cid=cid, prob=prob[j].astype(np.float16), gt=(lbl[j] == 1).numpy(),
                            valid=valid[j].numpy(), img=img[j].numpy().astype(np.float16)))
    return out


def evaluate_chips(chips, th=0.5):
    TP = FP = FN = TN = 0
    rows = []
    for c in chips:
        v, g = c["valid"], c["gt"] & c["valid"]
        pr = (c["prob"] >= th) & v
        tp, fp = int((pr & g).sum()), int((pr & ~g).sum())
        fn, tn = int((~pr & g).sum()), int((~pr & ~g & v).sum())
        TP, FP, FN, TN = TP + tp, FP + fp, FN + fn, TN + tn
        u = tp + fp + fn
        rows.append(dict(cid=c["cid"], iou=tp / u if u else np.nan, tp=tp, fp=fp, fn=fn, tn=tn,
                         valid_px=int(v.sum()), water_frac=g.sum() / max(v.sum(), 1)))
    res = scores(TP, FP, FN, TN)
    per = pd.DataFrame(rows)
    res["mean_iou_chips"] = float(np.nanmean(per["iou"])) if per["iou"].notna().any() else float("nan")
    return res, per


def sweep_threshold(chips, ths=np.round(np.arange(0.10, 0.91, 0.05), 2)):
    rows = []
    for t in ths:
        r, _ = evaluate_chips(chips, t)
        rows.append(dict(th=float(t), f1=r["f1"], iou=r["iou"], precision=r["precision"], recall=r["recall"]))
    df = pd.DataFrame(rows)
    return float(df.loc[df.f1.idxmax(), "th"]), df


def bootstrap_iou(per, n=2000, seed=0):
    """IC 95 % del IoU agregado remuestreando chips con reemplazo."""
    rng = np.random.default_rng(seed)
    tp, fp, fn = per.tp.values, per.fp.values, per.fn.values
    vals = []
    for _ in range(n):
        k = rng.integers(0, len(per), len(per))
        vals.append(tp[k].sum() / max(tp[k].sum() + fp[k].sum() + fn[k].sum(), 1))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))
