"""Análisis y figuras: ruido de las etiquetas débiles, composición de los conjuntos, tipos de error del SAR
y nubes en Sentinel-2."""
import json
from pathlib import Path

import cv2
import matplotlib
import rasterio
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .data import event_of, load_s2, read_label, to_db  # noqa: E402

DARK_DB = -15.0     # VV por debajo: retrodispersión típica de agua abierta o superficies lisas
BRIGHT_DB = -12.0   # VV por encima: demasiado brillante para agua abierta (doble rebote, vegetación)
MID_DB = -16.0      # entre MID_DB y BRIGHT_DB: retrodispersión intermedia (agua rugosa o turbia)
NEAR_PX = 5         # 5 px = 50 m del borde del agua real


# --- etiquetas débiles ---
def weak_label_noise(paths, ids):
    """Compara la etiqueta Otsu (S1OtsuLabelHand) con la dorada en los mismos chips."""
    rows = []
    for c in ids:
        po, pg = Path(paths.otsu_hand) / f"{c}_S1OtsuLabelHand.tif", Path(paths.lbl) / f"{c}_LabelHand.tif"
        if not po.exists():
            continue
        g, o = read_label(pg), read_label(po)
        v = (g != -1) & (o != -1)
        gt, ot = (g == 1) & v, (o == 1) & v
        rows.append(dict(cid=c, event=event_of(c), tp=int((gt & ot).sum()), fp=int((~gt & ot & v).sum()),
                         fn=int((gt & ~ot).sum()), agree=int(((gt == ot) & v).sum()), n=int(v.sum())))
    if not rows:
        return None
    df = pd.DataFrame(rows)

    def summ(d):
        tp, fp, fn = d.tp.sum(), d.fp.sum(), d.fn.sum()
        p, r = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        return pd.Series(dict(chips=len(d), iou=tp / max(tp + fp + fn, 1), precision=p, recall=r,
                              f1=2 * p * r / max(p + r, 1e-9), acuerdo_pixel=d.agree.sum() / max(d.n.sum(), 1)))

    overall = summ(df).to_dict()
    per_event = pd.DataFrame([dict(event=e, **summ(d)) for e, d in df.groupby("event")])
    return dict(overall=overall, per_event=per_event)


# --- composición de los conjuntos ---
def composition(ds, per, name):
    vv_w, vv_l = [], []
    for img, lbl, valid in ds.data:
        db = to_db(img[0])
        vv_w.append(db[(lbl == 1) & valid])
        vv_l.append(db[(lbl == 0) & valid])
    vv_w, vv_l = np.concatenate(vv_w), np.concatenate(vv_l)
    med_w = float(np.median(vv_w)) if vv_w.size else float("nan")
    return dict(conjunto=name, chips=len(ds), eventos=len({event_of(c) for c in ds.ids}),
                frac_agua=float(per.tp.add(per.fn).sum() / max(per.valid_px.sum(), 1)),
                chips_sin_agua=int((per.water_frac == 0).sum()),
                vv_agua_db=med_w, vv_tierra_db=float(np.median(vv_l)), contraste_db=float(np.median(vv_l)) - med_w,
                agua_brillante=float((vv_w > BRIGHT_DB).mean()) if vv_w.size else float("nan"))


def per_event_table(per):
    d = per.assign(event=per.cid.map(event_of)).groupby("event")[["tp", "fp", "fn"]].sum()
    d["iou"] = d.tp / (d.tp + d.fp + d.fn).clip(lower=1)
    d["chips"] = per.assign(event=per.cid.map(event_of)).groupby("event").size()
    return d.reset_index().sort_values("iou")


def load_metadata(path):
    p = Path(path)
    if not p.exists():
        return {}
    feats = json.loads(p.read_text())["features"]
    meta = {f["properties"]["location"]: f["properties"] for f in feats}
    if "Cambodia" in meta:            # los chips del evento de Camboya se llaman "Mekong_<id>"
        meta.setdefault("Mekong", meta["Cambodia"])
    return meta


# --- tipos de error ---
_K = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * NEAR_PX + 1, 2 * NEAR_PX + 1))


def _masks(chip, th):
    v, g = chip["valid"], chip["gt"] & chip["valid"]
    pr = (chip["prob"] >= th) & v
    vv = to_db(chip["img"][0])
    g8 = g.astype(np.uint8)
    near_out = cv2.dilate(g8, _K).astype(bool) & ~g          # franja de 50 m fuera del agua
    near_in = g & ~cv2.erode(g8, _K).astype(bool)             # franja de 50 m dentro del agua
    fp, fn = pr & ~g & v, ~pr & g
    return dict(v=v, g=g, pr=pr, vv=vv, tp=pr & g, fp=fp, fn=fn,
                fp_far_dark=fp & ~near_out & (vv < DARK_DB), fp_edge=fp & near_out,
                fn_bright=fn & (vv > BRIGHT_DB), fn_edge=fn & near_in & ~(vv > BRIGHT_DB),
                fn_mid=fn & ~near_in & (vv > MID_DB) & (vv <= BRIGHT_DB),
                fn_dark=fn & ~near_in & (vv <= MID_DB))


def error_taxonomy(chips, th):
    rows, tot = [], {}
    for c in chips:
        m = _masks(c, th)
        r = dict(cid=c["cid"], water_frac=m["g"].sum() / max(m["v"].sum(), 1))
        for k in ("tp", "fp", "fn", "fp_far_dark", "fp_edge", "fn_bright", "fn_edge", "fn_mid", "fn_dark"):
            r[k] = int(m[k].sum())
            tot[k] = tot.get(k, 0) + r[k]
        rows.append(r)
    df = pd.DataFrame(rows)
    fp, fn = max(tot["fp"], 1), max(tot["fn"], 1)
    breakdown = {
        f"FP oscuros (VV<{DARK_DB:.0f} dB) lejos del agua: sombra/superficie lisa": tot["fp_far_dark"] / fp,
        "FP en el borde del agua (suelo húmedo/píxel mixto)": tot["fp_edge"] / fp,
        f"FP lejos del agua con VV>={DARK_DB:.0f} dB (contexto/agua no etiquetada)": 1 - (tot["fp_far_dark"] + tot["fp_edge"]) / fp,
        f"FN brillantes VV>{BRIGHT_DB:.0f} dB (doble rebote/vegetación inundada)": tot["fn_bright"] / fn,
        "FN en el borde del agua": tot["fn_edge"] / fn,
        f"FN interiores con VV intermedio ({MID_DB:.0f} a {BRIGHT_DB:.0f} dB): agua rugosa/turbia": tot["fn_mid"] / fn,
        f"FN interiores oscuros (VV<={MID_DB:.0f} dB): firma de agua no aprovechada": tot["fn_dark"] / fn,
    }
    return df, breakdown, tot


def select_cases(tax, min_px=30, min_water=0.01):
    """Un chip distinto por tipo de error: el que tiene más píxeles de ese tipo (>= min_px si hay alguno)."""
    used, cases = set(), {}

    def pick(df, col, key):
        df = df[~df.cid.isin(used) & (df[col] > 0)].sort_values(col, ascending=False)
        if len(df):
            good = df[df[col] >= min_px]
            cid = (good if len(good) else df).cid.iloc[0]
            used.add(cid)
            cases[key] = cid

    pick(tax, "fp_far_dark", "A")
    pick(tax[tax.water_frac >= min_water], "fn_bright", "B")
    pick(tax.assign(edge=tax.fp_edge + tax.fn_edge), "edge", "C")
    pick(tax[tax.water_frac >= min_water], "fn_mid", "D")
    return cases


def _nd(a, b):
    return (a - b) / (a + b + 1e-6)


CASE_GROUP = {"A": "fp_far_dark", "B": "fn_bright", "C": "edge", "D": "fn_mid"}


def case_stats(chip, th, s2_path=None, key=None):
    """Mediciones del caso; 'cat' son los píxeles del tipo de error que define el caso (p. ej. FP oscuros lejanos)."""
    m = _masks(chip, th)
    m["edge"] = m["fp_edge"] | m["fn_edge"]
    m["tn"] = m["v"] & ~m["g"] & ~m["pr"]
    m["cat"] = m[CASE_GROUP.get(key, "fp")]
    med = lambda x, k: float(np.median(x[m[k]])) if m[k].any() else float("nan")  # noqa: E731
    out = dict(cid=chip["cid"], group=CASE_GROUP.get(key), vv_tp=med(m["vv"], "tp"), vv_tn=med(m["vv"], "tn"),
               vv_fp=med(m["vv"], "fp"), vv_fn=med(m["vv"], "fn"), vv_cat=med(m["vv"], "cat"),
               n_fp=int(m["fp"].sum()), n_fn=int(m["fn"].sum()), n_tp=int(m["tp"].sum()), n_cat=int(m["cat"].sum()))
    for k in ("fp_far_dark", "fp_edge", "fn_bright", "fn_edge", "fn_mid", "fn_dark"):
        out["n_" + k] = int(m[k].sum())
    if s2_path is not None and Path(s2_path).exists():
        s2 = load_s2(s2_path)
        ndvi, mndwi = _nd(s2[7], s2[3]), _nd(s2[2], s2[11])
        for grp in ("cat", "tp", "tn"):
            out[f"ndvi_{grp}"] = float(np.median(ndvi[m[grp]])) if m[grp].any() else float("nan")
            out[f"mndwi_{grp}"] = float(np.median(mndwi[m[grp]])) if m[grp].any() else float("nan")
    return out


def cloud_fraction(s2_path):
    """Píxel nube = muy brillante en el azul (B2 > 0.2) y en el visible (> 0.25). S2Hand no trae máscara QA."""
    s2 = load_s2(s2_path)
    return float(((s2[1] > 0.2) & (s2[[1, 2, 3]].mean(0) > 0.25)).mean())


# --- figuras ---
def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def fig_curves(histories, path):
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
    for name, h in histories.items():
        if not h:
            continue
        h = pd.DataFrame(h)
        ax[0].plot(h.epoch, h.train_loss, label=name)
        ax[1].plot(h.epoch, h.val_iou, label=name)
    ax[0].set_title("Pérdida de entrenamiento"); ax[1].set_title("IoU en validación dorada (umbral 0.5)")
    for a in ax:
        a.set_xlabel("época"); a.grid(alpha=.3); a.legend(fontsize=7)
    return _save(fig, path)


def fig_sweep(df, th, path):
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for k in ("f1", "iou", "precision", "recall"):
        ax.plot(df.th, df[k], marker="o", ms=3, label=k)
    ax.axvline(th, ls="--", c="k", lw=1); ax.text(th, 0.02, f" umbral={th:.2f}")
    ax.set_xlabel("umbral"); ax.set_title("Barrido de umbral (validación dorada)"); ax.grid(alpha=.3); ax.legend()
    return _save(fig, path)


def fig_confusion(r, th, title, path):
    cm = np.array([[r["tn"], r["fp"]], [r["fn"], r["tp"]]], float)
    pct = cm / cm.sum(1, keepdims=True).clip(min=1) * 100
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(pct, cmap="Blues", vmin=0, vmax=100)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{int(cm[i, j]):,}\n({pct[i, j]:.1f}%)", ha="center", va="center",
                    color="white" if pct[i, j] > 50 else "black")
    ax.set_xticks([0, 1], ["Pred. no-agua", "Pred. agua"]); ax.set_yticks([0, 1], ["Real no-agua", "Real agua"])
    ax.set_title(f"{title} (umbral {th:.2f})")
    return _save(fig, path)


def fig_bars(df, path, title, metrics=("iou", "mean_iou_chips", "f1")):
    fig, ax = plt.subplots(figsize=(1.6 + 1.9 * len(df), 3.8))
    x, w = np.arange(len(df)), 0.8 / len(metrics)
    for i, k in enumerate(metrics):
        b = ax.bar(x + i * w, df[k], w, label=k)
        ax.bar_label(b, fmt="%.3f", fontsize=7)
    ax.set_xticks(x + w * (len(metrics) - 1) / 2, df["modelo"], rotation=12, ha="right", fontsize=8)
    ax.set_ylim(0, 1); ax.set_title(title); ax.legend(fontsize=8); ax.grid(axis="y", alpha=.3)
    return _save(fig, path)


def fig_gap(rows, path):
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(1.6 + 1.9 * len(df), 3.8))
    x = np.arange(len(df))
    b1 = ax.bar(x - .2, df.test, .4, label="Test (eventos vistos)")
    b2 = ax.bar(x + .2, df.bolivia, .4, label="Bolivia (evento no visto)")
    ax.bar_label(b1, fmt="%.3f", fontsize=7); ax.bar_label(b2, fmt="%.3f", fontsize=7)
    ax.set_xticks(x, df.modelo, rotation=12, ha="right", fontsize=8); ax.set_ylim(0, 1)
    ax.set_ylabel("IoU agregado"); ax.set_title("Generalización geográfica"); ax.legend(fontsize=8)
    return _save(fig, path)


def s2_rgb(s2_path):
    if s2_path is None or not Path(s2_path).exists():
        return None
    s2 = load_s2(s2_path)
    return np.clip(np.stack([s2[3], s2[2], s2[1]], -1) * 3.5, 0, 1)


def fig_case(chip, th, title, path, s2_path=None):
    m = _masks(chip, th)
    err = np.zeros((*m["g"].shape, 3), np.uint8)
    err[m["v"] & ~m["g"] & ~m["pr"]] = (60, 60, 60)
    err[m["tp"]] = (0, 190, 0); err[m["fn"]] = (230, 0, 0); err[m["fp"]] = (255, 165, 0)
    rgb = s2_rgb(s2_path)
    vh = to_db(chip["img"][1])
    fig, ax = plt.subplots(1, 5, figsize=(23, 4.6))
    ax[0].imshow(m["vv"], cmap="gray", vmin=-25, vmax=0); ax[0].set_title("Sentinel-1 VV (dB)")
    ax[1].imshow(vh, cmap="gray", vmin=-30, vmax=-5); ax[1].set_title("Sentinel-1 VH (dB)")
    ax[2].imshow(rgb if rgb is not None else np.zeros_like(err)); ax[2].set_title("Sentinel-2 RGB (contexto)")
    ax[3].imshow(err); ax[3].set_title("verde TP · rojo FN · naranja FP · gris TN · negro sin dato", fontsize=10)
    for a in ax[:4]:
        a.axis("off")
    for k, lab, col in (("tp", "TP", "green"), ("fn", "FN", "red"), ("fp", "FP", "orange")):
        if m[k].sum() > 20:
            ax[4].hist(m["vv"][m[k]], bins=60, range=(-35, 5), density=True, histtype="step", lw=1.5, color=col,
                       label=f"{lab} ({m[k].sum():,} px)")
    tn = m["v"] & ~m["g"] & ~m["pr"]
    if tn.sum() > 20:
        ax[4].hist(m["vv"][tn], bins=60, range=(-35, 5), density=True, histtype="step", color="gray", label="TN")
    ax[4].axvline(DARK_DB, ls=":", c="k"); ax[4].axvline(BRIGHT_DB, ls="--", c="k")
    ax[4].set_xlabel("VV (dB)"); ax[4].legend(fontsize=8); ax[4].set_title("Retrodispersión por tipo de píxel")
    fig.suptitle(f"{title} — {chip['cid']}", fontsize=13)
    return _save(fig, path)


def fig_examples(chips, th, path, n=6, title="Predicciones"):
    chips = sorted(chips, key=lambda c: -c["gt"].sum())[:n]
    fig, ax = plt.subplots(len(chips), 3, figsize=(10, 3.3 * len(chips)))
    ax = np.atleast_2d(ax)
    for i, c in enumerate(chips):
        m = _masks(c, th)
        err = np.zeros((*m["g"].shape, 3), np.uint8)
        err[m["v"] & ~m["g"] & ~m["pr"]] = (60, 60, 60)
        err[m["tp"]] = (0, 190, 0); err[m["fn"]] = (230, 0, 0); err[m["fp"]] = (255, 165, 0)
        ax[i, 0].imshow(m["vv"], cmap="gray", vmin=-25, vmax=0); ax[i, 0].set_title(f"{c['cid']} — VV")
        ax[i, 1].imshow(np.where(m["v"], m["g"], np.nan), cmap="Blues", vmin=0, vmax=1); ax[i, 1].set_title("Etiqueta dorada")
        ax[i, 2].imshow(err); ax[i, 2].set_title("Errores (verde TP, rojo FN, naranja FP, negro sin dato)", fontsize=9)
        for a in ax[i]:
            a.axis("off")
    fig.suptitle(title)
    return _save(fig, path)


def fig_breakdown(breakdown, path):
    k, v = list(breakdown), list(breakdown.values())
    fig, ax = plt.subplots(figsize=(8, 3.6))
    cols = ["#e69f00"] * 3 + ["#d55e00"] * 4
    b = ax.barh(k[::-1], [x * 100 for x in v[::-1]], color=cols[::-1])
    ax.bar_label(b, fmt="%.1f%%", fontsize=8)
    ax.set_xlabel("% de los FP (naranja) o de los FN (rojo) del test"); ax.set_title("Taxonomía de errores")
    ax.tick_params(axis="y", labelsize=8)
    return _save(fig, path)


# --- inventario del dataset ---
EXPECTED = dict(eventos=11, dorados=446, debiles=4385, s2_bandas=13, size=512, res_m=10)


def _raster_props(path):
    with rasterio.open(path) as src:
        a = src.read()
        res_m = abs(src.transform.e) * 111320        # tamaño N-S del píxel (EPSG:4326, de grados a metros)
        return dict(count=src.count, dtype=str(src.dtypes[0]), h=src.height, w=src.width, res_m=float(res_m),
                    crs=str(src.crs), desc=src.descriptions, arr=a)


def _weak_status(n, expected):
    if n == expected:
        return "✔"
    return f"≈ ({n - expected:+d}, ver nota)" if abs(n - expected) <= 1 else "⚠ difiere"


def inventory(P, S, s1m, lblm, s2m, otsum, w1, wl1, wl2, weak_used):
    """Compara las características del dataset que da el enunciado con lo que hay en los datos descargados."""
    E = EXPECTED
    bucket = json.loads(Path(P.inventory).read_text()) if Path(P.inventory).exists() else {}
    cnt = lambda layer, local: bucket.get(layer, {}).get("count", local)  # noqa: E731
    gold = sorted(set(s1m) & set(lblm))
    events = sorted({event_of(c) for c in gold})
    weak_bucket = Path(P.weak_bucket).read_text().split() if Path(P.weak_bucket).exists() else sorted(w1)
    labels_vals = set()
    for c in gold:
        with rasterio.open(lblm[c]) as src:
            labels_vals |= set(np.unique(src.read(1)).tolist())
    r1 = _raster_props(s1m[gold[0]])
    vv = r1["arr"][0][np.isfinite(r1["arr"][0])]
    r2 = _raster_props(s2m[gold[0]]) if s2m else None
    bol_all = [c for c in gold if c.startswith("Bolivia")]
    ok = lambda cond: "✔" if cond else "⚠ difiere"  # noqa: E731
    wb, md = bucket.get("weak", {}), bucket.get("metadata", {})
    md_txt = f" (la metadata oficial suma {md['chips_debiles']:,})" if md.get("chips_debiles") else ""
    n1, n2 = cnt("S1OtsuLabelWeak", len(wl1)), cnt("S2IndexLabelWeak", len(wl2))
    otros2 = bucket.get("S2IndexLabelWeak", {}).get("otros", [])
    s2w = bucket.get("S2Weak", {})
    rows = [
        dict(Característica="Eventos de inundación (chips dorados)", Enunciado=f"{E['eventos']}",
             Encontrado=f"{len(events)}: {', '.join(events)}", Estado=ok(len(events) == E["eventos"])),
        dict(Característica="Bandas Sentinel-1", Enunciado="VV y VH, en dB",
             Encontrado=f"{r1['count']} bandas" + (f" ({', '.join(d for d in r1['desc'] if d)})" if any(r1["desc"]) else "")
                        + f", float32 en dB, VV mediano {np.median(vv):.1f} dB",
             Estado=ok(r1["count"] == 2 and -40 < np.median(vv) < 5)),
        dict(Característica="Bandas Sentinel-2", Enunciado=f"{E['s2_bandas']} bandas espectrales",
             Encontrado=(f"{r2['count']} bandas ({r2['desc'][0]} a {r2['desc'][-1]}, incluye B8A), reflectancia TOA x 10000"
                         if r2 and r2["desc"][0] else f"{r2['count']} bandas") if r2 else "no descargado",
             Estado=ok(r2 is not None and r2["count"] == 13)),
        dict(Característica="Tamaño y resolución de los chips", Enunciado="512×512 px a 10 m, GeoTIFF",
             Encontrado=f"{r1['h']}×{r1['w']} px, píxel de {r1['res_m']:.1f} m ({r1['crs']}), GeoTIFF",
             Estado=ok(r1["h"] == r1["w"] == E["size"] and abs(r1["res_m"] - 10) < 1.0)),
        dict(Característica="Chips con etiqueta dorada", Enunciado=f"{E['dorados']}",
             Encontrado=f"{cnt('LabelHand', len(gold))} en el bucket, {len(gold)} usados "
                        f"(train {len(S['train'])}, valid {len(S['val'])}, test {len(S['test'])}, Bolivia {len(S['bolivia'])})",
             Estado=ok(len(gold) == E["dorados"] and sum(len(v) for v in S.values()) == len(gold))),
        dict(Característica="Valores de la máscara dorada", Enunciado="agua / no agua / sin datos",
             Encontrado=f"{sorted(labels_vals)} (1 agua, 0 no agua, -1 sin datos)", Estado=ok(labels_vals <= {-1, 0, 1})),
        dict(Característica="Chips débiles derivados de Sentinel-1 (Otsu)", Enunciado=f"{E['debiles']:,}",
             Encontrado=f"{n1:,} GeoTIFF en el bucket{md_txt}; {len([c for c in weak_used if c in wl1]):,} usados",
             Estado=_weak_status(n1, E["debiles"])),
        dict(Característica="Chips débiles derivados de Sentinel-2 (índice)", Enunciado=f"{E['debiles']:,}",
             Encontrado=f"{n2:,} GeoTIFF en el bucket"
                        + (f" (+{len(otros2)} archivo no imagen: {', '.join(otros2)})" if otros2 else "")
                        + f"; {len([c for c in weak_used if c in wl2]):,} usados",
             Estado=_weak_status(n2, E["debiles"])),
        dict(Característica="Débiles excluidos por ser de Bolivia", Enunciado="(necesario para el entregable 4)",
             Encontrado=f"{wb.get('bolivia_excluidos', len([c for c in weak_bucket if c.startswith('Bolivia')])):,}",
             Estado="✔"),
        dict(Característica="Bolivia como prueba geográficamente independiente", Enunciado="todos los chips dorados de Bolivia",
             Encontrado=f"{len(S['bolivia'])} de {len(bol_all)} chips dorados de Bolivia están en el split de prueba; "
                        f"0 en train/valid", Estado=ok(sorted(S["bolivia"]) == bol_all)),
        dict(Característica="Métrica de referencia de los autores", Enunciado="IoU promedio sobre los chips + omisión y comisión",
             Encontrado="se reportan (además de IoU agregado, F1, precisión y recall)", Estado="✔"),
    ]
    if s2w.get("count"):
        rows.insert(8, dict(Característica="Imágenes Sentinel-2 de los chips débiles (S2Weak)", Enunciado="(no lo pide ningún entregable)",
                            Encontrado=f"{s2w['count']:,} en el bucket ({s2w['bytes'] / 1e9:.1f} GB); no se descargan porque "
                                       "los modelos con débiles usan SAR como entrada", Estado="—"))
    # tabla por evento
    split_of = {c: k for k, v in S.items() for c in v}
    ev_rows = []
    weak_ev = pd.Series([event_of(c) for c in weak_bucket]).value_counts() if weak_bucket else pd.Series(dtype=int)
    used_ev = pd.Series([event_of(c) for c in weak_used]).value_counts() if weak_used else pd.Series(dtype=int)
    for e in sorted(set(events) | set(weak_ev.index)):
        cs = [c for c in gold if event_of(c) == e]
        ev_rows.append(dict(evento=e, train=sum(split_of.get(c) == "train" for c in cs),
                            valid=sum(split_of.get(c) == "val" for c in cs), test=sum(split_of.get(c) == "test" for c in cs),
                            bolivia=sum(split_of.get(c) == "bolivia" for c in cs), dorados=len(cs),
                            debiles_bucket=int(weak_ev.get(e, 0)), debiles_usados=int(used_ev.get(e, 0))))
    return rows, pd.DataFrame(ev_rows)


def fig_events(ev, path):
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.2))
    x = np.arange(len(ev))
    bottom = np.zeros(len(ev))
    for col, c in (("train", "#4c72b0"), ("valid", "#55a868"), ("test", "#c44e52"), ("bolivia", "#8172b2")):
        ax[0].bar(x, ev[col], bottom=bottom, color=c, label=col)
        bottom += ev[col].values
    ax[0].set_xticks(x, ev.evento, rotation=40, ha="right"); ax[0].set_title(f"Chips dorados por evento ({int(ev.dorados.sum())})")
    ax[0].legend(fontsize=8)
    ax[1].bar(x - .2, ev.debiles_bucket, .4, color="#bbbbbb", label="en el bucket")
    ax[1].bar(x + .2, ev.debiles_usados, .4, color="#dd8452", label="usados")
    ax[1].set_xticks(x, ev.evento, rotation=40, ha="right")
    ax[1].set_title(f"Chips débiles por evento ({int(ev.debiles_bucket.sum()):,} en el bucket, {int(ev.debiles_usados.sum()):,} usados)")
    ax[1].legend(fontsize=8)
    return _save(fig, path)


def forbidden_imports(src_dir):
    """Busca imports de librerías de segmentación en src/ (el entregable 1 pide no usarlas)."""
    bad = ("segmentation_models", "torchvision.models.segmentation", "monai", "smp.", "mmseg", "detectron2")
    hits = []
    for f in Path(src_dir).rglob("*.py"):
        for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            if line.lstrip().startswith(("import", "from")) and any(b in line for b in bad):
                hits.append(f"{f.name}:{i}")
    return hits
