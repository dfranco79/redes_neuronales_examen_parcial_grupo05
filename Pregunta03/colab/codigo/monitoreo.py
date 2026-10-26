# -*- coding: utf-8 -*-
"""monitoreo.py — Módulo de monitoreo ecológico por reglas (Tabla 6).

Sobre una cámara NO vista (test_trans): detecta con umbral de confianza, agrupa por especie y periodo y cuenta un
*evento* cuando una secuencia tiene la especie en >= min_frames fotogramas.
"""
import pandas as pd
import colab_comun as C

def monitoreo(cfg, model, exp, split="test_trans", conf=0.25, min_frames=2, periodo="D", loc=None, lote=32, log=print):
    """periodo: 'H' hora, 'D' día, 'M' mes. loc=None -> la cámara con más imágenes. Devuelve (tab6, act)."""
    import matplotlib; import matplotlib.pyplot as plt
    name = C.nombre_exp(cfg, exp); classes = C.cargar_clases(cfg)
    meta = C.cargar_meta(cfg); meta = meta[meta.split == split]
    loc = str(loc) if loc is not None else meta.location.astype(str).value_counts().index[0]
    cam_df = meta[meta.location.astype(str) == loc].sort_values(["dt", "frame"]).reset_index(drop=True)
    log(f"Cámara: {loc} | imágenes: {len(cam_df)}")

    recs = []
    fl = [str(cfg.DATA / "images" / split / f"{i}.jpg") for i in cam_df.id]
    for i in range(0, len(fl), lote):
        for j, pr in enumerate(model.predict(fl[i:i+lote], imgsz=cfg.IMGSZ, conf=conf, iou=0.6, device=cfg.DEVICE, verbose=False)):
            r = cam_df.iloc[i + j]
            for c, s_ in zip(pr.boxes.cls, pr.boxes.conf):
                recs.append(dict(id=r["id"], dt=r["dt"], seq_id=r["seq_id"], especie=classes[int(c)], conf=float(s_)))
    L = pd.DataFrame(recs)
    n_img, n_fauna = len(cam_df), L.id.nunique() if len(L) else 0
    log(str(dict(imagenes=n_img, con_fauna=n_fauna, vacias=n_img - n_fauna, tasa_vacias=round((n_img - n_fauna) / max(n_img, 1), 3))))
    if not len(L):
        log("Sin detecciones."); return None, None
    ev = L.groupby(["seq_id", "especie"]).id.nunique().reset_index(name="frames")
    ev = ev[ev.frames >= min_frames].groupby("especie").size().rename("eventos")
    tab6 = L.groupby("especie").agg(detecciones=("conf", "size"), fotogramas=("id", "nunique"), conf_media=("conf", "mean"),
                                    primera=("dt", "min"), ultima=("dt", "max")).join(ev).fillna({"eventos": 0})
    tab6["eventos"] = tab6.eventos.astype(int)
    tab6 = tab6.sort_values("detecciones", ascending=False)
    log("Tabla 6 — monitoreo por especie:\n" + str(tab6.round({"conf_media": 3}))); tab6.to_csv(cfg.RES / f"tabla6_monitoreo_{name}.csv")
    L["periodo"] = L["dt"].dt.to_period(periodo)
    act = L.pivot_table(index="periodo", columns="especie", values="id", aggfunc="count", fill_value=0)
    act.to_csv(cfg.RES / f"monitoreo_por_periodo_{name}.csv")
    ax = act.plot(figsize=(10, 4), title=f"Actividad por periodo — cámara {loc}")
    plt.ylabel("detecciones"); plt.tight_layout(); plt.savefig(cfg.RES / f"monitoreo_actividad_{name}.png", dpi=150); plt.close(ax.figure)
    return tab6, act
