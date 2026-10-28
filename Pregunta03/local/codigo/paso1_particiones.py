# -*- coding: utf-8 -*-
"""PASO 1 — Lee los 3 JSON de LILA, indexa las imágenes y construye las particiones cis/trans.

Salida en <work>/: particiones.csv, anotaciones.csv, categorias.json
    python codigo/paso1_particiones.py
"""
import argparse, json, random
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image
import cct_comun as C

def ejecutar(cfg, log=print):
    rng = random.Random(cfg.seed)
    log("Leyendo JSON ...")
    C.comprobar_entradas(cfg)
    bb = json.load(open(cfg.json_dir / cfg.bbox_json, encoding="utf-8"))
    sp = json.load(open(cfg.json_dir / cfg.splits_json, encoding="utf-8"))
    ip = cfg.json_dir / cfg.images_json
    im_js = json.load(open(ip, encoding="utf-8")) if ip.exists() else None
    if im_js is None:
        log("AVISO: no se encontró el JSON de imágenes; no habrá imágenes vacías adicionales.")

    cats = {c["id"]: c["name"] for c in bb["categories"]}
    empty_cat = {i for i, n in cats.items() if n.lower() == "empty"}
    img_bb = C.norm_images(bb["images"])
    ann = pd.DataFrame(bb["annotations"]); ann["image_id"] = ann["image_id"].astype(str)
    ann = ann[~ann["category_id"].isin(empty_cat)]
    ann = ann[ann["bbox"].notna()].copy()
    log(f"bboxes: {len(img_bb)} imágenes, {len(ann)} cajas, {img_bb.location.nunique()} ubicaciones")

    pos = img_bb[img_bb.id.isin(set(ann.image_id))].copy()
    empty = pd.DataFrame(columns=pos.columns)
    if im_js:
        img_all = C.norm_images(im_js["images"])
        ann_all = pd.DataFrame(im_js["annotations"]); ann_all["image_id"] = ann_all["image_id"].astype(str)
        cats_all = {c["id"]: c["name"] for c in im_js["categories"]}
        empty_ids = set(ann_all[ann_all.category_id.map(cats_all).str.lower() == "empty"].image_id)
        meta = img_all.set_index("id")
        for col in ["dt", "seq_id", "frame", "location"]:
            miss = pos[col].isna() | pos[col].astype(str).isin(["None", "nan", "NaT"])
            if miss.any():
                pos.loc[miss, col] = pos.loc[miss, "id"].map(meta[col])
        empty = img_all[img_all.id.isin(empty_ids) & img_all.location.isin(set(pos.location))].copy()
        log(f"vacías disponibles en esas ubicaciones: {len(empty)}")

    log(f"Indexando .jpg bajo {cfg.img_dir} (puede tardar varios minutos en un disco externo) ...")
    index = C.indexar_jpg(cfg.img_dir, excluir=cfg.work)
    log(f"{len(index)} imágenes .jpg encontradas")
    if not index:
        raise FileNotFoundError(f"No hay .jpg bajo {cfg.img_dir}. Revisa --base / --img-dir.")
    for df in (pos, empty):
        df["path"] = df.file.str.lower().map(index)
    for nombre, df in (("con animal", pos), ("vacías", empty)):
        log(f"{nombre}: {len(df)} en JSON, {int(df.path.isna().sum())} no están en disco")
    pos, empty = pos[pos.path.notna()].copy(), empty[empty.path.notna()].copy()
    pos["has_animal"], empty["has_animal"] = True, False

    S = sp.get("splits", sp)
    tr_key, va_key = cfg.split_keys
    tr_set, va_set = set(map(str, S[tr_key])), set(map(str, S[va_key]))
    all_imgs = pd.concat([pos, empty], ignore_index=True)
    locs, ids = set(all_imgs.location), set(all_imgs.id)
    log(f"{tr_key}: {len(tr_set)} únicos | {va_key}: {len(va_set)} únicos | comunes: {len(tr_set & va_set)}")
    key = "location" if len((tr_set | va_set) & locs) >= len((tr_set | va_set) & ids) else "id"
    va_set -= tr_set
    col = all_imgs[key].astype(str)
    all_imgs["split_raw"] = np.where(col.isin(tr_set), "train", np.where(col.isin(va_set), "trans", None))
    all_imgs = all_imgs[all_imgs.split_raw.notna()].copy()

    # trans: ubicaciones nuevas, repartidas POR UBICACIÓN entre validación y prueba
    all_imgs["split"] = all_imgs.split_raw.map({"train": "train"})
    tl = sorted(all_imgs[all_imgs.split_raw == "trans"].location.unique()); rng.shuffle(tl)
    nv = max(1, round(len(tl) * cfg.trans_val_frac)) if len(tl) > 1 else len(tl)
    val_locs, test_locs = set(tl[:nv]), set(tl[nv:])
    all_imgs.loc[all_imgs.location.isin(val_locs), "split"] = "val_trans"
    all_imgs.loc[all_imgs.location.isin(test_locs), "split"] = "test_trans"
    log(f"val_trans: {sorted(val_locs)} | test_trans: {sorted(test_locs)}")
    # cis: secuencias completas reservadas de las ubicaciones de entrenamiento
    seqs = sorted(all_imgs[all_imgs.split == "train"].seq_id.unique()); rng.shuffle(seqs)
    n = int(len(seqs) * cfg.cis_frac)
    cis_val, cis_test = set(seqs[:n]), set(seqs[n:2 * n])
    all_imgs.loc[(all_imgs.split == "train") & all_imgs.seq_id.isin(cis_val),  "split"] = "val_cis"
    all_imgs.loc[(all_imgs.split == "train") & all_imgs.seq_id.isin(cis_test), "split"] = "test_cis"
    data = all_imgs.dropna(subset=["split"]).copy()
    if not len(data):
        raise RuntimeError("No quedó ninguna imagen: revisa la ruta de imágenes y las claves de splits.")

    def sub_empty(df, ratio):
        p, e = df[df.has_animal], df[~df.has_animal]
        if ratio >= 0:
            e = e.sample(n=min(len(e), int(ratio * len(p))), random_state=cfg.seed)
        return pd.concat([p, e])
    data = pd.concat([sub_empty(g, cfg.empty_ratio if s == "train" else cfg.eval_empty_ratio)
                      for s, g in data.groupby("split")], ignore_index=True)

    for i in data.index[data.W.isna() | data.H.isna()]:      # W/H faltantes -> leer de la imagen
        with Image.open(data.at[i, "path"]) as im:
            data.at[i, "W"], data.at[i, "H"] = im.size

    ann = ann.merge(data[["id", "split", "file", "W", "H"]], left_on="image_id", right_on="id", how="inner")
    ann["especie"] = ann.category_id.map(cats)
    log("\nImágenes por partición:\n" + str(data.groupby(["split", "has_animal"]).size().unstack(fill_value=0)))
    C.guardar_estado(cfg, data, ann, cats)
    log(f"\nGuardado: {cfg.work / 'particiones.csv'}")
    return data, ann, cats

def main():
    ap = C.agregar_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    ejecutar(C.resolver(ap.parse_args()))

if __name__ == "__main__":
    main()
