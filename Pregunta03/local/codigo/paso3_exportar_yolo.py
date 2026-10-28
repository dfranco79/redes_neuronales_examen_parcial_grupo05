# -*- coding: utf-8 -*-
"""PASO 3 — Exporta a formato YOLO con imágenes REDUCIDAS (lado mayor = --resize px). Reanudable.

Lee <work>/particiones.csv y anotaciones.csv (paso 1). Escribe <work>/yolo_data/:
images/<split>/*.jpg, labels/<split>/*.txt, meta.csv, anotaciones.csv, classes.json, export_info.json, *.yaml
    python codigo/paso3_exportar_yolo.py --resize 640
"""
import argparse, json, os, shutil, time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import numpy as np
from PIL import Image
import cct_comun as C

def _yolo_line(c, b, W, H):
    x1, y1, x2, y2 = max(0, b[0]), max(0, b[1]), min(W, b[0] + b[2]), min(H, b[1] + b[3])
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return f"{c} {(x1+x2)/2/W:.6f} {(y1+y2)/2/H:.6f} {(x2-x1)/W:.6f} {(y2-y1)/H:.6f}"

def _guardar_reducida(src, dst, max_side, quality):
    """Reduce src a max_side px (lado mayor) y la guarda en dst de forma atómica (0 = copiar sin reducir)."""
    tmp = str(dst) + ".part"
    with Image.open(src) as im:
        W, H = im.size
        if max_side and max(W, H) > max_side:
            sc = max_side / max(W, H)
            im.draft("RGB", (max(1, int(W * sc)), max(1, int(H * sc))))     # decodificación JPEG a escala reducida
            im = im.convert("RGB")
            im.thumbnail((max_side, max_side), Image.LANCZOS)
            im.save(tmp, "JPEG", quality=quality, optimize=True)
        else:
            shutil.copyfile(src, tmp)
    os.replace(tmp, dst)

def ejecutar(cfg, data=None, ann=None, log=print):
    if data is None:
        data, ann, _ = C.cargar_estado(cfg)
    data_dir = cfg.data_dir
    classes = sorted(ann.especie.unique()); cls_id = {n: i for i, n in enumerate(classes)}
    log(f"{len(classes)} clases: {classes}")

    lines = defaultdict(list)
    for r in ann.itertuples():
        ln = _yolo_line(cls_id[r.especie], r.bbox, r.W, r.H)
        if ln:
            lines[r.image_id].append((cls_id[r.especie], ln))

    factor = {}
    if cfg.oversample_max > 1:
        cnt = ann[ann.split == "train"].groupby("especie").image_id.nunique()
        med = float(np.median(cnt.values))
        cf = {cls_id[e]: int(min(cfg.oversample_max, max(1, round(med / n)))) for e, n in cnt.items()}
        for iid, l in lines.items():
            factor[iid] = max(cf[c] for c, _ in l)

    if cfg.reexport and data_dir.exists():
        log(f"--reexport -> borrando {data_dir}")
        shutil.rmtree(data_dir)
    for s in data.split.unique():
        (data_dir / "images" / s).mkdir(parents=True, exist_ok=True)
        (data_dir / "labels" / s).mkdir(parents=True, exist_ok=True)

    tareas = []
    for r in data.itertuples():
        txt = "\n".join(l for _, l in lines.get(r.id, []))
        tareas.append((r.path, r.split, r.id, txt + ("\n" if txt else ""),
                       factor.get(r.id, 1) if r.split == "train" else 1))

    def trabajo(t):
        src, split, iid, txt, nd = t
        d_img = data_dir / "images" / split; d_lab = data_dir / "labels" / split
        dst0 = d_img / f"{iid}.jpg"; lab0 = d_lab / f"{iid}.txt"
        hecho = dst0.exists() and dst0.stat().st_size > 0 and lab0.exists()
        if not hecho:
            _guardar_reducida(src, dst0, cfg.resize, cfg.quality)
            lab0.write_text(txt)
        for k in range(1, nd):                              # duplicados para sobremuestreo
            dk = d_img / f"{iid}_dup{k}.jpg"
            if not dk.exists():
                shutil.copyfile(dst0, dk)
            (d_lab / f"{iid}_dup{k}.txt").write_text(txt)
        return 0 if hecho else 1

    t0 = time.time(); nuevos = 0; errores = []
    log(f"Exportando {len(tareas)} imágenes (lado mayor <= {cfg.resize or 'original'} px, {cfg.workers} hilos) ...")
    with ThreadPoolExecutor(max_workers=cfg.workers) as ex:
        futs = {ex.submit(trabajo, t): t for t in tareas}
        for i, f in enumerate(as_completed(futs), 1):
            try:
                nuevos += f.result()
            except Exception as e:
                errores.append((futs[f][2], repr(e)))
            if i % 2000 == 0 or i == len(tareas):
                el = time.time() - t0
                log(f"  {i}/{len(tareas)}  nuevas={nuevos}  {el/60:.1f} min  (~{el/i*(len(tareas)-i)/60:.1f} min restantes)")
    if errores:
        log(f"ERRORES en {len(errores)} imágenes (primeras 5): {errores[:5]}")
        raise RuntimeError("Hubo errores al exportar. Vuelve a ejecutar: lo ya escrito se reutiliza.")

    # metadatos livianos que viajan en el zip (fecha/secuencia/cámara sirven para el monitoreo en Colab)
    m = data.drop(columns=["path"]).copy()
    m["dt"] = m["dt"].astype(str)
    m["img_rel"] = "images/" + m.split + "/" + m.id + ".jpg"
    m.to_csv(data_dir / "meta.csv", index=False)
    a = ann[["image_id", "split", "especie", "category_id", "bbox", "W", "H"]].copy()
    a["bbox"] = a.bbox.map(lambda b: json.dumps([float(x) for x in b]))
    a.to_csv(data_dir / "anotaciones.csv", index=False)
    json.dump(classes, open(data_dir / "classes.json", "w", encoding="utf-8"), ensure_ascii=False)
    json.dump({"resize_max": cfg.resize, "quality": cfg.quality, "oversample_max": cfg.oversample_max,
               "n_imagenes": int(len(data))}, open(data_dir / "export_info.json", "w"))
    C.write_yamls(data_dir, classes)

    log("\nArchivos escritos por partición (deben coincidir con la Tabla 1):")
    esperado = data.groupby("split").size().to_dict()
    for s in sorted(esperado):
        n_img = len(list((data_dir / "images" / s).glob("*.jpg")))
        n_lab = len(list((data_dir / "labels" / s).glob("*.txt")))
        ok = "OK" if (n_img == n_lab and n_img >= esperado[s]) else "REVISAR"
        log(f"  {s:11s} imágenes={n_img:6d} etiquetas={n_lab:6d} esperado>={esperado[s]:6d}  {ok}")
    return classes

def main():
    ap = C.agregar_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    ejecutar(C.resolver(ap.parse_args()))

if __name__ == "__main__":
    main()
