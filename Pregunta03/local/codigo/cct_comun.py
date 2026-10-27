# -*- coding: utf-8 -*-
"""Utilidades comunes de los pasos 1-4 (preparación LOCAL, Pregunta 3: YOLO sobre CCT-20).

Estructura (carpeta 'local'):  codigo/  json/  cct_images/  work_q3/   (las dos últimas no se suben a GitHub)

Cada paso guarda su resultado en <work>/ (particiones.csv, anotaciones.csv, yolo_data/, zip), de modo que
puede ejecutarse solo (python codigo/paso3_exportar_yolo.py) o desde preparar_local.py.
"""
import json, os
from pathlib import Path
from types import SimpleNamespace
import pandas as pd
import yaml

# Carpeta base = la carpeta 'local' (la que contiene codigo/, json/, cct_images/ y work_q3/).
# Se calcula a partir de la ubicación de ESTE archivo, así funciona igual en cualquier PC y desde cualquier directorio actual.
BASE_DEFECTO = Path(__file__).resolve().parent.parent

COLS = dict(file=["file_name"], location=["location"], seq=["seq_id"], frame=["frame_num"],
            dt=["date_captured", "datetime"], w=["width"], h=["height"])

def agregar_args(ap):
    """Argumentos compartidos por todos los pasos."""
    ap.add_argument("--base", "--root", dest="base", default=str(BASE_DEFECTO),
                    help="carpeta 'local' (por defecto la que contiene a codigo/); json/, cct_images/ y work_q3/ cuelgan de ella")
    ap.add_argument("--json-dir", default=None, help="carpeta de los 3 JSON (por defecto <base>/json; si no existe, <base>)")
    ap.add_argument("--img-dir", default=None, help="carpeta de imágenes (por defecto <base>/cct_images; útil si están en un disco externo)")
    ap.add_argument("--work", default=None, help="carpeta de salida (por defecto <base>/work_q3)")
    ap.add_argument("--bbox-json", default="caltech_bboxes_20200316.json")
    ap.add_argument("--splits-json", default="CaltechCameraTrapsSplits_v0.json")
    ap.add_argument("--images-json", default="caltech_images_20210113.json")
    ap.add_argument("--split-keys", nargs=2, default=["eccv_train", "eccv_val"], help="claves del JSON de splits (CCT-20)")
    ap.add_argument("--trans-val-frac", type=float, default=0.5, help="fracción de ubicaciones nuevas que van a val_trans")
    ap.add_argument("--cis-frac", type=float, default=0.10, help="fracción de secuencias de train para val_cis y para test_cis (c/u)")
    ap.add_argument("--empty-ratio", type=float, default=1.0, help="vacías por imagen con animal en train (-1 = todas)")
    ap.add_argument("--eval-empty-ratio", type=float, default=-1, help="vacías en val/test (-1 = todas)")
    ap.add_argument("--oversample-max", type=int, default=1, help=">1 activa sobremuestreo de especies raras")
    ap.add_argument("--resize", type=int, default=640, help="lado mayor en px de las imágenes exportadas (0 = sin reducir)")
    ap.add_argument("--quality", type=int, default=90, help="calidad JPEG de las imágenes reducidas")
    ap.add_argument("--workers", type=int, default=4, help="hilos de exportación")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--reexport", action="store_true", help="borra yolo_data y lo regenera (si cambias splits, semilla o razones)")
    ap.add_argument("--zip-name", default="yolo_data_cct20.zip")
    ap.add_argument("--recomprimir", action="store_true", help="rehace el zip aunque ya exista y esté al día")
    return ap

def resolver(a):
    """Convierte los argumentos en rutas ABSOLUTAS (evita el error de ruta relativa con discos externos)."""
    base = Path(a.base).resolve()
    work = Path(a.work).resolve() if a.work else base / "work_q3"
    cfg = SimpleNamespace(**vars(a))
    cfg.base = base
    cfg.json_dir = Path(a.json_dir).resolve() if a.json_dir else (base / "json" if (base / "json").is_dir() else base)
    cfg.img_dir = Path(a.img_dir).resolve() if a.img_dir else base / "cct_images"
    cfg.work = work
    cfg.data_dir = work / "yolo_data"
    cfg.res_dir = work / "results"
    cfg.zip_path = work / a.zip_name
    cfg.work.mkdir(parents=True, exist_ok=True); cfg.res_dir.mkdir(parents=True, exist_ok=True)
    return cfg

def mostrar_rutas(cfg, log=print):
    log(f"Carpeta base : {cfg.base}\nJSON         : {cfg.json_dir}\nImágenes     : {cfg.img_dir}\nSalida       : {cfg.work}")

def comprobar_entradas(cfg):
    """Crea las carpetas que falten y detiene el programa con instrucciones claras si faltan datos de entrada."""
    cfg.json_dir.mkdir(parents=True, exist_ok=True); cfg.img_dir.mkdir(parents=True, exist_ok=True)
    faltan = [n for n in (cfg.bbox_json, cfg.splits_json) if not (cfg.json_dir / n).exists()]
    msg = []
    if faltan:
        msg.append(f"Faltan estos JSON en {cfg.json_dir}: {faltan}. Descárgalos de LILA (Caltech Camera Traps) y colócalos ahí.")
    if not any(cfg.img_dir.rglob("*.jpg")):
        msg.append(f"No hay imágenes .jpg en {cfg.img_dir}. Descomprime ahí las imágenes de Caltech Camera Traps "
                   f"(o usa --img-dir para indicar otra carpeta, p. ej. un disco externo).")
    if msg:
        raise FileNotFoundError("\n".join(msg))

def norm_images(images):
    """Normaliza los nombres de columnas de la lista 'images' de un JSON COCO/LILA."""
    def pick(df, names):
        for n in names:
            if n in df.columns: return df[n]
        return pd.Series([None] * len(df), index=df.index)
    df = pd.DataFrame(images)
    out = pd.DataFrame({"id": df["id"].astype(str)})
    out["file"]     = pick(df, COLS["file"]).astype(str).map(os.path.basename)
    out["location"] = pick(df, COLS["location"]).astype(str)
    out["seq_id"]   = pick(df, COLS["seq"]).astype(str)
    out["frame"]    = pd.to_numeric(pick(df, COLS["frame"]), errors="coerce")
    out["dt"]       = pd.to_datetime(pick(df, COLS["dt"]), errors="coerce")
    out["W"]        = pd.to_numeric(pick(df, COLS["w"]), errors="coerce")
    out["H"]        = pd.to_numeric(pick(df, COLS["h"]), errors="coerce")
    return out

def indexar_jpg(img_dir, excluir=None):
    """Recorre img_dir recursivamente y devuelve {nombre_en_minúsculas: ruta_completa}."""
    excluir = Path(excluir).resolve() if excluir else None
    index = {}
    for dp, dn, fn in os.walk(img_dir):
        if excluir is not None:
            dn[:] = [d for d in dn if Path(dp, d).resolve() != excluir]
        for f in fn:
            if os.path.splitext(f)[1].lower() in (".jpg", ".jpeg"):
                index.setdefault(f.lower(), os.path.join(dp, f))
    return index

# ---- estado entre pasos (CSV en <work>) -------------------------------------------------------------
def guardar_estado(cfg, data, ann, cats):
    d = data.copy(); d["dt"] = d["dt"].astype(str)
    d.to_csv(cfg.work / "particiones.csv", index=False)
    a = ann.copy(); a["bbox"] = a["bbox"].map(lambda b: json.dumps([float(x) for x in b]))
    a.to_csv(cfg.work / "anotaciones.csv", index=False)
    json.dump({str(k): v for k, v in cats.items()}, open(cfg.work / "categorias.json", "w", encoding="utf-8"), ensure_ascii=False)

def cargar_estado(cfg):
    p, a_ = cfg.work / "particiones.csv", cfg.work / "anotaciones.csv"
    if not p.exists() or not a_.exists():
        raise FileNotFoundError(f"Falta {p.name}/{a_.name} en {cfg.work}. Ejecuta antes el paso 1 (paso1_particiones.py).")
    data = pd.read_csv(p, dtype={"id": str, "location": str, "seq_id": str, "file": str, "path": str, "split": str, "split_raw": str})
    data["dt"] = pd.to_datetime(data["dt"], errors="coerce")
    data["has_animal"] = data["has_animal"].astype(bool)
    ann = pd.read_csv(a_, dtype={"image_id": str, "id": str, "split": str, "file": str, "especie": str})
    ann["bbox"] = ann["bbox"].map(json.loads)
    cats = {int(k): v for k, v in json.load(open(cfg.work / "categorias.json", encoding="utf-8")).items()}
    return data, ann, cats

def write_yamls(data_dir, classes, splits=None):
    """Escribe un .yaml por partición con la ruta ABSOLUTA actual de data_dir (se vuelve a llamar en Colab)."""
    data_dir = Path(data_dir).resolve()
    splits = splits or sorted(p.name for p in (data_dir / "images").iterdir() if p.is_dir())
    for s in splits:
        yaml.safe_dump({"path": str(data_dir), "train": "images/train", "val": f"images/{s}",
                        "names": {i: n for i, n in enumerate(classes)}},
                       open(data_dir / f"{s}.yaml", "w", encoding="utf-8"), allow_unicode=True)
