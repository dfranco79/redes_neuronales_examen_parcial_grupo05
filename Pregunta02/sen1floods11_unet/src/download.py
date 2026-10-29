"""Descarga de Sen1Floods11 v1.1 desde el bucket público gs://sen1floods11.

Lista los objetos con la API JSON de GCS y los baja por HTTPS en paralelo (no hace falta gsutil ni credenciales).
Se descargan los 446 chips dorados (S1Hand, LabelHand, S1OtsuLabelHand y S2Hand), todos los chips débiles que
no son de Bolivia (S1Weak, S1OtsuLabelWeak y S2IndexLabelWeak; con --max_weak N se puede submuestrear), la
partición oficial (splits/flood_handlabeled) y la metadata de los eventos.
S2Weak (~26 GB) solo se cuenta para el inventario, porque ningún experimento usa Sentinel-2 en los chips débiles.
"""
import json
import os
import re
import shutil
import subprocess
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

GCS_API = os.environ.get("SEN1FLOODS_GCS_API", "https://storage.googleapis.com/storage/v1/b/sen1floods11/o")
GCS_DL = os.environ.get("SEN1FLOODS_GCS_DL", "https://storage.googleapis.com/sen1floods11")
META_GITHUB = "https://raw.githubusercontent.com/cloudtostreet/Sen1Floods11/master/Sen1Floods11_Metadata.geojson"
PREFIX = "v1.1/"
HAND = PREFIX + "data/flood_events/HandLabeled/"
WEAK = PREFIX + "data/flood_events/WeaklyLabeled/"
SPLITS = PREFIX + "splits/flood_handlabeled/"
META = PREFIX + "Sen1Floods11_Metadata.geojson"


def _get(url, retries=5, timeout=60):
    for k in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                return r.read()
        except Exception as e:  # en Colab la red a veces falla: se reintenta esperando cada vez más
            if k == retries - 1:
                raise RuntimeError(f"No se pudo descargar {url}: {e}") from e
            time.sleep(2 ** k)


def _gsutil_list(prefix):
    """Alternativa si la API JSON no responde: gsutil (viene instalado en Colab)."""
    out = subprocess.run(["gsutil", "ls", "-l", f"gs://sen1floods11/{prefix}**"], capture_output=True, text=True, check=True).stdout
    return [(m.group(2), int(m.group(1))) for m in re.finditer(r"^\s*(\d+)\s+\S+\s+gs://sen1floods11/(\S+)$", out, re.M)]


def list_objects(prefix):
    try:
        return _list_objects_api(prefix)
    except Exception as e:
        if shutil.which("gsutil") is None:
            raise
        print(f"  (API JSON no disponible: {e}; uso gsutil)")
        return _gsutil_list(prefix)


def _list_objects_api(prefix):
    items, token = [], None
    while True:
        q = {"prefix": prefix, "fields": "nextPageToken,items(name,size)", "maxResults": 1000}
        if token:
            q["pageToken"] = token
        data = json.loads(_get(f"{GCS_API}?{urllib.parse.urlencode(q)}"))
        items += [(it["name"], int(it["size"])) for it in data.get("items", []) if not it["name"].endswith("/")]
        token = data.get("nextPageToken")
        if not token:
            return items


def list_chips(prefix, inv=None, layer=None):
    """Lista los .tif de una carpeta; los demás objetos (p. ej. un .DS_Store) se anotan en el inventario."""
    objs = list_objects(prefix)
    tifs = [(n, s) for n, s in objs if n.endswith(".tif")]
    if inv is not None:
        inv[layer] = dict(count=len(tifs), bytes=sum(s for _, s in tifs), objetos=len(objs),
                          otros=[Path(n).name for n, _ in objs if not n.endswith(".tif")])
    return tifs


def _local(data_dir, name):
    return Path(data_dir) / name[len(PREFIX):]


def _download_one(name, size, data_dir):
    dst = _local(data_dir, name)
    if dst.exists() and dst.stat().st_size == size:
        return 0
    dst.parent.mkdir(parents=True, exist_ok=True)
    url = f"{GCS_DL}/{urllib.parse.quote(name)}"
    tmp = dst.with_suffix(dst.suffix + ".part")
    try:
        tmp.write_bytes(_get(url))
    except RuntimeError:
        if shutil.which("gsutil") is None:
            raise
        subprocess.run(["gsutil", "-q", "cp", f"gs://sen1floods11/{name}", str(tmp)], check=True)
    tmp.replace(dst)
    return size


def download_objects(objs, data_dir, workers=16, desc="descarga"):
    todo = [(n, s) for n, s in objs if not (_local(data_dir, n).exists() and _local(data_dir, n).stat().st_size == s)]
    if not todo:
        print(f"  {desc}: {len(objs)} archivos ya presentes")
        return
    total = sum(s for _, s in todo)
    with ThreadPoolExecutor(workers) as ex, tqdm(total=total, unit="B", unit_scale=True, desc=desc) as bar:
        futs = [ex.submit(_download_one, n, s, data_dir) for n, s in todo]
        for f in as_completed(futs):
            bar.update(f.result())


def _cid(name):
    return Path(name).name.rsplit("_", 1)[0]


def download_dataset(cfg, need_s2=True):
    """Descarga los datos que usa el proyecto y guarda un inventario del bucket."""
    data_dir = Path(cfg.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"Descargando Sen1Floods11 v1.1 desde gs://sen1floods11 en {data_dir}")
    inv = {}

    split_objs = list_objects(SPLITS)
    download_objects(split_objs, data_dir, cfg.workers, "splits")

    # chips dorados: SAR, etiqueta manual, etiqueta Otsu del mismo chip y Sentinel-2
    for layer in ["S1Hand", "LabelHand", "S1OtsuLabelHand"] + (["S2Hand"] if need_s2 else []):
        objs = list_chips(HAND + layer + "/", inv, layer)
        download_objects(objs, data_dir, cfg.workers, f"{layer} ({len(objs)})")

    # chips débiles: SAR + etiqueta Otsu (S1) + etiqueta por índice espectral (S2)
    blocked = set()
    for f in (data_dir / "splits" / "flood_handlabeled").glob("*.csv"):
        blocked |= {_cid(line.split(",")[0]) for line in f.read_text().splitlines() if line.strip()}
    s1w = {_cid(n): (n, s) for n, s in list_chips(WEAK + "S1Weak/", inv, "S1Weak")}
    lbw = {_cid(n): (n, s) for n, s in list_chips(WEAK + "S1OtsuLabelWeak/", inv, "S1OtsuLabelWeak")}
    s2w = ({_cid(n): (n, s) for n, s in list_chips(WEAK + "S2IndexLabelWeak/", inv, "S2IndexLabelWeak")}
           if cfg.use_s2_weak else {})
    list_chips(WEAK + "S2Weak/", inv, "S2Weak")             # solo para el inventario, no se descarga
    inv["S2Weak"]["descargado"] = False
    universe = sorted(s1w)
    bol = [c for c in universe if c.startswith("Bolivia")]
    ids = sorted(c for c in universe if c not in blocked and not c.startswith("Bolivia") and (c in lbw or c in s2w))
    inv["weak"] = dict(bucket=len(universe), bolivia_excluidos=len(bol),
                       en_splits_dorados=len([c for c in universe if c in blocked]), usables=len(ids))
    print(f"  chips débiles en el bucket: {len(universe)} | excluidos por ser de Bolivia: {len(bol)} | usables: {len(ids)}")
    if cfg.max_weak and len(ids) > cfg.max_weak:
        ids = sorted(np.random.default_rng(cfg.seed).choice(ids, cfg.max_weak, replace=False).tolist())
        print(f"  submuestreo con --max_weak {cfg.max_weak}: {len(ids)} chips")
    inv["weak"]["usados"] = len(ids)
    (data_dir / "weak_ids_bucket.txt").write_text("\n".join(universe))
    (data_dir / "weak_ids_usados.txt").write_text("\n".join(ids))
    download_objects([s1w[c] for c in ids], data_dir, cfg.workers, f"S1Weak ({len(ids)})")
    download_objects([lbw[c] for c in ids if c in lbw], data_dir, cfg.workers, "S1OtsuLabelWeak")
    if s2w:
        download_objects([s2w[c] for c in ids if c in s2w], data_dir, cfg.workers, "S2IndexLabelWeak")

    # metadata de los eventos (órbita, fechas, chips por evento)
    meta = data_dir / "Sen1Floods11_Metadata.geojson"
    if not meta.exists():
        for url in (f"{GCS_DL}/{META}", META_GITHUB):
            try:
                meta.write_bytes(_get(url))
                break
            except Exception as e:
                print(f"  no se pudo bajar la metadata desde {url}: {e}")
    if meta.exists():
        feats = json.loads(meta.read_text())["features"]
        inv["metadata"] = dict(eventos=len(feats), chips_debiles=int(sum(f["properties"].get("train_chip", 0) for f in feats)),
                               chips_dorados=int(sum(f["properties"].get("val_chip", 0) for f in feats)))
    (data_dir / "inventario_bucket.json").write_text(json.dumps(inv, indent=2))
    print("Descarga completa.")
