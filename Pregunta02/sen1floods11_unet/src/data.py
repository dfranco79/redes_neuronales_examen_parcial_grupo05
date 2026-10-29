"""Lectura, normalización y Dataset de Sen1Floods11.

- SAR en dB: NaN -> -50 dB, recorte a [-50, 1] dB y escala lineal a [0, 1] (igual que el notebook oficial del dataset).
- valid = SAR sin NaN y etiqueta en {0, 1}. El nodata (-1 en la máscara dorada, 255 en las débiles) no entra
  ni en la pérdida ni en las métricas.
- Entrenamiento con recortes aleatorios de 256x256 a resolución nativa (10 m) + volteos/rotaciones; la evaluación
  usa el chip completo de 512x512 (la U-Net es totalmente convolucional).
- Los 446 chips dorados se precargan en RAM (float16); los débiles se leen del disco en cada época porque no
  entran en la RAM de Colab.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm


def chip_id(p):
    """'Ghana_103272_S1Hand.tif' -> 'Ghana_103272'."""
    return Path(p).name.rsplit("_", 1)[0]


def event_of(cid):
    return cid.rsplit("_", 1)[0]


class Paths:
    def __init__(self, data_dir):
        b = Path(data_dir)
        hand, weak = b / "data" / "flood_events" / "HandLabeled", b / "data" / "flood_events" / "WeaklyLabeled"
        self.s1, self.lbl, self.s2 = hand / "S1Hand", hand / "LabelHand", hand / "S2Hand"
        self.otsu_hand = hand / "S1OtsuLabelHand"
        self.s1_weak, self.lbl_weak, self.s2_weak = weak / "S1Weak", weak / "S1OtsuLabelWeak", weak / "S2IndexLabelWeak"
        self.splits = b / "splits" / "flood_handlabeled"
        self.meta = b / "Sen1Floods11_Metadata.geojson"
        self.weak_list = b / "weak_ids_usados.txt"
        self.weak_bucket = b / "weak_ids_bucket.txt"
        self.inventory = b / "inventario_bucket.json"


def build_map(d):
    d = Path(d)
    return {chip_id(p): p for p in sorted(d.glob("*.tif"))} if d.exists() else {}


def read_split(splits_dir, name):
    df = pd.read_csv(Path(splits_dir) / name, header=None)   # los CSV oficiales no tienen encabezado
    return [chip_id(f) for f in df[0]]


def load_splits(paths):
    return dict(train=read_split(paths.splits, "flood_train_data.csv"),
                val=read_split(paths.splits, "flood_valid_data.csv"),
                test=read_split(paths.splits, "flood_test_data.csv"),
                bolivia=read_split(paths.splits, "flood_bolivia_data.csv"))


def normalize_sar(raw):
    img = np.nan_to_num(raw, nan=-50.0, posinf=1.0, neginf=-50.0)
    return ((np.clip(img, -50.0, 1.0) + 50.0) / 51.0).astype(np.float32)


def to_db(x):
    """Inversa de normalize_sar (para analizar la retrodispersión en dB)."""
    return np.asarray(x, np.float32) * 51.0 - 50.0


def load_s2(path):
    """13 bandas S2 L1C (reflectancia TOA x 10000) -> [0, 1] con escala fija."""
    with rasterio.open(path) as src:
        s2 = src.read().astype(np.float32)
    s2 = np.nan_to_num(s2, nan=0.0)
    if s2.max() > 50:
        s2 /= 10000.0
    return np.clip(s2, 0.0, 1.0)


def read_label(path):
    with rasterio.open(path) as src:
        lbl = src.read(1).astype(np.int16)
    return np.where(np.isin(lbl, [0, 1]), lbl, -1).astype(np.int8)


def load_sample(s1_path, lbl_path, s2_path=None):
    with rasterio.open(s1_path) as src:
        raw = src.read().astype(np.float32)               # (2,H,W) [VV, VH] en dB
    img = normalize_sar(raw)
    if s2_path is not None:
        img = np.concatenate([img, load_s2(s2_path)], 0)  # (15,H,W)
    lbl = read_label(lbl_path)
    valid = np.isfinite(raw).all(0) & (lbl != -1)
    return img.astype(np.float16), lbl, valid


class FloodDataset(Dataset):
    """Devuelve (img[C,H,W] float32, lbl[H,W] int64 {-1,0,1}, valid[H,W] bool, chip_id)."""

    def __init__(self, ids, s1m, lblm, s2m=None, crop=None, augment=False, name="", preload=True):
        self.ids, self.crop, self.augment, self.preload = list(ids), crop, augment, preload
        self.paths = [(s1m[c], lblm[c], s2m[c] if s2m is not None else None) for c in self.ids]
        # preload=True carga todo en RAM (chips dorados); con False se lee del disco en cada uso (chips débiles)
        self.data = ([load_sample(*p) for p in tqdm(self.paths, desc=f"cargando {name}", leave=False)]
                     if preload else None)

    def __len__(self):
        return len(self.ids)

    def _random_crop(self, img, lbl, valid):
        H, W = lbl.shape
        S = min(self.crop, H, W)
        for _ in range(10):                                # evita recortes casi sin datos válidos
            i, j = np.random.randint(0, H - S + 1), np.random.randint(0, W - S + 1)
            if valid[i:i + S, j:j + S].mean() > 0.1:
                break
        return img[:, i:i + S, j:j + S], lbl[i:i + S, j:j + S], valid[i:i + S, j:j + S]

    def __getitem__(self, k):
        img, lbl, valid = self.data[k] if self.preload else load_sample(*self.paths[k])
        if self.crop:
            img, lbl, valid = self._random_crop(img, lbl, valid)
        if self.augment:
            if np.random.rand() < 0.5:
                img, lbl, valid = img[:, :, ::-1], lbl[:, ::-1], valid[:, ::-1]
            if np.random.rand() < 0.5:
                img, lbl, valid = img[:, ::-1, :], lbl[::-1, :], valid[::-1, :]
            r = np.random.randint(4)
            if r:
                img, lbl, valid = np.rot90(img, r, (1, 2)), np.rot90(lbl, r), np.rot90(valid, r)
        return (torch.from_numpy(np.ascontiguousarray(img, dtype=np.float32)),
                torch.from_numpy(np.ascontiguousarray(lbl, dtype=np.int64)),
                torch.from_numpy(np.ascontiguousarray(valid)),
                self.ids[k])
