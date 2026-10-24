"""Dataset de volúmenes preprocesados (.npz) con etiquetas suaves y pesos de confianza."""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # para importar configuracion.py (carpeta Pregunta01)

import configuracion as C

MEDIA_IMAGENET = np.array([0.485, 0.456, 0.406], np.float32)[:, None, None]
DESV_IMAGENET = np.array([0.229, 0.224, 0.225], np.float32)[:, None, None]


def a_tripletes(volumen: np.ndarray, paso: int) -> np.ndarray:
    """(D,H,W) uint8 -> (D/paso, 3, H, W): cada 'imagen' 2.5D = corte anterior, actual y siguiente."""
    d = len(volumen)
    centros = np.arange(1, d - 1, paso)
    return np.stack([volumen[[c - 1, c, c + 1]] for c in centros])


class DatasetRodilla(Dataset):
    def __init__(self, seleccion: pd.DataFrame, dir_tensores, entrenamiento: bool, modo_etiquetas: str = "suave"):
        """modo_etiquetas: 'suave' (0/0.5/1 con peso de confianza) o 'consenso'
        (las celdas en desacuerdo entre fuentes se ignoran: peso 0)."""
        disponibles = {p.stem for p in dir_tensores.glob("*.npz")}
        self.tabla = seleccion[seleccion.StudyInstanceUID.isin(disponibles)].reset_index(drop=True)
        self.dir, self.entrenamiento = dir_tensores, entrenamiento
        self.y = self.tabla[C.ETIQUETAS].values.astype(np.float32)
        self.peso = self.tabla[[f"conf_{e}" for e in C.ETIQUETAS]].values.astype(np.float32)
        if modo_etiquetas == "consenso":
            self.peso = np.where(self.y == 0.5, 0.0, 1.0).astype(np.float32)

    def __len__(self):
        return len(self.tabla)

    def _aumentar(self, x: np.ndarray) -> np.ndarray:
        """Recorte aleatorio + jitter de brillo/contraste (sin voltear: cambiaría la anatomía)."""
        h = x.shape[-1]; r = C.TAMANO_RECORTE
        if self.entrenamiento:
            i, j = np.random.randint(0, h - r + 1, 2)
            x = x[..., i:i + r, j:j + r]
            x = np.clip(x * np.random.uniform(0.85, 1.15) + np.random.uniform(-0.08, 0.08), 0, 1)
        else:
            m = (h - r) // 2
            x = x[..., m:m + r, m:m + r]
        return x

    def __getitem__(self, i):
        fila = self.tabla.iloc[i]
        volumen = np.load(self.dir / f"{fila.StudyInstanceUID}.npz")["volumen"]
        x = a_tripletes(volumen, C.PASO_CORTES).astype(np.float32) / 255.0
        x = (self._aumentar(x) - MEDIA_IMAGENET) / DESV_IMAGENET
        return torch.from_numpy(x.astype(np.float32)), torch.from_numpy(self.y[i]), torch.from_numpy(self.peso[i])
