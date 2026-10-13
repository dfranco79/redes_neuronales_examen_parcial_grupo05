"""
Paso 2: DICOM -> volumen compacto por estudio (.npz uint8, N_CORTES x 256 x 256).

Se corre UNA vez donde están los DICOM (tu PC con el disco D:). El resultado (~1 MB por
estudio, ~560 MB para 558 estudios) se sube a Google Drive y Colab entrena sobre eso.

Por estudio:
1. Toma la serie elegida en seleccion_estudios.csv (si hay varias candidatas, la de más cortes;
   si el estudio no tiene serie del plano principal, usa la alterna).
2. Ordena los cortes por posición física (ImagePositionPatient proyectada sobre la normal
   del plano), no por nombre de archivo (los nombres son UIDs aleatorios).
3. Normaliza intensidad por volumen (percentiles 1-99.5 -> 0-255), redimensiona a 256x256
   y remuestrea el eje de cortes a N_CORTES.

Uso:  python 02_preprocesar_dicom.py [--plano principal|alterna] [--dicom RUTA] [--limite N]
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pydicom

import configuracion as C


def posicion_corte(cabecera) -> float:
    """Distancia del corte a lo largo de la normal del plano (orden anatómico real)."""
    fila, columna = np.array(cabecera.ImageOrientationPatient, float).reshape(2, 3)
    return float(np.dot(np.cross(fila, columna), np.array(cabecera.ImagePositionPatient, float)))


def leer_serie(carpeta: Path) -> np.ndarray:
    """Lee una serie y devuelve el volumen float32 (cortes, alto, ancho) ordenado."""
    archivos = list(carpeta.glob("*.dcm"))
    cabeceras = [(pydicom.dcmread(a, stop_before_pixels=True), a) for a in archivos]
    cabeceras.sort(key=lambda x: posicion_corte(x[0]))
    cortes, forma = [], None
    for _, archivo in cabeceras:
        img = pydicom.dcmread(archivo).pixel_array.astype(np.float32)
        forma = forma or img.shape
        if img.shape != forma:                       # cortes con otra matriz: se ajustan
            img = cv2.resize(img, forma[::-1], interpolation=cv2.INTER_AREA)
        cortes.append(img)
    return np.stack(cortes)


def normalizar_y_redimensionar(volumen: np.ndarray) -> np.ndarray:
    bajo, alto = np.percentile(volumen, [C.PERCENTIL_BAJO, C.PERCENTIL_ALTO])
    volumen = np.clip((volumen - bajo) / max(alto - bajo, 1e-6), 0, 1)
    volumen = np.stack([cv2.resize(c, (C.TAMANO_IMAGEN, C.TAMANO_IMAGEN), interpolation=cv2.INTER_AREA)
                        for c in volumen])
    # remuestreo lineal del eje de cortes a N_CORTES
    origen = np.linspace(0, len(volumen) - 1, C.N_CORTES)
    i0 = np.floor(origen).astype(int); i1 = np.minimum(i0 + 1, len(volumen) - 1); w = (origen - i0)[:, None, None]
    volumen = volumen[i0] * (1 - w) + volumen[i1] * w
    return (volumen * 255).round().astype(np.uint8)


def procesar_estudio(estudio: str, series: str, dir_dicom: Path, dir_salida: Path) -> dict:
    destino = dir_salida / f"{estudio}.npz"
    if destino.exists():
        return {"estudio": estudio, "estado": "ya_existia"}
    try:
        carpetas = [dir_dicom / estudio / s for s in series.split("|")]
        carpetas = [c for c in carpetas if c.exists()]
        if not carpetas:
            return {"estudio": estudio, "estado": "sin_carpeta"}
        carpeta = max(carpetas, key=lambda c: sum(1 for _ in c.glob("*.dcm")))   # la de más cortes
        volumen = leer_serie(carpeta)
        np.savez_compressed(destino, volumen=normalizar_y_redimensionar(volumen))
        return {"estudio": estudio, "estado": "ok", "serie": carpeta.name, "cortes_originales": len(volumen)}
    except Exception as error:   # un DICOM corrupto no debe detener todo el lote
        return {"estudio": estudio, "estado": f"error: {error}"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--plano", choices=["principal", "alterna"], default="principal")
    p.add_argument("--dicom", type=Path, default=C.DIR_DICOM)
    p.add_argument("--limite", type=int, default=None, help="procesar solo N estudios (prueba)")
    args = p.parse_args()

    seleccion = pd.read_csv(C.CSV_SELECCION)
    col, otra = (f"series_{args.plano}", "series_alterna" if args.plano == "principal" else "series_principal")
    seleccion["serie_a_usar"] = seleccion[col].fillna(seleccion[otra])       # respaldo si falta el plano
    tareas = seleccion.dropna(subset=["serie_a_usar"]).head(args.limite) if args.limite else seleccion.dropna(subset=["serie_a_usar"])
    dir_salida = C.DIR_TENSORES / args.plano
    dir_salida.mkdir(parents=True, exist_ok=True)

    resultados = []
    with ProcessPoolExecutor(C.N_PROCESOS) as grupo:
        futuros = [grupo.submit(procesar_estudio, r.StudyInstanceUID, r.serie_a_usar, args.dicom, dir_salida)
                   for r in tareas.itertuples()]
        for n, f in enumerate(as_completed(futuros), 1):
            resultados.append(f.result())
            if n % 25 == 0 or n == len(futuros):
                print(f"{n}/{len(futuros)} estudios")
    registro = pd.DataFrame(resultados)
    registro.to_csv(dir_salida / "registro_preprocesamiento.csv", index=False)
    print(registro.estado.str.split(":").str[0].value_counts().to_string())


if __name__ == "__main__":
    main()
