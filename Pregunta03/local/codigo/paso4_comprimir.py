# -*- coding: utf-8 -*-
"""PASO 4 — Empaqueta <work>/yolo_data en un .zip SIN recomprimir (los JPG ya están comprimidos).

Los .yaml no se incluyen: se regeneran en Colab con la ruta correcta.
    python codigo/paso4_comprimir.py
"""
import argparse, os, zipfile
from pathlib import Path
import cct_comun as C

def _mtime_mas_reciente(carpeta):
    """Fecha de modificación más reciente entre los archivos de images/ y labels/ (rápido: solo lee metadatos)."""
    m = 0.0
    for sub in ("images", "labels"):
        for dp, _, fn in os.walk(Path(carpeta) / sub):
            for f in fn:
                m = max(m, os.stat(os.path.join(dp, f)).st_mtime)
    return m

def ejecutar(cfg, log=print):
    data_dir, zip_path = cfg.data_dir, cfg.zip_path
    if not (data_dir / "images").exists():
        raise FileNotFoundError(f"No existe {data_dir / 'images'}. Ejecuta antes el paso 3.")
    if zip_path.exists() and not cfg.recomprimir:
        if zip_path.stat().st_mtime >= _mtime_mas_reciente(data_dir):
            log(f"El zip ya existe y está al día -> se omite: {zip_path} ({zip_path.stat().st_size/1e9:.2f} GB). "
                f"Usa --recomprimir para rehacerlo.")
            return zip_path
        log("El zip existe pero hay archivos más nuevos en yolo_data -> se rehace.")
    tmp = zip_path.with_suffix(".zip.part")
    n = 0
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as z:
        for p in sorted(data_dir.rglob("*")):
            if p.is_file() and p.suffix != ".yaml":
                z.write(p, Path(data_dir.name) / p.relative_to(data_dir)); n += 1
        for extra in (cfg.res_dir / "tabla1_dataset.csv", cfg.res_dir / "desbalance.png"):
            if extra.exists():
                z.write(extra, Path(data_dir.name) / extra.name)
    os.replace(tmp, zip_path)
    gb = zip_path.stat().st_size / 1e9
    log(f"Zip creado: {zip_path}  ({n} archivos, {gb:.2f} GB)")
    if gb > 14:
        log("AVISO: pesa más de ~14 GB; el Drive gratuito tiene 15 GB en total. Usa --resize 512 o --eval-empty-ratio 0.5 y --reexport.")
    return zip_path

def main():
    ap = C.agregar_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    ejecutar(C.resolver(ap.parse_args()))

if __name__ == "__main__":
    main()
