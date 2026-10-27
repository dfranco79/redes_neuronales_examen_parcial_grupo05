# -*- coding: utf-8 -*-
"""PASO 2 — Controles anti-fuga + estadísticas (Tabla 1) + gráfico de desbalance.

Lee <work>/particiones.csv (paso 1). Lanza AssertionError si detecta fuga entre particiones.
Salida en <work>/results/: tabla1_dataset.csv, desbalance_train.csv, desbalance.png
    python codigo/paso2_verificar.py
"""
import argparse
import cct_comun as C

def ejecutar(cfg, data=None, ann=None, log=print):
    if data is None:
        data, ann, _ = C.cargar_estado(cfg)
    tr_locs = set(data[data.split == "train"].location)
    for s in ["val_trans", "test_trans"]:
        comunes = set(data[data.split == s].location) & tr_locs
        log(f"{s}: ubicaciones compartidas con train = {len(comunes)}  (debe ser 0)")
        assert not comunes, f"FUGA: {s} comparte ubicaciones con train: {sorted(comunes)}"
    rep = int((data.groupby("seq_id").split.nunique() > 1).sum())
    log(f"secuencias repartidas en >1 partición: {rep} (debe ser 0)")
    assert rep == 0, "FUGA: hay secuencias repartidas en más de una partición"
    dup = int(data.id.duplicated().sum())
    assert dup == 0, f"Hay {dup} imágenes repetidas"
    for s in ["train", "val_cis", "test_cis", "val_trans", "test_trans"]:
        assert (data.split == s).any(), f"La partición {s} quedó vacía"

    tab1 = data.groupby("split").agg(imagenes=("id", "size"), con_animal=("has_animal", "sum"),
                                     ubicaciones=("location", "nunique"))
    tab1["especies"] = ann.groupby("split").especie.nunique()
    log("\nTabla 1:\n" + str(tab1))
    tab1.to_csv(cfg.res_dir / "tabla1_dataset.csv")

    cnt = ann[ann.split == "train"].groupby("especie").image_id.nunique().sort_values(ascending=False)
    cnt.to_csv(cfg.res_dir / "desbalance_train.csv", header=["imagenes_train"])
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 4)); cnt.plot.bar(ax=ax, log=True)
        ax.set_ylabel("imágenes de train (escala log)"); ax.set_title("Desbalance de clases en entrenamiento")
        plt.tight_layout(); plt.savefig(cfg.res_dir / "desbalance.png", dpi=150); plt.close(fig)
    except Exception as e:                     # matplotlib es opcional
        log(f"(gráfico de desbalance omitido: {e})")
    return tab1

def main():
    ap = C.agregar_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    ejecutar(C.resolver(ap.parse_args()))

if __name__ == "__main__":
    main()
