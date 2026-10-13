"""
Paso 5: análisis para el informe (se corre después de ejecutar_colab.sh).

- Intervalos de confianza 95 % del AUC macro por bootstrap (2.000 remuestreos) y
  diferencia de cada experimento contra el modelo base, sobre los mismos remuestreos.
- Figuras: curvas de validación, AUC por hallazgo, prevalencia y paneles Grad-CAM.
- Tabla comparativa del AUC macro de todos los experimentos.
Salida: salidas/intervalos_confianza.csv, salidas/comparacion_experimentos.csv y salidas/figuras/*.pdf
Uso:    python 05_analisis_informe.py
"""
import glob
import json

import cv2
import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import configuracion as C

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

N_BOOTSTRAP = 2000
EXPERIMENTO_BASE = "resnet50_bce_suave_ninguno_principal"
EXPERIMENTO_CERO = "cero_bce_suave_ninguno_principal"
COLORES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
NOMBRES = ["LCA", "LCM", "Menisco medial", "Menisco lateral", "OA medial", "OA lateral", "OA patelofemoral",
           "Derrame", "Sinovitis", "Quiste de Baker", "Contusión", "Fractura"]
DIR_FIG = C.DIR_SALIDAS / "figuras"


def auc_macro(y, p):
    return np.mean([roc_auc_score(y[:, j], p[:, j]) for j in range(y.shape[1]) if 0 < y[:, j].sum() < len(y)])


def intervalos_confianza():
    oro = pd.read_csv(C.CSV_TRAIN_ORIGINAL).set_index("StudyInstanceUID")
    experimentos = sorted(p.split("/")[-2] for p in glob.glob(str(C.DIR_SALIDAS / "*/predicciones_oro.csv")))
    pred = {e: pd.read_csv(C.DIR_SALIDAS / e / "predicciones_oro.csv").set_index("StudyInstanceUID") for e in experimentos}
    ids = pred[EXPERIMENTO_BASE].index
    y = oro.loc[ids, C.ETIQUETAS].astype(int).values
    rng = np.random.default_rng(C.SEMILLA)
    muestras = [rng.integers(0, len(ids), len(ids)) for _ in range(N_BOOTSTRAP)]
    boot = {e: np.array([auc_macro(y[m], pred[e].loc[ids, C.ETIQUETAS].values[m]) for m in muestras]) for e in experimentos}
    filas = []
    for e in experimentos:
        d = boot[EXPERIMENTO_BASE] - boot[e]
        filas.append({"experimento": e, "n_oro": len(ids),
                      "auc_macro": auc_macro(y, pred[e].loc[ids, C.ETIQUETAS].values),
                      "ic95_inf": np.percentile(boot[e], 2.5), "ic95_sup": np.percentile(boot[e], 97.5),
                      "dif_base_menos_exp": np.mean(d), "dif_ic95_inf": np.percentile(d, 2.5), "dif_ic95_sup": np.percentile(d, 97.5)})
    tabla = pd.DataFrame(filas).round(4)
    tabla.to_csv(C.DIR_SALIDAS / "intervalos_confianza.csv", index=False)
    print(tabla.to_string(index=False))


def tabla_comparativa():
    """AUC macro dorado de cada experimento que tenga resumen.json (se rehace en cada ejecución)."""
    filas = [{"experimento": p.parent.name, "auc_macro_oro": json.load(open(p))["auc_macro_oro"]}
             for p in C.DIR_SALIDAS.glob("*/resumen.json")]
    tabla = pd.DataFrame(filas).sort_values("auc_macro_oro", ascending=False)
    tabla.to_csv(C.DIR_SALIDAS / "comparacion_experimentos.csv", index=False)
    print(tabla.to_string(index=False))


def estilo():
    plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6, "savefig.dpi": 220})


def historial(e):
    return pd.DataFrame(json.load(open(C.DIR_SALIDAS / e / "resumen.json"))["historial"])


def figuras():
    estilo(); DIR_FIG.mkdir(exist_ok=True)
    # curvas de AUC en validación débil
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    exps = sorted(p.split("/")[-2] for p in glob.glob(str(C.DIR_SALIDAS / "*/resumen.json")))
    for e, c in zip(exps, COLORES * 2):
        h = historial(e); ax.plot(h.epoca, h.auc_validacion_debil, lw=2, color=c, label=e.replace("_principal", ""))
    ax.set_xlabel("Época"); ax.set_ylabel("AUC macro (validación débil)"); ax.legend(frameon=False, fontsize=6.5)
    fig.tight_layout(); fig.savefig(DIR_FIG / "curvas.pdf"); plt.close(fig)
    # AUC por hallazgo: desde cero vs base
    m0 = pd.read_csv(C.DIR_SALIDAS / EXPERIMENTO_CERO / "metricas_oro.csv").set_index("etiqueta")
    m1 = pd.read_csv(C.DIR_SALIDAS / EXPERIMENTO_BASE / "metricas_oro.csv").set_index("etiqueta")
    y = np.arange(C.N_ETIQUETAS); h = 0.38
    fig, ax = plt.subplots(figsize=(6.4, 3.3))
    ax.barh(y - h / 2, m0.loc[C.ETIQUETAS, "auc_roc"], h, color=COLORES[0], label="CNN desde cero")
    ax.barh(y + h / 2, m1.loc[C.ETIQUETAS, "auc_roc"], h, color=COLORES[1], label="ResNet-50 (BCE)")
    ax.axvline(0.5, color="#52514e", lw=1, ls="--"); ax.set_yticks(y, NOMBRES); ax.invert_yaxis(); ax.set_xlim(0.4, 1)
    ax.set_xlabel("AUC-ROC sobre el conjunto dorado"); ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    fig.tight_layout(); fig.savefig(DIR_FIG / "auc_etiqueta.pdf"); plt.close(fig)
    # paneles Grad-CAM: primer mapa de cada hallazgo
    for e in [EXPERIMENTO_BASE, EXPERIMENTO_CERO]:
        rutas = sorted(glob.glob(str(C.DIR_SALIDAS / e / "gradcam" / "*.png")))
        primeros = {}
        for r in rutas:
            primeros.setdefault(r.split("/")[-1].rsplit("_", 3)[0], r)
        if primeros:
            fig, ax = plt.subplots(1, len(primeros), figsize=(7.2, 1.95))
            for a, (hallazgo, r) in zip(np.atleast_1d(ax), primeros.items()):
                a.imshow(cv2.cvtColor(cv2.imread(r), cv2.COLOR_BGR2RGB)); a.set_title(hallazgo, fontsize=7.5); a.axis("off")
            fig.tight_layout(pad=0.3); fig.savefig(DIR_FIG / f"gradcam_{e.split('_')[0]}.pdf"); plt.close(fig)
    print(f"Figuras guardadas en {DIR_FIG}")


if __name__ == "__main__":
    intervalos_confianza()
    tabla_comparativa()
    figuras()
