"""
Paso 1: etiquetas débiles + selección estratificada de estudios.

Qué hace:
1. Extrae etiquetas por REGLAS (extractor_etiquetas.py) de los 4.407 reportes.
2. Las combina con las etiquetas curadas (datos/entrada/etiquetas_curadas_v1.3.csv) en una etiqueta "suave":
   ambas fuentes = 1 -> 1.0 ; ambas = 0 -> 0.0 ; discrepan -> 0.5 (con confianza baja).
3. Valida el extractor por reglas contra las 58 etiquetas doradas (entregable 1).
4. Elige la serie a usar por estudio (sagital sensible a líquido con supresión grasa)
   y arma un subconjunto estratificado: N_ENTRENAMIENTO débiles + 58 dorados.
Salida (datos/generados/): seleccion_estudios.csv, validacion_etiquetas_debiles.csv y prevalencia_seleccion.csv
"""
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, f1_score

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # para importar configuracion.py (carpeta Pregunta01)

import configuracion as C
from extractor_etiquetas import etiquetar_informe


def cargar_tablas():
    original = pd.read_csv(C.CSV_TRAIN_ORIGINAL)
    curadas = pd.read_csv(C.CSV_ETIQUETAS_CURADAS)
    series = pd.read_csv(C.CSV_SERIES)
    assert (original.StudyInstanceUID == curadas.StudyInstanceUID).all()
    return original, curadas, series


def etiquetas_por_reglas(reportes: pd.Series) -> pd.DataFrame:
    return pd.DataFrame([etiquetar_informe(t) for t in reportes], index=reportes.index)[C.ETIQUETAS]


def validar_contra_oro(reglas: pd.DataFrame, oro: pd.DataFrame) -> pd.DataFrame:
    """Acuerdo, F1 y kappa por etiqueta del extractor frente a las etiquetas doradas."""
    filas = []
    for e in C.ETIQUETAS:
        y, p = oro[e].astype(int), reglas.loc[oro.index, e].astype(int)
        filas.append({"etiqueta": e, "positivos_oro": int(y.sum()), "acuerdo": (y == p).mean(),
                      "f1": f1_score(y, p, zero_division=0), "kappa": cohen_kappa_score(y, p)})
    tabla = pd.DataFrame(filas)
    tabla.loc[len(tabla)] = ["MACRO", int(oro.values.sum()), tabla.acuerdo.mean(), tabla.f1.mean(), tabla.kappa.mean()]
    return tabla.round(3)


def elegir_series(series: pd.DataFrame, plano: str) -> pd.DataFrame:
    """Una fila por estudio con la(s) serie(s) candidatas del plano pedido.
    Prioridad: sensible a líquido + supresión grasa; si no hay, cualquier serie del plano."""
    s = series[series.Anatomical_Plane == plano].copy()
    s["prioridad"] = s.Fluid_Sensitive + s.Fat_Suppression
    s = s.sort_values(["StudyInstanceUID", "prioridad"], ascending=[True, False])
    mejores = s.groupby("StudyInstanceUID").head(1)
    # se guardan todas las candidatas de máxima prioridad; el preprocesador toma la de más cortes
    candidatas = s.merge(mejores[["StudyInstanceUID", "prioridad"]], on=["StudyInstanceUID", "prioridad"])
    return (candidatas.groupby("StudyInstanceUID")
            .agg(series=("SeriesInstanceUID", "|".join), prioridad=("prioridad", "first")).reset_index())


def muestreo_estratificado(debiles: pd.DataFrame, rng) -> pd.Index:
    """Garantiza MIN_POSITIVOS por hallazgo (de raro a común) y completa al azar."""
    elegidos = set()
    prevalencia = (debiles[C.ETIQUETAS] == 1).sum().sort_values()
    for e in prevalencia.index:
        positivos = debiles.index[debiles[e] == 1].difference(list(elegidos))
        faltan = C.MIN_POSITIVOS - int((debiles.loc[list(elegidos), e] == 1).sum()) if elegidos else C.MIN_POSITIVOS
        if faltan > 0:
            elegidos |= set(rng.choice(positivos, min(faltan, len(positivos)), replace=False))
    resto = debiles.index.difference(list(elegidos))
    faltan = max(0, C.N_ENTRENAMIENTO - len(elegidos))
    elegidos |= set(rng.choice(resto, min(faltan, len(resto)), replace=False))
    return pd.Index(sorted(elegidos))


def main():
    rng = np.random.default_rng(C.SEMILLA)
    original, curadas, series = cargar_tablas()
    es_oro = original[C.ETIQUETAS[0]].notna()

    reglas = etiquetas_por_reglas(original.Report)
    validacion = validar_contra_oro(reglas, original.loc[es_oro, C.ETIQUETAS])
    C.DIR_DATOS_GENERADOS.mkdir(parents=True, exist_ok=True)
    validacion.to_csv(C.DIR_DATOS_GENERADOS / "validacion_etiquetas_debiles.csv", index=False)
    print("Extractor por reglas vs etiqueta dorada:\n", validacion.to_string(index=False))

    # etiqueta suave y confianza (solo estudios sin etiqueta dorada)
    # en el CSV curado las 58 filas doradas van vacías: ahí siempre manda la etiqueta dorada de train.csv
    curadas_e = curadas[C.ETIQUETAS].fillna(0).astype(int) if C.FUENTE_ETIQUETAS == "combinada" else reglas
    suave = (reglas + curadas_e) / 2.0
    confianza = np.where(reglas.values == curadas_e.values, 1.0, 0.5)
    suave[es_oro.values] = original.loc[es_oro, C.ETIQUETAS].astype(float).values   # oro = verdad
    confianza[es_oro.values] = 1.0

    tabla = original[["StudyInstanceUID"]].join(suave)
    for j, e in enumerate(C.ETIQUETAS):
        tabla[f"conf_{e}"] = confianza[:, j]
    principal = elegir_series(series, C.PLANO_PRINCIPAL).rename(columns={"series": "series_principal", "prioridad": "prioridad_principal"})
    alterna = elegir_series(series, C.PLANO_ALTERNATIVO).rename(columns={"series": "series_alterna", "prioridad": "prioridad_alterna"})
    tabla = tabla.merge(principal, on="StudyInstanceUID", how="left").merge(alterna, on="StudyInstanceUID", how="left")

    # entrenamiento: estudios sin oro y con serie principal sensible a líquido + FS (prioridad 2)
    candidatos = tabla[(~es_oro.values) & (tabla.prioridad_principal == 2)]
    # solo estudios que existen en disco (la descarga local de Kaggle puede estar incompleta)
    if C.DIR_DICOM.exists():
        en_disco = {p.name for p in C.DIR_DICOM.iterdir() if p.is_dir()}
        faltan_oro = int((~tabla.loc[es_oro.values, "StudyInstanceUID"].isin(en_disco)).sum())
        candidatos = candidatos[candidatos.StudyInstanceUID.isin(en_disco)]
        print(f"Estudios en disco: {len(en_disco)} | candidatos de entrenamiento: {len(candidatos)} | "
              f"dorados sin carpeta en disco: {faltan_oro}")
    consenso = candidatos[C.ETIQUETAS].where(candidatos[C.ETIQUETAS] != 0.5, 0)   # para estratificar
    idx = muestreo_estratificado(consenso.round().astype(int), rng)
    entrenamiento = tabla.loc[idx].copy()
    n_val = int(len(entrenamiento) * C.FRACCION_VALIDACION_DEBIL)
    entrenamiento["particion"] = "entrenamiento"
    entrenamiento.loc[rng.choice(entrenamiento.index, n_val, replace=False), "particion"] = "validacion_debil"
    oro = tabla[es_oro.values].copy()
    oro["particion"] = "oro"
    seleccion = pd.concat([entrenamiento, oro])
    seleccion.to_csv(C.CSV_SELECCION, index=False)

    # prevalencia (entregable 6)
    prev = seleccion.groupby("particion")[C.ETIQUETAS].apply(lambda d: (d >= 0.5).sum()).T
    prev["%_entrenamiento"] = (100 * prev["entrenamiento"] / (seleccion.particion == "entrenamiento").sum()).round(1)
    prev.to_csv(C.DIR_DATOS_GENERADOS / "prevalencia_seleccion.csv")
    print("\nEstudios por partición:", seleccion.particion.value_counts().to_dict())
    print("Dorados sin serie sagital FS (usan alternativa):", int((oro.prioridad_principal != 2).sum()))
    print("\nPrevalencia (positivos, etiqueta suave >= 0.5):\n", prev.to_string())


if __name__ == "__main__":
    main()
