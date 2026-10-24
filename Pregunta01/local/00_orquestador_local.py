"""
Paso 00: orquestador de la parte LOCAL (PC con los DICOM).

Ejecuta en orden, cada uno en su propio proceso:
  01   python 01_seleccion_estudios.py                  etiquetas débiles y selección de estudios
  02p  python 02_preprocesar_dicom.py --plano principal  DICOM sagital -> tensores/principal/*.npz
  02a  python 02_preprocesar_dicom.py --plano alterna    DICOM coronal -> tensores/alterna/*.npz

Qué controla:
- Antes de empezar, que existan los CSV de entrada y la carpeta de DICOM.
- Muestra la salida de cada paso en pantalla y la guarda en datos/generados/logs/.
- Si un paso falla, se detiene y dice desde dónde retomar.
- Después de cada paso 02 resume el registro (ok / ya_existia / sin_carpeta / error).
- Al final muestra el tiempo de cada paso y qué subir a Google Drive.

Como cada paso corre en un proceso aparte, funciona en Windows tanto desde una terminal como
desde un notebook (el procesamiento en paralelo del paso 02 no se ve afectado).

Uso:
  python local/00_orquestador_local.py                     todo (01, 02p, 02a)
  python local/00_orquestador_local.py --desde 02p         retoma desde un paso
  python local/00_orquestador_local.py --solo 01           un solo paso
  python local/00_orquestador_local.py --limite 10         prueba rápida: 10 estudios por paso 02
  python local/00_orquestador_local.py --dicom "E:\\otra\\ruta\\train_series"
"""
import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # para importar configuracion.py (carpeta Pregunta01)

import configuracion as C

# orden fijo de los pasos: clave -> (descripción, script, argumentos)
PASOS = {
    "01": ("Selección de estudios y etiquetas débiles", "01_seleccion_estudios.py", []),
    "02p": ("Preprocesamiento DICOM, plano principal (sagital)", "02_preprocesar_dicom.py", ["--plano", "principal"]),
    "02a": ("Preprocesamiento DICOM, plano alterno (coronal)", "02_preprocesar_dicom.py", ["--plano", "alterna"]),
}
ARCHIVOS_ENTRADA = [C.CSV_TRAIN_ORIGINAL, C.CSV_SERIES, C.CSV_ETIQUETAS_CURADAS]
DIR_LOGS_LOCAL = C.DIR_DATOS_GENERADOS / "logs"


def argumentos():
    p = argparse.ArgumentParser(description="Orquestador de los pasos locales 01 y 02.")
    grupo = p.add_mutually_exclusive_group()
    grupo.add_argument("--desde", choices=list(PASOS), help="empieza en este paso y sigue con los siguientes")
    grupo.add_argument("--solo", choices=list(PASOS), help="ejecuta solo este paso")
    p.add_argument("--dicom", type=Path, default=C.DIR_DICOM, help="carpeta train_series (por defecto la de configuracion.py)")
    p.add_argument("--limite", type=int, default=None, help="solo N estudios en cada paso 02 (para probar)")
    return p.parse_args()


def pasos_a_ejecutar(a) -> list:
    claves = list(PASOS)
    if a.solo:
        return [a.solo]
    return claves[claves.index(a.desde):] if a.desde else claves


def verificar_requisitos(claves: list, dir_dicom: Path) -> list:
    """Devuelve la lista de problemas encontrados (vacía si todo está bien)."""
    problemas = [f"Falta el archivo de entrada: {f}" for f in ARCHIVOS_ENTRADA if not f.exists()]
    usa_dicom = any(k.startswith("02") for k in claves)
    if usa_dicom and not dir_dicom.exists():
        problemas.append(f"No existe la carpeta de DICOM: {dir_dicom} (revisa DIR_DICOM en configuracion.py o usa --dicom)")
    if "01" not in claves and usa_dicom and not C.CSV_SELECCION.exists():
        problemas.append(f"El paso 02 necesita {C.CSV_SELECCION}; ejecuta antes el paso 01.")
    if "01" in claves and not dir_dicom.exists():
        print(f"Aviso: no existe {dir_dicom}; el paso 01 no podrá filtrar los estudios que faltan en disco.")
    return problemas


def ejecutar_paso(clave: str, a, log) -> tuple:
    """Corre un paso en un proceso aparte, muestra y guarda su salida. Devuelve (código, segundos)."""
    descripcion, script, args = PASOS[clave]
    if script.startswith("02"):
        args = args + ["--dicom", str(a.dicom)] + (["--limite", str(a.limite)] if a.limite else [])
    comando = [sys.executable, "-u", str(Path(__file__).resolve().parent / script), *args]
    cabecera = f"\n{'=' * 70}\n[{clave}] {descripcion}\n$ python {Path(comando[2]).parent.name}/{' '.join([Path(comando[2]).name] + comando[3:])}\n{'=' * 70}"
    print(cabecera); log.write(cabecera + "\n")
    entorno = dict(os.environ, PYTHONIOENCODING="utf-8", DIR_DICOM=str(a.dicom))   # 01 y 02 usan la misma ruta
    inicio = time.time()
    with subprocess.Popen(comando, cwd=C.RAIZ, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, encoding="utf-8", errors="replace", env=entorno) as proceso:
        for linea in proceso.stdout:
            print(linea, end=""); log.write(linea); log.flush()
    return proceso.returncode, time.time() - inicio


def resumen_registro(plano: str) -> str:
    """Cuenta los estados del registro del paso 02 y avisa si hubo errores."""
    ruta = C.DIR_TENSORES / plano / "registro_preprocesamiento.csv"
    if not ruta.exists():
        return "sin registro"
    estados = pd.read_csv(ruta).estado.str.split(":").str[0].value_counts()
    texto = ", ".join(f"{k}: {v}" for k, v in estados.items())
    if estados.get("error", 0):
        texto += f"  <- revisar las filas 'error' en {ruta}"
    return texto


def main():
    a = argumentos()
    claves = pasos_a_ejecutar(a)
    problemas = verificar_requisitos(claves, a.dicom)
    if problemas:
        print("No se puede empezar:\n  - " + "\n  - ".join(problemas))
        sys.exit(1)

    DIR_LOGS_LOCAL.mkdir(parents=True, exist_ok=True)
    ruta_log = DIR_LOGS_LOCAL / f"orquestador_{datetime.now():%Y-%m-%d_%H%M}.txt"
    resumen = []
    with open(ruta_log, "w", encoding="utf-8") as log:
        print(f"Pipeline {C.VERSION} | pasos: {', '.join(claves)}")
        log.write(f"Inicio {datetime.now():%Y-%m-%d %H:%M:%S} | pasos: {', '.join(claves)} | DICOM: {a.dicom}\n")
        for clave in claves:
            codigo, segundos = ejecutar_paso(clave, a, log)
            detalle = resumen_registro(PASOS[clave][2][1]) if clave.startswith("02") and codigo == 0 else ""
            resumen.append((clave, "ok" if codigo == 0 else f"FALLÓ (código {codigo})", segundos, detalle))
            if codigo != 0:
                break
        tabla = "\n".join(f"  {c:<4} {e:<20} {s / 60:6.1f} min  {d}" for c, e, s, d in resumen)
        final = f"\n{'=' * 70}\nResumen\n{tabla}\nRegistro completo: {ruta_log}"
        print(final); log.write(final + "\n")

    if resumen[-1][1] != "ok":
        print(f"\nCorrige el problema y retoma con:  python local/00_orquestador_local.py --desde {resumen[-1][0]}")
        sys.exit(1)
    print("\nParte local terminada. Sube a Google Drive, dentro de la carpeta Pregunta01:\n"
          "  datos/   (entrada y generados)\n  tensores/  (principal y alterna)\n"
          "y continúa con la parte de Colab del README.")


if __name__ == "__main__":
    main()
