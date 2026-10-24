"""
Constantes del proyecto (Pregunta 1: clasificación multietiqueta de RM de rodilla).
Cambia aquí rutas e hiperparámetros; el resto del código solo las importa.
"""
import os
from pathlib import Path

VERSION = "v4"                                     # versión del pipeline (taxonomía de carpetas por ejecución)

# ---------------------------------------------------------------- rutas
# Taxonomía de carpetas (v4):
#   datos/entrada/     CSV de entrada (no se modifican)
#   datos/generados/   salidas del paso 01 (selección, validación de etiquetas, prevalencia)
#   tensores/          volúmenes .npz del paso 02 (se reutilizan entre ejecuciones)
#   ejecuciones/<ID>/  resultados de una corrida de Colab (pasos 03 a 05), una carpeta por corrida
RAIZ = Path(__file__).resolve().parent             # carpeta Pregunta01
DIR_DATOS_ENTRADA = RAIZ / "datos" / "entrada"
DIR_DATOS_GENERADOS = RAIZ / "datos" / "generados"
DIR_TENSORES = RAIZ / "tensores"

# Identificador de la corrida. Se toma de la variable de entorno ID_EJECUCION
# (ejecutar_colab.sh la define); si no existe, se usa este valor por defecto.
ID_EJECUCION = os.environ.get("ID_EJECUCION", "ejecucion_manual")
DIR_EJECUCION = RAIZ / "ejecuciones" / ID_EJECUCION
DIR_EXPERIMENTOS = DIR_EJECUCION / "experimentos"  # un subdirectorio por experimento (03 y 04)
DIR_ANALISIS = DIR_EJECUCION / "analisis"          # intervalos, comparación y figuras (05)
DIR_LOGS = DIR_EJECUCION / "logs"                  # registro de la corrida y copia de la configuración

# DICOM originales (solo los usan los pasos 01 y 02, en la PC con los DICOM)
# (el orquestador puede cambiarla con --dicom, que la pasa como variable de entorno DIR_DICOM)
DIR_DICOM = Path(os.environ.get("DIR_DICOM", r"D:\kaggle\rsna-knee-abnormality-detection\train_series"))

CSV_TRAIN_ORIGINAL = DIR_DATOS_ENTRADA / "train.csv"                       # Kaggle: 58 filas con etiqueta dorada
CSV_SERIES = DIR_DATOS_ENTRADA / "train_series.csv"                        # Kaggle: estudio -> series + plano
CSV_ETIQUETAS_CURADAS = DIR_DATOS_ENTRADA / "etiquetas_curadas_v1.3.csv"   # grupo: etiquetas curadas (fuente 2)
CSV_SELECCION = DIR_DATOS_GENERADOS / "seleccion_estudios.csv"            # salida del paso 01

# ---------------------------------------------------------------- etiquetas
ETIQUETAS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
             "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
N_ETIQUETAS = len(ETIQUETAS)

# ---------------------------------------------------------------- etiquetas débiles
# "combinada": reglas + CSV curado v1.3 (etiqueta suave 0/0.5/1 con confianza)
# "reglas": solo el extractor por reglas (100 % código propio del grupo)
FUENTE_ETIQUETAS = "combinada"

# ---------------------------------------------------------------- selección de estudios
N_ENTRENAMIENTO = 500          # estudios con etiqueta débil (el PDF sugiere 300-600)
MIN_POSITIVOS = 40             # mínimo de positivos por hallazgo en la muestra (si existen)
FRACCION_VALIDACION_DEBIL = 0.15   # parte del entrenamiento reservada para early stopping/umbral
SEMILLA = 42

# Secuencia/plano principal y alternativa (ablación de plano)
PLANO_PRINCIPAL = "Sagittal"
PLANO_ALTERNATIVO = "Coronal"

# ---------------------------------------------------------------- preprocesamiento
N_CORTES = 24                  # cortes por volumen tras remuestrear el eje de cortes
TAMANO_IMAGEN = 256            # alto = ancho tras redimensionar
PERCENTIL_BAJO, PERCENTIL_ALTO = 1.0, 99.5   # recorte de intensidad por volumen
N_PROCESOS = 6                 # procesos en paralelo para leer DICOM

# ---------------------------------------------------------------- entrenamiento
TAMANO_RECORTE = 224           # recorte aleatorio (train) / central (eval)
PASO_CORTES = 2                # usa 1 de cada 2 tripletes de cortes (D/2 imágenes por estudio)
TAMANO_LOTE = 4                # estudios por lote
EPOCAS = 15
TASA_APRENDIZAJE = 3e-4
DECAIMIENTO_PESOS = 1e-4
PACIENCIA = 4                  # early stopping sobre AUC de validación débil
GAMMA_FOCAL = 2.0
N_TRABAJADORES = 2             # DataLoader
