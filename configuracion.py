"""
Constantes del proyecto (Pregunta 1: clasificación multietiqueta de RM de rodilla).
Cambia aquí rutas e hiperparámetros; el resto del código solo las importa.
"""
from pathlib import Path

# ---------------------------------------------------------------- rutas
# Carpeta del proyecto (en Colab: /content/drive/MyDrive/rsna_rodilla)
RAIZ = Path(__file__).resolve().parent
DIR_DATOS = RAIZ / "datos"                       # CSV de entrada y selección
DIR_TENSORES = RAIZ / "tensores"                 # volúmenes preprocesados (.npz)
DIR_SALIDAS = RAIZ / "salidas"                   # métricas, modelos, Grad-CAM

# DICOM originales (solo los usa 02_preprocesar_dicom.py, en la PC con el disco D:)
DIR_DICOM = Path(r"D:\kaggle\rsna-knee-abnormality-detection\train_series")

CSV_TRAIN_ORIGINAL = DIR_DATOS / "train.csv"                  # 58 filas con etiqueta dorada
CSV_SERIES = DIR_DATOS / "train_series.csv"                   # estudio -> series + plano/secuencia
CSV_ETIQUETAS_LLM = DIR_DATOS / "train_etiquetado_v1.3.csv"   # etiquetas curadas (fuente 2)
CSV_SELECCION = DIR_DATOS / "seleccion_estudios.csv"          # salida de 01_seleccion_estudios.py

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
