#!/usr/bin/env bash
# Ejecuta en Colab los pasos 03 a 05 de la Pregunta 1 (requiere datos/ y tensores/ ya subidos).
# Uso:  bash colab/ejecutar_colab.sh <ID_EJECUCION>
#       bash colab/ejecutar_colab.sh                  (sin argumento usa la fecha y hora: AAAA-MM-DD_HHMM)
# Todo se guarda en ejecuciones/<ID_EJECUCION>/ ; una corrida nunca pisa a otra.
set -eo pipefail
cd "$(dirname "$0")/.."      # trabaja desde la carpeta Pregunta01

export ID_EJECUCION="${1:-$(date +%Y-%m-%d_%H%M)}"
DIR_EJECUCION="ejecuciones/${ID_EJECUCION}"
if [ -d "${DIR_EJECUCION}/experimentos" ]; then
  echo "ERROR: ${DIR_EJECUCION} ya tiene experimentos. Usa otro ID_EJECUCION para no mezclar corridas."
  exit 1
fi
mkdir -p "${DIR_EJECUCION}/logs"
LOG="${DIR_EJECUCION}/logs/log_colab.txt"

# trazabilidad: configuración usada y GPU asignada
cp configuracion.py "${DIR_EJECUCION}/logs/configuracion_usada.py"
nvidia-smi --query-gpu=name,memory.total --format=csv > "${DIR_EJECUCION}/logs/gpu.txt" 2>/dev/null || echo "sin GPU" > "${DIR_EJECUCION}/logs/gpu.txt"

{
  echo "== Ejecución ${ID_EJECUCION} | inicio $(date '+%Y-%m-%d %H:%M:%S')"
  pip install -q pydicom timm
  python colab/03_entrenar.py --arquitectura cero
  python colab/03_entrenar.py --arquitectura resnet50
  python colab/03_entrenar.py --arquitectura resnet50 --perdida focal
  python colab/03_entrenar.py --arquitectura resnet50 --etiquetas consenso
  python colab/03_entrenar.py --arquitectura resnet50 --balanceo pos_weight
  if [ -d tensores/alterna ]; then python colab/03_entrenar.py --arquitectura resnet50 --plano alterna; fi
  python colab/04_gradcam.py --experimento resnet50_bce_suave_ninguno_principal --arquitectura resnet50
  python colab/04_gradcam.py --experimento cero_bce_suave_ninguno_principal --arquitectura cero
  python colab/05_analisis_informe.py     # intervalos de confianza, tabla comparativa y figuras
  echo "== Fin $(date '+%Y-%m-%d %H:%M:%S')"
} 2>&1 | tee "${LOG}"
