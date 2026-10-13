#!/usr/bin/env bash
# Ejecuta en Colab todos los experimentos de la Pregunta 1 (requiere tensores/ ya subidos).
set -e
pip install -q pydicom timm
python 03_entrenar.py --arquitectura cero
python 03_entrenar.py --arquitectura resnet50
python 03_entrenar.py --arquitectura resnet50 --perdida focal
python 03_entrenar.py --arquitectura resnet50 --etiquetas consenso
python 03_entrenar.py --arquitectura resnet50 --balanceo pos_weight
if [ -d tensores/alterna ]; then python 03_entrenar.py --arquitectura resnet50 --plano alterna; fi
python 04_gradcam.py --experimento resnet50_bce_suave_ninguno_principal --arquitectura resnet50
python 04_gradcam.py --experimento cero_bce_suave_ninguno_principal --arquitectura cero
python 05_analisis_informe.py     # intervalos de confianza, tabla comparativa y figuras
