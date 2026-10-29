# Sen1Floods11: mapeo de inundaciones con una U-Net desde cero (Pregunta 2)

Segmentación del agua de inundación píxel a píxel con radar Sentinel-1 (VV, VH) sobre el dataset
[Sen1Floods11](https://github.com/cloudtostreet/Sen1Floods11) (Cloud to Street y NASA), con una U-Net implementada
desde cero en PyTorch.

## Cómo correrlo

```bash
pip install -r requirements.txt
python main.py
```

`main.py` descarga los datos de `gs://sen1floods11` (versión v1.1 con la partición oficial; no hace falta gsutil ni
credenciales), entrena y evalúa todos los modelos y guarda en `outputs/`:

| Ruta | Contenido |
|---|---|
| `outputs/results/resultados.json` | todas las métricas |
| `outputs/results/*.csv` | tablas (resumen de modelos, IoU por evento, ruido de etiquetas, tipos de error) |
| `outputs/figures/` | figuras del informe |
| `outputs/analisis.md` | análisis de los resultados |
| `outputs/informe/informe_p2.tex` | informe en LaTeX (plantilla NeurIPS) con figuras |
| `outputs/checkpoints/` | pesos de cada modelo (si la ejecución se corta, al volver a correr se reutilizan) |

Por defecto se usan todos los datos: los 446 chips dorados (con su imagen Sentinel-2) y los
4,160 chips débiles fuera de Bolivia con sus dos etiquetas automáticas (Otsu de Sentinel-1 e índice
espectral de Sentinel-2). Algunas opciones: `--quick true` (prueba rápida de pocos minutos), `--max_weak 1000` (usar menos
chips débiles), `--run_s2_bonus false`, `--epochs 60`. El experimento completo necesita GPU (esta corrida tomó
55 min en `cuda`).

## Estructura

```
main.py            # pipeline completo
src/download.py    # descarga desde la API pública de GCS
src/data.py        # lectura, normalización y Dataset con recortes y aumentos
src/model.py       # U-Net propia
src/losses.py      # BCE / Dice / BCE+Dice con máscara
src/metrics.py     # IoU, F1, precisión, recall, omisión, comisión, IC bootstrap
src/train.py       # entrenamiento con checkpoints reutilizables
src/analysis.py    # ruido de etiquetas, composición, tipos de error del SAR, figuras
src/report.py      # análisis, README e informe a partir de los resultados
notebooks/         # cuaderno de Colab que escribe estos archivos y ejecuta main.py
```

## Configuración usada

U-Net 32/4 (7.85 M parámetros), recortes de 256 px, 40 épocas,
AdamW lr=0.001, pos_weight=3.0, 4160 chips débiles, semilla 42.

## Resultados

Modelo final (elegido por IoU de validación): **Dorados + débiles S2 (índice)**

| Modelo                                  |   Umbral |   IoU test |   IoU medio/chip test |   F1 test |   IoU Bolivia |   F1 Bolivia |
|:----------------------------------------|---------:|-----------:|----------------------:|----------:|--------------:|-------------:|
| Solo dorados · BCE+Dice                 |     0.55 |      0.654 |                 0.413 |     0.79  |         0.583 |        0.736 |
| Solo dorados · BCE                      |     0.55 |      0.662 |                 0.386 |     0.796 |         0.642 |        0.782 |
| Solo dorados · Dice                     |     0.1  |      0.64  |                 0.385 |     0.78  |         0.658 |        0.793 |
| Dorados + débiles S1 (Otsu)             |     0.2  |      0.604 |                 0.332 |     0.753 |         0.706 |        0.828 |
| Dorados + débiles S2 (índice)           |     0.45 |      0.669 |                 0.421 |     0.802 |         0.664 |        0.798 |
| Pre-entreno débiles S1 → ajuste dorados |     0.55 |      0.647 |                 0.391 |     0.786 |         0.659 |        0.795 |
| Sin nubes · SAR (2 canales)             |     0.5  |      0.671 |                 0.431 |     0.803 |         0.685 |        0.813 |
| Sin nubes · SAR+S2 (15 canales)         |     0.55 |      0.875 |                 0.664 |     0.933 |         0.814 |        0.897 |

Análisis completo: `outputs/analisis.md`.
