# Pregunta 1: análisis y decisiones

## 1. Qué pide el examen (en corto)
Un modelo que, a partir de la RM de rodilla, prediga a la vez 12 hallazgos (multietiqueta).
Hay que entregar:
- un extractor de etiquetas desde el texto, validado;
- dos arquitecturas: una CNN propia y una con backbone preentrenado;
- AUC-ROC macro sobre el conjunto dorado;
- ablaciones, Grad-CAM, análisis de desbalance y discusión de sesgo.

## 2. Qué es la "etiqueta dorada"
Son los **58 estudios cuyo `train.csv` original de Kaggle trae las 12 columnas llenas**. Son las
etiquetas que anotaron radiólogos. Los otros 4.349 estudios traen esas columnas vacías y solo tienen el
reporte en texto. El PDF lo dice así: *"Solo del orden de 58 estudios cuentan con etiquetas doradas"*.
Lo confirmé en los datos: exactamente 58 filas con valores. Por eso la v1.3 conserva esas 58 sin tocar.

Uso correcto:
- Las 58 son el **conjunto de prueba limpio**: no se entrena con ellas, no se eligen umbrales con ellas
  y no se hace early stopping con ellas.
- Todo lo demás se entrena con etiquetas **débiles**, derivadas del texto.

## 3. Cómo se relacionan los archivos (verificado leyendo los DICOM)
```
train_series/<StudyInstanceUID>/<SeriesInstanceUID>/<SOPInstanceUID>.dcm
```
- `train_etiquetado_v1.3.csv`: 1 fila por estudio (4.407), `StudyInstanceUID` + reporte + 12 etiquetas.
- `train_series.csv`: 24.371 series (3 a 14 por estudio). Para cada serie indica el plano
  (Sagittal/Coronal/Axial) y si es sensible a líquido (`Fluid_Sensitive`) y con supresión grasa
  (`Fat_Suppression`). En este CSV las dos columnas coinciden siempre.
- Dentro de cada serie, un `.dcm` por corte. **El nombre del archivo es un UID aleatorio y no
  indica el orden.** El orden real sale de la cabecera: `ImagePositionPatient` proyectada sobre la normal
  de `ImageOrientationPatient`. Lo verifiqué en una serie de 36 cortes: el espaciado es
  constante de 2,87 mm.
- Ejemplo real de cabecera: Philips 1,5 T, `SeriesDescription = "DP SPIR CS_SAG"` (densidad
  protónica con supresión grasa, sagital), 512x512 o 640x640, 22-36 cortes, sin compresión.
  Pesa entre 0,2 y 0,8 MB por corte.

## 4. Reducción de escala (lo que hace el código)
| Decisión | Valor | Por qué |
|---|---|---|
| Plano/secuencia principal | Sagital, sensible a líquido, con supresión grasa | Es la recomendación del PDF y muestra LCA, meniscos, derrame y Baker. La tienen 4.150 de 4.407 estudios. |
| Plano para la ablación | Coronal FS (4.248 estudios) | Mejor para el LCM y los compartimentos medial y lateral. |
| Muestra | 500 débiles (425 entrenamiento + 75 validación débil) + 58 oro | El PDF sugiere entre 300 y 600. |
| Estratificación | Mínimo 40 positivos por hallazgo, empezando por los más raros | Asegura casos positivos de Fractura, MCL, etc. |
| Volumen por estudio | 24 cortes x 256x256 uint8 (~1,1 MB comprimido) | Unos 620 MB en total: cabe en Drive y en la RAM de Colab. |
| Entrada al modelo | Tripletes de cortes contiguos (2.5D), 1 de cada 2, recorte de 224 | Así se puede usar un backbone 2D de ImageNet. |

Dos de los 58 dorados no tienen serie sagital FS. El preprocesador usa la mejor serie sagital
disponible o, si no hay ninguna, la coronal.

**PySpark no se usa.** Con 558 estudios, Python con procesos en paralelo es más simple y rápido; Spark
solo agregaría arranque y dependencias sin ganancia. Tendría sentido si se procesaran los 800 mil DICOM.

## 5. Etiquetas débiles (entregable 1)
Fuentes disponibles:
1. **Extractor por reglas multilingüe** (`extractor_etiquetas.py`): código propio. Maneja negaciones,
   9 idiomas y asigna cada lesión a la estructura más cercana del texto.
2. **CSV curado v1.3**: reportes revisados con un LLM aplicando una rúbrica bibliográfica.
   **Esto debe declararse en el informe.** Si el profesor exige que todo sea código propio, pon
   `FUENTE_ETIQUETAS = "reglas"` en `configuracion.py`.

Validación contra las 58 doradas (datos reales, `salidas/validacion_etiquetas_debiles.csv`):

| Fuente | Acuerdo por celda | F1 macro | Kappa macro |
|---|---|---|---|
| Reglas | 0,809 | 0,717 | 0,553 |
| LLM con rúbrica estricta (lectura ciega) | 0,823 | 0,701 | n/d |

**Tasa de ruido estimada: alrededor del 18-19% de celdas en desacuerdo con el oro.**
Los más ruidosos son:

| Hallazgo | Kappa |
|---|---|
| Sinovitis | 0,32 |
| Derrame | 0,31 |
| Lateral OA | 0,38 |

El oro usa un criterio más amplio que la bibliografía estricta: cuenta la condropatía como OA y
marca sinovitis o derrame con hallazgos indirectos. Ese desfase de criterio es parte del ruido, y es
un buen punto para la discusión del informe.

Cómo se usa el ruido en el entrenamiento:
- **Etiqueta suave:** si las dos fuentes coinciden vale 0 o 1 con confianza 1; si discrepan vale
  0,5 con confianza 0,5.
- **Ablación "consenso":** en lugar de la etiqueta suave, se ignoran las celdas en desacuerdo
  (peso 0). Es exactamente la comparación que sugiere el PDF: ponderar por confianza frente a usar
  solo alta confianza.

## 6. Arquitecturas (entregable 2)
Las dos comparten la misma idea. Cada triplete de cortes pasa por un extractor 2D. Luego un
**pooling por atención (MIL)** combina los cortes en un vector del estudio y salen **12 sigmoides
independientes**. El pooling permite entrenar con una etiqueta por estudio sin saber en qué corte
está la lesión. Además, sus pesos dicen qué corte miró el modelo, y eso alimenta el Grad-CAM.

- `cero`: CNN propia de 5 bloques (dos convoluciones con BatchNorm y ReLU + MaxPool) con 32, 64,
  128, 256 y 512 canales.
- `resnet50` o `efficientnet_b0`: backbone de timm preentrenado en ImageNet, con la misma cabeza.

## 7. Evaluación, ablaciones, desbalance (entregables 3, 4, 6)
**Evaluación:** AUC-ROC macro y por etiqueta sobre las 58 doradas. Precisión, recall y F1 se
calculan con un umbral por etiqueta. Ese umbral es el que maximiza F1 en la validación débil, y así
se justifica sin tocar el oro.

**Ablaciones listas por línea de comandos:**
- pérdida: BCE o focal;
- etiquetas: suaves o consenso;
- plano: sagital o coronal;
- arquitectura: desde cero, ResNet-50 o EfficientNet.

**Desbalance:**
- `salidas/prevalencia_seleccion.csv` da la prevalencia por partición. En entrenamiento va de
  15,5% (Fractura) a 66,6% (Derrame).
- Hay dos mitigaciones para comparar: `--balanceo pos_weight` (negativos/positivos por etiqueta,
  acotado a 10) y la pérdida focal.

## 8. Interpretabilidad (entregable 5)
`04_gradcam.py` toma, para cada hallazgo (por defecto LCA, menisco medial, derrame y Baker), los
estudios dorados positivos con mayor probabilidad. Elige el corte con más atención y guarda la imagen
original junto al mapa superpuesto. Lo que hay que discutir en el informe:
- si el mapa del menisco cae en el menisco;
- si el del LCA cae en la escotadura intercondílea;
- si el del derrame cae en el receso suprapatelar.

## 9. Qué está probado y qué no
Probado en el sandbox:
- la selección corre sobre los datos reales;
- la lectura, el orden y la normalización de una serie DICOM real de un estudio dorado, con imagen
  revisada visualmente;
- entrenamiento y evaluación de punta a punta con las dos arquitecturas, la pérdida focal,
  `pos_weight` y las etiquetas de consenso, más Grad-CAM, usando volúmenes de prueba en CPU.

**No probado:** el entrenamiento real con GPU. Las cifras de AUC las tendrás al correrlo en Colab.

Además, ResNet-50 preentrenado no se pudo bajar en el sandbox (la red estaba bloqueada), así que se
probó con ResNet-18 sin pesos. En Colab descarga los pesos de ImageNet normalmente.

## 10. Riesgo de sesgo lingüístico (insumo para el entregable 7)
El extractor por reglas se validó solo con las 58 doradas, repartidas así por idioma:

| Idioma | Estudios dorados |
|---|---|
| Inglés | 30 |
| Castellano | 8 |
| Turco | 6 |
| Croata | 4 |
| Griego | 3 |
| Búlgaro | 3 |
| Neerlandés | 2 |
| Alemán | 2 |

Con 2 o 3 casos no se puede estimar el ruido por idioma. El ruido de etiqueta probablemente es mayor
en los idiomas con menos vocabulario cubierto, así que el modelo podría rendir peor en esas
poblaciones sin que la métrica global lo muestre.
Recomendación: reportar el acuerdo por idioma cuando haya oro disponible y estratificar la muestra
por centro o idioma.
