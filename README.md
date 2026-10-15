# Pregunta 1: Clasificación multietiqueta de anomalías de rodilla en RM

Guía de ejecución para reproducir los resultados del informe.

La ejecución tiene dos partes:

| Parte | Dónde | Pasos | Qué hace |
|---|---|---|---|
| A | Local (PC con los DICOM) | 01 y 02 | Selecciona los estudios y convierte los DICOM en volúmenes compactos (sagital y coronal, ~620 MB cada plano). |
| B | Google Colab con GPU | 03, 04 y 05 | Entrena, evalúa, genera Grad-CAM e intervalos de confianza. |

---

## 0. Requisitos

- Python 3.10 o superior.
- Cuenta en Kaggle con las reglas de la competencia aceptadas:
  https://www.kaggle.com/competitions/rsna-knee-abnormality-detection
- Unos 600 GB libres en disco para los DICOM de entrenamiento (`train_series`).
- Una cuenta de Google con espacio en Drive (~1 GB) para la parte B.

Descarga del repositorio e instalación de dependencias en local:

```bash
git clone https://github.com/dfranco79/redes_neuronales_examen_parcial_grupo05.git
cd redes_neuronales_examen_parcial_grupo05
pip install -r requirements.txt
```

## 1. Datos de entrada

Coloca estos archivos en la carpeta `datos/` del repositorio:

| Archivo | Origen |
|---|---|
| `datos/train.csv` | Kaggle, pestaña *Data* de la competencia (contiene las 58 etiquetas doradas). |
| `datos/train_series.csv` | Kaggle, pestaña *Data* de la competencia. |
| `datos/train_etiquetado_v1.3.csv` | Etiquetas débiles curadas por el grupo (segunda fuente). |

Los DICOM se descargan de Kaggle y se descomprimen en una carpeta local, por ejemplo:

```
D:\kaggle\rsna-knee-abnormality-detection\train_series\<StudyInstanceUID>\<SeriesInstanceUID>\*.dcm
```

Si tu ruta es otra, cámbiala en `configuracion.py`:

```python
DIR_DICOM = Path(r"D:\kaggle\rsna-knee-abnormality-detection\train_series")
```

Si se quiere usar solo el extractor por reglas (sin la segunda fuente), cambia en `configuracion.py`:

```python
FUENTE_ETIQUETAS = "reglas"
```

## 2. Parte A: ejecución local

Ejecutar desde una **terminal** (CMD, PowerShell o terminal de VS Code), dentro de la carpeta del
repositorio. No ejecutar desde un notebook de Jupyter: en Windows el procesamiento en paralelo
del paso 02 no funciona dentro de un notebook.

```bash
# Paso 01: etiquetas débiles, validación contra el oro y selección de 558 estudios
python 01_seleccion_estudios.py

# Paso 02: DICOM sagital con supresión grasa -> tensores/principal/*.npz
python 02_preprocesar_dicom.py

# Paso 02 para la ablación de plano: serie coronal -> tensores/alterna/*.npz
# (necesario para reproducir la Ablación 4 del informe; si no se ejecuta, esa ablación se omite)
python 02_preprocesar_dicom.py --plano alterna
```

Salidas de la parte A:

```
datos/seleccion_estudios.csv                     estudios elegidos, etiquetas suaves y partición
salidas/validacion_etiquetas_debiles.csv         extractor por reglas frente a las 58 etiquetas doradas
salidas/prevalencia_seleccion.csv                prevalencia por hallazgo y partición
tensores/principal/*.npz                         un volumen por estudio (24 x 256 x 256, uint8)
tensores/principal/registro_preprocesamiento.csv estado de cada estudio (ok / error / sin_carpeta)
tensores/alterna/*.npz                           lo mismo con la serie coronal
tensores/alterna/registro_preprocesamiento.csv   estado de cada estudio en el plano coronal
```

Notas:

- Antes de pasar a Colab, revisar ambos `registro_preprocesamiento.csv`: todas las filas deben decir `ok` o `ya_existia`, salvo los dorados que falten en disco (`sin_carpeta`).
- El paso 02 muestra el avance cada 25 estudios. Si se interrumpe, al volver a ejecutarlo salta los `.npz` que ya existen.
- El paso 01 solo elige estudios cuya carpeta exista en el disco. Si la descarga de Kaggle está incompleta, algunos estudios dorados aparecerán como `sin_carpeta` en el registro del paso 02. En nuestra ejecución se evaluaron 53 de los 58 dorados.

## 3. Parte B: ejecución en Google Colab

1. Subir a Google Drive la carpeta completa del repositorio, **incluyendo** `datos/` y `tensores/` (con sus dos subcarpetas, `principal` y `alterna`), por ejemplo a `MyDrive/rsna_rodilla`. Los DICOM no se suben.
2. Abrir un notebook nuevo en https://colab.research.google.com
3. Activar la GPU: **Entorno de ejecución > Cambiar tipo de entorno de ejecución > GPU (T4) > Guardar**.
4. Ejecutar en una celda (ajustar la ruta si la carpeta tiene otro nombre):

```python
from google.colab import drive; drive.mount('/content/drive')
%cd /content/drive/MyDrive/rsna_rodilla
!mkdir -p salidas
!bash ejecutar_colab.sh 2>&1 | tee salidas/log_colab.txt
```

`ejecutar_colab.sh` ejecuta en orden:

| Paso | Comando | Qué hace |
|---|---|---|
| 03 | `python 03_entrenar.py --arquitectura cero` | CNN 2.5D desde cero |
| 03 | `python 03_entrenar.py --arquitectura resnet50` | ResNet-50 preentrenada (modelo base) |
| 03 | `python 03_entrenar.py --arquitectura resnet50 --perdida focal` | Ablación: pérdida focal |
| 03 | `python 03_entrenar.py --arquitectura resnet50 --etiquetas consenso` | Ablación: solo etiquetas de consenso |
| 03 | `python 03_entrenar.py --arquitectura resnet50 --balanceo pos_weight` | Mitigación del desbalance |
| 03 | `python 03_entrenar.py --arquitectura resnet50 --plano alterna` | Ablación de plano (solo si existe `tensores/alterna`) |
| 04 | `python 04_gradcam.py --experimento ... --arquitectura ...` | Mapas Grad-CAM de ambos modelos |
| 05 | `python 05_analisis_informe.py` | Intervalos de confianza bootstrap, tabla comparativa y figuras |

Tiempo aproximado en una GPU T4: entre 19 y 29 segundos por época, unos 6 minutos por experimento.

### Correr un solo experimento

Si ya se ejecutó el flujo completo y solo se quiere agregar o repetir un experimento (por ejemplo, la ablación de plano después de subir `tensores/alterna`), no hace falta volver a correr todo:

```python
from google.colab import drive; drive.mount('/content/drive')
%cd /content/drive/MyDrive/rsna_rodilla
!pip install -q pydicom timm
!python 03_entrenar.py --arquitectura resnet50 --plano alterna 2>&1 | tee salidas/log_alterna.txt
!python 05_analisis_informe.py
```

El paso 05 siempre se vuelve a correr al final, porque recalcula los intervalos de confianza y la tabla comparativa con todos los experimentos que existan en `salidas/`.

Opciones de `03_entrenar.py`:

| Opción | Valores | Por defecto |
|---|---|---|
| `--arquitectura` | `cero`, `resnet50` (o cualquier modelo de `timm`) | `resnet50` |
| `--perdida` | `bce`, `focal` | `bce` |
| `--etiquetas` | `suave`, `consenso` | `suave` |
| `--balanceo` | `ninguno`, `pos_weight` | `ninguno` |
| `--plano` | `principal` (sagital), `alterna` (coronal) | `principal` |
| `--epocas` | número entero | 15 |
| `--sin_preentrenar` | (bandera) | usa pesos de ImageNet |

El nombre del experimento se arma solo: `<arquitectura>_<perdida>_<etiquetas>_<balanceo>_<plano>`.

Para verificar que todos los experimentos se evaluaron con el mismo número de estudios dorados (en nuestra ejecución, 53 en los 6):

```python
import pandas as pd, glob
for f in glob.glob('salidas/*/predicciones_oro.csv'):
    print(len(pd.read_csv(f)), f.split('/')[1])
```

## 4. Resultados que se generan

```
salidas/<experimento>/metricas_oro.csv       AUC-ROC, precisión, recall, F1 y umbral por hallazgo
salidas/<experimento>/predicciones_oro.csv   probabilidades por estudio dorado
salidas/<experimento>/resumen.json           argumentos, AUC macro y pérdida/AUC por época
salidas/<experimento>/mejor_modelo.pt        pesos del mejor modelo
salidas/<experimento>/gradcam/*.png          mapas Grad-CAM (solo modelo base y desde cero)
salidas/comparacion_experimentos.csv         AUC macro de todos los experimentos
salidas/intervalos_confianza.csv             IC 95 % bootstrap y diferencias contra el modelo base
salidas/figuras/*.pdf                        curvas, AUC por hallazgo y paneles Grad-CAM
salidas/log_colab.txt                        registro completo de la ejecución
```

Para descargar los resultados sin los pesos de los modelos (que pesan mucho):

```python
!zip -r salidas_rodilla.zip salidas -x "*.pt"
```

## 5. Reglas de evaluación

- Las etiquetas doradas solo se usan para la evaluación final. No se usan para entrenar, ni para la parada temprana, ni para elegir umbrales.
- La parada temprana y los umbrales (máximo F1 por hallazgo) se calculan sobre la validación débil (75 estudios).
- Métrica principal: AUC-ROC macro sobre las 12 etiquetas, igual que la competencia.
- Semilla fija: `SEMILLA = 42` en `configuracion.py`.
