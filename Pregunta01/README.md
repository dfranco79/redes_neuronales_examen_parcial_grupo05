# Pregunta 1: Clasificación multietiqueta de anomalías de rodilla en RM (v4)

Repositorio: https://github.com/dfranco79/redes_neuronales_examen_parcial_grupo05
Carpeta de esta pregunta: `Pregunta01/`

Guía de ejecución para reproducir los resultados del informe. La ejecución tiene dos partes:

| Parte | Dónde | Pasos | Qué hace |
|---|---|---|---|
| A | Local (PC con los DICOM) | `local/00_orquestador_local.py` (orquesta 01 y 02) | Selecciona los estudios y convierte los DICOM en volúmenes compactos (sagital y coronal, unos 620 MB por plano). |
| B | Google Colab con GPU | notebook `colab/00_orquestador_colab.ipynb` (orquesta 03, 04 y 05) | Entrena, evalúa, genera Grad-CAM, intervalos de confianza y figuras. |

La parte A se hace en local porque los DICOM pesan cientos de GB y no caben en Colab ni en Drive; la parte B
solo necesita los volúmenes ya procesados y aprovecha la GPU de Colab.

---

## 0. Estructura de carpetas

```
Pregunta01/
├── README.md, requirements.txt, .gitignore
├── configuracion.py                 constantes compartidas (rutas, hiperparámetros, semilla)
│
├── local/                           parte A: se ejecuta en la PC con los DICOM
│   ├── 00_orquestador_local.py      ejecuta 01 y 02 en orden y controla errores (único comando)
│   ├── 01_seleccion_estudios.py     etiquetas débiles y selección de estudios
│   ├── 02_preprocesar_dicom.py      DICOM -> volúmenes .npz
│   └── extractor_etiquetas.py       extractor multilingüe por reglas (lo usa 01)
│
├── colab/                           parte B: se ejecuta en Google Colab con GPU
│   ├── 00_orquestador_colab.ipynb   notebook que ejecuta 03, 04 y 05 en orden y controla errores
│   ├── 03_entrenar.py               entrenamiento y evaluación en los dorados
│   ├── 04_gradcam.py                mapas Grad-CAM
│   ├── 05_analisis_informe.py       intervalos de confianza, tabla comparativa y figuras
│   └── datos.py, modelos.py         dataset 2.5D, arquitecturas y pérdidas
│
├── datos/
│   ├── entrada/                     CSV de entrada (no se modifican)
│   └── generados/                   salidas del paso 01 y registro del orquestador (logs/)
├── tensores/                        salidas del paso 02 (se reutilizan entre corridas)
│   ├── principal/                   serie sagital
│   └── alterna/                     serie coronal
└── ejecuciones/                     una carpeta por corrida de Colab; ninguna pisa a otra
    └── <ID_EJECUCION>/
        ├── experimentos/<experimento>/   modelo, métricas, predicciones, resumen y gradcam/
        ├── analisis/                     intervalos, tabla comparativa y figuras/
        └── logs/                         log de Colab, configuración usada y GPU
```

## 1. Requisitos

- Python 3.10 o superior.
- Cuenta en Kaggle con las reglas de la competencia aceptadas:
  https://www.kaggle.com/competitions/rsna-knee-abnormality-detection
- Espacio en disco para los DICOM de entrenamiento (`train_series`, del orden de cientos de GB).
- Una cuenta de Google con espacio en Drive para la parte B: unos 1.3 GB de datos y tensores, más unos 500 MB por corrida (pesos de los modelos).

Descarga del código e instalación de dependencias en local:

```bash
git clone https://github.com/dfranco79/redes_neuronales_examen_parcial_grupo05.git
cd redes_neuronales_examen_parcial_grupo05/Pregunta01
pip install -r requirements.txt
```

## 2. Datos de entrada

Coloca en `datos/entrada/`:

| Archivo | Origen |
|---|---|
| `train.csv` | Kaggle, pestaña *Data* de la competencia (incluye las 58 etiquetas doradas). No se incluye en el repositorio. |
| `train_series.csv` | Kaggle, pestaña *Data* de la competencia. No se incluye en el repositorio. |
| `etiquetas_curadas_v1.3.csv` | Incluido en el repositorio. Etiquetas débiles curadas por el grupo (segunda fuente): `StudyInstanceUID` y las 12 etiquetas, sin el texto de los reportes. Las 58 filas doradas van vacías, porque para ellas siempre manda la etiqueta de `train.csv`. |

Los DICOM se descargan de Kaggle y se descomprimen en una carpeta local, por ejemplo:

```
D:\kaggle\rsna-knee-abnormality-detection\train_series\<StudyInstanceUID>\<SeriesInstanceUID>\*.dcm
```

Si tu ruta es otra, cámbiala en `configuracion.py` (`DIR_DICOM`) o pásala al orquestador con `--dicom`.

Para entrenar solo con el extractor por reglas (sin la segunda fuente), cambia en `configuracion.py`:

```python
FUENTE_ETIQUETAS = "reglas"
```

## 3. Parte A: ejecución local

Desde una terminal (CMD, PowerShell o la de VS Code), dentro de la carpeta `Pregunta01`, un solo comando:

```bash
python local/00_orquestador_local.py
```

El orquestador ejecuta en orden, cada uno en su propio proceso:

| Clave | Comando | Qué hace |
|---|---|---|
| `01` | `python local/01_seleccion_estudios.py` | Etiquetas débiles, validación contra los dorados y selección de 500 estudios + dorados |
| `02p` | `python local/02_preprocesar_dicom.py --plano principal` | Serie sagital con supresión grasa -> `tensores/principal/*.npz` |
| `02a` | `python local/02_preprocesar_dicom.py --plano alterna` | Serie coronal (ablación de plano) -> `tensores/alterna/*.npz` |

Antes de empezar verifica que existan los CSV de entrada y la carpeta de DICOM. Si un paso falla, se detiene y
muestra desde dónde retomar. Al final muestra el tiempo de cada paso y el resumen del registro del paso 02.

Opciones:

```bash
python local/00_orquestador_local.py --desde 02p          # retomar desde un paso
python local/00_orquestador_local.py --solo 01            # un solo paso
python local/00_orquestador_local.py --limite 10          # prueba rápida: 10 estudios por paso 02
python local/00_orquestador_local.py --dicom "E:\ruta\train_series"
```

Salidas de la parte A:

```
datos/generados/seleccion_estudios.csv           estudios elegidos, etiquetas suaves y partición
datos/generados/validacion_etiquetas_debiles.csv extractor por reglas frente a las 58 etiquetas doradas
datos/generados/prevalencia_seleccion.csv        prevalencia por hallazgo y partición
datos/generados/logs/orquestador_<fecha>.txt     registro completo de la parte A
tensores/principal/*.npz                         un volumen por estudio (24 x 256 x 256, uint8)
tensores/alterna/*.npz                           lo mismo con la serie coronal
tensores/<plano>/registro_preprocesamiento.csv   estado de cada estudio (ok / ya_existia / sin_carpeta / error)
```

Notas:

- El paso 02 salta los `.npz` que ya existen; si se interrumpe, basta con volver a ejecutarlo.
- El paso 01 solo elige estudios de entrenamiento cuya carpeta exista en el disco. Si la descarga de Kaggle está
  incompleta, algunos estudios dorados aparecerán como `sin_carpeta` y se evalúa con los disponibles
  (en nuestra ejecución, 53 de 58).
- En el código, el conjunto dorado aparece como la partición `oro`.

## 4. Parte B: ejecución en Google Colab

1. Subir a Google Drive la carpeta `Pregunta01` completa, **incluyendo** `datos/` y `tensores/`
   (con `principal` y `alterna`). Los DICOM no se suben.
2. En Drive, abrir `Pregunta01/colab/00_orquestador_colab.ipynb` con
   **clic derecho > Abrir con > Google Colaboratory**.
3. Activar la GPU: **Entorno de ejecución > Cambiar tipo de entorno de ejecución > GPU (T4) > Guardar**.
4. En la celda **1. Parámetros**, ajustar la ruta y el nombre de la corrida:

```python
RUTA_PROYECTO = "/content/drive/MyDrive/Pregunta01"   # ubicación de Pregunta01 en tu Drive
ID_EJECUCION = "corrida_01"                           # cada corrida va a ejecuciones/<ID_EJECUCION>/
DESDE = None                                          # para retomar: nombre del paso, por ejemplo "gradcam_base"
```

5. **Entorno de ejecución > Ejecutar todas.** Al final el notebook descarga `resultados_<ID_EJECUCION>.zip`
   (resultados de la corrida y `datos/generados`, sin los pesos `.pt`).

El notebook tiene cinco secciones:

| Sección | Qué hace |
|---|---|
| 1. Parámetros | Ruta del proyecto, nombre de la corrida y paso desde el que retomar |
| 2. Conexión y verificación | Monta Drive, verifica `datos/generados/seleccion_estudios.csv`, `tensores/principal` y la GPU, e instala `pydicom` y `timm` si faltan. Si la corrida ya tiene experimentos, no empieza, para no mezclar corridas |
| 3. Pasos | Define el orden de los pasos (tabla de abajo) |
| 4. Ejecución | Corre cada paso en su propio proceso, muestra y guarda su salida, y se detiene si uno falla |
| 5. Descarga | Arma y descarga el zip de resultados |

Pasos que ejecuta, en orden:

| Clave | Comando | Qué hace |
|---|---|---|
| `cero` | `python colab/03_entrenar.py --arquitectura cero` | CNN 2.5D desde cero |
| `base` | `python colab/03_entrenar.py --arquitectura resnet50` | ResNet-50 preentrenada (modelo base) |
| `focal` | `... --perdida focal` | Ablación 1: pérdida focal |
| `consenso` | `... --etiquetas consenso` | Ablación 2: solo etiquetas de consenso |
| `pos_weight` | `... --balanceo pos_weight` | Ablación 3: mitigación del desbalance |
| `coronal` | `... --plano alterna` | Ablación 4: plano coronal (se omite si no existe `tensores/alterna`) |
| `gradcam_base` | `python colab/04_gradcam.py ...` | Grad-CAM del modelo base |
| `gradcam_cero` | `python colab/04_gradcam.py ...` | Grad-CAM de la CNN desde cero |
| `analisis` | `python colab/05_analisis_informe.py` | Intervalos de confianza bootstrap, tabla comparativa y figuras |

En `ejecuciones/<ID_EJECUCION>/logs/` quedan el log completo, una copia de `configuracion.py` y la GPU asignada.
Si un paso falla, el notebook indica el nombre del paso: se pone en `DESDE` y se vuelve a **Ejecutar todas**;
los pasos anteriores no se repiten.

Tiempo aproximado en una GPU T4: entre 19 y 29 segundos por época, unos 6 minutos por experimento.
Mantener la pestaña de Colab activa durante la corrida para que no se desconecte Drive.

### Correr un solo experimento

Para agregar o repetir un experimento dentro de una corrida existente, en una celda nueva del notebook
(después de ejecutar la sección 2, que deja la carpeta `Pregunta01` como carpeta de trabajo), se fija el identificador con `%env`
y se vuelve a correr el paso 05 al final (recalcula intervalos y tabla con todos los experimentos de esa corrida):

```python
%env ID_EJECUCION=corrida_01
!python colab/03_entrenar.py --arquitectura resnet50 --plano alterna
!python colab/05_analisis_informe.py
```

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

### Verificación y descarga de resultados

Comprobar que todos los experimentos se evaluaron con el mismo número de estudios dorados:

```python
import pandas as pd, glob
for f in sorted(glob.glob('ejecuciones/corrida_01/experimentos/*/predicciones_oro.csv')):
    print(len(pd.read_csv(f)), f.split('/')[-2])
```

Descargar los resultados de la corrida sin los pesos de los modelos:

```python
!zip -r resultados_corrida_01.zip ejecuciones/corrida_01 datos/generados -x "*.pt"
```

## 5. Resultados que se generan (por corrida)

```
ejecuciones/<ID>/experimentos/<exp>/metricas_oro.csv      AUC-ROC, precisión, recall, F1 y umbral por hallazgo
ejecuciones/<ID>/experimentos/<exp>/predicciones_oro.csv  probabilidades por estudio dorado
ejecuciones/<ID>/experimentos/<exp>/resumen.json          argumentos, AUC macro y pérdida/AUC por época
ejecuciones/<ID>/experimentos/<exp>/mejor_modelo.pt       pesos del mejor modelo (no se suben a GitHub)
ejecuciones/<ID>/experimentos/<exp>/gradcam/*.png         mapas Grad-CAM (modelo base y desde cero)
ejecuciones/<ID>/analisis/intervalos_confianza.csv        IC 95 % bootstrap y diferencias contra el modelo base
ejecuciones/<ID>/analisis/comparacion_experimentos.csv    AUC macro de todos los experimentos
ejecuciones/<ID>/analisis/figuras/*.pdf                   curvas, AUC por hallazgo y paneles Grad-CAM
ejecuciones/<ID>/logs/                                    log_colab.txt, configuracion_usada.py y gpu.txt
```

## 6. Reglas de evaluación

- Las etiquetas doradas solo se usan para la evaluación final. No se usan para entrenar, ni para la parada
  temprana, ni para elegir umbrales.
- La parada temprana y los umbrales (máximo F1 por hallazgo) se calculan sobre la validación débil (75 estudios).
- Métrica principal: AUC-ROC macro sobre las 12 etiquetas, igual que la competencia.
- Semilla fija: `SEMILLA = 42` en `configuracion.py`. En GPU los resultados pueden variar ligeramente entre
  corridas aunque la semilla sea la misma; por eso cada corrida se guarda en su propia carpeta.
