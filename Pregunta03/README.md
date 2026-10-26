# Pregunta 3 — Monitoreo de fauna silvestre con cámaras trampa (YOLOv8 sobre CCT-20)

Examen Parcial de Redes Neuronales y Aprendizaje Profundo · FIIS, Universidad Nacional de Ingeniería (UNI) · 2026-II · Grupo 05

**Integrantes:** Guillermo Cajo · Andre Carasas · Daniel Franco · Ivonne Lazo · David Grabiel · Joel Renojo

Este directorio contiene el código y las instrucciones para reproducir los resultados del informe de la Pregunta 3: un detector YOLOv8s que localiza y clasifica 15 categorías (14 especies y vehículos) en fotografías de cámaras trampa del dataset **Caltech Camera Traps (CCT-20)**, un estudio de generalización entre ubicaciones de cámara y un módulo de monitoreo ecológico por reglas.

---

## 1. Resumen del desarrollo

El flujo tiene dos partes: la **preparación de datos en tu PC** (sin GPU) y el **entrenamiento y análisis en Google Colab** (con GPU).

| Etapa | Qué se hace | Archivo |
|---|---|---|
| 1. Particiones | Lee los 3 JSON de LILA, indexa las imágenes y construye las particiones *cis*/*trans* | `paso1_particiones.py` |
| 2. Verificación | Controles anti-fuga, resumen por partición (Tabla 1 del código) y gráfico de desbalance | `paso2_verificar.py` |
| 3. Exportación | Reduce las imágenes a ≤ 640 px y las exporta con etiquetas en formato YOLO (reanudable) | `paso3_exportar_yolo.py` |
| 4. Empaquetado | Comprime todo en `yolo_data_cct20.zip` | `paso4_comprimir.py` |
| — | Ejecuta los pasos 1 a 4 en orden | `preparar_local.py` |
| — | Utilidades compartidas por los pasos 1 a 4 | `cct_comun.py` |
| 5. Entrenamiento | YOLOv8s preentrenado en COCO; se omite si ya terminó y se reanuda si quedó a medias | `entrenar.py` |
| 6. Evaluación | mAP@0.5 y mAP@0.5:0.95 por partición y por especie; brecha *cis* − *trans*; Spearman | `evaluar.py` |
| 7. Robustez | Recall por tamaño del animal y por un indicador aproximado de oclusión | `robustez.py` |
| 8. Interpretabilidad | Grad-CAM (capa 9, SPPF): energía en caja, enriquecimiento y galería | `gradcam_cct.py` |
| 9. Monitoreo | Reporte por especie y periodo sobre una cámara no vista | `monitoreo.py` |
| — | Configuración, copia y verificación de datos, experimentos A/B/C | `colab_comun.py` |
| — | Orquesta las etapas 5 a 9 en Colab | `cct20_colab.ipynb` |

Los módulos de Colab contienen la lógica; el notebook solo los llama.

---

## 2. Estructura esperada

### En tu PC (parte local)

```
local/
├── json/                      # los 3 JSON de LILA (ver nombres abajo)
├── cct_images/                # imágenes .jpg de CCT-20 (≈ 244 584 archivos; se buscan en subcarpetas)
├── codigo/
│   ├── cct_comun.py
│   ├── paso1_particiones.py
│   ├── paso2_verificar.py
│   ├── paso3_exportar_yolo.py
│   ├── paso4_comprimir.py
│   └── preparar_local.py
├── requirements.txt           # dependencias (alternativa: environment.yml)
├── environment.yml
└── work_q3/                   # salida (se crea sola)
    ├── particiones.csv, anotaciones.csv, categorias.json
    ├── results/               # tabla1_dataset.csv, desbalance_train.csv, desbalance.png
    ├── yolo_data/             # imágenes reducidas, etiquetas YOLO, meta.csv, classes.json, .yaml
    └── yolo_data_cct20.zip    # ≈ 3.95 GB, 97 576 archivos
```

Los tres JSON (nombres por defecto, editables con `--bbox-json`, `--splits-json` y `--images-json`):

| Archivo | Contenido |
|---|---|
| `caltech_bboxes_20200316.json` | Imágenes con cajas y categorías |
| `CaltechCameraTrapsSplits_v0.json` | Particiones oficiales (`eccv_train`, `eccv_val`) |
| `caltech_images_20210113.json` | Metadatos de todas las imágenes (de aquí salen las vacías) |

`json/`, `cct_images/` y `work_q3/` pesan mucho y no se suben a GitHub.

### En Google Drive (parte Colab)

```
MyDrive/POSGRADO_IA/2026_2/REDES_NEURONALES/EXAMEN_PARCIAL/P03/
├── yolo_data_cct20.zip          # generado por preparar_local.py
├── cct20_colab.ipynb            # notebook orquestador
├── codigo/
│   ├── colab_comun.py
│   ├── entrenar.py
│   ├── evaluar.py
│   ├── robustez.py
│   ├── gradcam_cct.py
│   ├── monitoreo.py
│   └── LEEME_colab.md
└── resultados/                  # se crea solo
    ├── runs/                    # corridas de Ultralytics (pesos, curvas)
    ├── tablas/                  # tablas CSV y figuras
    └── modelos/                 # copia estable de best.pt
```

---

## 3. Cómo reproducir

### Requisitos

- **Datos:** CCT-20 de [LILA BC](https://lila.science/datasets/caltech-camera-traps) (los tres JSON y las imágenes). Revisa la licencia y las condiciones de uso en esa página.
- **Parte local:** Python 3.13. No necesita GPU. Los pasos 1 y 3 recorren y reducen unas 49 000 imágenes, por lo que conviene un disco rápido.
- **Parte Colab:** cuenta de Google con Drive y una **GPU** (se usó una Tesla T4).
- **Versiones usadas en la corrida del informe:** Python 3.13, PyTorch 2.11 (CUDA 13.0) y Ultralytics 8.4.188.

### Instalar el entorno

Con **pip** (`requirements.txt`):

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate        Linux/macOS:  source .venv/bin/activate
pip install -r requirements.txt
```

O con **conda** (`environment.yml`):

```bash
conda env create -f environment.yml
conda activate cct20-p3
```

**Solo para la parte local** no hace falta instalar PyTorch ni Ultralytics (pesan varios GB). Basta con:

```bash
pip install numpy pandas pillow pyyaml matplotlib
```

**En Colab** no hay que instalar nada a mano: casi todo ya viene instalado y la celda 2 del notebook ejecuta `pip install ultralytics`. Para reproducir la versión exacta, cambia esa línea a `%pip install -q ultralytics==8.4.188`.

### Dependencias y dónde se usan

| Paquete (nombre de importación) | Se usa en | Obligatorio en |
|---|---|---|
| `numpy` | `paso1`, `paso3`, `colab_comun`, `robustez`, `gradcam_cct` | local y Colab |
| `pandas` | todos los módulos | local y Colab |
| `pyyaml` (`yaml`) | `cct_comun`, `colab_comun` | local y Colab |
| `pillow` (`PIL`) | `paso1`, `paso3` | local |
| `matplotlib` | `entrenar` (curvas), `gradcam_cct`, `monitoreo`; `paso2` (opcional) | Colab |
| `opencv-python` (`cv2`) | `robustez`, `gradcam_cct` | Colab |
| `ultralytics` | `entrenar`, `evaluar`, `robustez`, `gradcam_cct`, `monitoreo` | Colab |
| `torch` | `colab_comun`, `gradcam_cct` (lo instala `ultralytics`) | Colab |
| `scipy` | lo usa pandas en `Series.corr(method="spearman")` de `evaluar` | Colab |
| `google.colab` | Montar Drive en el notebook | solo Colab (no se instala con pip) |

### Paso 1 — Preparar los datos en tu PC

Con los JSON en `local/json/` y las imágenes en `local/cct_images/`, ejecuta desde cualquier carpeta:

```bash
python codigo/preparar_local.py --resize 640
```

Las rutas se calculan a partir de la ubicación del script (la carpeta `local` es la que contiene a `codigo/`), así que funciona igual en cualquier PC. Si las imágenes están en otro disco, usa `--img-dir`.

Ejecuta los cuatro pasos en orden:

1. Leer los JSON, indexar las imágenes y armar las particiones.
2. Verificar que no haya fuga (ubicaciones compartidas y secuencias repartidas deben ser 0; si no, el script se detiene) y generar la tabla de resumen y el gráfico de desbalance.
3. Exportar a YOLO con imágenes reducidas. Es **reanudable**: lo ya escrito se reutiliza.
4. Comprimir en `work_q3/yolo_data_cct20.zip`. Los `.yaml` no se incluyen; se regeneran en Colab con la ruta correcta.

Opciones útiles:

| Opción | Por defecto | Para qué |
|---|---|---|
| `--pasos 3 4` | `1 2 3 4` | Ejecutar solo algunos pasos (usa lo guardado por los anteriores) |
| `--base` | carpeta `local` | Carpeta que contiene `json/`, `cct_images/` y `work_q3/` |
| `--json-dir`, `--img-dir`, `--work` | `<base>/json`, `<base>/cct_images`, `<base>/work_q3` | Cambiar cada ruta (p. ej. imágenes en un disco externo) |
| `--resize` | 640 | Lado mayor en px de las imágenes exportadas (0 = sin reducir) |
| `--quality` | 90 | Calidad JPEG de las imágenes reducidas |
| `--workers` | 4 | Hilos de exportación |
| `--seed` | 42 | Semilla de las particiones |
| `--trans-val-frac` | 0.5 | Fracción de ubicaciones nuevas que va a `val_trans` |
| `--cis-frac` | 0.10 | Fracción de secuencias de *train* para `val_cis` y para `test_cis` (cada una) |
| `--empty-ratio` | 1.0 | Vacías por imagen con animal en *train* (−1 = todas) |
| `--eval-empty-ratio` | −1 | Vacías en validación y prueba (−1 = todas) |
| `--reexport` | — | Borra `yolo_data` y lo regenera (si cambias particiones, semilla o razones) |
| `--recomprimir` | — | Rehace el zip aunque ya exista y esté al día |
| `--zip-name` | `yolo_data_cct20.zip` | Nombre del zip |

Cada paso también se puede ejecutar solo, por ejemplo `python codigo/paso3_exportar_yolo.py --resize 640`.

La primera ejecución completa tardó ≈ 113 minutos (el paso 3 es el más largo: ≈ 82 min con 4 hilos). Con los datos ya exportados, una segunda ejecución tarda ≈ 2.6 min y omite lo que está al día. Si el zip pesa más de ~14 GB, el script avisa (el Drive gratuito tiene 15 GB); usa entonces `--resize 512` o `--eval-empty-ratio 0.5` con `--reexport`.

Al terminar, sube `work_q3/yolo_data_cct20.zip` a la carpeta `P03` de tu Drive.

### Paso 2 — Ejecutar el notebook en Colab

1. Sube la carpeta `codigo/` (los módulos de Colab) y `cct20_colab.ipynb` a `P03` en Drive. Opcional: guarda `yolov8s.pt` junto al código para no descargarlo.
2. En Colab: *Entorno de ejecución → Cambiar tipo de entorno → GPU (T4)*.
3. En la celda 1 revisa `DRIVE_BASE` y `EXPERIMENTO`.
4. Ejecuta las celdas en orden. El notebook:
   - monta Drive e instala `ultralytics`;
   - copia el zip al disco local de Colab (`/content/cct`), lo descomprime, regenera los `.yaml` y **verifica los conteos contra la tabla de resumen** (cada partición debe decir `OK`);
   - entrena (o reutiliza / reanuda una corrida previa);
   - evalúa, mide robustez, calcula Grad-CAM y genera el reporte de monitoreo.

**Si Colab se desconecta:** vuelve a ejecutar desde la celda 1. La preparación de datos se repite (el disco de Colab es temporal), pero el entrenamiento **se omite si ya terminó** y **se reanuda si quedó a medias**. Con `RETRAIN = True` se fuerza un entrenamiento nuevo (la corrida anterior se conserva renombrada).

### Experimentos disponibles

Se cambian editando `EXPERIMENTO` en la celda 1 y repitiendo desde la celda 4:

| `EXPERIMENTO` | Descripción | Estado |
|---|---|---|
| `B_strong` | Aumento fuerte (mosaic, mixup, copy-paste) | **Ejecutado** (resultados del informe) |
| `A_base` | Aumento suave, sin mosaic/mixup/copy-paste | Previsto, **no ejecutado** |
| `C_strong_os3` | Aumento fuerte + sobremuestreo de especies raras (factor máximo 3) | Previsto, **no ejecutado** |

`C_strong_os3` crea en Colab las carpetas `images/train_os3` y `labels/train_os3` con copias de las imágenes de especies raras; no hace falta volver a exportar los datos en tu PC. `evaluar.comparar_experimentos(cfg)` junta las métricas de las configuraciones que ya se hayan evaluado.

---

## 4. Configuración usada (corrida `yolov8s_B_strong`)

| Aspecto | Valor |
|---|---|
| Modelo | YOLOv8s preentrenado en COCO (11.1 M de parámetros, 28.5 GFLOPs) |
| Resolución / lote / épocas | 640 px · 32 · 50 (todas completadas; paciencia 15 no se activó) |
| Datos de entrenamiento | 22 487 imágenes (todo el *train*), 1 704 de ellas vacías |
| Optimizador | Automático de Ultralytics (MuSGD, lr 0.01), decaimiento coseno, AMP |
| Aumento (*strong*) | mosaic 1.0, mixup 0.15, copy-paste 0.3, escala 0.6, HSV (0.03, 0.7, 0.5), rotación 5°, volteo 0.5 |
| Mosaic | Activo en las épocas 1 a 40; se desactiva en las últimas 10 (valor por defecto) |
| Semilla | 42, modo determinista |
| Cómputo | Tesla T4 en Colab, 4.9 h |
| Selección del mejor modelo | Por *val_cis* (≈ época 39) |

### Particiones

Se usa la partición oficial `eccv_train` / `eccv_val`, que separa ubicaciones vistas de ubicaciones nuevas:

| Partición | Imágenes | Con animal | Vacías | Ubicaciones |
|---|---|---|---|---|
| Entrenamiento | 22 487 | 20 783 | 1 704 | 10 |
| Val. *cis* | 2 775 | 2 555 | 220 | 10 |
| Val. *trans* | 10 638 | 9 473 | 1 165 | 5 |
| Prueba *cis* | 2 859 | 2 653 | 206 | 10 |
| Prueba *trans* | 10 027 | 9 405 | 622 | 5 |

- *trans*: las 10 ubicaciones de `eccv_val` se reparten por ubicación en `val_trans` (7, 28, 46, 105, 125) y `test_trans` (0, 40, 78, 100, 130), con semilla 42.
- *cis*: de las ubicaciones de entrenamiento se reservan **secuencias completas** (10 % para `val_cis` y 10 % para `test_cis`), no fotogramas sueltos.
- Verificaciones del paso 2: 0 ubicaciones *trans* compartidas con entrenamiento y 0 secuencias en más de una partición.

---

## 5. Resultados principales

**mAP por partición** (punto de control elegido en `val_cis`, por lo que esa cifra es optimista):

| Partición | mAP@0.5 | mAP@0.5:0.95 | P | R |
|---|---|---|---|---|
| Val. *cis* | 0.963 | 0.665 | 0.889 | 0.952 |
| Val. *trans* | 0.484 | 0.307 | 0.719 | 0.454 |
| Prueba *cis* | 0.901 | 0.635 | 0.929 | 0.876 |
| Prueba *trans* | 0.563 | 0.384 | 0.734 | 0.511 |
| **Brecha validación** (*cis* − *trans*) | **+0.478** | **+0.358** | | |
| **Brecha prueba** (*cis* − *trans*) | **+0.338** | **+0.251** | | |

**Robustez** (recall en `test_trans`, IoU ≥ 0.5, conf ≥ 0.25):

| Factor | Grupo | n | Recall |
|---|---|---|---|
| Tamaño | pequeño (< 1 %) | 709 | 0.495 |
| Tamaño | mediano (1–8 %) | 7 032 | 0.831 |
| Tamaño | grande (> 8 %) | 2 107 | 0.736 |
| Oclusión (indicador aproximado) | bajo / medio / alto | 2 267 / 2 917 / 4 664 | 0.775 / 0.841 / 0.758 |

El indicador de oclusión combina el contraste de la caja frente a un anillo de fondo con el contacto con el borde de la imagen: nivel *alto* si la caja toca el borde o su contraste está en el tercil inferior, *medio* en el tercil intermedio y *bajo* en el superior.

**Grad-CAM** (capa 9, SPPF; 100 imágenes por partición): enriquecimiento de **1.040** en ubicaciones vistas y **0.833** en ubicaciones nuevas (≈ 1 significa atención repartida según el área, sin preferencia por el animal).

**Monitoreo** (cámara 100, no vista en entrenamiento): 4 320 imágenes, fauna detectada en 3 948, tasa estimada de vacías de **0.086**, 11 especies distintas.

El desglose por especie, el análisis del desbalance, la discusión y las limitaciones están en el informe (`informe_P3_v2.docx`).

---

## 6. Archivos que se generan

En `resultados/` de Drive (con `<exp>` = por ejemplo `yolov8s_B_strong`):

| Archivo | Producido por | Tabla del informe |
|---|---|---|
| `tabla1_dataset.csv` (en `work_q3/results/`) | `paso2_verificar.py` | Tabla 2 |
| `tablas/tabla2_metricas_<exp>.csv` | `evaluar.py` | Tabla 3 |
| `tablas/tabla3_ap_por_especie_<exp>.csv`, `tablas/ap_por_especie_<exp>.csv` | `evaluar.py` | Tabla 4 |
| `tablas/tabla4_robustez_<exp>.csv`, `tablas/detalle_robustez_<exp>.csv` | `robustez.py` | Tabla 5 |
| `tablas/tabla5_gradcam_<exp>.csv` | `gradcam_cct.py` | Tabla 6 |
| `tablas/tabla6_monitoreo_<exp>.csv`, `tablas/monitoreo_por_periodo_<exp>.csv` | `monitoreo.py` | Tabla 7 (apéndice) |
| `tablas/curvas_<exp>.png` | `entrenar.py` | Figura 1 |
| `work_q3/results/desbalance.png` | `paso2_verificar.py` | Figura 2 |
| `tablas/gradcam_galeria_<exp>.png` y carpeta `tablas/gradcam_<exp>/` | `gradcam_cct.py` | Figura 3 |
| `tablas/monitoreo_actividad_<exp>.png` | `monitoreo.py` | Figura 4 |
| `runs/<exp>/weights/best.pt` y `last.pt`, y copia en `modelos/<exp>_best.pt` | `entrenar.py` | — |
| `tablas/comparacion_configuraciones.csv` | `evaluar.comparar_experimentos` | — |

La numeración de las tablas dentro del código (Tabla 1 a 6) no coincide con la del informe, que añade una tabla de decisiones de entrenamiento al inicio.

---

## 7. Limitaciones conocidas

- Solo se ejecutó la configuración `B_strong` con una semilla: no hay comparación de aumentos ni evaluación de la mitigación del desbalance.
- `val_cis` se usó para elegir el punto de control, por lo que sus cifras son optimistas.
- Las particiones *trans* tienen solo 5 ubicaciones cada una y sus especies difieren; las especies con muy pocas instancias (*fox*, *badger*) dan métricas inestables.
- Solo entre 6 % y 11 % de las imágenes de evaluación son vacías (≈ 70 % en el dataset completo).
- El indicador de oclusión es aproximado y no resultó fiable.
- Grad-CAM se evaluó en 100 imágenes por partición, con la puntuación de clase máxima de la salida como objetivo, y el mapa del backbone puede no reflejar la decisión de la cabeza de detección.
- Las imágenes se reducen a ≤ 640 px antes de entrenar, lo que puede perjudicar a los animales pequeños.

---

## 8. Pendientes de entrega

- [ ] El enunciado pide ejecución de extremo a extremo con **un único comando**. Hoy la parte local se lanza con uno (`python codigo/preparar_local.py`), pero la parte de Colab se ejecuta desde un notebook; documentar aquí el comando final que se acuerde.
- [ ] *(Opcional, no lo exige el enunciado de la Pregunta 3)* Ejecutar `A_base` y `C_strong_os3` para justificar con datos la elección del aumento y probar una mitigación del desbalance.

---

## 9. Declaración de uso de IA

Se usó un asistente de IA para apoyar la redacción y el formato del informe y para revisar y ajustar partes del código de los notebooks. Las decisiones de diseño, la ejecución de los experimentos y el análisis de resultados fueron revisados por el grupo. *(Ajustar a lo que el grupo realmente haya hecho.)*

---

## 10. Referencias

1. Beery, S., Van Horn, G., Perona, P. (2018). *Recognition in Terra Incognita*. ECCV.
2. Norouzzadeh, M. S., et al. (2018). Automatically identifying, counting, and describing wild animals in camera-trap images with deep learning. *PNAS*.
3. Beery, S., Morris, D., Yang, S. (2019). *Efficient Pipeline for Camera Trap Image Review*.
4. Jocher, G., Chaurasia, A., Qiu, J. (2023). *Ultralytics YOLOv8*. https://github.com/ultralytics/ultralytics
5. Selvaraju, R. R., et al. (2017). Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization. *ICCV*.
