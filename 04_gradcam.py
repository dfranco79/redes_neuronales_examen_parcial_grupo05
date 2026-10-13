"""
Paso 4: Grad-CAM (Selvaraju et al. 2017) sobre estudios dorados positivos.

Para cada hallazgo pedido toma los estudios oro positivos con mayor probabilidad, elige el
corte con más peso de atención y superpone el mapa de activación de la última capa conv.
Uso: python 03_entrenar.py ...  y luego
     python 04_gradcam.py --experimento resnet50_bce_suave_ninguno_principal --arquitectura resnet50
"""
import argparse

import cv2
import numpy as np
import pandas as pd
import torch

import configuracion as C
from datos import DatasetRodilla
from modelos import crear_modelo

HALLAZGOS_POR_DEFECTO = ["ACL", "Medial Meniscus", "Effusion", "Baker's"]


def gradcam(modelo, x, indice_etiqueta):
    """Devuelve (mapas por corte (D,h,w) normalizados, pesos de atención, probabilidad)."""
    guardado = {}
    h1 = modelo.capa_gradcam.register_forward_hook(lambda m, i, o: guardado.__setitem__("act", o))
    h2 = modelo.capa_gradcam.register_full_backward_hook(lambda m, gi, go: guardado.__setitem__("grad", go[0]))
    modelo.zero_grad()
    logits, pesos = modelo(x)
    logits[0, indice_etiqueta].backward()
    h1.remove(); h2.remove()
    act, grad = guardado["act"], guardado["grad"]                    # (D, canales, h, w)
    cam = torch.relu((grad.mean((2, 3), keepdim=True) * act).sum(1))
    cam = cam / cam.amax((1, 2), keepdim=True).clamp(min=1e-6)
    return cam.detach().cpu().numpy(), pesos[0].detach().cpu().numpy(), torch.sigmoid(logits[0, indice_etiqueta]).item()


def superponer(corte: np.ndarray, mapa: np.ndarray) -> np.ndarray:
    gris = cv2.cvtColor((corte * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    color = cv2.applyColorMap((cv2.resize(mapa, corte.shape[::-1]) * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.addWeighted(gris, 0.6, color, 0.4, 0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--experimento", required=True)
    p.add_argument("--arquitectura", required=True)
    p.add_argument("--plano", default="principal")
    p.add_argument("--hallazgos", nargs="+", default=HALLAZGOS_POR_DEFECTO)
    p.add_argument("--casos", type=int, default=3)
    a = p.parse_args()

    dir_exp = C.DIR_SALIDAS / a.experimento
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelo = crear_modelo(a.arquitectura, preentrenado=False)
    modelo.load_state_dict(torch.load(dir_exp / "mejor_modelo.pt", map_location=dispositivo))
    modelo.to(dispositivo).eval()
    seleccion = pd.read_csv(C.CSV_SELECCION)
    oro = DatasetRodilla(seleccion[seleccion.particion == "oro"], C.DIR_TENSORES / a.plano, entrenamiento=False)
    # alinear predicciones con el dataset por StudyInstanceUID (no por posición de fila)
    predicciones = (pd.read_csv(dir_exp / "predicciones_oro.csv").set_index("StudyInstanceUID")
                    .reindex(oro.tabla.StudyInstanceUID).reset_index())
    faltan = int(predicciones[C.ETIQUETAS[0]].isna().sum())
    if faltan:
        print(f"Aviso: {faltan} estudios oro no tienen predicción en este experimento (se omiten).")
    salida = dir_exp / "gradcam"; salida.mkdir(exist_ok=True)

    for hallazgo in a.hallazgos:
        j = C.ETIQUETAS.index(hallazgo)
        positivos = np.where((oro.y[:, j] >= 0.5) & predicciones[hallazgo].notna().values)[0]
        orden = positivos[np.argsort(-predicciones[hallazgo].values[positivos])][: a.casos]
        for i in orden:
            x, _, _ = oro[i]
            mapas, pesos, prob = gradcam(modelo, x.unsqueeze(0).to(dispositivo), j)
            k = int(np.argmax(pesos))                                      # corte más atendido
            corte = x[k, 1].numpy() * 0.229 + 0.485                        # canal central, sin normalizar
            imagen = np.hstack([cv2.cvtColor((np.clip(corte, 0, 1) * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR),
                                superponer(np.clip(corte, 0, 1), mapas[k])])
            estudio = oro.tabla.StudyInstanceUID[i][-8:]
            cv2.imwrite(str(salida / f"{hallazgo.replace(' ', '_')}_{estudio}_p{prob:.2f}_corte{k}.png"), imagen)
        print(f"{hallazgo}: {len(orden)} mapas guardados en {salida}")


if __name__ == "__main__":
    main()
