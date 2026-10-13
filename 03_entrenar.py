"""
Paso 3: entrenamiento y evaluación (correr en Colab con GPU).

Entrena con etiquetas débiles, hace early stopping y elige umbrales en la validación débil,
y evalúa UNA vez sobre las 58 etiquetas doradas (nunca se usan para ajustar nada).

Ejemplos:
  python 03_entrenar.py --arquitectura cero
  python 03_entrenar.py --arquitectura resnet50
  python 03_entrenar.py --arquitectura resnet50 --perdida focal            # ablación pérdida
  python 03_entrenar.py --arquitectura resnet50 --etiquetas consenso       # ablación ruido
  python 03_entrenar.py --arquitectura resnet50 --plano alterna            # ablación plano
  python 03_entrenar.py --arquitectura resnet50 --balanceo pos_weight      # mitigación desbalance
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader

import configuracion as C
from datos import DatasetRodilla
from modelos import crear_modelo, perdida_bce, perdida_focal


def argumentos():
    p = argparse.ArgumentParser()
    p.add_argument("--arquitectura", default="resnet50", help="cero | resnet50 | resnet18 | efficientnet_b0 ...")
    p.add_argument("--perdida", choices=["bce", "focal"], default="bce")
    p.add_argument("--etiquetas", choices=["suave", "consenso"], default="suave")
    p.add_argument("--balanceo", choices=["ninguno", "pos_weight"], default="ninguno")
    p.add_argument("--plano", choices=["principal", "alterna"], default="principal")
    p.add_argument("--epocas", type=int, default=C.EPOCAS)
    p.add_argument("--sin_preentrenar", action="store_true")
    p.add_argument("--nombre", default=None, help="nombre del experimento (carpeta de salida)")
    return p.parse_args()


def auc_por_etiqueta(y, prob) -> dict:
    """AUC-ROC por etiqueta (se omite si la etiqueta tiene una sola clase) y macro-promedio."""
    y = (y >= 0.5).astype(int)
    aucs = {e: roc_auc_score(y[:, j], prob[:, j]) for j, e in enumerate(C.ETIQUETAS) if 0 < y[:, j].sum() < len(y)}
    aucs["MACRO"] = float(np.mean(list(aucs.values())))
    return aucs


@torch.no_grad()
def predecir(modelo, cargador, dispositivo):
    modelo.eval(); probs, ys = [], []
    for x, y, _ in cargador:
        with torch.autocast(dispositivo.type, enabled=dispositivo.type == "cuda"):
            logits, _ = modelo(x.to(dispositivo, non_blocking=True))
        probs.append(torch.sigmoid(logits.float()).cpu().numpy()); ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(probs)


def umbrales_f1(y, prob) -> np.ndarray:
    """Umbral por etiqueta que maximiza F1 en la validación débil (no en oro: evita fuga)."""
    y = (y >= 0.5).astype(int); candidatos = np.linspace(0.05, 0.95, 19)
    return np.array([candidatos[np.argmax([f1_score(y[:, j], prob[:, j] >= u, zero_division=0) for u in candidatos])]
                     for j in range(C.N_ETIQUETAS)])


def metricas_umbral(y, prob, umbrales) -> pd.DataFrame:
    y = (y >= 0.5).astype(int); pred = (prob >= umbrales).astype(int)
    tabla = pd.DataFrame({"etiqueta": C.ETIQUETAS, "umbral": umbrales,
                          "precision": precision_score(y, pred, average=None, zero_division=0),
                          "recall": recall_score(y, pred, average=None, zero_division=0),
                          "f1": f1_score(y, pred, average=None, zero_division=0)})
    tabla.loc[len(tabla)] = ["MACRO", np.nan, *tabla[["precision", "recall", "f1"]].mean()]
    return tabla.round(3)


def main():
    a = argumentos()
    torch.manual_seed(C.SEMILLA); np.random.seed(C.SEMILLA)
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nombre = a.nombre or f"{a.arquitectura}_{a.perdida}_{a.etiquetas}_{a.balanceo}_{a.plano}"
    salida = C.DIR_SALIDAS / nombre; salida.mkdir(parents=True, exist_ok=True)

    seleccion = pd.read_csv(C.CSV_SELECCION)
    dir_tensores = C.DIR_TENSORES / a.plano
    conjuntos = {p: DatasetRodilla(seleccion[seleccion.particion == p], dir_tensores, p == "entrenamiento", a.etiquetas)
                 for p in ["entrenamiento", "validacion_debil", "oro"]}
    print({p: len(d) for p, d in conjuntos.items()}, "| dispositivo:", dispositivo)
    cargar = lambda p, mezclar: DataLoader(conjuntos[p], C.TAMANO_LOTE, shuffle=mezclar, num_workers=C.N_TRABAJADORES,
                                           pin_memory=True, drop_last=mezclar and len(conjuntos[p]) > C.TAMANO_LOTE)

    modelo = crear_modelo(a.arquitectura, not a.sin_preentrenar).to(dispositivo)
    optimizador = torch.optim.AdamW(modelo.parameters(), C.TASA_APRENDIZAJE, weight_decay=C.DECAIMIENTO_PESOS)
    planificador = torch.optim.lr_scheduler.CosineAnnealingLR(optimizador, a.epocas)
    escalador = torch.amp.GradScaler(enabled=dispositivo.type == "cuda")
    fn_perdida = perdida_focal if a.perdida == "focal" else perdida_bce
    pos_weight = None
    if a.balanceo == "pos_weight":     # negativos/positivos por etiqueta, acotado para no desestabilizar
        y = conjuntos["entrenamiento"].y
        pos_weight = torch.tensor(np.clip((1 - y).sum(0) / np.maximum(y.sum(0), 1), 1, 10), device=dispositivo)

    mejor, sin_mejora, historial = -1, 0, []
    for epoca in range(a.epocas):
        modelo.train(); inicio, total = time.time(), 0.0
        for x, y, peso in cargar("entrenamiento", True):
            x, y, peso = (t.to(dispositivo, non_blocking=True) for t in (x, y, peso))
            with torch.autocast(dispositivo.type, enabled=dispositivo.type == "cuda"):
                logits, _ = modelo(x)
            perdida = fn_perdida(logits.float(), y, peso, pos_weight)
            optimizador.zero_grad(set_to_none=True)
            escalador.scale(perdida).backward(); escalador.step(optimizador); escalador.update()
            total += perdida.item()
        planificador.step()
        auc_val = auc_por_etiqueta(*predecir(modelo, cargar("validacion_debil", False), dispositivo))["MACRO"]
        historial.append({"epoca": epoca, "perdida": total, "auc_validacion_debil": auc_val})
        print(f"época {epoca:2d} | pérdida {total:.3f} | AUC val débil {auc_val:.4f} | {time.time() - inicio:.0f}s")
        if auc_val > mejor:
            mejor, sin_mejora = auc_val, 0
            torch.save(modelo.state_dict(), salida / "mejor_modelo.pt")
        else:
            sin_mejora += 1
            if sin_mejora >= C.PACIENCIA:
                break

    # evaluación final: umbrales desde validación débil, métricas sobre oro
    modelo.load_state_dict(torch.load(salida / "mejor_modelo.pt", map_location=dispositivo))
    umbrales = umbrales_f1(*predecir(modelo, cargar("validacion_debil", False), dispositivo))
    y_oro, p_oro = predecir(modelo, cargar("oro", False), dispositivo)
    aucs = auc_por_etiqueta(y_oro, p_oro)
    tabla = metricas_umbral(y_oro, p_oro, umbrales)
    tabla["auc_roc"] = tabla.etiqueta.map(aucs).round(4)
    tabla.to_csv(salida / "metricas_oro.csv", index=False)
    pd.DataFrame(p_oro, columns=C.ETIQUETAS).assign(StudyInstanceUID=conjuntos["oro"].tabla.StudyInstanceUID).to_csv(salida / "predicciones_oro.csv", index=False)
    json.dump({"argumentos": vars(a), "auc_macro_oro": aucs["MACRO"], "historial": historial},
              open(salida / "resumen.json", "w"), indent=2, default=str)
    print(f"\nAUC-ROC macro sobre oro: {aucs['MACRO']:.4f}\n", tabla.to_string(index=False))


if __name__ == "__main__":
    main()
