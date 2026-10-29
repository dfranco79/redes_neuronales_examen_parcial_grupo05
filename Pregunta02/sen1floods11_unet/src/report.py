"""Genera, a partir de resultados.json, el análisis en Markdown, el README con resultados y el informe en
LaTeX (plantilla NeurIPS). Los textos usan los números de la corrida, así que cambian si cambian los resultados."""
import math
import re
import shutil
from pathlib import Path

import pandas as pd

from .analysis import BRIGHT_DB, DARK_DB, MID_DB

REPO = Path(__file__).resolve().parents[1]
KEYS = [("iou", "IoU"), ("mean_iou_chips", "IoU medio/chip"), ("f1", "F1"), ("precision", "Precisión"),
        ("recall", "Recall"), ("omission", "Omisión"), ("commission", "Comisión")]


def f3(x):
    return "n/d" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.3f}"


def pc(x):
    return "n/d" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:.1f} %"


def ep(n):
    return f"{n} época" if n == 1 else f"{n} épocas"


def db(x):
    return "n/d" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.1f} dB"



# ============================================================================ resumen por entregable
def checklist(R):
    """Una fila por cada punto del enunciado, con el resultado obtenido en esta corrida."""
    M, D, V = R["models"], R["data"], R.get("verificacion", {})
    fin = R["final_model"]
    F = M[fin]
    rows = []
    add = lambda e, req, ev: rows.append(dict(Entregable=e, Requisito=req, Resultado=ev))  # noqa: E731
    libs = V.get("seg_libs", [])
    add("1", "U-Net propia en PyTorch",
        f"`src/model.py`: clases DoubleConv, Down, Up y UNet, {R['n_params'] / 1e6:.2f} M de parámetros")
    add("1", "Sin librerías de segmentación",
        "Revisión de los imports de `src/`: " + ("no se usa ninguna" if not libs else ", ".join(libs)))
    add("1", "Entrada con las dos bandas SAR (VV, VH)", "`in_channels=2` (VV, VH en dB); 15 canales en el bono")
    weak = R["groups"].get("weak", [])
    add("2a", "Entrenar solo con los chips dorados",
        f"Modelo «Solo dorados · BCE+Dice»: {D['train']} chips de train y {D['val']} de validación (los {D['test']} de "
        f"test y {D['bolivia']} de Bolivia quedan para evaluar; en total {D['train'] + D['val'] + D['test'] + D['bolivia']})")
    add("2a", "Añadir los chips con etiqueta débil",
        f"{D.get('weak_s1', 0):,} chips débiles S1 (Otsu) y {D.get('weak_s2', 0):,} S2 (índice): "
        + "; ".join(weak[1:]) if weak else "no se ejecutó")
    if weak:
        bw = max(weak, key=lambda n: M[n]["val"]["iou"])
        add("2a", "Decidir cómo combinar ambas fuentes",
            f"Mejor estrategia por IoU de validación: «{bw}» (val {M[bw]['val']['iou']:.3f}, test {M[bw]['test']['iou']:.3f})")
    loss = R["groups"].get("loss", [])
    if loss:
        bl = max(loss, key=lambda n: M[n]["val"]["iou"])
        add("2b", "BCE vs. Dice vs. combinada",
            "IoU en test: " + ", ".join(f"{n.split('· ')[1]} {M[n]['test']['iou']:.3f}" for n in loss)
            + f". Mejor en validación: {bl.split('· ')[1]}")
    else:
        add("2b", "BCE vs. Dice vs. combinada", "no se ejecutó")
    t = F["test"]
    add("3", "Evaluar en chips dorados de eventos vistos",
        f"Test oficial: {D['test']} chips; todos sus eventos están en train: {'sí' if V.get('test_events_seen') else 'no'}")
    add("3", "IoU, F1, precisión y recall de la clase agua",
        f"IoU {t['iou']:.3f}, F1 {t['f1']:.3f}, precisión {t['precision']:.3f}, recall {t['recall']:.3f} "
        f"(IoU medio por chip {t['mean_iou_chips']:.3f}; omisión {t['omission']:.3f}; comisión {t['commission']:.3f})")
    add("3", "Matriz de confusión comentada", "Figura de la matriz y comentario con los conteos (sección del entregable 3)")
    add("4", "Entrenar sin Bolivia",
        f"Chips de Bolivia usados al entrenar: {V.get('bolivia_train', 0)} de train, {V.get('bolivia_val', 0)} de "
        f"validación y {V.get('bolivia_weak', 0)} débiles")
    add("4", "Evaluar solo en los chips dorados de Bolivia",
        f"{V.get('bolivia_eval', D['bolivia'])} chips de `flood_bolivia_data.csv` (partición oficial)")
    b = F["bolivia"]
    lo, hi = F["bolivia_ci"]
    add("4", "Brecha frente al punto 3",
        f"IoU test {t['iou']:.3f} vs. Bolivia {b['iou']:.3f} (diferencia {b['iou'] - t['iou']:+.3f}; IC 95 % en Bolivia "
        f"{lo:.3f}-{hi:.3f})")
    add("4", "Al menos dos hipótesis",
        "5 hipótesis (composición, contraste, cobertura del suelo, geometría de adquisición y tamaño de muestra), "
        "revisadas con datos")
    cases = R.get("cases", {})
    add("5", "Al menos tres tipos de error del SAR",
        f"{len(cases)} tipos: " + "; ".join(c["title"].split(") ", 1)[1] for c in cases.values()))
    add("5", "Ejemplos visuales",
        f"{sum(1 for k in R['figures'] if k.startswith('caso_'))} figuras (VV, VH, Sentinel-2, mapa de errores, histograma)")
    s2 = R["groups"].get("s2", [])
    if len(s2) == 2:
        d = M[s2[1]]["test"]["iou"] - M[s2[0]]["test"]["iou"]
        add("6 (bono)", "Bandas de Sentinel-2 como canales extra", "U-Net de 15 canales: VV, VH y las 13 bandas S2")
        add("6 (bono)", "Solo chips sin nubes",
            f"{R['s2']['clear']['train']} de train, {R['s2']['clear']['val']} de validación y {R['s2']['clear']['test']} "
            f"de test con menos de {100 * R['s2']['cloud_max']:.0f} % de nubes")
        add("6 (bono)", "Medir si la fusión mejora el IoU",
            f"Mismos chips: SAR {M[s2[0]]['test']['iou']:.3f} vs. SAR+S2 {M[s2[1]]['test']['iou']:.3f} (ΔIoU {d:+.3f})")
    return rows


# ============================================================================ bloques de contenido
def build_blocks(R):
    B = []
    h = lambda t, lvl=2: B.append(("h", lvl, t))          # noqa: E731
    p = lambda t: B.append(("p", t))                      # noqa: E731
    ul = lambda items, tex=True: B.append(("ul", items, tex))  # noqa: E731
    fig = lambda key, cap: B.append(("fig", key, cap)) if key in R["figures"] else None  # noqa: E731
    M, cfg, D = R["models"], R["config"], R["data"]
    fin = R["final_model"]
    F = M[fin]
    t = F["test"]
    tot = t["tp"] + t["fp"] + t["fn"] + t["tn"]
    wf = (t["tp"] + t["fn"]) / max(tot, 1)
    lo, hi = F["test_ci"]

    CHK = pd.DataFrame(checklist(R))

    h("Resumen por entregable")
    p("Qué se hizo para cada punto del enunciado y el resultado principal. El detalle está en las secciones siguientes.")
    B.append(("table", CHK, "Puntos del enunciado y resultado obtenido.", False))

    def table(names, split="test", cap="", tex=True):
        rows = [dict(Modelo=n, Umbral=f"{M[n]['threshold']:.2f}", **{lab: f3(M[n][split][k]) for k, lab in KEYS})
                for n in names if n in M]
        B.append(("table", pd.DataFrame(rows), cap, tex))

    # --- datos y ruido de las etiquetas débiles
    h("Contexto: Inventario del dataset y ruido de las etiquetas débiles")
    p("Comparación entre las características de Sen1Floods11 que da el enunciado y lo que se encontró al descargar "
      "los datos.")
    if R.get("inventario"):
        B.append(("table", pd.DataFrame(R["inventario"]), "Características del dataset: enunciado vs. datos descargados."))
    bk = R.get("bucket", {})
    n_w, n_md = bk.get("S1Weak", {}).get("count"), bk.get("metadata", {}).get("chips_debiles")
    otros = bk.get("S2IndexLabelWeak", {}).get("otros", [])
    if n_w and n_w != 4385:
        p(f"**Número de chips débiles.** El enunciado y el artículo original hablan de 4,385 chips débiles, pero el "
          f"bucket v1.1 tiene {n_w:,} GeoTIFF en cada carpeta de chips débiles"
          + (f", lo mismo que suma la metadata oficial del dataset ({n_md:,})" if n_md else "")
          + (f". La carpeta de etiquetas de Sentinel-2 tiene {n_w + len(otros):,} objetos, pero {len(otros)} no es una "
             f"imagen ({', '.join(otros)})" if otros else "")
          + ". Se usan todos los chips débiles disponibles menos los de Bolivia.")
    if R.get("eventos"):
        evd = pd.DataFrame(R["eventos"])
        B.append(("table", evd.rename(columns=dict(evento="Evento", dorados="Dorados", debiles_bucket="Débiles (bucket)",
                                                   debiles_usados="Débiles (usados)", orbita="Órbita", fecha_s1="Fecha S1")),
                  "Chips por evento de inundación y partición.", False))
        solo_debiles = [e for e, d in zip(evd.evento, evd.dorados) if d == 0]
        p(f"Los chips dorados cubren {int((evd.dorados > 0).sum())} eventos y suman {int(evd.dorados.sum())}; de los "
          f"{int(evd.debiles_bucket.sum()):,} chips débiles del bucket se usan {int(evd.debiles_usados.sum()):,}. "
          + ((f"{solo_debiles[0]} solo tiene chips débiles. " if len(solo_debiles) == 1 else
              f"{', '.join(solo_debiles)} solo tienen chips débiles. ") if solo_debiles else "")
          + "Los chips débiles de Bolivia se dejan fuera para que el entregable 4 mida la generalización a un evento "
            "que el modelo de verdad no vio.")
    fig("eventos", "Chips dorados por evento y partición (izquierda) y chips débiles por evento (derecha).")
    p(f"Partición oficial: {D['train']} chips dorados de entrenamiento, {D['val']} de validación, {D['test']} de "
      f"test (eventos vistos) y {D['bolivia']} de Bolivia (evento reservado). Para el entregable 2a se usan "
      f"{D.get('weak_s1', 0):,} chips débiles con etiqueta Otsu de Sentinel-1 y {D.get('weak_s2', 0):,} con etiqueta "
      "por índice espectral de Sentinel-2, que son las dos fuentes de etiquetas débiles del dataset.")
    wn = R.get("weak_noise")
    if wn:
        o = wn["overall"]
        sesgo = ("marca más agua de la que realmente hay (más comisión que omisión)" if o["precision"] < o["recall"]
                 else "marca menos agua de la que realmente hay (más omisión que comisión)")
        pe = pd.DataFrame(wn["per_event"]).sort_values("iou")
        p(f"**Ruido de las etiquetas débiles.** El dataset también trae la etiqueta Otsu de los chips dorados "
          f"(S1OtsuLabelHand), así que se puede comparar directamente con la etiqueta manual: IoU = {f3(o['iou'])}, "
          f"precisión = {f3(o['precision'])} y recall = {f3(o['recall'])} en {int(o['chips'])} chips (coinciden en el "
          f"{pc(o['acuerdo_pixel'])} de los píxeles). En otras palabras, la umbralización {sesgo}. El ruido cambia bastante entre "
          f"eventos: el peor es {pe.iloc[0]['event']} (IoU {f3(pe.iloc[0]['iou'])}) y el mejor {pe.iloc[-1]['event']} "
          f"(IoU {f3(pe.iloc[-1]['iou'])}). Las etiquetas débiles son muchas pero imperfectas, y eso es lo que se pone a "
          f"prueba en el entregable 2a. Para las etiquetas débiles de Sentinel-2 el dataset no trae una versión sobre los "
          f"chips dorados, así que su calidad solo se puede ver de forma indirecta en los resultados del 2a.")

    # --- modelo
    h("Entregable 1: U-Net propia y entrenamiento")
    p(f"La U-Net está escrita desde cero en PyTorch, sin librerías de segmentación. Tiene {cfg['depth']} niveles de "
      f"submuestreo, {cfg['base_ch']} canales en el primer nivel y {R['n_params'] / 1e6:.2f} M de parámetros. Como "
      f"entrada usa VV y VH en dB, normalizados a [0, 1]. Se entrena con recortes aleatorios de {cfg['crop']}x{cfg['crop']} px "
      f"a resolución nativa (10 m), volteos y rotaciones de 90°, y se evalúa sobre el chip completo de 512x512. "
      f"Optimizador AdamW (lr={cfg['lr']}, weight decay={cfg['weight_decay']}) con cosine annealing, "
      f"{ep(cfg['epochs'])} por modelo y batch de {cfg['batch_size']}. Los píxeles sin dato (SAR vacío o etiqueta -1) "
      f"no cuentan en la pérdida ni en las métricas. De cada modelo se guarda la época con mayor IoU en la validación "
      f"dorada, y el umbral de decisión se elige maximizando F1 en validación; el test y Bolivia solo se usan para "
      f"reportar. El modelo final es el de mayor IoU en validación: **{fin}**.")
    fig("curvas", "Curvas de entrenamiento: pérdida y IoU de validación por época de cada modelo.")

    # --- 2a
    if "weak" in R["groups"]:
        names = R["groups"]["weak"]
        h("Entregable 2a: Solo dorados vs. dorados + débiles")
        p(f"Todas las estrategias usan la misma U-Net y la pérdida BCE+Dice, y se validan y evalúan solo con chips "
          f"dorados: (1) solo los chips dorados de entrenamiento; (2) dorados + {D.get('weak_s1', 0):,} chips débiles "
          f"con etiqueta Otsu de Sentinel-1; (3) dorados + {D.get('weak_s2', 0):,} chips débiles con etiqueta por índice "
          f"de Sentinel-2; (4) pre-entrenamiento con los débiles S1 y ajuste final con los dorados. Con los débiles cada "
          f"época tiene ~{D.get('weak_s1', 0) + D['train']:,} chips, por eso estos modelos se entrenan "
          f"{ep(cfg.get('weak_epochs', 0))} en lugar de {cfg['epochs']} (aun así hacen más pasos de optimización).")
        table(names, "test", "Estrategias de supervisión evaluadas en el test (eventos vistos).")
        table(names, "bolivia", "Las mismas estrategias evaluadas en Bolivia (evento no visto).", tex=False)
        fig("ablacion_debiles", "Entregable 2a: IoU en test de cada estrategia de supervisión.")
        g = M["Solo dorados · BCE+Dice"]
        lines = []
        for n in names[1:]:
            dt, dbv = M[n]["test"]["iou"] - g["test"]["iou"], M[n]["bolivia"]["iou"] - g["bolivia"]["iou"]
            lines.append(f"**{n}** frente a solo dorados: ΔIoU test {dt:+.3f}, ΔIoU Bolivia {dbv:+.3f}, "
                         f"IoU de validación {f3(M[n]['val']['iou'])} (solo dorados: {f3(g['val']['iou'])})")
        ul(lines)
        bv = max(names, key=lambda n: M[n]["val"]["iou"])
        wfrac, wfrac2 = R["data"].get("weak_water_frac"), R["data"].get("weak_s2_water_frac")
        s1n, s2n = "Dorados + débiles S1 (Otsu)", "Dorados + débiles S2 (índice)"
        comp = ""
        if s1n in M and s2n in M:
            mejor = s1n if M[s1n]["val"]["iou"] >= M[s2n]["val"]["iou"] else s2n
            comp = (f"De las dos fuentes de etiquetas débiles, las **{mejor.split('+ ')[1]}** dan mejor resultado en validación "
                    f"({f3(M[s1n]['val']['iou'])} S1 vs. {f3(M[s2n]['val']['iou'])} S2). Una explicación es que las "
                    "etiquetas S1 salen del mismo radar y heredan sus errores (sombras, doble rebote), que el modelo termina "
                    "aprendiendo; las S2 no dependen del radar, aunque sí de que la escena óptica esté despejada y de que "
                    "su fecha coincida con la del SAR. ")
        p(f"Según la validación, la mejor estrategia es **{bv}**. "
          + (f"Las etiquetas débiles S1 tienen en promedio {pc(wfrac)} de agua"
             + (f" (S2: {pc(wfrac2)})" if wfrac2 is not None else "")
             + (f" y la etiqueta Otsu tiene un IoU de {f3(wn['overall']['iou'])} frente a la etiqueta manual" if wn else "")
             + ": aportan muchos más ejemplos y más variedad de eventos, pero con errores sistemáticos en bordes, sombras "
               "y vegetación. " if wfrac is not None else "")
          + comp
          + "La idea del pre-entrenamiento con ajuste es aprovechar la cantidad de datos débiles para aprender patrones "
            "generales de retrodispersión y dejar que al final las etiquetas manuales corrijan los sesgos de la "
            "umbralización; al mezclarlos con los dorados, en cambio, el modelo ve ese ruido durante todo el entrenamiento. "
          + ("En nuestros resultados la supervisión débil **mejora** al modelo que solo usa chips dorados, así que vale "
             "la pena usarla." if bv != "Solo dorados · BCE+Dice" else
             "En nuestros resultados ninguna estrategia con débiles supera en validación al modelo que solo usa chips "
             "dorados: con este número de épocas pesa más el ruido que la cantidad extra de datos."))

    # --- 2b
    if "loss" in R["groups"]:
        names = R["groups"]["loss"]
        h("Entregable 2b: BCE vs. Dice vs. BCE+Dice")
        table(names, "test", "Comparación de pérdidas (solo chips dorados, mismas épocas; umbral elegido en validación).")
        fig("ablacion_perdida", "Comparación de pérdidas en test.")
        bv = max(names, key=lambda n: M[n]["val"]["iou"])
        bt = max(names, key=lambda n: M[n]["test"]["iou"])
        bce, dice = M.get("Solo dorados · BCE"), M.get("Solo dorados · Dice")
        if bv == bt:
            txt = (f"**{bv}** es el mejor tanto en validación, que es el criterio de selección (IoU "
                   f"{f3(M[bv]['val']['iou'])}), como en test (IoU {f3(M[bt]['test']['iou'])}). ")
        else:
            txt = (f"En validación, que es el criterio de selección, gana **{bv}** (IoU {f3(M[bv]['val']['iou'])}); en "
                   f"test el mejor es **{bt}** (IoU {f3(M[bt]['test']['iou'])}). ")
        if bce and dice:
            txt += (f"La BCE con pos_weight={cfg['pos_weight']} logra precisión {f3(bce['test']['precision'])} y recall "
                    f"{f3(bce['test']['recall'])}, y la Dice {f3(dice['test']['precision'])} y "
                    f"{f3(dice['test']['recall'])}. La BCE promedia el error de todos los píxeles y, como hay "
                    f"~{round(1 / max(wf, 1e-3))} píxeles válidos por cada píxel de agua, sin el pos_weight su gradiente "
                    "quedaría dominado por la tierra. La Dice mide directamente el solapamiento de la clase agua y no "
                    "depende de los verdaderos negativos, así que en principio maneja mejor el desbalance. La combinación "
                    "busca juntar el gradiente estable de la BCE con el enfoque en el solapamiento de la Dice.")
        p(txt)

    # --- evaluación en distribución
    h("Entregable 3: Evaluación en distribución (test, eventos vistos)")
    table([fin], "test", f"Métricas de la clase agua en test del modelo final (umbral {F['threshold']:.2f}).")
    fig("umbral", "Barrido del umbral de decisión en validación; la línea discontinua marca el umbral elegido.")
    fig("confusion", "Matriz de confusión por píxel en el test (conteos y porcentaje por fila).")
    dom = ("la **omisión** (agua que no se detecta) es mayor que la comisión, así que el modelo es conservador"
           if t["omission"] > t["commission"] else
           "la **comisión** (falsas alarmas) es mayor que la omisión, así que el modelo tiende a marcar agua de más")
    p(f"**Comentario de la matriz de confusión.** El agua es solo el {pc(wf)} de los {tot:,} píxeles válidos del test, "
      f"así que la exactitud no dice mucho: un modelo que nunca predijera agua tendría {pc(1 - wf)} de exactitud (el "
      f"nuestro tiene {pc(t['accuracy'])}). Por eso usamos métricas de la clase agua. De cada 100 píxeles de agua real el "
      f"modelo detecta {round(100 * t['recall'])} y se le escapan {round(100 * t['omission'])}; de cada 100 píxeles que "
      f"marca como agua, {round(100 * t['precision'])} lo son. Hay {t['fp'] / max(t['tp'], 1):.2f} falsos positivos por "
      f"cada verdadero positivo y {dom}. El IoU agregado es {f3(t['iou'])} (IC 95 % por bootstrap sobre chips: "
      f"{f3(lo)}-{f3(hi)}) y el IoU promedio por chip es {f3(t['mean_iou_chips'])}"
      + (": la diferencia indica que al modelo le va bien en los chips con mucha agua y peor en los que tienen poca, que "
         "en el promedio por chip pesan lo mismo." if t["iou"] - t["mean_iou_chips"] > 0.05 else
         ": son parecidos, así que el desempeño es similar en chips con poca y con mucha agua.")
      )
    ev = pd.DataFrame(R.get("per_event_test", []))
    if len(ev):
        ev = ev.sort_values("iou")
        B.append(("table", ev.assign(iou=ev.iou.map(f3))[["event", "chips", "iou", "orbita"]]
                  .rename(columns=dict(event="Evento", chips="Chips", iou="IoU", orbita="Órbita S1")),
                  "IoU agregado del modelo final por evento de inundación en el test.", False))
        p(f"El evento más difícil es **{ev.iloc[0]['event']}** (IoU {f3(ev.iloc[0]['iou'])}) y el más fácil "
          f"**{ev.iloc[-1]['event']}** (IoU {f3(ev.iloc[-1]['iou'])}).")
    fig("ejemplos_test", "Ejemplos del test: VV, etiqueta dorada y mapa de errores.")

    # --- Bolivia
    h("Entregable 4: Generalización geográfica (Bolivia)")
    b = F["bolivia"]
    blo, bhi = F["bolivia_ci"]
    gap = b["iou"] - t["iou"]
    flat = abs(gap) < 0.01
    rows = [dict(Métrica=lab, Test=f3(t[k]), Bolivia=f3(b[k]), Brecha=f"{b[k] - t[k]:+.3f}")
            for k, lab in KEYS[:5]]
    B.append(("table", pd.DataFrame(rows), f"Brecha de generalización del modelo final, {fin}."))
    fig("brecha", "IoU en test y en Bolivia de todos los modelos.")
    fig("ejemplos_bolivia", "Ejemplos de Bolivia: VV, etiqueta dorada y mapa de errores.")
    overlap = not (bhi < lo or hi < blo)
    p(f"Siguiendo la partición oficial, ningún chip de Bolivia (dorado o débil) se usó para entrenar ni para elegir "
      f"hiperparámetros. El IoU en Bolivia es {f3(b['iou'])} (IC 95 %: {f3(blo)}-{f3(bhi)}) y en el test "
      f"{f3(t['iou'])}, una brecha de {gap:+.3f}. "
      + ("Los intervalos de confianza se solapan, así que con solo "
         f"{R['data']['bolivia']} chips la diferencia no es estadísticamente concluyente. " if overlap else
         "Los intervalos de confianza no se solapan, así que la diferencia se mantiene aunque cambien los chips. ")
      + ("En la práctica el modelo rinde igual en Bolivia que en los eventos vistos, aunque nunca vio esa zona al "
         "entrenar." if flat else
         "Contra lo esperado, el modelo rinde **mejor** en el evento no visto; abajo se revisa por qué." if gap > 0 else
         "Como era de esperar, el rendimiento **baja** en el evento que no se vio al entrenar."))
    comp = {c["conjunto"]: c for c in R["composition"]}
    ct, cb = comp["test"], comp["bolivia"]
    B.append(("table", pd.DataFrame([dict(Conjunto=c["conjunto"], Chips=c["chips"], Eventos=c["eventos"],
                                          **{"Frac. agua": pc(c["frac_agua"]), "Chips sin agua": c["chips_sin_agua"],
                                             "VV agua": db(c["vv_agua_db"]), "VV tierra": db(c["vv_tierra_db"]),
                                             "Contraste": db(c["contraste_db"]),
                                             "Agua brillante": pc(c["agua_brillante"])}) for c in (ct, cb)]),
              "Composición del test y de Bolivia (datos para revisar las hipótesis)."))
    bm = R.get("bolivia_meta", {})
    worst = ev.iloc[0] if len(ev) else None

    def verdict(favors_bol, negligible):
        if negligible:
            return " La diferencia es pequeña, así que este factor por sí solo no explica la brecha."
        if flat:
            return (" Este factor favorece a Bolivia y puede compensar que sea un evento nuevo." if favors_bol else
                    " Este factor perjudica a Bolivia; que el IoU no baje sugiere que otros factores lo compensan.")
        if favors_bol == (gap > 0):
            return " Va en el mismo sentido que la brecha observada, así que es una explicación posible."
        return " Va en sentido contrario a la brecha observada, así que no la explica (sin este factor la brecha sería mayor)."

    fw_t, fw_b = ct["frac_agua"], cb["frac_agua"]
    c_t, c_b = ct["contraste_db"], cb["contraste_db"]
    ab_t, ab_b = ct["agua_brillante"], cb["agua_brillante"]
    hyp = [
        f"**Composición y tamaño de los cuerpos de agua.** Fracción de agua: test {pc(fw_t)} vs. Bolivia {pc(fw_b)}; chips "
        f"sin agua: {ct['chips_sin_agua']} vs. {cb['chips_sin_agua']}. Las superficies de agua grandes y continuas (como las "
        "llanuras inundadas de la Amazonía boliviana) son más fáciles de segmentar que los parches pequeños, porque los "
        "errores de borde pesan menos frente al área total; y en un chip sin agua solo se pueden cometer falsas alarmas."
        + verdict(fw_b > fw_t, abs(fw_b - fw_t) < 0.1 * max(fw_t, 1e-6)),
        f"**Contraste radiométrico.** Diferencia mediana de VV entre tierra y agua: {db(c_t)} en test vs. {db(c_b)} en "
        "Bolivia. Con más contraste, el agua (lisa y oscura) se separa mejor de la tierra (más brillante)."
        + verdict(c_b > c_t, abs(c_b - c_t) < 0.5),
        f"**Cobertura del suelo: zonas urbanas y vegetación inundada.** Porcentaje del agua con VV > {BRIGHT_DB:.0f} dB "
        f"(agua 'brillante', típica del doble rebote en ciudades o con vegetación que sobresale): test {pc(ab_t)} vs. "
        f"Bolivia {pc(ab_b)}. Esa agua es casi invisible para un modelo que se basa en la retrodispersión."
        + verdict(ab_b < ab_t, abs(ab_b - ab_t) < 0.02)
        + (f" El evento más difícil del test es {worst['event']} (IoU {f3(worst['iou'])})." if worst is not None else ""),
        "**Geometría de adquisición y ángulo de incidencia.** "
        + (f"La escena de Bolivia es de órbita {bm.get('orbit', '?')} (órbita relativa {int(bm['rel_orbit_num'])}). "
           if bm.get("rel_orbit_num") else "")
        + "El ángulo de incidencia cambia a lo ancho del swath de Sentinel-1 y modifica la retrodispersión de la tierra y "
          "del agua, así que el umbral que el modelo aprende en otros eventos puede encajar mejor o peor en una escena "
          "nueva. Los chips no traen el ángulo, por lo que no se puede comprobar aquí; habría que agregarlo como canal "
          "de entrada.",
        f"**Varianza por tamaño de muestra.** Bolivia tiene solo {R['data']['bolivia']} chips; el IC bootstrap "
        f"({f3(blo)}-{f3(bhi)}) muestra cuánto puede moverse la métrica según qué chips entren.",
    ]
    p("**Hipótesis sobre la brecha**, cada una comparada con los datos de la tabla anterior:")
    ul(hyp)

    # --- errores del SAR
    h("Entregable 5: Errores característicos del SAR")
    p(f"Cada píxel mal clasificado del test se asignó a un tipo de error con reglas simples sobre VV y la distancia al "
      f"agua: FP oscuros (VV < {DARK_DB:.0f} dB) a más de 50 m del agua real (sombra de radar o superficies muy lisas), "
      f"FP en la franja de 50 m alrededor del agua (suelo saturado y píxeles mixtos), FN brillantes (VV > "
      f"{BRIGHT_DB:.0f} dB: doble rebote en zonas urbanas o vegetación inundada), FN dentro del agua con VV intermedio "
      f"({MID_DB:.0f} a {BRIGHT_DB:.0f} dB: agua rugosa por viento o con sedimentos) y FN dentro del agua con VV oscuro "
      "(el agua tiene su firma típica pero el modelo no la marca, por ejemplo en cuerpos de agua muy pequeños o "
      "aislados).")
    fig("taxonomia", "Distribución de los errores del test por tipo.")
    eb = R["error_breakdown"]
    ul([f"{k}: {pc(v)}" for k, v in eb.items()], tex=False)   # en el PDF basta con la figura
    def s2_txt(c):
        if "mndwi_cat" not in c or math.isnan(c["mndwi_cat"]):
            return ""
        w = c["mndwi_cat"]
        wtxt = ("negativo: en el óptico no parecen agua" if w < -0.05 else
                "positivo: en el óptico parecen agua o suelo muy húmedo" if w > 0.05 else "cercano a cero: ambiguo en el óptico")
        v = c["ndvi_cat"]
        vtxt = "vegetación densa" if v > 0.4 else "vegetación moderada" if v > 0.2 else "suelo desnudo, zona urbana o agua"
        return (f" En Sentinel-2 esos píxeles tienen MNDWI mediano {f3(w)} ({wtxt}; en el agua bien detectada: "
                f"{f3(c.get('mndwi_tp'))}) y NDVI {f3(v)} ({vtxt}).")

    for k, c in R.get("cases", {}).items():
        base = (f"**{c['title']}** (chip {c['cid']}). En este chip el VV mediano del agua bien detectada es "
                f"{db(c['vv_tp'])} y el de la tierra bien clasificada {db(c['vv_tn'])}. ")
        vc = c["vv_cat"]
        if k == "A":
            txt = (f"Hay {c['n_cat']:,} falsos positivos oscuros a más de 50 m del agua real, con VV mediano {db(vc)}"
                   + (", prácticamente igual al del agua" if abs(vc - c["vv_tp"]) < 3 else ", mucho más oscuro que la tierra")
                   + ": al sensor le llega tan poca señal como desde agua abierta. Esto pasa en laderas que quedan ocultas "
                     "para el radar (sombra de radar) y en superficies lisas como pistas o suelo desnudo húmedo o recién "
                     "arado, que reflejan el pulso lejos del satélite." + s2_txt(c)
                   + " Para separarlos haría falta información adicional (modelo de elevación, Sentinel-2 o series de tiempo).")
        elif k == "B":
            txt = (f"Hay {c['n_cat']:,} falsos negativos con VV > {BRIGHT_DB:.0f} dB (mediana {db(vc)}, "
                   f"{vc - c['vv_tp']:+.1f} dB respecto al agua detectada). Si hay estructuras verticales sobre el agua, el "
                   "pulso rebota en el agua, luego en una pared o un tronco, y vuelve al sensor (doble rebote): el agua se ve "
                   "brillante y el modelo la confunde con tierra." + s2_txt(c)
                   + (" Un NDVI alto apunta a vegetación inundada." if c.get("ndvi_cat", 0) > 0.3 else
                      " Un NDVI bajo con retrodispersión alta es compatible con zonas urbanas inundadas." if "ndvi_cat" in c else ""))
        elif k == "C":
            tot_e = max(c["n_fp"] + c["n_fn"], 1)
            txt = (f"{c['n_cat']:,} de los {c['n_fp'] + c['n_fn']:,} errores del chip ({pc(c['n_cat'] / tot_e)}) están a "
                   f"menos de 50 m del borde del agua real, con VV mediano {db(vc)}. En esa franja el suelo saturado tiene "
                   "una retrodispersión intermedia, los píxeles de 10 m mezclan agua y tierra y hasta la etiqueta manual es "
                   "dudosa; el speckle del SAR empeora la confusión." + s2_txt(c))
        else:
            txt = (f"Hay {c['n_cat']:,} falsos negativos dentro del agua con VV intermedio (mediana {db(vc)}, entre el agua "
                   "abierta y la tierra): el viento o la corriente vuelven rugosa la superficie y aumentan la "
                   "retrodispersión, igual que los sedimentos en suspensión o la vegetación baja que sobresale." + s2_txt(c))
        p(base + txt)
        fig(f"caso_{k}", c["title"])

    # --- S2
    s2names = R["groups"].get("s2", [])
    if len(s2names) == 2:
        h("Entregable 6 (bono): Fusión SAR + Sentinel-2")
        s2 = R["s2"]
        table(s2names, "test", "Modelos entrenados y evaluados solo con chips sin nubes (los mismos para ambos).")
        d = M[s2names[1]]["test"]["iou"] - M[s2names[0]]["test"]["iou"]
        p(f"Las imágenes S2Hand no traen máscara de nubes, así que se marcó como nube todo píxel con reflectancia azul "
          f"> 0.2 y visible medio > 0.25, y se usaron los chips con menos de {pc(s2['cloud_max'])} de nubes: "
          f"{s2['clear']['train']} de train, {s2['clear']['val']} de validación y {s2['clear']['test']} de test. Para que "
          f"la comparación sea justa, el modelo solo-SAR se volvió a entrenar con exactamente esos chips. Agregar las 13 "
          f"bandas ópticas "
          + (f"**mejora** el IoU en {d:+.3f}: el infrarrojo cercano y el SWIR separan el agua del suelo húmedo y de las "
             "sombras, que en el SAR se confunden." if d > 0 else
             f"**no mejora** el IoU ({d:+.3f}): con pocos chips sin nubes, 13 canales más aumentan el riesgo de sobreajuste.")
          + " El problema en la práctica es que durante las tormentas casi no hay imágenes ópticas despejadas, que es "
            "justamente la razón para usar SAR." + (" " + s2["bolivia_note"] if s2.get("bolivia_note") else ""))

    # --- limitaciones
    h("Limitaciones")
    usados, disp = R["data"].get("weak", 0), R.get("bucket", {}).get("weak", {}).get("usables")
    ul(["Se usó una sola semilla por configuración, así que las diferencias pequeñas entre variantes pueden deberse a la "
        "variabilidad del entrenamiento.",
        f"Bolivia tiene {R['data']['bolivia']} chips, por lo que sus intervalos de confianza son anchos.",
        (f"Se usaron los {usados:,} chips débiles disponibles fuera de Bolivia, pero por tiempo de cómputo los modelos con "
         f"débiles se entrenaron {ep(cfg.get('weak_epochs', 0))} (frente a {cfg['epochs']} de los modelos solo con "
         "dorados); con más épocas el resultado del 2a podría cambiar."
         if not disp or usados >= disp else
         f"Se usó un subconjunto de {usados:,} de los {disp:,} chips débiles disponibles (--max_weak); con todos, el "
         "resultado del 2a podría cambiar."),
        "La etiqueta dorada también es incierta en bordes y zonas urbanas, lo que limita el IoU que se puede alcanzar.",
        "El filtro de nubes de Sentinel-2 es una regla aproximada (S2Hand no trae máscara de nubes).",
        "No se usa el ángulo de incidencia ni un modelo de elevación, que ayudarían a descartar sombras de radar."])
    return B


# ============================================================================ Markdown
def to_md(B, R, base):
    out = []
    for blk in B:
        if blk[0] == "h":
            out.append("#" * (blk[1] + 1) + " " + blk[2])
        elif blk[0] == "p":
            out.append(blk[1])
        elif blk[0] == "ul":
            out.append("\n".join(f"- {x}" for x in blk[1]))
        elif blk[0] == "table":
            df, cap = blk[1], blk[2]
            out.append(f"*{cap}*\n\n" + df.to_markdown(index=False, disable_numparse=True))  # números tal como se formatearon
        elif blk[0] == "fig":
            rel = Path(R["figures"][blk[1]]).resolve().relative_to(base.resolve())
            out.append(f"![{blk[2]}]({rel.as_posix()})\n\n*{blk[2]}*")
    return "\n\n".join(out) + "\n"


# ============================================================================ LaTeX
_REPL = {"→": r"$\rightarrow$", "≈": r"$\approx$", "·": r"$\cdot$", "Δ": r"$\Delta$", "×": r"$\times$",
         "≥": r"$\geq$", "≤": r"$\leq$", "−": "-", "±": r"$\pm$", "°": r"$^\circ$", "~": r"$\sim$",
         "✔": r"$\checkmark$", "⚠": r"(!)", "✘": r"$\times$"}


def tex(s):
    s = str(s)
    s = s.replace("\\", r"\textbackslash{}")
    for a, b in [("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"), ("_", r"\_"), ("{", r"\{"), ("}", r"\}"),
                 ("^", r"\^{}")]:
        s = s.replace(a, b)
    s = s.replace(r"\textbackslash\{\}", r"\textbackslash{}")
    for a, b in _REPL.items():
        s = s.replace(a, b)
    s = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", s)
    return s


FIG_WIDTH = dict(curvas=0.85, umbral=0.5, confusion=0.4, eventos=0.85, ejemplos_test=0.55, ejemplos_bolivia=0.55,
                 ablacion_perdida=0.6, ablacion_debiles=0.65, brecha=0.7, taxonomia=0.7, caso=1.0)


# estas figuras se quedan en analisis.md y en el cuaderno; en el PDF no entran para no pasar de 10 páginas
TEX_SKIP = {"ejemplos_test", "ejemplos_bolivia", "caso_D", "ablacion_perdida", "ablacion_debiles"}


def to_tex_body(B, R):
    out, skip_section = [], False
    for blk in B:
        if skip_section and blk[0] != "h":
            continue
        if blk[0] == "h":
            if blk[2].startswith("Resumen por entregable"):
                skip_section = True
                continue
            skip_section = False
            title = re.sub(r"^(Entregable [^:]+|Contexto): ", "", blk[2])
            out.append(("\\subsection{" if blk[1] > 2 else "\\subsection{") + tex(title) + "}")
        elif blk[0] == "p":
            out.append(tex(blk[1]))
        elif blk[0] == "ul":
            if len(blk) > 2 and not blk[2]:
                continue
            out.append("\\begin{itemize}\n" + "\n".join(f"  \\item {tex(x)}" for x in blk[1]) + "\n\\end{itemize}")
        elif blk[0] == "table":
            if len(blk) > 3 and not blk[3]:
                continue
            df, cap = blk[1], blk[2]
            cols = "l" + "c" * (df.shape[1] - 1)
            head = " & ".join(f"\\textbf{{{tex(c)}}}" for c in df.columns) + r" \\"
            body = "\n".join(" & ".join(tex(v) for v in r) + r" \\" for r in df.astype(str).values)
            out.append("\\begin{table}[!htbp]\n\\centering\\small\n\\caption{" + tex(cap) + "}\n\\begin{adjustbox}{max width=\\linewidth}\n"
                       f"\\begin{{tabular}}{{{cols}}}\n\\toprule\n{head}\n\\midrule\n{body}\n\\bottomrule\n\\end{{tabular}}\n\\end{{adjustbox}}\n\\end{{table}}")
        elif blk[0] == "fig":
            if blk[1] in TEX_SKIP:
                continue
            name = Path(R["figures"][blk[1]]).name
            w = FIG_WIDTH.get(blk[1].split("_")[0] if blk[1].startswith("caso") else blk[1], 0.8)
            out.append("\\begin{figure}[!htbp]\n\\centering\n\\includegraphics[width=" + f"{w}" + "\\linewidth]{" + name + "}\n\\caption{"
                       + tex(blk[2]) + "}\n\\end{figure}")
    return "\n\n".join(out)


REFS = r"""
\begin{thebibliography}{99}\small
\bibitem{bonafilia2020} D.~Bonafilia, B.~Tellman, T.~Anderson, E.~Issenberg. Sen1Floods11: a georeferenced dataset to train and test deep learning flood algorithms for Sentinel-1. \emph{IEEE/CVF CVPR Workshops}, pp.~210--211, 2020.
\bibitem{ronneberger2015} O.~Ronneberger, P.~Fischer, T.~Brox. U-Net: Convolutional networks for biomedical image segmentation. \emph{MICCAI}, LNCS 9351, pp.~234--241, 2015.
\bibitem{oktay2018} O.~Oktay et al. Attention U-Net: Learning where to look for the pancreas. \emph{Medical Imaging with Deep Learning (MIDL)}, 2018. arXiv:1804.03999.
\bibitem{zhou2018} Z.~Zhou, M.~M.~R.~Siddiquee, N.~Tajbakhsh, J.~Liang. UNet++: A nested U-Net architecture for medical image segmentation. \emph{DLMIA, MICCAI Workshops}, LNCS 11045, pp.~3--11, 2018.
\bibitem{milletari2016} F.~Milletari, N.~Navab, S.-A.~Ahmadi. V-Net: Fully convolutional neural networks for volumetric medical image segmentation. \emph{3DV}, pp.~565--571, 2016.
\bibitem{otsu1979} N.~Otsu. A threshold selection method from gray-level histograms. \emph{IEEE Trans. Systems, Man, and Cybernetics}, 9(1):62--66, 1979.
\bibitem{martinis2009} S.~Martinis, A.~Twele, S.~Voigt. Towards operational near real-time flood detection using a split-based automatic thresholding procedure on high resolution TerraSAR-X data. \emph{Natural Hazards and Earth System Sciences}, 9:303--314, 2009.
\bibitem{twele2016} A.~Twele, W.~Cao, S.~Plank, S.~Martinis. Sentinel-1-based flood mapping: a fully automated processing chain. \emph{International Journal of Remote Sensing}, 37(13):2990--3004, 2016.
\bibitem{konapala2021} G.~Konapala, S.~V.~Kumar, S.~K.~Ahmad. Exploring Sentinel-1 and Sentinel-2 diversity for flood inundation mapping using deep learning. \emph{ISPRS Journal of Photogrammetry and Remote Sensing}, 180:163--173, 2021.
\bibitem{mateo2021} G.~Mateo-Garcia et al. Towards global flood mapping onboard low cost satellites with machine learning. \emph{Scientific Reports}, 11:7249, 2021.
\bibitem{loshchilov2019} I.~Loshchilov, F.~Hutter. Decoupled weight decay regularization. \emph{ICLR}, 2019.
\end{thebibliography}
"""

RELATED = r"""
\textbf{Umbralización clásica.} El agua abierta refleja el pulso del radar de forma especular y por eso se ve oscura en las imágenes SAR. Los métodos operativos aprovechan esto y umbralizan la retrodispersión, muchas veces con el método de Otsu~\cite{otsu1979}, aplicado por bloques~\cite{martinis2009} o dentro de cadenas automáticas para Sentinel-1~\cite{twele2016}. Son rápidos y no necesitan etiquetas, pero fallan en sombras de radar, superficies lisas, zonas urbanas y vegetación inundada. Las etiquetas débiles de Sen1Floods11 se generaron justamente con este tipo de umbralización.

\textbf{Segmentación con redes convolucionales.} La U-Net~\cite{ronneberger2015} combina un codificador que captura el contexto con un decodificador que recupera el detalle espacial gracias a las conexiones de salto, y es la arquitectura de referencia en segmentación. Variantes como Attention U-Net~\cite{oktay2018} y UNet++~\cite{zhou2018} modifican esas conexiones. Para clases desbalanceadas, la pérdida Dice~\cite{milletari2016} optimiza directamente el solapamiento. Como optimizador usamos AdamW~\cite{loshchilov2019}, que separa el decaimiento de pesos de la actualización del gradiente.

\textbf{Aprendizaje profundo para inundaciones.} Sen1Floods11~\cite{bonafilia2020} publica 446 chips etiquetados a mano y 4,385 con etiquetas automáticas de 11 eventos, y muestra que las redes superan a la umbralización y que generalizar a un evento no visto (Bolivia) es la parte difícil. Konapala et al.~\cite{konapala2021} estudian la combinación de Sentinel-1 y Sentinel-2 en este mismo dataset, y Mateo-Garcia et al.~\cite{mateo2021} entrenan modelos globales de detección de agua con etiquetas imperfectas a gran escala.
"""


def to_tex(B, R):
    t, b, fin = R["models"][R["final_model"]]["test"], R["models"][R["final_model"]]["bolivia"], R["final_model"]
    abstract = tex(
        f"Las nubes que acompañan a las tormentas dejan sin uso a los satélites ópticos justo cuando más se necesitan, "
        f"pero el radar de apertura sintética (SAR) de Sentinel-1 puede ver a través de ellas. En este trabajo entrenamos "
        f"desde cero una U-Net en PyTorch para segmentar el agua de inundación píxel a píxel en Sen1Floods11 usando las "
        f"bandas VV y VH. Estudiamos cómo combinar los 446 chips etiquetados a mano con miles de etiquetas débiles "
        f"obtenidas por umbralización, comparamos tres funciones de pérdida y medimos la generalización a un evento no "
        f"visto (Bolivia). El modelo final, {fin}, obtiene IoU = {t['iou']:.3f} y F1 = {t['f1']:.3f} en el test, e "
        f"IoU = {b['iou']:.3f} en Bolivia. También analizamos los errores típicos del SAR (sombra de radar, doble rebote "
        f"y bordes con suelo húmedo) usando reglas físicas simples.")
    intro = tex(
        "Durante una inundación es clave saber dónde está el agua en las primeras horas para evacuar, llevar ayuda y "
        "estimar daños. En el Perú, eventos como El Niño costero causan inundaciones repentinas que hay que mapear con las "
        "primeras imágenes disponibles. Los sensores ópticos no ven a través de las nubes de la tormenta, mientras que el "
        "radar de apertura sintética (SAR) de Sentinel-1 toma imágenes de día y de noche aunque esté nublado. En SAR el "
        "agua abierta se ve oscura porque refleja el pulso lejos del sensor, pero la sombra de radar, las superficies "
        "lisas, el doble rebote en zonas urbanas y la vegetación inundada rompen esa regla. En este trabajo entrenamos una "
        "U-Net propia sobre Sen1Floods11 para responder tres preguntas: (i) cómo combinar pocas etiquetas manuales con "
        "muchas etiquetas automáticas pero ruidosas, (ii) qué función de pérdida conviene con un desbalance de clases tan "
        "fuerte y (iii) cuánto empeora el modelo en un evento que no vio durante el entrenamiento.")
    body = to_tex_body(B, R)
    return r"""\documentclass{article}
% Si neurips_2024.sty está en la misma carpeta se usa el formato oficial; si no, un formato parecido.
\IfFileExists{neurips_2024.sty}{\usepackage[preprint]{neurips_2024}}{\usepackage[letterpaper,textwidth=5.5in,textheight=9in]{geometry}\IfFileExists{mathptmx.sty}{\usepackage{mathptmx}}{}\AtBeginDocument{\renewcommand{\And}{\and}}}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\IfFileExists{spanish.ldf}{\usepackage[spanish,es-noshorthands]{babel}}{}
\usepackage{graphicx,booktabs,amsmath,amssymb,float,adjustbox}
\usepackage{hyperref}
\graphicspath{{figures/}{../figures/}}
\title{Mapeo de inundaciones con radar Sentinel-1 y una U-Net entrenada desde cero: etiquetas débiles, pérdidas y generalización geográfica}
\author{Integrante 1 \And Integrante 2 \And Integrante 3 \And Integrante 4 \And Integrante 5\\ Redes Neuronales y Aprendizaje Profundo -- UNI FIIS 2026-II}
\begin{document}
\maketitle
\begin{abstract}
""" + abstract + r"""
\end{abstract}

\section{Introducción}
""" + intro + r"""

\section{Trabajos relacionados}
""" + RELATED + r"""

\section{Metodología, experimentos y resultados}
""" + body + r"""

\section{Conclusiones}
""" + tex(
        f"Una U-Net compacta entrenada desde cero con VV y VH segmenta el agua de inundación con IoU {t['iou']:.3f} en "
        f"eventos vistos y {b['iou']:.3f} en Bolivia. Las decisiones sobre la supervisión y la pérdida se tomaron con la "
        "validación dorada, y los errores que quedan se concentran donde la física del SAR es ambigua: sombras y "
        "superficies lisas, doble rebote y bordes con suelo húmedo. Como trabajo futuro, agregar el ángulo de incidencia, "
        "un modelo de elevación o la coherencia interferométrica parecen las mejoras más prometedoras.") + r"""

\section*{Reproducibilidad}
Todo el proyecto se ejecuta con \texttt{python main.py} (ver el README del repositorio): descarga los datos de \texttt{gs://sen1floods11}, entrena todos los modelos, los evalúa y vuelve a generar este informe con los números obtenidos.

""" + REFS + r"""
\end{document}
"""


# ============================================================================ README
def readme(R, md_results):
    cfg = R["config"]
    return f"""# Sen1Floods11: mapeo de inundaciones con una U-Net desde cero (Pregunta 2)

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
{R['data'].get('weak', 0):,} chips débiles fuera de Bolivia con sus dos etiquetas automáticas (Otsu de Sentinel-1 e índice
espectral de Sentinel-2). Algunas opciones: `--quick true` (prueba rápida de pocos minutos), `--max_weak 1000` (usar menos
chips débiles), `--run_s2_bonus false`, `--epochs 60`. El experimento completo necesita GPU (esta corrida tomó
{R.get('minutes', 0):.0f} min en `{R['device']}`).

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

U-Net {cfg['base_ch']}/{cfg['depth']} ({R['n_params'] / 1e6:.2f} M parámetros), recortes de {cfg['crop']} px, {cfg['epochs']} épocas,
AdamW lr={cfg['lr']}, pos_weight={cfg['pos_weight']}, {R['data'].get('weak', 0)} chips débiles, semilla {cfg['seed']}.

## Resultados

{md_results}
"""


def write_reports(R, out):
    out = Path(out)
    B = build_blocks(R)
    md = to_md(B, R, out)
    (out / "analisis.md").write_text("# Pregunta 2: análisis de resultados\n\n" + md, encoding="utf-8")
    rows = [dict(Modelo=n, Umbral=f"{v['threshold']:.2f}", **{"IoU test": f3(v["test"]["iou"]),
            "IoU medio/chip test": f3(v["test"]["mean_iou_chips"]), "F1 test": f3(v["test"]["f1"]),
            "IoU Bolivia": f3(v["bolivia"]["iou"]), "F1 Bolivia": f3(v["bolivia"]["f1"])}) for n, v in R["models"].items()]
    res_md = (f"Modelo final (elegido por IoU de validación): **{R['final_model']}**\n\n"
              + pd.DataFrame(rows).to_markdown(index=False) + "\n\nAnálisis completo: `outputs/analisis.md`.")
    (REPO / "README.md").write_text(readme(R, res_md), encoding="utf-8")
    inf = out / "informe"
    (inf / "figures").mkdir(parents=True, exist_ok=True)
    for pth in R["figures"].values():
        shutil.copy(pth, inf / "figures" / Path(pth).name)
    (inf / "informe_p2.tex").write_text(to_tex(B, R), encoding="utf-8")
    print(f"  análisis: {out / 'analisis.md'}\n  informe : {inf / 'informe_p2.tex'}\n  README  : {REPO / 'README.md'}")
