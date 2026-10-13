"""
Etiquetador multilingüe de informes de RM de rodilla.

Lee un CSV con la columna 'Report' (texto libre del informe radiológico) y marca
12 etiquetas (1 = hallazgo presente, 0 = ausente):
ACL, MCL, Medial Meniscus, Lateral Meniscus, Medial OA, Lateral OA, PF OA,
Effusion, Synovitis, Baker's, Contusion, Fracture.

Idiomas cubiertos: inglés, castellano, francés, alemán, neerlandés, croata, turco.

Versión 1.1 (doble check): grado 1 de LCA/LCM cuenta como lesión, derrame/Baker mínimos cuentan,
Hoffa sin inflamación no es sinovitis, más vocabulario (búlgaro, turco, griego, croata).

Lógica general (por cada informe):
1. Normalizar el texto (minúsculas, sin tildes).
2. Quitar la parte de antecedentes / indicación clínica (allí aparecen dudas
   del médico solicitante, no hallazgos).
3. Partir en cláusulas (frases) y, dentro de cada una:
   - detectar términos de lesión y descartarlos si están negados
     ("sin rotura", "no fracture", "keine Fraktur", "geen scheur"...)
   - asignar cada lesión a la estructura anatómica más cercana
     (LCA, LCM, menisco medial/lateral, etc.)
   - para artrosis, decidir el compartimento (medial, lateral, patelofemoral).
4. Las filas que ya traen etiquetas en el CSV original se respetan.
"""

import re
import sys
import unicodedata
import pandas as pd

# ----------------------------------------------------------------------------
# CONSTANTES PRINCIPALES (modificables)
# ----------------------------------------------------------------------------
RUTA_ENTRADA = "train.csv"
RUTA_SALIDA = "train_etiquetado.csv"
COLUMNA_TEXTO = "Report"
ETIQUETAS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
             "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
             "Contusion", "Fracture"]
RESPETAR_ETIQUETAS_EXISTENTES = True   # no sobrescribir filas ya etiquetadas
ALCANCE_NEGACION_PREVIA = 70           # caracteres que cubre "no/sin/kein..." hacia adelante
ALCANCE_NEGACION_POSTERIOR = 45        # caracteres tras el término para "no se observa", "izlenmedi"...
DISTANCIA_MAX_ENTIDAD = 130            # distancia máxima lesión -> estructura anatómica
VENTANA_LADO_MENISCO = 35              # ventana para buscar "medial/lateral" junto a "menisco"
VERSION = "1.1"
EXCLUIR_LIGAMENTO_GRADO_1 = False      # v1.1: esguince grado 1 SÍ es lesión de LCA/LCM
CONTAR_DERRAME_MINIMO = True           # v1.1: derrame/Baker "mínimo/trace" informado cuenta como hallazgo
INFERIR_ESGUINCE_LCM_POR_EDEMA = False # v1.1: edema periligamentoso solo no se toma como lesión del LCM
HOFFA_COMO_SINOVITIS = False           # v1.1: edema/pinzamiento de Hoffa sin inflamación no es sinovitis


# ----------------------------------------------------------------------------
# NORMALIZACIÓN
# ----------------------------------------------------------------------------
_MAPA_ESPECIAL = str.maketrans({"∆": "Δ", "ß": "ss", "ı": "i", "đ": "d", "ø": "o", "æ": "ae",
                                "’": "'", "`": "'", "´": "'"})


def normalizar(texto: str) -> str:
    """Minúsculas, sin tildes ni diacríticos, espacios simples por línea."""
    texto = str(texto).translate(_MAPA_ESPECIAL)
    # une líneas cortadas a mitad de frase (texto extraído de PDF): salto seguido de minúscula
    texto = re.sub(r"[ \t]*\r?\n[ \t]*(?=[a-zà-öø-ÿα-ωа-я(])", " ", texto)
    texto = texto.lower()
    # "Fractures :\n Aucune" -> misma cláusula, para que la negación posterior funcione
    texto = re.sub(r"[ \t]*;[ \t]*(?=(none|nil|absent|normal)\b)", " : ", texto)
    texto = re.sub(r"[ \t]*:[ \t]*\r?\n[ \t]*(?=(aucun|none|nil|no |ninguno|ninguna|keine?|geen|nema|yok|pas |absent|normal))", " : ", texto)
    texto = unicodedata.normalize("NFKD", texto)
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    texto = re.sub(r"[ \t\xa0]+", " ", texto)
    return texto


# ----------------------------------------------------------------------------
# LIMPIEZA DE ANTECEDENTES / INDICACIÓN
# ----------------------------------------------------------------------------
RE_INICIO_HALLAZGOS = re.compile(
    r"\b(findings|hallazgos|bevindingen|befunde?|resultados|bulgular|nalaz|"
    r"interpretation|constatations|resultats|ευρηματα|мр находка)\s*:?")

RE_ENCABEZADO_CLINICO = re.compile(
    r"\b(indications?|indicacion(es)?|history|clinical (history|information|indication|data|question)|"
    r"reason for (exam|study)|motivo( de consulta)?|antecedentes( clinicos)?|datos clinicos|"
    r"informacion clinica|klinische (inlichtingen|angaben|informatie|fragestellung)|"
    r"fragestellung|anamnese|anamnesis|(diagnostische )?vraagstel+ing|renseignements cliniques|"
    r"indication clinique|motif|klinicki podaci|uputna dijagnoza|klinik bilgi|endikasyon|"
    r"comparison|comparacion|technique|tecnica|scanprotocol|protocol[eo]?|tetkik protokolu)\s*:")


def quitar_antecedentes(texto: str) -> str:
    """Si hay un marcador de hallazgos ('Findings:', 'Hallazgos:'...), se descarta lo previo.
    Además se eliminan las cláusulas que empiezan con encabezados clínicos."""
    m = RE_INICIO_HALLAZGOS.search(texto)
    if m and RE_ENCABEZADO_CLINICO.search(texto[:m.start()]):
        texto = texto[m.start():]
    return texto


# ----------------------------------------------------------------------------
# SEGMENTACIÓN EN CLÁUSULAS
# ----------------------------------------------------------------------------
RE_SEPARADOR = re.compile(r"(?<!\d)\.(?!\d)|[;\n!?]|\s>\s|\s--\s|\*")


RE_PUNTO_NO_FINAL = re.compile(r"\b([ivx]{1,4}|\d{1,2})\.(?=\s?(stupnj\w*|stup|grad\w*|stad\w*|степен|°))|"
                               r"\b(v|m|mm|lig|ca|bzw|z|st|n|dd|vs|cf|sr|img|ser|approx|aprox)\.(?=\s?\w)")


def partir_clausulas(texto: str):
    texto = RE_PUNTO_NO_FINAL.sub(lambda m: m.group(0)[:-1], texto)
    return [c.strip() for c in RE_SEPARADOR.split(texto) if c and c.strip()]


# ----------------------------------------------------------------------------
# NEGACIÓN E INCERTIDUMBRE
# ----------------------------------------------------------------------------
# "no se puede descartar", "cannot be excluded" => hallazgo posible => se cuenta como positivo
RE_INCERTIDUMBRE = re.compile(
    r"(cannot be (excluded|ruled out)|can ?not be (excluded|ruled out)|can't be (totally )?(excluded|ruled out)|"
    r"not (be )?(totally )?excluded|no (se )?puede(n)? descartar(se)?|no se descarta|"
    r"nicht (sicher )?auszuschliessen|nicht (sicher )?ausgeschlossen|ne peut etre exclu\w*|"
    r"non exclu\w*|niet (uit te sluiten|uitgesloten)|ne moze se (sa sigurnoscu )?iskljucit\w*|"
    r"ekarte edilemez|dislanamaz)")

RE_NEGACION_PREVIA = re.compile(
    r"\b(no|not|without|negative for|absence of|absent|free of|neither|nor|sin|ni|ausencia de|"
    r"descart\w*|pas|sans|aucune?|kein\w*|ohne|nicht|weder|geen|zonder|niet|bez|nema|ne|nije|"
    r"nisu|nao|sem|nem|non|rule out|r/o|unassociated with|no longer|πλεον|δεν|χωρις|ουτε|μη|ελλειψη|απουσια|няма|без|не|липса)\b")

RE_CORTE_NEGACION = re.compile(
    r"\b(but|however|although|except|apart from|other than|pero|aunque|sin embargo|excepto|salvo|"
    r"mais|cependant|aber|jedoch|allerdings|maar|wel|doch|evenwel|ali|medutim|osim|ancak|fakat|"
    r"with|con|mit|avec|met|uz|ile|there is|there are|hay|se observa\w*|se aprecia\w*|se identifica\w*|"
    r"presenta|existe|il existe|on note|es besteht|er is|postoji|vidi se|nalazi se|αλλα|ομως|με|но|със|с|вижда се)\b")

RE_INICIO_POSITIVO = re.compile(
    r"^\s*(small|mild|moderate|large|massive|minimal|trace|focal|leve|moderad\w*|pequen\w*|"
    r"discret\w*|minim\w*|importante|abundante|petit\w*|modere\w*|important\w*|"
    r"klein\w*|gering\w*|deutlich\w*|ausgepragt\w*|beperkt\w*|matig\w*|uitgesproken|"
    r"manj\w*|blag\w*|izrazen\w*|hafif|belirgin)\b")

RE_NEGACION_POSTERIOR = re.compile(
    r"\b(not (seen|identified|demonstrated|evident|present|detected|visualized|noted)|is excluded|"
    r"are excluded|excluded|ruled out|no (se )?(observa\w*|identifica\w*|aprecia\w*|evidencia\w*|"
    r"visualiza\w*|demuestra\w*)|descartad\w*|ausente|n'est pas|ne sont pas|absente?s?|"
    r"nicht (nachweisbar|abgrenzbar|erkennbar|nachzuweisen|sichtbar)|ausgeschlossen|"
    r"niet (aanwezig|aantoonbaar|weerhouden|zichtbaar)|ne (nalazi|prikazuje|vidi)|"
    r"nije (prisutn|vidljiv)\w*|izlen(medi|memis\w*)|saptan(madi|mamis\w*)|yok|gozlenmedi|"
    r"aucun\w*|none|nil|ninguno|ninguna|pas de|no hay|keine?|geen|nema|"
    r"etmiyor|bulunmamakta\w*|rastlanmadi|degil\w*|δεν (παρατηρ|αναγνωρ|απεικον)\w*|не (се )?(вижда|установява|намира|визуализира)\w*)")


RE_NEGACION_FUERTE = re.compile(r"\b(no evidence of|no signs? of|kein (nachweis|hinweis)\w*|sin (evidencia|signos) de|"
                                r"geen (aanwijzing|teken)\w*|pas de (signe|preuve)\w*|nema znakova|bez znakova|δεν (υπαρχει|παρατηρ)\w*|няма данни)")
RE_CONTRASTE = re.compile(r"\b(but|however|pero|sin embargo|mais|aber|jedoch|maar|ali|medutim|ancak|αλλα|но)\b|[;:]")


def esta_negado(clausula: str, inicio: int, fin: int) -> bool:
    """True si el término [inicio:fin] está bajo una negación dentro de la cláusula."""
    # negación previa: cue -> ... -> término, sin palabras de corte entre medio
    desde = max(0, inicio - ALCANCE_NEGACION_PREVIA)
    previo = clausula[desde:inicio]
    for m in RE_NEGACION_PREVIA.finditer(previo):
        tramo = previo[m.end():]
        # "no hay", "ne nalazi se", "no se observa": la frase existencial pegada a la negación no corta
        if any(c.start() > 2 for c in RE_CORTE_NEGACION.finditer(tramo)):
            continue
        # si tras una coma el segmento empieza con "leve/small/klein..." se rompe el alcance
        partes = tramo.split(",")
        if len(partes) > 1 and RE_INICIO_POSITIVO.match(partes[-1]):
            continue
        return True
    # negación fuerte ("no evidence of", "kein Nachweis"...): alcance hasta el inicio de la cláusula
    for m in RE_NEGACION_FUERTE.finditer(clausula[:inicio]):
        tramo = clausula[m.end():inicio]
        if not RE_CONTRASTE.search(tramo) and len(tramo) < 160 and \
           not any(c.start() > 2 for c in RE_CORTE_NEGACION.finditer(tramo)) and \
           not any(RE_INICIO_POSITIVO.match(p) for p in tramo.split(",")[1:]):
            return True
    # negación posterior: término -> "no se observa", "izlenmedi"...
    posterior = clausula[fin:fin + ALCANCE_NEGACION_POSTERIOR]
    m = RE_NEGACION_POSTERIOR.search(posterior)
    if m and not re.search(r"[,]|\b(and|y|et|und|en|i|ve)\b", posterior[:m.start()]):
        return True
    return False


# ----------------------------------------------------------------------------
# ESTRUCTURAS ANATÓMICAS (para atribuir cada lesión a la más cercana)
# ----------------------------------------------------------------------------
ENTIDADES = {
    "ACL": r"\bacl|\blca\b|\blcae\b|ligamento cruzado anterior|cruzado anterior|anterior cruciate|"
           r"ligament croise anterieur|croise anterieur|vorder\w* kreuzband|\bvkb|voorste kruisband|"
           r"prednj\w* kri\w*|\bpkl\b|on capraz bag|\bocb\b|anterior capraz|προσθι\w* χιαστ\w*|предн\w* кръстн\w*|\bпкв\b",
    "PCL": r"\bpcl|\blcp\b|cruzado posterior|posterior cruciate|croise posterieur|hinter\w* kreuzband|"
           r"\bhkb|achterste kruisband|straznj\w* kri\w*|\bskl\b|arka capraz|\bacb\b|οπισθι\w* χιαστ\w*|задн\w* кръстн\w*|\bзкв\b",
    "MCL": r"\bmcl|\blcm\b|\blli\b|medial collateral|tibial collateral|colateral medial|colateral interno|"
           r"lateral interno|collateral (interne|medial)|collateral\w* interne|ligament lateral interne|"
           r"innenband|medial\w* (kollateral|seiten)band|mediale collaterale|medial\w* collateral\w* (band|ligament)|"
           r"\bmcb\b|medijaln\w* kolateraln\w*|medial kollateral|medyal kollateral|ic yan bag|tibial kollateral|"
           r"εσω πλαγι\w*|медиал\w* колатерал\w*",
    "LCL": r"\blcl|\bfcl\b|\blle\b|lateral collateral|fibular collateral|colateral lateral|colateral externo|"
           r"lateral externo|collateral (externe|lateral)|ligament lateral externe|aussenband|"
           r"lateral\w* (kollateral|seiten)band|laterale collaterale|lateraln\w* kolateraln\w*|"
           r"lateral kollateral|dis yan bag|posterolateral corner|εξω πλαγι\w*|латерал\w* колатерал\w*",
    "MENISCO": r"menisc(?!o-?(femoral|tibial|poplit))|menisk|mensik|menisq|meniscal|meniskal|\bim-|\bam-|μηνισκ|менискус|менискуса",
    "TENDON": r"tendon|tendao|tendin|sehne|pees|pezen|tetiv|tendo|mpfl|patellofemoral ligament|"
              r"retinacul|popliteus|popliteo|iliotibial|cintilla|pes anserin|pata de ganso|τενοντ|сухожил|ретинакул",
    "CARTILAGO": r"cartilag|cartilago|knorpel|kraakbeen|hrskav|kikirdak|chondr|condr|hondr|χονδρ|хрущял|хондр",
    "HUESO": r"\bbone\b|\bhueso|\bos\b|osseu|knochen|\bbot\b|kost|kemik|condyle|condilo|kondyl|"
             r"plateau|platillo|meseta|tibia|femur|patella|rotula|fibula|perone|οστ|κονδυλ|κνημ|μηρια|κост|кондил|тибия|фемур|пател",
    "MUSCULO": r"muscle|muscul|musculo|muskel|spier|misic|\bkas\b",
}
RE_ENTIDADES = {k: re.compile(v) for k, v in ENTIDADES.items()}

# términos de lesión (ligamento / menisco)
RE_LESION = re.compile(
    r"\b(tear|tears|torn|tearing|rupture\w*|disrupt\w*|discontinu\w*|sprain\w*|injur\w*|avuls\w*|avulz\w*|"
    r"deficien\w*|insufficien\w*|incompeten\w*|rotur\w*|roto|rota|ruptur\w*|desgarr\w*|esguince\w*|"
    r"lesion\w*|desinsercion\w*|elongacion|distension|rompu\w*|dechirure\w*|entorse\w*|"
    r"desinsertion\w*|elongation|perdida (parcial |total )?de (la )?continuidad|lasion\w*|zerrung|distorsion|uberdehnung|insuffizien\w*|gerissen|"
    r"verstuiking|distorsie|elongatie|gescheurd|letsel\w*|lesie\w*|lezij\w*|ozljed\w*|distenzij\w*|"
    r"distorzij\w*|istegnu\w*|yirti\w*|hasar\w*|kopu\w*|zedelen\w*|zorlanma\w*|lezyon\w*|fissur\w*|"
    r"fisura\w*|bucket|asa de (balde|cubo)|anse de seau|flap|colgajo|macerat\w*|root tear|ramp|"
    r"parrot|kova sapi|ρηξη\w*|ρηξεω\w*|ασυνεχει\w*|κακωση\w*|διαστρεμμ\w*|"
    r"скъсан\w*|devamsiz\w*|butunluk kayb\w*|devamlilik kayb\w*|ruprut\w*|скъсван\w*|разкъсван\w*|руптур\w*|лезия\w*|лезии|увреда\w*|травм\w* увред\w*)"
    r"|riss|ruptur|scheur|ruptuur|korbhenkel|emmerhengsel|lasion")

RE_GRADO_1 = re.compile(r"\b(grad[eo]?|grado|stupnj\w*|derece|тип|степен|стадии|βαθμ\w*)\s*(1|i)\b|\b(1|i)\.? ?(grad|stupnj)|interstitial|intersticial")
RE_GRADO_ALTO = re.compile(r"\b(grad[eo]?|grado|stupnj\w*)\s*(2|3|ii|iii)\b|complet\w*|partial|parcial|partiel\w*")
RE_GRADO_BAJO_MENISCO = re.compile(r"\b(grad[eo]?|grado|stupnj\w*|тип|степен|стадии|βαθμ\w*)\s*(1|2|i|ii)[abcабв]?\b|"
                                   r"\b(1|2|i|ii)[abcабв]?\.?\s*(степен|стадии|стодии|stupnj\w*|grad\w*)|[–-]\s*(1|2)[abcабв]?\b")
RE_NO_ALCANZA = re.compile(r"не достига|недостига\w*|(does not|doesn't|not) (reach|extend\w*|communicat\w*) (to |with )?(the )?(articular )?surface|"
                           r"no (alcanza|contacta|impresiona contactar)|sans atteindre|n'atteint pas|ne doseze|ulasmayan|ulasmiyor")
RE_GRADO_ALTO_MENISCO = re.compile(r"\b(grad[eo]?|grado|stupnj\w*|тип|степен|стадии|βαθμ\w*)\s*(3|iii)\b|(?<!не )\bдостига\w* (\w+ ){0,3}(ставн|повърхност|края|ръба)")
RE_DEGENERACION_G3 = re.compile(r"(grade|grado|derece)\s*(3|iii)\s*(dejenerasyon\w*|meniskopati\w*|degenera\w*|signal|senal|lesion\w*)|"
                                r"(dejenerasyon\w*|meniskopati\w*)\s*(grade|grado)?\s*(3|iii)\b")
RE_LLEGA_SUPERFICIE = re.compile(r"(?<!не )\bдостига\w* (\w+ ){0,3}(ставн|повърхност|края)|extend\w* to the (articular )?surface|"
                                 r"contacta\w* (con )?la superficie|reach\w* the (articular )?surface")
RE_TERMINO_ROTURA_FRANCA = re.compile(
    r"tear|torn|rotur|ruptur|desgarr|riss|scheur|ruptuur|yirti|dechirure|bucket|asa de|anse|korbhenkel|"
    r"emmerhengsel|flap|colgajo|macerat|root|ramp|kova|fissur|fisura|ρηξ|скъсван|скъсан|разкъсван|руптур")

# lateralidad para meniscos
RE_LADO_MEDIAL = re.compile(r"mediyal|medial|interno|interne|innen|mediaal|medijal\w*|medyal|\bim-|\bεσω\b|медиал|вътрешн")
RE_LADO_LATERAL = re.compile(r"lateral|externo|externe|aussen|lateraal|lateraln\w*|\bam-|\bεξω\b|латерал|външн")
RE_AMBOS = re.compile(r"\b(ambos|both|beide[n]?|les deux|oba|her iki|bilateral\w*)\b.{0,20}menis|"
                      r"menisc\w* (medial|interno) y (lateral|externo)|medial and lateral menisc|"
                      r"medial (y|and|et|und|en|i|ve) lateral\w* menis|menisques? interne et externe|"
                      r"menisco interno y lateral|menisco medial y lateral|inner\w* und aussen\w*|"
                      r"interno y externo|εσω και εξω μηνισκ|медиал\w* и латерал\w* менискус|двата менискус")


def lado_de_menisco(clausula: str, inicio: int, fin: int, contexto: str, ultimo_lado: str = ""):
    """Devuelve {'medial'}, {'lateral'}, ambos o vacío, según palabras de lado cercanas."""
    desde = max(0, inicio - VENTANA_LADO_MENISCO)
    ventana = clausula[desde:fin + VENTANA_LADO_MENISCO]
    if RE_AMBOS.search(ventana):
        return {"medial", "lateral"}
    # palabra de lado más cercana (incluye compuestos como "innenmeniskus")
    candidatos = []
    for lado, rx in (("medial", RE_LADO_MEDIAL), ("lateral", RE_LADO_LATERAL)):
        for m in rx.finditer(ventana):
            pos = desde + m.start()
            if inicio - 8 <= pos <= fin:
                dist = 0
            elif pos < inicio:
                dist = inicio - pos
            else:
                dist = pos - fin + 12  # v1.1: el adjetivo previo ("medial meniscus") tiene prioridad
            # evitar "tibial lateral", "condilo lateral": si entre medio hay hueso, penalizar
            candidatos.append((dist, lado))
    if candidatos:
        candidatos.sort()
        return {candidatos[0][1]}
    # respaldo: lado unido a "menisco" en cualquier parte de la cláusula
    lados = set()
    if re.search(r"(medial|interno|interne|innen|mediaa?l\w*|medijal\w*|medyal|εσω|медиал\w*)\s+(\w+\s+)?(menisc|menisk|mensik|menisq|μηνισκ|менискус)|"
                 r"(menisc|menisk|mensik|menisq|μηνισκ|менискус)\w*\s+(\w+\s+)?(medial|interno|interne|mediaa?l\b)", clausula):
        lados.add("medial")
    if re.search(r"(lateral|externo|externe|aussen|lateraa?l\w*|lateraln\w*|εξω|латерал\w*)\s+(\w+\s+)?(menisc|menisk|mensik|menisq|μηνισκ|менискус)|"
                 r"(menisc|menisk|mensik|menisq|μηνισκ|менискус)\w*\s+(\w+\s+)?(lateral|externo|externe|lateraa?l\b)", clausula):
        lados.add("lateral")
    if len(lados) == 1:
        return lados
    if contexto in ("medial", "lateral"):
        return {contexto}
    # v1.1: frase sin lado que continúa la descripción del menisco citado en la cláusula previa
    if ultimo_lado in ("medial", "lateral"):
        return {ultimo_lado}
    return set()


RE_MENISCO_CON_LADO_MED = re.compile(r"(mediyal|medial|interno|interne|innen|mediaa?l\w*|medijal\w*|medyal|εσω|медиал\w*)\s+(\w+\s+)?(menisc|menisk|mensik|menisq|μηνισκ|менискус)|innenmenisk|"
                                     r"(menisc|menisk|menisq|μηνισκ|менискус)\w*\s+(medial|interno|interne)")
RE_MENISCO_CON_LADO_LAT = re.compile(r"(lateral|externo|externe|aussen|lateraa?l\w*|lateraln\w*|εξω|латерал\w*)\s+(\w+\s+)?(menisc|menisk|mensik|menisq|μηνισκ|менискус)|aussenmenisk|"
                                     r"(menisc|menisk|menisq|μηνισκ|менискус)\w*\s+(lateral|externo|externe)")


def lado_citado_en(clausula: str, previo: str) -> str:
    """Devuelve el lado del menisco nombrado en la cláusula (si hay uno solo); si no, conserva el previo."""
    med, lat = bool(RE_MENISCO_CON_LADO_MED.search(clausula)), bool(RE_MENISCO_CON_LADO_LAT.search(clausula))
    if med and not lat:
        return "medial"
    if lat and not med:
        return "lateral"
    if med and lat:
        return ""
    return previo


def entidad_mas_cercana(clausula: str, inicio: int, fin: int):
    """Busca la estructura anatómica más cercana al término de lesión."""
    mejor = (10 ** 6, None, None)
    for nombre, rx in RE_ENTIDADES.items():
        for m in rx.finditer(clausula):
            if m.start() <= inicio < m.end() or inicio <= m.start() < fin:
                dist = 0          # compuestos tipo "innenmeniskusriss"
            elif m.end() <= inicio:
                dist = inicio - m.end()
            else:
                dist = m.start() - fin + 3   # leve preferencia a la entidad que precede
            if nombre == "HUESO":
                dist += 40      # los huesos rara vez son el destino de "rotura/esguince"
            if dist < mejor[0]:
                mejor = (dist, nombre, m)
    if mejor[1] is None or mejor[0] > DISTANCIA_MAX_ENTIDAD:
        return None, None
    return mejor[1], mejor[2]


# ----------------------------------------------------------------------------
# ARTROSIS / CONDROPATÍA Y COMPARTIMENTOS
# ----------------------------------------------------------------------------
RE_ARTROSIS = re.compile(
    r"osteoarthr\w*|arthros\w*|artros\w*|gonartro\w*|gonarthro\w*|osteoartr\w*|\boa\b|artroz\w*|"
    r"degenerative (joint|arthr|change\w* of the (joint|cartilage))|osteofit\w*|osteophyt\w*|osteofyt\w*|"
    r"osteophytar\w*|chondropath\w*|chondropat\w*|condropat\w*|chondromalac\w*|condromalac\w*|"
    r"hondromalac\w*|hondropat\w*|kondromalaz\w*|kondropat\w*|kraakbeenlijden|chondrosis|condrosis|"
    r"chondral\w* (loss|defect\w*|lesion\w*|lasion\w*|ulcer\w*|injur\w*|thinning|fissur\w*|wear|schad\w*)|knorpelschad\w*|osteochondral (defect|lesion)|"
    r"cartilage (loss|thinning|defect\w*|fissur\w*|wear|degenerat\w*|lesion\w*|irregular\w*|damage)|"
    r"(loss|thinning|fissur\w*|defect\w*|irregular\w*) of (the )?[a-z ]{0,35}cartilage|"
    r"full[- ]thickness (cartilage|chondral)|high[- ]grade (cartilage|chondral)|partial[- ]thickness (cartilage|chondral)|"
    r"ulcera\w* condral\w*|lesion\w* condral\w*|defecto\w* condral\w*|perdida (de )?cartilag\w*|"
    r"adelgazamiento (del |de )?cartilag\w*|adelgazamiento condral|fisura\w* (del |de )?cartilag\w*|"
    r"knorpel(schaden|verlust|glatze|ausdunnung|defekt|ulzer\w*|lasion\w*|aufbrauch\w*|"
    r"degeneration)|aufgebraucht\w* knorpel|knorpelschicht|"
    r"kraakbeen\w* (verdun\w*|defect\w*|letsel\w*|onregelmat\w*|fissu\w*|lijden)|verdund\w* (voorkomen )?kraakbeen|"
    r"stanjen\w* (artikulacijsk\w* )?hrskavic\w*|hrskavic\w* (stanjen|ostecen|defekt|nepravil)\w*|"
    r"kikirdak\w* (incel|kayip|defekt|duzensiz|hasar)\w*|kondral (defekt|lezyon|incel)\w*|"
    r"subchondral sclerosis|eburnation|joint space narrowing|pinzamiento (articular|femorotibial)|"
    r"amincissement (du )?cartilag\w*|chondrolyse|fissure\w* cartilagin\w*|ulceration\w* cartilagin\w*|"
    r"atteinte cartilagin\w*|dejeneratif (artrit|eklem)\w*|dejeneratif eklem hastalig\w*|chondral\w* ossifikation|"
    r"οστεοαρθρ\w*|αρθροπαθ\w*|οστεοφυτ\w*|χονδροπαθ\w*|χονδρομαλακ\w*|"
    r"(λεπτυνση|διαβρωση|απωλεια|εξαλειψη|ρωγμ\w*|αποπτωση)\w* (\w+ ){0,3}(αρθρικ\w* )?χονδρ\w*|icrs|"
    r"degenerativn\w* promjen\w*|osteoartrotsk\w*|osteofitoz\w*|osteofitik\w*|"
    r"гонартроз\w*|артроз\w*|остеоартр\w*|остеофит\w*|хондромалаци\w*|хондропати\w*|"
    r"лизиран\w* (\w+ ){0,2}хрущял\w*|дефект\w* в (\w+ )?хрущял\w*|стесн\w* ставн\w* междин\w*|ставн\w* междина (\w+ ){0,2}стеснена|"
    r"изтън\w* хрущял\w*|хрущял\w* (лезия|дефект)\w*|(лезия|лезиия|дефект)\w* на хрущял\w*|хрущялн\w* (лезия|дефект)\w*")
RE_GRADO_1_CARTILAGO = re.compile(r"(\b(grad[eo]?|grado|stupnj\w*|icrs|stage|tip|тип|степен)\s*(1|i)\b(?!\s*(-|a|y|and|und|to|et|i)\s*(2|3|ii|iii))"
                                  r"|\b(1|i)\s*°|\b(1|i)\s*\.?\s*(stupnj\w*|st\b|grad\w*)|\b(1|i)\.?\s*stupnj\w*)")

# Compartimentos. Primero se neutralizan términos que contienen "patelar" pero no son cartílago PF.
RE_NO_PF = re.compile(
    r"(tendon|tendón|tendao|ligament\w*|lig\.?|sehne|pees|tetiv\w*) (rotulian\w*|patel\w*|de la rotula|"
    r"rotuliano)|patellar (tendon|ligament)|patellarsehne|patellapees|patellasehne|patelarn\w* ligament\w*|"
    r"(infra|supra|pre)patel\w*|(infra|supra|pre)rotulian\w*|patella (alta|baja|baja|bipartita|infera)|"
    r"medial patellofemoral ligament|mpfl|patellar retinacul\w*|patellofemoral ligament")
RE_PF = re.compile(
    r"patell\w*|patel\w*|pater\w*femor\w*|rotul\w*|troch\w*|trocl\w*|retropatell\w*|femoropatel\w*|patelofemor\w*|"
    r"femoro-patel\w*|patello-?femor\w*|facet\w*|gleitlager|knieschijf|kniescheibe|trohle\w*|"
    r"\bpf\b|\bfp\b|diz kapag\w*|femoro-?patellair\w*|επιγονατιδ\w*|τροχιλι\w*|μηροεπιγονατιδ\w*|"
    r"προσθιο διαμερισμ\w*|пател\w*|трохле\w*")
RE_ANATOMIA_FT = (r"(condyl\w*|condil\w*|kondyl\w*|kondil\w*|plateau|platillo|plato|platoa?|meseta|compart\w*|kompart\w*|"
                  r"femorotib\w*|tibiofem\w*|femoro-?tibia\w*|tibia\w*|femora\w*|femur\w*|joint (line|space)|"
                  r"interlinea|gelenkspalt|gewrichtsspleet|zglob\w*|weight-?bearing|zona de carga|tragzone|hauptbelastungszone|κονδυλ\w*|διαμερισμ\w*|κνημιαι\w*|μηριαι\w*|"
                  r"кондил\w*|тибиал\w*|феморал\w*|компартмент\w*|тибия\w*|фемур\w*)")
RE_MEDIAL_FT = re.compile(r"\b((postero|antero|infero|supero)?-?medial\w*|intern[oae]s?|innen\w*|mediaa?l\w*|medijal\w*|medyal\w*)\b.{0,25}?\b" + RE_ANATOMIA_FT +
                          r"|\b" + RE_ANATOMIA_FT + r".{0,25}?\b(medial\w*|intern[oae]s?|mediaa?l\w*|medijal\w*|medyal\w*)\b"
                          r"|\b(mfc|mtp)\b|varus|variser\w*|medialseitig|\bεσω (\w+ ){0,2}(κονδυλ|διαμερισμ|κνημιαι|μηριαι)\w*|медиал\w* (\w+ )?(кондил|компартмент|гонартроз|тибиал|феморал|страна)\w*")
RE_LATERAL_FT = re.compile(r"\b((postero|antero|infero|supero)?-?lateral\w*|extern[oae]s?|aussen\w*|lateraa?l\w*|lateraln\w*)\b.{0,25}?\b" + RE_ANATOMIA_FT +
                           r"|\b" + RE_ANATOMIA_FT + r".{0,25}?\b(lateral\w*|extern[oae]s?|lateraa?l\w*|lateraln\w*)\b"
                           r"|\b(lfc|ltp)\b|valgus|valgiser\w*|lateralseitig|\bεξω (\w+ ){0,2}(κονδυλ|διαμερισμ|κνημιαι|μηριαι)\w*|латерал\w* (\w+ )?(кондил|компартмент|гонартроз|тибиал|феморал|страна)\w*")
RE_OA_LADO_DIRECTO_MED = re.compile(r"\b(medial|interna?)\s+(oa|osteoarthr\w*|artros\w*)|\b(oa|artrosis|gonartrosis|arthrose|gonarthrose)\s+(medial|interna?)\b")
RE_OA_LADO_DIRECTO_LAT = re.compile(r"\b(lateral|externa?)\s+(oa|osteoarthr\w*|artros\w*)|\b(oa|artrosis|gonartrosis|arthrose|gonarthrose)\s+(lateral|externa?)\b")
RE_TRICOMPARTIMENTAL = re.compile(r"tri-?(c|k)ompart\w*|three compartments|todos los compartimentos|tous les compartiments|"
                                  r"alle(n)? (drei )?kompartiment\w*|sva tri|pan-?compart\w*|τρι\w* διαμερισμ\w*|трите компартмент\w*")
RE_FT_SIN_LADO = re.compile(r"femorotib\w*|tibiofem\w*|femoro-?tibia\w*|bicompart\w*|\bft\b|obostrano|фемура и тибията|μηριαι\w* και κνημιαι\w*|femur(s)? (and|und|y) tibia")


def compartimentos_artrosis(clausula: str, contexto: str):
    """Devuelve el conjunto de compartimentos ('medial','lateral','pf') citados en una cláusula de artrosis."""
    if RE_TRICOMPARTIMENTAL.search(clausula):
        return {"medial", "lateral", "pf"}
    limpio = RE_NO_PF.sub(" xx ", clausula)
    limpio = re.sub(r"(proximal )?tibio-?fibular\w*( joint)?|peroneotibial\w*|tibioperone\w*", " xx ", limpio)
    salida = set()
    if RE_OA_LADO_DIRECTO_MED.search(limpio):
        salida.add("medial")
    if RE_OA_LADO_DIRECTO_LAT.search(limpio):
        salida.add("lateral")
    if RE_PF.search(limpio):
        salida.add("pf")
    # para medial/lateral femorotibial se retiran las menciones PF ("faceta medial de la rótula")
    sin_pf = re.sub(r"(medial\w*|lateral\w*|intern\w*|extern\w*)\s+(patell\w*|patel\w*|rotul\w*|troch\w*|trocl\w*|facet\w*)"
                    r"|(patell\w*|patel\w*|rotul\w*|troch\w*|trocl\w*|facet\w*|faceta)\s+((?!femorotib|tibiofem|condil|condyl|kondyl|plateau|platillo|compart|kompart)\w+\s+){0,3}(medial\w*|lateral\w*|intern\w*|extern\w*)"
                    r"(?!\s+(\w+\s+)?(compart\w*|kompart\w*|femorotib\w*|tibiofem\w*|condyl\w*|condil\w*|plateau|platillo|tibia\w*|femor\w*|weight))",
                    " xx ", limpio)
    if RE_MEDIAL_FT.search(sin_pf):
        salida.add("medial")
    if RE_LATERAL_FT.search(sin_pf):
        salida.add("lateral")
    if not ({"medial", "lateral"} & salida) and RE_FT_SIN_LADO.search(sin_pf):
        salida |= {"medial", "lateral"}
    if not salida and contexto:
        salida.add(contexto)
    return salida


# ----------------------------------------------------------------------------
# DERRAME, SINOVITIS, BAKER, CONTUSIÓN, FRACTURA
# ----------------------------------------------------------------------------
RE_DERRAME = re.compile(
    r"effusion\w*|derrame\w*|epanchement\w*|erguss\w*|hydrops|izljev\w*|efuzij\w*|effuzyon\w*|efuzyon\w*|"
    r"h(a)?emarthros\w*|hemartros\w*|hamarthros\w*|lipoh(a)?emarthros\w*|lipohemartros\w*|effusie|"
    r"efusion|(\w+ )?ποσοτητα (\w+ )?(ενδαρθρικ\w* )?υγρου|υγρου εντος (της |τη )?αρθρωσ\w*|"
    r"kolicin\w* (\w+ )?tekucin\w* (u|uz) (\w+ ){0,3}(zglob\w*|suprapatel\w*|koljen\w*|recesus\w*)|"
    r"υδραρθρ\w*|αιμαρθρ\w*|(ενδαρθρικ\w* )?συλλογ\w* υγρου|συλλογ\w* υγρου|ενδαρθρικ\w* συλλογ\w*|"
    r"(ставен |ставния )?излив\w*|хемартроз\w*|течност в супрапателар\w*|"
    r"(increased|excess\w*|large amount of|moderate amount of|abundant\w*) (joint |intra-?articular )?fluid|"
    r"(aumento|incremento) (de|del) liquido (articular|intra-?articular)|liquido (intra-?)?articular aumentado|"
    r"(vermehrt\w*|akzentuiert\w*|deutlich\w*) (gelenk)?flussigkeit|gelenkflussigkeit (vermehrt|akzentuiert)|"
    r"(toegenomen|veel|verhoogd\w*) gewrichtsvocht|gewrichtsvocht (toegenomen|verhoogd)|"
    r"sivi\w* (miktari )?([\w-]+ )?([\w-]+ )?art(mis|is)\w*|"
    r"(distend\w*|distension|distensie|opzetting|distenzij\w*) (of |de |van |du |)?(the |la |el |le |)?"
    r"(suprapatel\w*|recess\w*|receso\w*|bursa suprapatel\w*)|suprapatel\w* (recess\w*|bursa|receso) (is )?distend\w*")
RE_FISIOLOGICO = re.compile(r"physiolog\w*|fisiolog\w*|fysiolog\w*|physiologi\w*|fizyolojik|fiziolosk\w*|φυσιολογικ\w*|физиологич\w*")

RE_SINOVITIS = re.compile(
    r"synovit\w*|sinovit\w*|synovialit\w*|sinovijalit\w*|synoviitis|"
    r"synovial (thickening|hypertroph\w*|prolifer\w*|enhancement|inflammation)|"
    r"(thicken\w*|hypertroph\w*|prolifer\w*) (of )?(the )?synovi\w*|"
    r"(engrosamiento|hipertrofia|proliferacion|realce) (de la |del )?(membrana )?sinovial|"
    r"epaississement synovial|hypertrophie synoviale|synovia(lis)?verdickung|verdickung der synovia|"
    r"synovial\w* verdik\w*|verdikking (van (het|de) )?synovi\w*|zadeblj\w* sinovij\w*|sinovij\w* zadeblj\w*|"
    r"verdickt\w* synovia|synovia (\w+ )?verdickt|synovial\w* proliferation\w*|hipertrofij\w* sinovij\w*|"
    r"sinovij\w* (\w+ )?(hipertrof|zadeblj)\w*|(παχυνση|υπερπλασια|υπερτροφια) (του )?(αρθρικου )?υμενα|"
    r"sinoviyal (kalinlas|proliferasyon)\w*|pvns|villonodul\w*|hoffitis|hoffit\w*|υμενιτιδ\w*|συνοβιτιδ\w*|синовит\w*")
RE_HOFFA = re.compile(r"hoffa|infrapatellar fat pad|grasa infrarotulian\w*|almohadilla grasa|graisse de hoffa|"
                      r"infrapatellar\w* fett|hoffa\w* fett|vetlichaam van hoffa|corps adipeux|infrapatellar yag yastik\w*|υποεπιγονατιδ\w* λιπωδ\w*")
RE_HOFFA_ANORMAL = re.compile(r"edema|oedem\w*|odem\w*|impinge\w*|pinzamiento|inflam\w*|syndrom\w*|sindrom\w*|"
                              r"stranding|maladie|conflit|irritat\w*|reticula\w*|enfermedad|reizung|entzund\w*|"
                              r"aumento de (la )?senal|increased (t2 )?signal|hyperintens\w*|signalanhebung|"
                              r"signal (alteration|abnormal)|alteracion\w* de (la )?senal|ozem|edem\w*|οιδημ\w*|προσκρουσ\w*|едем\w*")
RE_HOFFA_NORMAL = re.compile(r"normal|regelrecht\w*|unauffallig\w*|sin alteraciones|preserved|intact\w*|conservad\w*|"
                             r"uredan|urednog|sans anomalie|zonder bijzonder\w*")

RE_BAKER = re.compile(
    r"baker\w*|popliteal (cyst|fossa cyst)|poplitea?l\w* cyst\w*|quiste\w* popliteo\w*|kyste\w* poplite\w*|"
    r"poplitealzyste|popliteazyste|popliteacyste|popliteale cyste|poplitealn\w* cist\w*|popliteal kist\w*|"
    r"cisto popliteo|ιγνυακ\w* κυστ\w*|бе[йи]кер\w*|бекер\w*|поплитеалн\w* кист\w*|semimembranos\w*.{0,20}gastrocnem\w*.{0,15}(bursa|cyst|quiste|bursitis|kist|zyste|cyste)|"
    r"gastrocnem\w*.{0,20}semimembranos\w*.{0,15}(bursa|cyst|quiste|bursitis|kist|zyste|cyste)")

RE_MUSCULO_CERCA = re.compile(r"(muscle|muscul\w*|musculo|muskel\w*|spier\w*|misic\w*|\bkas\w*|subcutane\w*|"
                              r"soft tissue|partes blandas|tejido celular|weichteil\w*|weke delen|potkoz\w*|cilt alti|μυ(ος|ων|ικ|ες)\w*|мек\w* тъкан\w*|мускул\w*)")
RE_CONTUSION = re.compile(
    r"contusi\w*|bruise\w*|bone bruising|kontusion\w*|knochenprellung\w*|prellung\w*|kneuzing\w*|contusie\w*|"
    r"bone bruit\w*|(edema oseo|edema de medula osea).{0,120}fractura\w* (trabecular|subcortical)\w*|trabekularn\w* impakcij\w*|"
    r"посттравматич\w* костномоз\w* едем\w*|kontuzij\w*|postkontuzij\w*|kontuzyon\w*|contusao|μωλωπ\w*|θλασ\w*|контузи\w*|"
    r"micro-?fractur\w*|mikrofraktur\w*|microfracturas? trabecular\w*|trabecular (micro-?)?fractur\w*|"
    r"(bone marrow|osseous|bone|marrow) (edema|oedema).{0,60}(traum\w*|impact\w*|pivot\w*|kissing|post-?traum\w*)|"
    r"(edema|edema oseo|edema de medula osea).{0,40}(traumat\w*|contus\w*|impacto)|"
    r"(oedeme osseux|oedeme medullaire).{0,40}(traumat\w*|contus\w*)|kissing (contusion|lesion|bone)|"
    r"(posttraumat\w*|traumat\w*) (knochenmark\w*|bone marrow|botoedeem|edema)|"
    r"botoedeem.{0,40}(traum\w*|contus\w*|pivot\w*)")

RE_FRACTURA = re.compile(
    r"fractur\w*|fraktur\w*|fractuur\w*|fratura\w*|kiri[gk]\w*|avulsion of the (tibial spine|tibial eminence|fibular head|fibular)\w*|prijelom\w*|segond|καταγμ\w*|фрактур\w*|счупван\w*|"
    r"avulsion fracture|avulsion fragment|fragmento (de )?avulsion|avulsi\w* (osea|ossea|osseous|bony)|"
    r"arrancamiento oseo|arrachement osseux|avulzij\w* (fragment|prijelom)\w*|knocherne\w* (ausriss|avulsion)|"
    r"knocherner ausriss|benige avulsie|osteochondral\w* (fragment|fracture) desplazad\w*|"
    r"impaction\w* (injury|deformity)|insufficiency fracture|sifk|stress reaction|"
    r"fracture\w*|frakturnachweis|frakturlinie")
RE_NO_FRACTURA = re.compile(r"micro-?fractur\w*|mikrofraktur\w*|microfracturas?|trabecular (micro-?)?fractur\w*|"
                            r"microfracture (surgery|procedure|technique)")


# ----------------------------------------------------------------------------
# CONTEXTO DE SECCIÓN (informes estructurados "MEDIAL COMPARTMENT:")
# ----------------------------------------------------------------------------
RE_SECCION_MEDIAL = re.compile(r"^(.{0,25}εσω διαμερισμ|(in |at |from )?(the )?(medial (compartment|knee)|(στο )?εσω διαμερισμ\w*|compartimento (medial|interno)|compartiment (medial|interne)|"
                               r"medial\w* kompartiment|mediaal compartiment|medijaln\w* kompartment\w*|medial kompartman))")
RE_SECCION_LATERAL = re.compile(r"^(.{0,25}εξω διαμερισμ|(in |at |from )?(the )?(lateral (compartment|knee)|(απο τον ελεγχο του )?εξω διαμερισμ\w*|compartimento (lateral|externo)|compartiment (lateral|externe)|"
                                r"lateral\w* kompartiment|lateraal compartiment|lateraln\w* kompartment\w*|lateral kompartman))")
RE_SECCION_PF = re.compile(r"^(.{0,25}προσθι\w* διαμερισμ|(in |at |from )?(the )?(patellofemoral|patelofemoral|femoropatell\w*|compartimento (patelofemoral|femoropatelar)|"
                           r"compartiment femoro-?patell\w*|femoropatellar\w*|patellofemoraal|femoropatelarn\w*|extensor))")
RE_SECCION_RESET = re.compile(r"^(in |at )?(the )?(intercondylar|cruciate|collateral|extensor|ligament|osseous|bone marrow|joint space|fluid)")
RE_SECCION_OTRA = re.compile(r"^[^\W\d][\w /()&,-]{2,60}:")


def actualizar_contexto(clausula: str, contexto: str) -> str:
    """Si la cláusula empieza con un encabezado de compartimento, cambia el contexto."""
    if RE_SECCION_MEDIAL.match(clausula):
        return "medial"
    if RE_SECCION_LATERAL.match(clausula):
        return "lateral"
    if RE_SECCION_PF.match(clausula):
        return "pf"
    if RE_SECCION_RESET.match(clausula):
        return ""
    if RE_SECCION_OTRA.match(clausula):
        cabecera = clausula.split(":")[0]
        if not re.search(r"menisc|cartilag|knorpel|kraakbeen", cabecera):
            return ""
    return contexto


# ----------------------------------------------------------------------------
# DETECTORES POR CLÁUSULA
# ----------------------------------------------------------------------------
def hay_positivo(rx, clausula, filtro=None) -> bool:
    """True si algún match de rx no está negado (y pasa el filtro opcional)."""
    for m in rx.finditer(clausula):
        if filtro and not filtro(clausula, m):
            continue
        if not esta_negado(clausula, m.start(), m.end()):
            return True
    return False


def detectar_lesiones_estructurales(clausula, contexto, resultado, ultimo_lado="", informe_grado_bajo_menisco=False):
    """Atribuye cada término de lesión a LCA, LCM o menisco medial/lateral."""
    for m in RE_LESION.finditer(clausula):
        if esta_negado(clausula, m.start(), m.end()):
            continue
        entidad, m_ent = entidad_mas_cercana(clausula, m.start(), m.end())
        if entidad is None:
            continue
        # contexto local (lesión + entidad) para leer el grado
        a, b = sorted([m.start(), m.end(), m_ent.start(), m_ent.end()])[::3]
        local = clausula[max(0, a - 40):b + 60]
        if entidad in ("ACL", "MCL"):
            if re.search(r"^(laxity|lax)$", clausula[m.start():m.end()]):
                continue
            if EXCLUIR_LIGAMENTO_GRADO_1 and RE_GRADO_1.search(local) and not RE_GRADO_ALTO.search(local):
                continue
            if re.search(r"mucoid|mucoide|mukoid|ganglion|reconstruc\w*|plastik|plastie|graft|injert\w*",
                         local) and not re.search(r"tear|rotur|ruptur|riss|scheur|dechir", clausula[m.start():m.end()]):
                continue
            resultado[entidad] = 1
        elif entidad == "MENISCO":
            # "lesión grado 2" o señal intrasustancia sin término de rotura franca -> no es rotura
            termino = clausula[m.start():m.end()]
            if re.match(r"distenzij|distension|elongac|elongat", termino):
                continue
            if RE_NO_ALCANZA.search(clausula) and not RE_LLEGA_SUPERFICIE.search(clausula):
                continue
            if not RE_TERMINO_ROTURA_FRANCA.search(termino):
                if informe_grado_bajo_menisco and not RE_GRADO_ALTO_MENISCO.search(clausula):
                    continue
                if RE_GRADO_BAJO_MENISCO.search(local) and not RE_LLEGA_SUPERFICIE.search(clausula):
                    continue
                if re.search(r"(кистичн|cystic|quistic|kystique|cisti|kisti|zystisch)\w*\s*$", clausula[max(0, m.start() - 20):m.start()]):
                    continue
            if re.search(r"menisc\w*ectom|meniskusteilresektion|resectie|resekcij", local) and \
               not RE_TERMINO_ROTURA_FRANCA.search(termino):
                continue
            lados = lado_de_menisco(clausula, m_ent.start(), m_ent.end(), contexto, ultimo_lado)
            if "medial" in lados:
                resultado["Medial Meniscus"] = 1
            if "lateral" in lados:
                resultado["Lateral Meniscus"] = 1


RE_LCA_NO_VISIBLE = re.compile(
    r"(" + ENTIDADES["ACL"] + r")[^,;]{0,40}(not (visualized|identified|seen)|no se (identifica|visualiza)|ne prikazuje|se ne prikazuje|"
    r"nicht (abgrenzbar|darstellbar|nachweisbar)|niet (zichtbaar|herkenbaar)|secilememis\w*|izlenmemekte\w*|не се проследява|absent)|"
    r"(δεν αναγνωριζεται|non visualise|no se identifica|nao se identifica|not visualized)[^,;]{0,20}(" + ENTIDADES["ACL"] + r")")
RE_LCM_EDEMA = re.compile(
    r"(" + ENTIDADES["MCL"] + r")[^.;]{0,50}(edema\w*|oedem\w*|odem\w*|οιδημ\w*|едем\w*|high si|hyperintens\w*|increased (t2 )?signal|"
    r"aumento de (la )?senal|signalanheb\w*|distendiran\w*|zadebljan\w* (\w+ ){0,3}signal)|"
    r"(edema\w*|oedem\w*|odem\w*|οιδημ\w*|periligament\w*|aumento de (la )?senal|increased (t2 )?signal|hyperintens\w*)[^.;]{0,40}(" + ENTIDADES["MCL"] + r")")


def detectar_ligamento_extra(clausula, resultado):
    """v1.1: LCA no visualizado = rotura; edema/señal alta del LCM = esguince (lesión)."""
    if RE_LCA_NO_VISIBLE.search(clausula):
        resultado["ACL"] = 1
    for m in (RE_LCM_EDEMA.finditer(clausula) if INFERIR_ESGUINCE_LCM_POR_EDEMA else []):
        if not esta_negado(clausula, m.start(), m.start() + 3) and not re.search(r"\b(no|without|sin|kein\w*|geen|bez|δεν|няма)\b", m.group()):
            resultado["MCL"] = 1


def detectar_artrosis(clausula, contexto, resultado):
    for m in RE_ARTROSIS.finditer(clausula):
        if esta_negado(clausula, m.start(), m.end()):
            continue
        local = clausula[max(0, m.start() - 30):m.end() + 40]
        if RE_GRADO_1_CARTILAGO.search(local) and not re.search(r"\b(2|3|4|ii|iii|iv)\b", local):
            continue
        for comp in compartimentos_artrosis(clausula, contexto):
            resultado[{"medial": "Medial OA", "lateral": "Lateral OA", "pf": "PF OA"}[comp]] = 1
        return


RE_MINIMO = re.compile(r"\b(trace|tiny|minimal|minimo|minima|minime|minimaal|minimaler|minimale|minimalni|minimalna|minimal\w*|"
                       r"μικρη ποσοτητα|минимал\w*|eser|minuscule|winzig\w*|petite? quantite)\b")


RE_TRAZA = re.compile(r"\b(trace|traces|spur|spoor|huella|indicios|eser)\b")


def filtro_derrame(clausula, m):
    entorno = clausula[max(0, m.start() - 30):m.end() + 30]
    previo = clausula[max(0, m.start() - 25):m.start()]
    posterior = clausula[m.end():m.end() + 30]
    if RE_FISIOLOGICO.search(entorno) or RE_TRAZA.search(previo + " " + clausula[m.end():m.end() + 12]):
        return False
    # líquido en una bursa o alrededor de un ligamento no es derrame articular
    if re.search(r"^\W*(\w+\W+){0,2}(u|in|en|dans|im|in de) (the |la |el |le |der |de )?(burz\w*|bursa\w*|bourse)", posterior) or \
       re.search(r"(ligament\w*|ligaman\w*|bag\w*)\s+(cevresinde|etrafinda)", clausula[max(0, m.start() - 45):m.start()]):
        return False
    return CONTAR_DERRAME_MINIMO or not RE_MINIMO.search(previo)


RE_SIN_DERRAME_SIGNIFICATIVO = re.compile(
    r"no (significant|relevant|appreciable|substantial) (\w+ )?(knee |joint )?effusion|without (significant|relevant) (joint )?effusion|"
    r"sin (evidente |significativo )?derrame (articular )?significativo|no hay derrame (articular )?significativo|"
    r"kein\w* (wesentlich\w*|relevant\w*|signifikant\w*) (\w+ )?(gelenk)?erguss|pas d'epanchement (articulaire )?significatif|"
    r"geen (significant|relevant)\w* (gewrichts)?(vocht|effusie|hydrops)|ne nalazi se signifikantn\w* izljev\w*|bez signifikantn\w* izljev\w*")


def filtro_baker(clausula, m):
    return CONTAR_DERRAME_MINIMO or not RE_MINIMO.search(clausula[max(0, m.start() - 20):m.start()])


RE_HUESO_PALABRA = re.compile(r"bone|osse|oseo|osea|osseu|knoch|bot|kost|kemik|οστ|кост|marrow|medula|condyl|condil|kondil|kondyl|"
                              r"plateau|platillo|plato|tibia|femur|femor|patella|rotula|fibula|epifis|metafis|κονδυλ|κνημ|кондил|тибия")


def filtro_contusion(clausula, m):
    if RE_MUSCULO_CERCA.search(clausula[max(0, m.start() - 30):m.start()]):
        return False
    # contusión de partes blandas / ligamento sin mención ósea en la cláusula -> no es contusión ósea
    termino = clausula[m.start():m.end()]
    # la contusión debe ser ósea: o el término ya lo es, o la cláusula nombra hueso
    if not re.search(r"bruise|bruit|prellung|kneuzing|micro-?fract|mikrofrakt|trabecul|trabekul|μωλωπ|osea|oseo|ossea|osseuse|knochen|kost|кост|kemik", termino) \
       and not RE_HUESO_PALABRA.search(clausula):
        return False
    if re.search(r"micro-?fractur|mikrofraktur", termino) and \
       re.search(r"(microfracture|microfractura|mikrofraktur)\w* (surgery|procedure|treatment|for|on|technique|cirug|operation)|(s/p|post|status post|after|tras|nach) (micro|mikro)", clausula):
        return False
    if re.search(r"tissus mous|parties molles|soft tissue|partes blandas|tejidos? blandos|weichteil|hoffa|ligament|ligamento|lig\b|menisc|menisk", clausula) \
       and not RE_HUESO_PALABRA.search(clausula):
        return False
    return True


def filtro_fractura(clausula, m):
    return not RE_NO_FRACTURA.search(clausula[max(0, m.start() - 12):m.end() + 12])


RE_HOFFA_INFLAMATORIO = re.compile(r"hoffitis|hoffit\w*|inflam\w*|inflammatoir\w*|entzund\w*|enfermedad de hoffa|hoffa'?s disease|maladie de hoffa")


def detectar_hoffa(clausula):
    """Pinzamiento / edema de la grasa de Hoffa se considera proceso sinovial-inflamatorio."""
    for m in RE_HOFFA.finditer(clausula):
        local = clausula[max(0, m.start() - 60):m.end() + 60]
        anormal = RE_HOFFA_ANORMAL.search(local) if HOFFA_COMO_SINOVITIS else RE_HOFFA_INFLAMATORIO.search(local)
        if anormal and not RE_HOFFA_NORMAL.search(local) \
           and not esta_negado(clausula, m.start(), m.end()):
            return True
    return False


# ----------------------------------------------------------------------------
# ORQUESTACIÓN
# ----------------------------------------------------------------------------
def etiquetar_informe(texto: str) -> dict:
    """Devuelve un dict {etiqueta: 0/1} para un informe."""
    resultado = {e: 0 for e in ETIQUETAS}
    if not isinstance(texto, str) or not texto.strip():
        return resultado
    texto = normalizar(texto)
    texto = RE_INCERTIDUMBRE.sub(" posible ", texto)
    # "quiste de Baker no complicado / non ruptured" no es una negación del quiste
    texto = re.sub(r"\b(non|not|niet met|nicht|no|sin|sans)[- ](ruptured|complicated|rompid\w*|complicad\w*|rompu\w*|"
                   r"kompliziert\w*|ruptuur verwikkeld\w*|rupturiert\w*|komplicirana)", " simple ", texto)
    texto = texto.replace("βaker", "baker")
    texto = quitar_antecedentes(texto)
    contexto = ""
    ultimo_lado = ""
    # informes que gradúan la lesión meniscal como 1/2 (escala búlgara 2a/2b, "grade II") sin grado 3
    informe_grado_bajo = bool(RE_GRADO_BAJO_MENISCO.search(texto)) and not RE_GRADO_ALTO_MENISCO.search(texto)
    derrame_leve = False
    for clausula in partir_clausulas(texto):
        if RE_ENCABEZADO_CLINICO.match(clausula):
            continue
        contexto = actualizar_contexto(clausula, contexto)
        ultimo_lado = lado_citado_en(clausula, ultimo_lado)
        detectar_lesiones_estructurales(clausula, contexto, resultado, ultimo_lado, informe_grado_bajo)
        # grado III de degeneración meniscal (escala de Stoller) = rotura
        for m in RE_DEGENERACION_G3.finditer(clausula):
            if not esta_negado(clausula, m.start(), m.end()):
                lado = ultimo_lado or contexto
                lados = {lado} if lado in ("medial", "lateral") else set()
                if RE_MENISCO_CON_LADO_MED.search(clausula): lados.add("medial")
                if RE_MENISCO_CON_LADO_LAT.search(clausula) and not RE_MENISCO_CON_LADO_MED.search(clausula): lados = {"lateral"}
                for l in lados:
                    resultado["Medial Meniscus" if l == "medial" else "Lateral Meniscus"] = 1
        detectar_ligamento_extra(clausula, resultado)
        detectar_artrosis(clausula, contexto, resultado)
        for m in RE_DERRAME.finditer(clausula):
            if filtro_derrame(clausula, m) and not esta_negado(clausula, m.start(), m.end()):
                if RE_MINIMO.search(clausula[max(0, m.start() - 30):m.start()]) or \
                   re.search(r"\b(etwas|diskret\w*|gering\w*|allenfalls)\b", clausula[max(0, m.start() - 30):m.start()]):
                    derrame_leve = True
                else:
                    resultado["Effusion"] = 1
        clausula_sin_teno = re.sub(r"teno-?s[iy]novit\w*|tendosynovit\w*|tenosinovijalit\w*", " ", clausula)
        if hay_positivo(RE_SINOVITIS, clausula_sin_teno) or detectar_hoffa(clausula_sin_teno):
            resultado["Synovitis"] = 1
        if hay_positivo(RE_BAKER, clausula, filtro_baker):
            resultado["Baker's"] = 1
        if hay_positivo(RE_CONTUSION, clausula, filtro_contusion):
            resultado["Contusion"] = 1
        if hay_positivo(RE_FRACTURA, clausula, filtro_fractura):
            resultado["Fracture"] = 1
    # derrame mínimo cuenta, salvo que el mismo informe diga "sin derrame significativo"
    if derrame_leve and not resultado["Effusion"] and not RE_SIN_DERRAME_SIGNIFICATIVO.search(texto):
        resultado["Effusion"] = 1
    return resultado


def etiquetar_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Etiqueta todas las filas; conserva las ya etiquetadas si así se configura."""
    salida = df.copy()
    predicciones = pd.DataFrame([etiquetar_informe(t) for t in df[COLUMNA_TEXTO]], index=df.index)
    for e in ETIQUETAS:
        if RESPETAR_ETIQUETAS_EXISTENTES and e in df.columns:
            salida[e] = df[e].fillna(predicciones[e])
        else:
            salida[e] = predicciones[e]
        salida[e] = salida[e].astype(int)
    return salida


if __name__ == "__main__":
    entrada = sys.argv[1] if len(sys.argv) > 1 else RUTA_ENTRADA
    salida = sys.argv[2] if len(sys.argv) > 2 else RUTA_SALIDA
    datos = pd.read_csv(entrada)
    resultado = etiquetar_dataframe(datos)
    resultado.to_csv(salida, index=False)
    print(f"Filas etiquetadas: {len(resultado)}")
    print(resultado[ETIQUETAS].sum().to_string())
