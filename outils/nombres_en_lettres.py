"""Normalisation des nombres : texte donné au modèle et comparaison Whisper."""
import re, unicodedata, difflib
from num2words import num2words

UNITES = {'fr': {'km': 'kilomètres', 'cm': 'centimètres', 'mm': 'millimètres', 'dm': 'décimètres',
                 'm': 'mètres', 'kg': 'kilogrammes', 'g': 'grammes', 'km/h': 'kilomètres heure'},
          'en': {'km': 'kilometres', 'cm': 'centimetres', 'mm': 'millimetres', 'm': 'metres', 'kg': 'kilograms'}}


def nombre(s, lang):
    """'3,14' -> 'trois virgule quatorze' ; '21' -> 'vingt et un'."""
    s = s.replace('.', ',') if lang == 'fr' else s
    if ',' in s and lang == 'fr':
        a, b = s.split(',', 1)
        dec = ' '.join(num2words(int(b), lang='fr').split()) if not b.startswith('0') else \
            ' '.join(num2words(int(c), lang='fr') for c in b)
        return num2words(int(a), lang='fr') + ' virgule ' + dec
    if '.' in s and lang == 'en':
        a, b = s.split('.', 1)
        return num2words(int(a), lang='en') + ' point ' + ' '.join(num2words(int(c), lang='en') for c in b)
    return num2words(int(s), lang=lang)


NUM = r'\d+(?:[,.]\d+)?'


OPERANDE = r"(?:\d+(?:,\d+)?[a-z]{0,2}|\b[a-zA-Z]|carré|cube|longueur|largeur|hauteur|\))"


def plusse(t):
    """« plus » d'opération -> « plusse » (la voix disait « plu ») ; « le plus grand » reste inchangé."""
    t = re.sub(r'(' + OPERANDE + r') plus\b(?!-)(?! de\b)(?! d.)', r'\1 plusse', t)
    t = re.sub(r'(' + OPERANDE + r') plus\b(?!-)(?! de\b)(?! d.)', r'\1 plusse', t)  # « 2 plus 3 plus 4 »
    t = re.sub(r'\b([dD]e) plus(?=\s*[:.,;!?]|\s*$)', r'\1 plusse', t)
    return t


def texte_modele(t, lang='fr', modele=False):
    """Texte donné à Chatterbox (modele=True) : tous les nombres en toutes lettres, « plus » d'opération -> « plusse ».
    Avec modele=False, même conversion sans « plusse » (sert à la comparaison avec Whisper)."""
    if modele and lang == 'fr':
        t = plusse(t)
    t = re.sub(r'\s?%', ' pour cent' if lang == 'fr' else ' percent', t)
    t = t.replace('≈', ' environ ' if lang == 'fr' else ' about ')
    t = re.sub(r'xⁿ⁻¹', 'x puissance n moins un', t)
    t = re.sub(r'xⁿ', 'x puissance n', t)
    t = re.sub(r"(\w) ' ", r'\1 prime ', t)
    t = re.sub(r'\b14-18\b', 'quatorze-dix-huit', t)
    t = re.sub(r'\b(\d+)-(\d+)\b', r'\1 à \2' if lang == 'fr' else r'\1 to \2', t)
    t = re.sub(r'(^|[\s(:])-(?=\d)', r'\1moins ' if lang == 'fr' else r'\1minus ', t)
    t = re.sub(r'\b(\d+)(er|re|e|ème)\b', lambda m: num2words(int(m.group(1)), lang='fr', to='ordinal'), t)
    for u, w in sorted(UNITES[lang].items(), key=lambda kv: -len(kv[0])):
        t = re.sub(r'(?<=[\d\s])' + re.escape(u) + r'²', w + (' carrés' if lang == 'fr' else ' squared'), t)
        t = re.sub(r'(?<=[\d\s])' + re.escape(u) + r'³', w + (' cubes' if lang == 'fr' else ' cubed'), t)
        t = re.sub(r'(?<=\b1 )' + re.escape(u) + r'(?=[\s.,;:!?]|$)', w[:-1] if w.endswith('s') else w, t)
        if u == 'm':
            t = re.sub(r'(?<=\d )m(?=[\s.,;:!?]|$)|(?<=[\d\s])m(?= au (?:carré|cube))', w, t)
        else:
            t = re.sub(r'(?<=[\d\s])' + re.escape(u) + r'(?=[\s.,;:!?]|$)', w, t)
    t = re.sub(r'(\d)([a-zA-Z]+)\b', lambda m: m.group(1) + ' ' + ' '.join(m.group(2)), t)  # 2ab -> 2 a b
    t = re.sub(r'(\d),\s*(\d)', r'\1,\2', t) if False else t
    t = re.sub(NUM if lang == 'fr' else r'\d+(?:\.\d+)?', lambda m: nombre(m.group(), lang), t)
    return re.sub(r'\s+', ' ', t).strip()


def mots(t):
    t = unicodedata.normalize('NFC', t.lower())
    t = t.replace('’', "'").replace('œ', 'oe')
    t = re.sub(r"[^\w']+", ' ', t)
    t = t.replace("'", ' ')
    w = t.split()
    out = []
    for x in w:
        if x == 'à':
            x = 'a'
        if x in ('vingts', 'cents'):
            x = x[:-1]
        out.append(x)
    return out


def jetons(t, lang='fr'):
    """Liste de (mot, indice_du_nombre ou None)."""
    res = []
    pos = 0
    tm = t
    # on repère les nombres du texte d'origine, puis on les convertit un à un
    pat = re.compile(r'(?:(?<=^)|(?<=[\s(:]))-?\d+(?:[,.]\d+)?(?:-\d+)?(?:e\b|er\b)?')
    k = 0
    for m in re.finditer(r'-?\d+(?:,\d+)?(?:-\d+)?(?:er\b|e\b|[a-z]{1,2}\b)?(?![\d])', tm):
        avant = tm[pos:m.start()]
        res += [(w, None) for w in mots(texte_modele(avant, lang))]
        s = m.group()
        if s.startswith('-') and m.start() > 0 and not tm[m.start() - 1].isspace() and tm[m.start() - 1] not in '(:':
            res += [(w, None) for w in mots(texte_modele('-', lang))]
            s = s[1:]
        res += [(w, k) for w in mots(texte_modele(s, lang))]
        k += 1
        pos = m.end()
    res += [(w, None) for w in mots(texte_modele(tm[pos:], lang))]
    return res, k


def norm_trans(t, lang='fr', ref=''):
    # séparateurs de milliers (« 8,850 » / « 8 850 » pour 8850 dans le texte attendu)
    for n in re.findall(r'\d{4,}', ref):
        v = re.escape(n[:-3]) + r'[ ,.\u202f\u00a0]' + re.escape(n[-3:])
        t = re.sub(r'\b' + v + r'\b', n, t)
    # « cm2 », « cm3 », « m² » écrits par Whisper
    t = re.sub(r'\b(c?m|dm|km)2\b', r'\1 au carré', t)
    t = re.sub(r'\b(c?m|dm|km)3\b', r'\1 au cube', t)
    t = re.sub(r'\b(c?m|dm|km)²', r'\1 au carré', t)
    t = re.sub(r'\b(c?m|dm|km)³', r'\1 au cube', t)
    t = t.replace('²', ' au carré').replace('³', ' au cube').replace('°', ' degrés').replace('π', 'pi')
    if not re.search(r'\d-\d', ref):
        t = re.sub(r'(\d)\s?-\s?(?=\d)', r'\1 moins ', t)
    return mots(texte_modele(t, lang))


def controle(texte, trans, lang='fr'):
    """Renvoie (similarité, liste des indices de nombres faux, liste des nombres du texte)."""
    J, n = jetons(texte, lang)
    a = [w for w, _ in J]
    b = norm_trans(trans, lang, texte)
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    ok = [False] * len(a)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == 'equal':
            for i in range(i1, i2):
                ok[i] = True
    faux = set()
    for i, (w, k) in enumerate(J):
        if k is not None and not ok[i]:
            faux.add(k)
    # insertions de mots-nombres collées à un nombre (ex. attendu « vingt », entendu « vingt et un »)
    numw = set(mots(' '.join(num2words(i, lang=lang) for i in list(range(0, 101)) + [1000, 1000000]))) | {'et', 'and'}
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == 'insert' and any(x in numw and x not in ('et', 'and') for x in b[j1:j2]):
            for i in (i1 - 1, i1):
                if 0 <= i < len(J) and J[i][1] is not None:
                    faux.add(J[i][1])
    return sm.ratio(), sorted(faux), n


def similarite(texte, trans, lang):
    a = mots(texte_modele(texte, lang))
    b = norm_trans(trans, lang, texte)
    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()


if __name__ == '__main__':
    for t in ["La bonne réponse est : 21.", "Aire égale pi fois r au carré égale 3,14 fois 100 ≈ 314 cm au carré.",
              "2x égale -2 moins 10 égale -12, donc x égale -12 divisé par 2 égale -6.",
              "1914-1918 et 1939-1945 : les deux guerres mondiales.", "Les verbes du 2e groupe font « -issons ».",
              "Comment appelle-t-on la vie des soldats de 14-18 dans les tranchées?",
              "3x au carré ' égale 6x et 9x ' égale 9, donc 6x plus 9.", "Voilà d'où vient le 2ab!",
              "Le volume s'écrit en unités cubes : cm au cube, m au cube. Et 1 litre, c'est 1 dm au cube, soit 1000 cm au cube.",
              "La dérivée de xⁿ est n fois xⁿ⁻¹, donc 5x puissance 4."]:
        print(texte_modele(t))
        print('  ', jetons(t))
    print(controle("La bonne réponse est : 21.", "La bonne réponse est 20."))
    print(controle("La bonne réponse est : 20.", "La bonne réponse est 21."))
    print(controle("La bonne réponse est : 72.", "La bonne réponse est : 72."))
    print(controle("Aire égale pi fois r au carré égale 3,14 fois 100 ≈ 314 cm au carré.", "Aire égale pi fois R au carré égale 3,14 fois 100, environ 314 cm au carré."))
    print(controle("1914-1918 : Première Guerre mondiale", "1914 à 1918, Première Guerre mondiale"))
