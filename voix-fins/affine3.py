import numpy as np
def affine3(a, fin):
    """Fin du dernier mot : prolongation sur l'énergie (trames 10 ms), puis rattachement des pics brefs
    (relâchement d'occlusive k/t/p, ≤ 0,15 s) qui commencent dans les 0,25 s suivantes ; arrêt au premier
    segment long (murmure)."""
    fr = lambda i: 20*np.log10(np.sqrt(np.mean(a[i:i+160]**2)) + 1e-9)
    f = int(fin*16000)
    voix = [x for x in (fr(i) for i in range(0, max(f-160, 0), 160)) if x > -40]
    if not voix: return fin
    niveau = float(np.median(voix)); seuil = niveau - 16
    last, trou, i = f, 0, f
    while i + 160 <= len(a) and i < f + int(0.35*16000):
        if fr(i) > seuil: last, trou = i + 160, 0
        else:
            trou += 1
            if trou >= 5: break
        i += 160
    s2 = niveau - 22
    fen = last + int(0.25*16000)
    e = [fr(j) > s2 for j in range(0, len(a) - 160, 160)]
    k = last // 160
    while k < len(e) and k*160 < fen:
        if not e[k]: k += 1; continue
        deb, trou, j = k, 0, k
        while j < len(e):
            if e[j]: fin_seg, trou = j + 1, 0
            else:
                trou += 1
                if trou >= 3: break
            j += 1
        if (fin_seg - deb) * 160 > int(0.15*16000):
            break  # segment long : murmure
        last = max(last, fin_seg*160); k = fin_seg + 1
    return last / 16000
