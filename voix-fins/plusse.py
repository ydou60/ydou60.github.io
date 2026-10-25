"""Réenregistre les phrases de plusse1.json (« plus » d'opération écrit « plusse » pour le modèle).

usage : plusse.py liste.json sortie.jsonl
Réglages : outils/REGLAGES_CONTROLE.md (branche voix-controle). Sortie : <dépôt>/plusse/<file>.
"""
import json, os, re, subprocess, sys, tempfile, time
import numpy as np
D = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, D)
import torch, torchaudio
from nombres_en_lettres import texte_modele, controle, mots
from faster_whisper import WhisperModel
from affine3 import affine3
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

REPO = '/home/user/ydou60.github.io'
REF = os.path.join(D, 'ref_fr.wav')
TMP = os.path.join(D, 'tmp')
VOLUME = 'acompressor=threshold=0.06:ratio=3:attack=5:release=80:makeup=5,alimiter=limit=0.94:level=disabled'
torch.set_num_threads(4)
os.makedirs(TMP, exist_ok=True)

model = ChatterboxMultilingualTTS.from_pretrained(device='cpu')
model.prepare_conditionals(REF, exaggeration=0.6)
whisper = WhisperModel('small', device='cpu', compute_type='int8', cpu_threads=4)
whisper_m = WhisperModel('medium', device='cpu', compute_type='int8', cpu_threads=4)


def morceaux(t):
    if len(t) <= 150:
        return [t]
    out = []
    for p in [p for p in re.split(r'(?<=[.!?])\s+', t) if p.strip()]:
        if len(p) > 150:
            cur = ''
            for q in re.split(r'(?<=[:;,])\s+', p):
                if cur and len(cur) + len(q) + 1 > 150:
                    out.append(cur)
                    cur = q
                else:
                    cur = (cur + ' ' + q).strip()
            if cur:
                out.append(cur)
        else:
            out.append(p)
    return out


def pcm(p):
    raw = subprocess.check_output(['ffmpeg', '-v', 'error', '-i', p, '-f', 's16le', '-ac', '1', '-ar', '16000', '-'])
    return np.frombuffer(raw, np.int16).astype(np.float32) / 32768


def affine(a, fin):
    """Prolonge la fin Whisper tant que l'énergie (trames 10 ms) reste proche du niveau de parole (même règle que la tâche 1)."""
    fr = lambda i: 20 * np.log10(np.sqrt(np.mean(a[i:i + 160] ** 2)) + 1e-9)
    f = int(fin * 16000)
    voix = [x for x in (fr(i) for i in range(0, max(f - 160, 0), 160)) if x > -40]
    if not voix:
        return fin
    seuil = float(np.median(voix)) - 16
    last, trou, i = f, 0, f
    while i + 160 <= len(a) and i < f + int(0.35 * 16000):
        if fr(i) > seuil:
            last, trou = i + 160, 0
        else:
            trou += 1
            if trou >= 5:
                break
        i += 160
    return last / 16000


def generer(texte_m, lang, graine, mp3):
    parts = []
    for m in morceaux(texte_m):
        torch.manual_seed(graine)
        wav = model.generate(m, language_id=lang, cfg_weight=0.3, exaggeration=0.6, temperature=0.8,
                             repetition_penalty=2.0, min_p=0.05, top_p=1.0)
        if parts:
            parts.append(torch.zeros(1, int(0.25 * model.sr)))
        parts.append(wav)
    tmp = tempfile.mktemp(suffix='.wav', dir=TMP)
    torchaudio.save(tmp, torch.cat(parts, dim=1), model.sr)
    tmp2 = tmp[:-4] + '_t.wav'
    if lang == 'fr':
        subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', tmp, '-af', 'atempo=0.92', tmp2], check=True)
    else:
        os.rename(tmp, tmp2)
    a = pcm(tmp2)
    segs, _ = whisper.transcribe(a, language=lang, beam_size=5, condition_on_previous_text=False, word_timestamps=True)
    fins = [w.end for s in segs for w in (s.words or [])]
    af, coupe = VOLUME, None
    if fins:
        fin = affine3(a, max(fins))
        if len(a) / 16000 - fin > 0.3:
            coupe = fin + 0.15
            af = f'atrim=0:{coupe:.3f},afade=t=out:st={coupe - 0.08:.3f}:d=0.08,' + VOLUME
    os.makedirs(os.path.dirname(mp3), exist_ok=True)
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', tmp2, '-af', af, '-ar', '24000', '-ac', '1',
                    '-b:a', '56k', mp3], check=True)
    for f in (tmp, tmp2):
        if os.path.exists(f):
            os.unlink(f)
    return coupe


def transcrire(mp3, lang, w=None):
    segs, _ = (w or whisper).transcribe(pcm(mp3), language=lang, beam_size=5, condition_on_previous_text=False)
    return ' '.join(s.text.strip() for s in segs)


def seuil(texte):
    return 0.9 if len(mots(texte)) <= 6 else 0.8


def score(sim, faux):
    return (-len(faux), sim)


def fin_presente(texte, tr):
    """Le dernier mot attendu figure parmi les derniers mots entendus (phrase non coupée)."""
    a = mots(texte_modele(texte, 'fr'))
    b = mots(texte_modele(tr, 'fr'))
    return bool(a and b) and a[-1] in b[-3:]


entree, sortie = sys.argv[1], sys.argv[2]
items = json.load(open(entree))
fait = {json.loads(l)['file'] for l in open(sortie)} if os.path.exists(sortie) else set()
with open(sortie, 'a') as out:
    for x in items:
        if x['file'] in fait:
            continue
        t0 = time.time()
        tm = texte_modele(x['text'], x['lang'], modele=True)
        meilleur, essais = None, []
        for g in list(range(1, 9)) + list(range(9, 17)):
            if meilleur and meilleur['ok']:
                break
            if g == 9 and meilleur and not meilleur['faux'] and meilleur['sim'] >= seuil(x['text']) - 0.05:
                break  # graines de rattrapage seulement si les 8 premières sont nettement mauvaises
            mp3 = os.path.join(D, 'essais', x['file'][:-4] + f'_g{g}.mp3')
            coupe = generer(tm, x['lang'], g, mp3)
            tr = transcrire(mp3, x['lang'])
            sim, faux, _ = controle(x['text'], tr, x['lang'])
            if faux or sim < seuil(x['text']):
                tr2 = transcrire(mp3, x['lang'], whisper_m)
                sim2, faux2, _ = controle(x['text'], tr2, x['lang'])
                if score(sim2, faux2) > score(sim, faux):
                    tr, sim, faux = 'medium: ' + tr2, sim2, faux2
            fin_ok = fin_presente(x['text'], tr.replace('medium: ', ''))
            e = dict(mp3=mp3, g=g, trans=tr, sim=round(sim, 3), faux=faux, coupe=coupe, fin_ok=fin_ok,
                     ok=(not faux and sim >= seuil(x['text']) and fin_ok))
            essais.append(e)
            if meilleur is None or (e['ok'], score(sim, faux)) > (meilleur['ok'], score(meilleur['sim'], meilleur['faux'])):
                meilleur = e
        dst = os.path.join(REPO, 'plusse', x['file'])
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        subprocess.run(['cp', meilleur['mp3'], dst], check=True)
        r = dict(x, texte_modele=tm, graine=meilleur['g'], trans=meilleur['trans'], sim=meilleur['sim'],
                 faux=meilleur['faux'], ok=meilleur['ok'], coupe=meilleur['coupe'],
                 essais=[{k: v for k, v in e.items() if k != 'mp3'} for e in essais], duree=round(time.time() - t0))
        out.write(json.dumps(r, ensure_ascii=False) + '\n')
        out.flush()
        print(r['ok'], meilleur['g'], r['sim'], r['faux'], x['text'][:60], '|', r['trans'][:60], r['duree'], 's', flush=True)
