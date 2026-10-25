"""Régénère des phrases avec Chatterbox (réglages REGLAGES.md + CONSIGNES.md) et garde la meilleure graine.

usage : regen.py liste.json sortie.jsonl [graines]
Chaque élément : {code, lang, text, file, avant: {sim, faux}}  (avant = score du fichier d'origine)
Les mp3 acceptés sont écrits dans w/neuf/<file>.
"""
import json, os, re, subprocess, sys, tempfile, time
D = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, D)
import torch, torchaudio
from nombres_en_lettres import texte_modele, controle, mots
from faster_whisper import WhisperModel
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

W = os.path.join(D, 'w')
REF = os.path.join(W, 'ref_fr.wav')
VOLUME = 'acompressor=threshold=0.06:ratio=3:attack=5:release=80:makeup=5,alimiter=limit=0.94:level=disabled'
torch.set_num_threads(4)

entree, sortie = sys.argv[1], sys.argv[2]
graines = [int(g) for g in sys.argv[3].split(',')] if len(sys.argv) > 3 else list(range(1, 9))

model = ChatterboxMultilingualTTS.from_pretrained(device='cpu')
model.prepare_conditionals(REF, exaggeration=0.6)
whisper = WhisperModel('small', device='cpu', compute_type='int8', cpu_threads=4)
whisper_m = WhisperModel('medium', device='cpu', compute_type='int8', cpu_threads=4)


def morceaux(t):
    if len(t) <= 150:
        return [t]
    ph = [p for p in re.split(r'(?<=[.!?])\s+', t) if p.strip()]
    out = []
    for p in ph:
        if len(p) > 150:  # phrase encore trop longue : on coupe aux deux-points / points-virgules / virgules
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


def generer(texte_m, lang, graine, mp3):
    parts = []
    for i, m in enumerate(morceaux(texte_m)):
        torch.manual_seed(graine)
        wav = model.generate(m, language_id=lang, cfg_weight=0.3, exaggeration=0.6, temperature=0.8,
                             repetition_penalty=2.0, min_p=0.05, top_p=1.0)
        if parts:
            parts.append(torch.zeros(1, int(0.25 * model.sr)))
        parts.append(wav)
    wav = torch.cat(parts, dim=1)
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        tmp = f.name
    torchaudio.save(tmp, wav, model.sr)
    tmp2 = tmp[:-4] + '_t.wav'
    if lang == 'fr':
        subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', tmp, '-af', 'atempo=0.92', tmp2], check=True)
    else:
        os.rename(tmp, tmp2)
    # coupe 0,15 s après la fin du dernier mot (murmure de fin), fondu de sortie de 80 ms, puis volume
    segs, _ = whisper.transcribe(tmp2, language=lang, beam_size=5, condition_on_previous_text=False,
                                 word_timestamps=True)
    fins = [w.end for s in segs for w in (s.words or [])]
    af = VOLUME
    if fins:
        fin = fins[-1] + 0.15
        af = f'atrim=end={fin:.3f},afade=t=out:st={max(0, fin - 0.08):.3f}:d=0.08,' + VOLUME
    os.makedirs(os.path.dirname(mp3), exist_ok=True)
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', tmp2, '-af', af, '-ar', '24000', '-ac', '1',
                    '-b:a', '56k', mp3], check=True)
    for f in (tmp, tmp2):
        if os.path.exists(f):
            os.unlink(f)


def transcrire(mp3, lang, w=None):
    segs, _ = (w or whisper).transcribe(mp3, language=lang, beam_size=5, condition_on_previous_text=False)
    return ' '.join(s.text.strip() for s in segs)


def seuil(texte):
    return 0.9 if len(mots(texte)) <= 6 else 0.8


def score(sim, faux):
    return (-len(faux), sim)


items = json.load(open(entree))
fait = set()
if os.path.exists(sortie):
    fait = {json.loads(l)['file'] for l in open(sortie)}
with open(sortie, 'a') as out:
    for x in items:
        if x['file'] in fait:
            continue
        t0 = time.time()
        tms = x.get('variantes') or [texte_modele(x['text'], x['lang'], modele=True)]
        meilleur = None
        essais = []
        for vi, g in [(vi, g) for vi in range(len(tms)) for g in graines]:
            tm = tms[vi]
            if meilleur and not meilleur[2] and meilleur[1] >= seuil(x['text']):
                break
            mp3 = os.path.join(W, 'essais', x['file'][:-4] + f'_v{vi}_g{g}.mp3')
            generer(tm, x['lang'], g, mp3)
            tr = transcrire(mp3, x['lang'])
            sim, faux, _ = controle(x['text'], tr, x['lang'])
            if faux or sim < seuil(x['text']):  # second avis : Whisper medium
                tr2 = transcrire(mp3, x['lang'], whisper_m)
                sim2, faux2, _ = controle(x['text'], tr2, x['lang'])
                if score(sim2, faux2) > score(sim, faux):
                    tr, sim, faux = 'medium: ' + tr2, sim2, faux2
            essais.append(dict(v=tm, g=g, trans=tr, sim=round(sim, 3), faux=faux))
            if meilleur is None or score(sim, faux) > score(meilleur[1], meilleur[2]):
                meilleur = (mp3, sim, faux, tr, g, tm)
        mp3, sim, faux, tr, g, tm = meilleur
        av = x.get('avant', {'sim': -1, 'faux': [0] * 99})
        mieux = score(sim, faux) > score(av['sim'], av['faux'])
        if mieux:
            dst = os.path.join(W, 'neuf', x['file'])
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            subprocess.run(['cp', mp3, dst], check=True)
        r = dict(x, texte_modele=tm, graine=g, trans=tr, sim=round(sim, 3), faux=faux,
                 ok=(not faux and sim >= seuil(x['text'])), remplace=mieux, essais=essais,
                 duree=round(time.time() - t0))
        out.write(json.dumps(r, ensure_ascii=False) + '\n')
        out.flush()
        print(r['ok'], r['remplace'], g, r['sim'], faux, x['text'][:60], '|', tr[:60], r['duree'], 's', flush=True)
