"""Mots anglais isolés : génération dans une phrase porteuse, découpe du mot, contrôle Whisper.

usage : mots_en.py liste.json sortie.jsonl
"""
import json, os, re, subprocess, sys, tempfile, time
D = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, D)
import torch, torchaudio
from nombres_en_lettres import mots
from faster_whisper import WhisperModel
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

W = os.path.join(D, 'w')
REF = os.path.join(W, 'ref_fr.wav')
VOLUME = 'acompressor=threshold=0.06:ratio=3:attack=5:release=80:makeup=5,alimiter=limit=0.94:level=disabled'
PORTEUSES = ['I said {w}.', 'The word is: {w}.', 'Say {w} again.', 'I said: {w}.', 'The word is... {w}.']
torch.set_num_threads(4)

model = ChatterboxMultilingualTTS.from_pretrained(device='cpu')
model.prepare_conditionals(REF, exaggeration=0.6)
whisper = WhisperModel('small', device='cpu', compute_type='int8', cpu_threads=4)


def transcrire(p, words=False):
    segs, _ = whisper.transcribe(p, language='en', beam_size=5, condition_on_previous_text=False,
                                 word_timestamps=words)
    segs = list(segs)
    return segs


def bornes(y, s, e, sr):
    """Affine les bornes Whisper sur l'énergie : région non silencieuse (trous < 60 ms) autour du mot."""
    f = int(0.01 * sr)
    n = len(y) // f
    rms = (y[:n * f].reshape(n, f) ** 2).mean(1).sqrt()
    seuil = 0.06 * float(rms.max())
    actif = (rms > seuil).tolist()
    m = min(n - 1, int((s + e) / 2 / 0.01))
    if not actif[m]:  # le milieu tombe dans un creux : on prend la trame active la plus proche
        cand = [i for i in range(int(s / 0.01), min(n, int(e / 0.01) + 1)) if actif[i]]
        if not cand:
            return s, e
        m = cand[len(cand) // 2]
    i = m
    trou = 0
    while i > 0 and trou < 6:
        i -= 1
        trou = 0 if actif[i] else trou + 1
    a = (i + trou) * 0.01
    j = m
    trou = 0
    while j < n - 1 and trou < 6:
        j += 1
        trou = 0 if actif[j] else trou + 1
    b = (j - trou + 1) * 0.01
    # on ne déborde pas trop sur les mots voisins détectés par Whisper
    return max(a, s - 0.15), min(b, e + 0.25)


entree, sortie = sys.argv[1], sys.argv[2]
graines = [int(g) for g in sys.argv[3].split(',')] if len(sys.argv) > 3 else list(range(1, 9))
items = json.load(open(entree))
fait = {json.loads(l)['file'] for l in open(sortie)} if os.path.exists(sortie) else set()
with open(sortie, 'a') as out:
    for x in items:
        if x['file'] in fait:
            continue
        w = x['text'].strip()
        cible = mots(w)
        t0 = time.time()
        essais, bon = [], None
        for porteuse in PORTEUSES:
            for g in graines:
                torch.manual_seed(g)
                wav = model.generate(porteuse.format(w=w), language_id='en', cfg_weight=0.3, exaggeration=0.6,
                                     temperature=0.8, repetition_penalty=2.0, min_p=0.05, top_p=1.0)
                tmp = tempfile.mktemp(suffix='.wav')
                torchaudio.save(tmp, wav, model.sr)
                segs = transcrire(tmp, words=True)
                ws = [ww for s in segs for ww in (s.words or [])]
                phrase = ' '.join(s.text.strip() for s in segs)
                hit = [ww for ww in ws if mots(ww.word) == cible]
                e = dict(porteuse=porteuse, g=g, phrase=phrase)
                if hit:
                    ww = hit[-1] if porteuse != 'Say {w} again.' else hit[0]
                    a, b = bornes(wav[0], ww.start, ww.end, model.sr)
                    a = max(0.0, a - 0.04)
                    b = min(wav.shape[1] / model.sr, b + 0.04)
                    d = b - a
                    mp3 = os.path.join(W, 'essais', x['file'][:-4] + f'_{PORTEUSES.index(porteuse)}_g{g}.mp3')
                    os.makedirs(os.path.dirname(mp3), exist_ok=True)
                    af = f'atrim=start={a:.3f}:end={b:.3f},asetpts=PTS-STARTPTS,afade=t=in:d=0.01,' \
                         f'afade=t=out:st={max(0, d - 0.01):.3f}:d=0.01,' + VOLUME
                    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', tmp, '-af', af, '-ar', '24000',
                                    '-ac', '1', '-b:a', '56k', mp3], check=True)
                    tr = ' '.join(s.text.strip() for s in transcrire(mp3))
                    e.update(trans=tr, debut=round(a, 3), fin=round(b, 3))
                    if mots(tr) == cible:
                        bon = (mp3, tr, porteuse, g)
                os.unlink(tmp)
                essais.append(e)
                print(' ', w, e, flush=True)
                if bon:
                    break
            if bon:
                break
        if bon:
            dst = os.path.join(W, 'neuf', x['file'])
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            subprocess.run(['cp', bon[0], dst], check=True)
        r = dict(x, ok=bool(bon), remplace=bool(bon), trans=bon[1] if bon else None,
                 porteuse=bon[2] if bon else None, graine=bon[3] if bon else None, essais=essais,
                 duree=round(time.time() - t0))
        out.write(json.dumps(r, ensure_ascii=False) + '\n')
        out.flush()
        print(r['ok'], w, r['trans'], r['porteuse'], r['graine'], r['duree'], 's', flush=True)
