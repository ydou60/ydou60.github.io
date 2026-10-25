"""Transcrit les mp3 dont le texte contient un chiffre et vérifie chaque nombre."""
import json, os, re, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nombres_en_lettres import controle
from faster_whisper import WhisperModel

W = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'w')
taille = sys.argv[1]           # small / medium
entree = sys.argv[2]           # liste json d'éléments
sortie = sys.argv[3]           # jsonl
racine = sys.argv[4] if len(sys.argv) > 4 else os.path.join(W, 'orig')

model = WhisperModel(taille, device='cpu', compute_type='int8', cpu_threads=4)
items = json.load(open(entree))
fait = set()
if os.path.exists(sortie):
    fait = {json.loads(l)['file'] for l in open(sortie)}
with open(sortie, 'a') as out:
    for n, x in enumerate(items):
        if x['file'] in fait:
            continue
        p = os.path.join(racine, x['file'])
        if not os.path.exists(p):
            continue
        t0 = time.time()
        segs, _ = model.transcribe(p, language=x['lang'], beam_size=5, condition_on_previous_text=False)
        tr = ' '.join(s.text.strip() for s in segs)
        sim, faux, nb = controle(x['text'], tr, x['lang'])
        out.write(json.dumps(dict(x, trans=tr, sim=round(sim, 3), faux=faux, nb=nb), ensure_ascii=False) + '\n')
        out.flush()
        if n % 50 == 0:
            print(n, round(time.time() - t0, 1), x['text'][:50], '|', tr[:50], faux, flush=True)
