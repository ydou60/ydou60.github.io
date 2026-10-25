"""Coupe le murmure de fin des mp3 d'Amara.
usage: python3 fins.py <sortie.jsonl> <worker_index> <n_workers> [limite]
Écrit les mp3 coupés dans le dépôt (en place) et une ligne JSON par fichier."""
import numpy as np
import json, os, re, subprocess, sys, unicodedata, difflib, tempfile
from faster_whisper import WhisperModel

REPO = '/home/user/ydou60.github.io'
W = os.path.dirname(os.path.abspath(__file__))
CHAIN = "acompressor=threshold=0.06:ratio=3:attack=5:release=80:makeup=5,alimiter=limit=0.94:level=disabled"
MARGE, FONDU, SEUIL = 0.15, 0.08, 0.3

out, wi, nw = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
limite = int(sys.argv[4]) if len(sys.argv) > 4 else None

entries = []
for i in range(1, 6):
    entries += json.load(open(f'{W}/lot{i}.json'))
entries.sort(key=lambda e: (e['code'] == 'en', e['file']))  # explications d'abord, anglais ensuite
controle = set(open(f'{W}/controle.txt').read().split())
entries = [e for e in entries if e['file'] not in controle][wi::nw]
if limite: entries = entries[:limite]
if os.environ.get('LISTE'):
    garde = set(open(os.environ['LISTE']).read().split())
    entries = [e for e in entries if e['file'] in garde]
done = set()
if os.path.exists(out):
    done = {json.loads(l)['file'] for l in open(out)}

NUM = {'0':'zero','1':'un','2':'deux','3':'trois','4':'quatre','5':'cinq','6':'six','7':'sept','8':'huit','9':'neuf','10':'dix'}
NUM_EN = {'1':'one','2':'two','3':'three','4':'four','5':'five','6':'six','7':'seven','8':'eight','9':'nine','10':'ten'}
def norm(w):
    w = unicodedata.normalize('NFD', w.lower())
    w = ''.join(ch for ch in w if unicodedata.category(ch) != 'Mn')
    return re.sub(r"[^a-z0-9]", '', w)

def mots(t):
    return [n for n in (norm(x) for x in re.split(r"[\s'’\-]+", t)) if n]

def pcm(p):
    raw = subprocess.check_output(['ffmpeg','-v','error','-i',p,'-f','s16le','-ac','1','-ar','16000','-'])
    return np.frombuffer(raw, np.int16).astype(np.float32) / 32768

def duree(p):
    return float(subprocess.check_output(['ffprobe','-v','error','-show_entries','format=duration','-of','csv=p=0',p]).strip())

def fin_ok(attendu, reconnu):
    """Le dernier mot attendu correspond-il à l'un des derniers mots reconnus ?"""
    a = mots(attendu); r = [norm(w.word) for w in reconnu]; r = [x for x in r if x]
    if not a or not r: return False, '', ''
    last_a = a[-1]; last_r = r[-1]
    def proche(x, y):
        if x == y: return True
        if NUM_EN.get(x) == y or NUM_EN.get(y) == x: return True
        if NUM.get(x) == y or NUM.get(y) == x: return True
        if x.isdigit() or y.isdigit():
            return x.isdigit() and y.isdigit() and (x.endswith(y) or y.endswith(x))
        return difflib.SequenceMatcher(None, x, y).ratio() >= 0.6 or x in y or y in x
    # tolère « 22 cm » reconnu « 22 centimètres », « 1 € », etc. : dernier mot reconnu = l'un des 2 derniers attendus
    if os.environ.get('RELAX') and len(a) == 1 and len(r) == 1:
        return True, last_a, last_r  # mot isolé : le seul mot entendu est forcément lui
    ok = proche(last_a, last_r)
    if not ok and os.environ.get('NORM'):
        # nombres et unités en lettres des deux côtés (outils de voix-controle)
        from nombres_en_lettres import texte_modele, mots as mots_c
        try:
            a2 = mots_c(texte_modele(attendu, LANGUE)); r2 = mots_c(texte_modele(''.join(w.word for w in reconnu), LANGUE))
        except Exception:
            a2 = r2 = []
        if a2 and r2:
            x, y = norm(a2[-1]), norm(r2[-1])
            ok = x == y or difflib.SequenceMatcher(None, x, y).ratio() >= 0.6
            last_a, last_r = a2[-1], r2[-1] or (len(a) > 1 and proche(a[-2], last_r) and len(last_a) <= 3)
    return ok, last_a, last_r

def affine(a, fin):
    """Prolonge la fin Whisper tant que l'énergie (trames 10 ms) reste proche du niveau de parole."""
    fr = lambda i: 20*np.log10(np.sqrt(np.mean(a[i:i+160]**2)) + 1e-9)
    f = int(fin*16000)
    avant = [fr(i) for i in range(0, max(f-160, 0), 160)]
    voix = [x for x in avant if x > -40]
    if not voix: return fin
    seuil = float(np.median(voix)) - 16
    last, trou, i = f, 0, f
    while i + 160 <= len(a) and i < f + int(0.35*16000):
        if fr(i) > seuil: last, trou = i + 160, 0
        else:
            trou += 1
            if trou >= 5: break
        i += 160
    return last / 16000

def transcrire(a, lang):
    segs, _ = model.transcribe(a, language=lang, word_timestamps=True, beam_size=5, condition_on_previous_text=False)
    return [w for s in segs for w in (s.words or [])]

model = WhisperModel('small', device='cpu', compute_type='int8', cpu_threads=int(os.environ.get('THREADS', '2')))
f_out = open(out, 'a')
for e in entries:
    if e['file'] in done: continue
    src = f"{REPO}/{e['file']}"
    d0 = duree(src)
    a = pcm(src)
    words = transcrire(a, e['lang'])
    rec = {'file': e['file'], 'avant': round(d0, 3)}
    LANGUE = e['lang']
    ok, la, lr = fin_ok(e['text'], words)
    rec.update(dernier_attendu=la, dernier_reconnu=lr)
    if not ok:
        rec['statut'] = 'dernier_mot_rate'
    else:
        fw = max(w.end for w in words)
        fin = affine(a, fw)
        rec['fin_whisper'] = round(fw, 3); rec['fin_mot'] = round(fin, 3)
        if d0 - fin <= SEUIL:
            rec['statut'] = 'inchange'
        else:
            cut = fin + MARGE
            tmp = tempfile.mktemp(suffix='.mp3', dir=W)
            af = f"atrim=0:{cut:.3f},afade=t=out:st={cut-FONDU:.3f}:d={FONDU},{CHAIN}"
            subprocess.run(['ffmpeg','-v','error','-y','-i',src,'-af',af,'-ar','24000','-ac','1','-b:a','56k',tmp], check=True)
            # contrôle Whisper du fichier coupé seulement si on retire plus de 0,4 s
            ok2, lr2 = (fin_ok(e['text'], transcrire(pcm(tmp), e['lang']))[::2]) if d0 - cut > 0.4 else (True, '')
            if not ok2:
                os.remove(tmp)
                rec['statut'] = 'verif_echouee'; rec['reconnu_apres'] = lr2
            else:
                os.replace(tmp, src)
                rec['statut'] = 'coupe'
                rec['apres'] = round(duree(src), 3)
    f_out.write(json.dumps(rec, ensure_ascii=False) + '\n'); f_out.flush()
