# Lèvres d'Amara : Wav2Lip (wav2lip_gan) + recollage ellipse bouche/menton (lanczos, unsharp, recalage Lab) + encodage addroi.
# Prérequis : /tmp/Wav2Lip (dépôt Rudrabha/Wav2Lip, audio.py corrigé pour librosa récent : filters.mel(sr=..., n_fft=...)),
# /tmp/w2l_w/wav2lip_gan.pth (Hugging Face Nekochu/Wav2Lip), /tmp/w2l/frames.npy et rects.npy produits par levres_visage.py.
# Usage : python levres.py liste.jsonl resultats.jsonl  (une entrée de levres.json par ligne)
# Réglage actuel ≈36 Ko/s (lot levres3a). Ancien réglage ≈61 Ko/s (lot levres-a, 672 premières vidéos) : addroi bouche −0,3 et visage −0,1, crf 36, preset slow, sans -x264-params, AAC 64k.
import shutil, sys, os, json, time, math, hashlib, subprocess, traceback
import numpy as np, cv2, torch
sys.path.insert(0, '/tmp/Wav2Lip')
import audio as w2a
from models import Wav2Lip

REPO = '/home/user/ydou60.github.io'
W, H, FPS = 720, 1174, 30
PAD_BOTTOM = 15
BATCH = 64
ENC = (['-crf', '0', '-preset', 'ultrafast'] if os.environ.get('LOSSLESS') else
       # débit concentré sur la bouche/menton puis le visage (zones fixes couvrant tous les cadres), reste plus compressé
       # réglage ≈35 Ko/s (lot levres3a) : bouche −0,5 puis visage −0,3 (≈ qoffset×25 en QP), corps et décor à crf 46
       ['-vf', 'addroi=x=290:y=120:w=170:h=175:qoffset=-0.5,addroi=x=270:y=10:w=210:h=300:qoffset=-0.3',
        '-profile:v', 'high', '-pix_fmt', 'yuv420p', '-crf', '46', '-preset', 'slow',
        '-x264-params', 'keyint=600:bframes=8:aq-mode=3'])
AUDIO_KBPS = '48k'

frames = np.load('/tmp/w2l/frames.npy')            # 219 images de talk.mp4
rects = np.load('/tmp/w2l/rects.npy')               # x1 y1 x2 y2
NF = len(frames)
PP = list(range(NF)) + list(range(NF - 2, 0, -1))   # aller-retour : 436 images, retombe sur 0

# lissage des cadres (fenêtre 5, bords en miroir = cohérent avec l'aller-retour)
pad = np.concatenate([rects[2:0:-1], rects, rects[-2:-4:-1]])
sm = np.array([pad[i:i + 5].mean(0) for i in range(NF)])
BOX = []
for x1, y1, x2, y2 in sm:
    BOX.append((int(round(y1)), min(H, int(round(y2)) + PAD_BOTTOM), int(round(x1)), int(round(x2))))

# masque ellipse bouche/menton par cadre (bords adoucis) + anneau pour la couleur
def masks(h, w):
    m = np.zeros((h, w), np.float32)
    cv2.ellipse(m, (int(w * 0.5), int(h * 0.75)), (int(w * 0.34), int(h * 0.21)), 0, 0, 360, 1.0, -1)
    k = max(3, int(w * 0.12)) | 1
    m = cv2.GaussianBlur(m, (k, k), 0)
    inner = np.zeros((h, w), np.float32)
    cv2.ellipse(inner, (int(w * 0.5), int(h * 0.74)), (int(w * 0.22), int(h * 0.13)), 0, 0, 360, 1.0, -1)
    ring = m * (1 - inner)
    return m[..., None], ring
MASKS = {}

torch.set_num_threads(int(os.environ.get('THREADS', '4')))
model = Wav2Lip()
ck = torch.load('/tmp/w2l_w/wav2lip_gan.pth', map_location='cpu', weights_only=False)
model.load_state_dict({k.replace('module.', ''): v for k, v in ck['state_dict'].items()})
model.eval()

def load_wav(mp3):
    raw = subprocess.run(['ffmpeg', '-v', 'error', '-i', mp3, '-f', 'f32le', '-ac', '1', '-ar', '16000', '-'],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32).copy()

def duration(mp3):
    return float(subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0', mp3],
                                capture_output=True, text=True, check=True).stdout)

def unsharp(img, amount=0.5, sigma=1.0):
    b = cv2.GaussianBlur(img, (0, 0), sigma)
    return cv2.addWeighted(img, 1 + amount, b, -amount, 0)

def composite(fi, pred):
    y1, y2, x1, x2 = BOX[fi]
    h, w = y2 - y1, x2 - x1
    if (h, w) not in MASKS: MASKS[(h, w)] = masks(h, w)
    m, ring = MASKS[(h, w)]
    out = frames[fi].copy()
    orig = out[y1:y2, x1:x2]
    p = cv2.resize(pred, (w, h), interpolation=cv2.INTER_LANCZOS4)
    p = unsharp(p)
    # correction couleur (Lab) : moyenne de l'anneau extérieur ramenée sur celle de la source
    pl = cv2.cvtColor(p, cv2.COLOR_BGR2LAB).astype(np.float32)
    ol = cv2.cvtColor(orig, cv2.COLOR_BGR2LAB).astype(np.float32)
    s = ring.sum() + 1e-6
    d = ((ol - pl) * ring[..., None]).sum((0, 1)) / s
    pl = np.clip(pl + d, 0, 255).astype(np.uint8)
    p = cv2.cvtColor(pl, cv2.COLOR_LAB2BGR).astype(np.float32)
    out[y1:y2, x1:x2] = np.clip(p * m + orig.astype(np.float32) * (1 - m) + 0.5, 0, 255).astype(np.uint8)
    return out

def render(mp3, out_mp4):
    dur = duration(mp3)
    N = max(1, math.ceil(dur * FPS - 1e-6))
    mel = w2a.melspectrogram(load_wav(mp3))
    if np.isnan(mel).any():
        raise ValueError('mel NaN')
    L = mel.shape[1]
    chunks = []
    for i in range(N):
        s = min(int(i * 80. / FPS), max(0, L - 16))
        c = mel[:, s:s + 16]
        if c.shape[1] < 16: c = np.pad(c, ((0, 0), (0, 16 - c.shape[1])), mode='edge')
        chunks.append(c)
    os.makedirs(os.path.dirname(out_mp4), exist_ok=True)
    tmp = '/tmp/w2l/tmp_' + os.path.basename(out_mp4)
    ff = subprocess.Popen(['ffmpeg', '-v', 'error', '-y',
                           '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
                           '-i', mp3, '-map', '0:v', '-map', '1:a',
                           '-c:v', 'libx264', *ENC,
                           '-r', str(FPS), '-af', 'apad', '-c:a', 'aac', '-ac', '1', '-b:a', AUDIO_KBPS,
                           '-t', f'{N / FPS:.6f}', '-movflags', '+faststart', tmp], stdin=subprocess.PIPE)
    for b0 in range(0, N, BATCH):
        idx = [PP[k % len(PP)] for k in range(b0, min(N, b0 + BATCH))]
        faces = np.array([cv2.resize(frames[fi][BOX[fi][0]:BOX[fi][1], BOX[fi][2]:BOX[fi][3]], (96, 96)) for fi in idx])
        masked = faces.copy(); masked[:, 48:] = 0
        img = np.concatenate((masked, faces), axis=3) / 255.
        mb = np.array(chunks[b0:b0 + len(idx)])[..., None]
        with torch.no_grad():
            pr = model(torch.FloatTensor(np.transpose(mb, (0, 3, 1, 2))),
                       torch.FloatTensor(np.transpose(img, (0, 3, 1, 2))))
        pr = (pr.numpy().transpose(0, 2, 3, 1) * 255.).astype(np.uint8)
        for fi, p in zip(idx, pr):
            ff.stdin.write(composite(fi, p).tobytes())
    ff.stdin.close()
    if ff.wait() != 0:
        raise RuntimeError('ffmpeg a échoué')
    shutil.move(tmp, out_mp4)
    return N, dur

if __name__ == '__main__':
    # usage : gen.py liste.jsonl resultats.jsonl
    items = [json.loads(l) for l in open(sys.argv[1])]
    res = open(sys.argv[2], 'a')
    for it in items:
        mp3 = os.path.join(REPO, it['audio']); out = os.path.join(REPO, it['video'])
        t = time.time()
        try:
            sha = hashlib.sha256(open(mp3, 'rb').read()).hexdigest()
            N, dur = render(mp3, out)
            r = dict(it, ok=True, sha=sha, frames=N, dur=dur, secs=round(time.time() - t, 1), size=os.path.getsize(out))
        except Exception as e:
            traceback.print_exc()
            r = dict(it, ok=False, err=repr(e), secs=round(time.time() - t, 1))
        res.write(json.dumps(r, ensure_ascii=False) + '\n'); res.flush()
        print(r['video'], r.get('ok'), r['secs'], flush=True)
