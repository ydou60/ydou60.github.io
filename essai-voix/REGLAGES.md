# Réglages de la voix D (Chatterbox Multilingual, référence française)

Ce sont les réglages exacts qui ont produit `chatterbox-ref-fr_*.mp3`.

## Référence

- Fichier : `essai-voix/ref_fr.wav` (7,68 s, 24 kHz, mono).
  SHA-256 `e45c208241ac4d69b39f42e418da9c8aa99f92e6a058b558b99350c90b404edd`.
- Contenu : « Additionner, c'est mettre ensemble. Si tu as quatre billes et qu'on t'en donne trois, tu as
  maintenant sept billes. »
- Comment il a été fabriqué : F5-TTS avec le modèle français RASPIAUDIO (vitesse 0,8, graine 1), à partir de
  `amara_parle.wav`, puis passage dans ChatterboxVC pour lui redonner le timbre d'Amara.
  Attention : les poids F5 sont sous licence CC-BY-NC 4.0 (voir NOTES.md).

## Modèle

- `chatterbox-tts` 0.1.7 (pip), classe `ChatterboxMultilingualTTS.from_pretrained(device='cpu')`.
- Python 3.11, torch 2.6.0 (CPU), avec `setuptools<81` pour que le tatouage Perth se charge.
- `torch.set_num_threads(4)`

## Appel

```python
torch.manual_seed(1)          # avant CHAQUE phrase
wav = model.generate(
    text,
    language_id='fr',         # ou 'en'
    audio_prompt_path='essai-voix/ref_fr.wav',
    cfg_weight=0.3,
    exaggeration=0.6,
    temperature=0.8,          # valeurs par défaut de la bibliothèque, inchangées :
    repetition_penalty=2.0,
    min_p=0.05,
    top_p=1.0,
)
torchaudio.save(out_wav, wav, model.sr)   # model.sr = 24000
```

## Conversion en mp3

```
ffmpeg -i out.wav -ac 1 -ar 24000 -c:a libmp3lame -b:a 64k out.mp3
```

## Points connus

- L'explication sortait un peu rapide (environ 4 mots/s). Pour un débit plus posé, on peut baisser
  `cfg_weight` (par exemple 0,2) ou ralentir légèrement le son avec `atempo`.
- Temps de calcul : environ 30 à 50 s par phrase sur 4 cœurs, plus environ 30 s de chargement du modèle (une seule fois).
