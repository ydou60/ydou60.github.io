# Contrôle qualité de la voix d'Amara (branche voix-controle)

## Fichiers

- `nombres_en_lettres.py` : conversion des nombres en toutes lettres et comparaison avec Whisper.
  - `texte_modele(texte, lang, modele=True)` : texte donné à Chatterbox (nombres en lettres via num2words,
    « 3,14 » → « trois virgule quatorze », « -12 » → « moins douze », « 2x » → « deux x », « 2ab » → « deux a b »,
    « ≈ » → « environ », « % » → « pour cent », cm/m/dm/km… en toutes lettres, « 1914-1918 » → « … à … »,
    « 14-18 » → « quatorze-dix-huit », « 2e » → « deuxième », « x ' » → « x prime »,
    « plus » d'opération → « plusse » ; « le plus grand », « plus de » restent « plus »).
  - `controle(texte, transcription, lang)` → (similarité, nombres faux, nb de nombres) : difflib autojunk=False,
    même normalisation des deux côtés ; un nombre est faux si un de ses mots n'est pas aligné, ou si un mot-nombre
    est inséré juste à côté (« vingt » attendu, « vingt et un » entendu).
- `controle_nombres.py taille liste.json sortie.jsonl` : transcription faster-whisper (small ou medium, int8, beam 5).
- `regenerer.py liste.json sortie.jsonl [graines]` : régénération (voir réglages ci-dessous), garde la meilleure graine.
- `mots_anglais.py liste.json sortie.jsonl` : mots anglais isolés (phrase porteuse + découpe).
- `liste_controle.txt` : les 128 phrases dont un nombre était faux (hachage, chemin, texte).

## Méthode du contrôle des nombres

1. Whisper small sur les 1 210 phrases contenant un chiffre → 216 suspectes.
2. Revérification Whisper medium des suspectes → 128 confirmées fausses (fausses pour small ET medium).

## Réglages de génération (identiques à REGLAGES.md + CONSIGNES.md)

- `chatterbox-tts` 0.1.7, CPU, Python 3.11, torch 2.6.0, `torch.set_num_threads(4)`.
- Correctif textes courts : `chatterbox/models/t3/inference/alignment_stream_analyzer.py` (~l.142),
  `A[self.completed_at:, :-5].numel() > 0 and` avant le `.max()`.
- `model.prepare_conditionals('essai-voix/ref_fr.wav', exaggeration=0.6)` une seule fois, puis
  `generate(texte, language_id=lang, cfg_weight=0.3, exaggeration=0.6, temperature=0.8, repetition_penalty=2.0,
  min_p=0.05, top_p=1.0)` sans `audio_prompt_path` ; `torch.manual_seed(graine)` avant chaque morceau ;
  graines 1 puis 2…8 (16 pour le rattrapage).
- Textes > 150 caractères : découpés en phrases, recollés avec 0,25 s de silence.
- Français : `atempo=0.92` ; anglais : rien.
- Fin d'audio : coupe 0,15 s après la fin du dernier mot (horodatage par mot faster-whisper small), fondu de
  sortie de 80 ms (appliqué à partir du paquet 4).
- Volume : `acompressor=threshold=0.06:ratio=3:attack=5:release=80:makeup=5,alimiter=limit=0.94:level=disabled`,
  `-ar 24000 -ac 1 -b:a 56k`.
- Acceptation : aucun nombre faux et similarité ≥ 0,8 (≥ 0,9 si ≤ 6 mots) avec Whisper small, second avis
  Whisper medium si small échoue. Un fichier n'est remplacé que si la nouvelle version a un meilleur score
  (moins de nombres faux, puis meilleure similarité) que l'original.
