# Réenregistrement de la voix d'Amara (voix « D »)

Chaque lot (`lotN.json`) est une liste de `{code, lang, text, file}`.

- Modèle : Chatterbox Multilingual (MIT), `language_id` = `lang` (fr ou en).
- Voix : clonée sur la référence française `essai-voix/ref_fr.wav` (branche `essai-levres`), la même pour toutes les phrases.
- Réglages identiques pour tous : ceux de l'échantillon validé `chatterbox-ref-fr_*` ; graine fixe.
- Débit : français un peu plus posé qu'à l'essai (viser 3 mots/s pour les explications) — baisser cfg_weight
  et/ou ffmpeg `atempo=0.92` sur le français. Anglais : débit naturel.
- Contrôle qualité : retranscrire chaque phrase avec faster-whisper (small, int8) ; si le texte reconnu
  s'écarte trop (similarité < 0,8) ou si la phrase est coupée, régénérer (graine différente, 2 essais max).
- Volume (identique au reste du site) : ffmpeg
  `-af "acompressor=threshold=0.06:ratio=3:attack=5:release=80:makeup=5,alimiter=limit=0.94:level=disabled" -ar 24000 -ac 1 -b:a 56k`
- Sortie : le chemin `file` exact (mp3), dans le dépôt.
