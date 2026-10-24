# Essai de voix clonée pour Amara

Référence : `amara_parle.wav` (8 s). Tout a été calculé sur processeur (4 cœurs, sans carte graphique).

## La voix de référence

- **Transcription Whisper (medium)** : langue détectée « anglais » (91 %), mais le texte n'a pas de sens :
  « The most in your wild come for the Millie again as it converged from challenge you got flyer humans and
  but one of the horses relaxed in today. »
  C'est très probablement un **pseudo-anglais généré avec la vidéo**, pas de vraies phrases.
- `amara_attente.wav` contient aussi de la parole du même genre (ni musique, ni silence). Ce n'est
  probablement pas tout à fait la même voix : ressemblance de 0,75 seulement avec `amara_parle.wav`.
- Le fond est propre (environ −50 dB, soit 37 dB sous la parole) : aucun nettoyage n'a été nécessaire.

## Comment j'ai mesuré

Je ne peux pas écouter. J'ai donc mesuré deux choses : la **langue reconnue par Whisper** sur chaque
résultat (qui révèle l'accent) et la **ressemblance de voix** avec la référence (de 0 à 1, cosinus des empreintes
vocales). Le débit est en mots par seconde : environ 3 mots/s est un débit posé, 4 mots/s un débit rapide.

## Résultats

| Fichiers | Modèle | Licence | Calcul par phrase | Français reconnu | Ressemblance | Débit FR |
|---|---|---|---|---|---|---|
| `chatterbox_*` | Chatterbox Multilingual, référence d'origine (cfg 0, exagération 0,5) | MIT | 26–48 s | 48–65 % (accent anglais) | 0,91–0,93 | 2,2–3,4 |
| `f5_*` | F5-TTS + modèle français RASPIAUDIO, vitesse 0,8 | code MIT, **poids CC-BY-NC 4.0** | 119–194 s | 100 % | 0,80–0,81 | 3,1–3,5 |
| `hybride-f5-vc_*` | F5 français, puis conversion de voix ChatterboxVC vers Amara | NC (à cause de F5) | F5 + 14–22 s | 96–99 % | 0,86–0,87 | 2,5–3,4 |
| `chatterbox-ref-fr_*` | Chatterbox Multilingual avec une **référence française** (l'extrait hybride) | MIT (mais l'extrait de référence vient de F5) | 29–49 s | **100 %** | 0,81–0,85 | 2,2–4,0 |
| — | XTTS-v2 | Coqui Public Model License (non commerciale) | non testé | | | |

Durée des phrases générées : explication 6,6–7,7 s, encouragement 3,5–4,5 s, phrase anglaise 2,8–3,9 s.
Il faut ajouter le chargement du modèle (environ 30 s, une seule fois).

## Mon avis

- **Chatterbox seul, avec la référence d'origine** : c'est la voix la plus proche de celle d'Amara, mais elle
  garde un accent anglais net en français. Whisper entend « Bravo to progress » pour « Bravo ! Tu progresses ».
  C'est le même problème qu'avec Kokoro, parce que la référence parle pseudo-anglais.
- **F5-TTS français** : l'accent français est parfait, avec une intonation de lecteur natif (le modèle a été appris
  sur des livres audio LibriVox). En revanche le timbre ressemble moins à Amara, c'est lent sur processeur
  (2 à 3 min par phrase), il ne sait pas parler anglais (la phrase anglaise donne « Mi na mi yamara… »), et
  ses poids sont **non commerciaux**.
- **Hybride F5 + conversion de voix** : il garde l'accent français et se rapproche nettement du timbre d'Amara.
- **Chatterbox avec la référence française** (piste la plus prometteuse) : il est rapide, parle un français
  reconnu à 100 % et un anglais correct, avec un timbre proche d'Amara. Une fois qu'on a un bon extrait de
  référence en français, Chatterbox (MIT) suffit pour toutes les phrases, dans les deux langues, avec toujours la même voix.
  - Problème de licence : l'extrait de référence actuel a été produit avec F5 (non commercial). Pour un usage
    public propre, il faudrait un extrait de référence français sans restriction, par exemple 10 s enregistrées
    par une vraie voix française qui donne son accord, ou un extrait généré par un modèle sous licence libre.
  - Le débit de l'explication est un peu rapide (4 mots/s). On peut le ralentir en baissant `cfg_weight` ou
    en étirant légèrement le son (atempo 0,9).

## XTTS-v2

Je ne l'ai pas testé : il faut accepter la Coqui Public Model License (non commerciale), et je ne peux pas
l'accepter à la place de l'utilisateur. De toute façon, cette licence ne permettrait pas un usage commercial.

## Chatterbox et tatouage

Chatterbox ajoute à chaque fichier un tatouage audio inaudible (Perth, de Resemble AI) qui signale une voix
générée. Je l'ai laissé en place.
