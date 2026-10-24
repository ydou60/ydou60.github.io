# Essai de synchronisation labiale – Amara

**Résultat :** `additions.mp4` (720x1174, 30 i/s, H.264 + AAC, 1,77 s, 250 Ko).
Phrase : « Le cours : Additions. » (`audio/fr-sub/af_bella/e9a9bf4a.mp3`) appliquée sur `prof/talk.mp4`.

## Outil

- **Wav2Lip** (dépôt officiel Rudrabha/Wav2Lip), poids `wav2lip_gan.pth` depuis Hugging Face (`Nekochu/Wav2Lip`),
  détecteur de visage S3FD. PyTorch 2.14 version CPU, Python 3.13, 4 cœurs, sans GPU.
- Petits correctifs de compatibilité (librosa 1.0, `torch.load(weights_only=False)`, `np.int`).
- Audio : 0,15 s de silence ajoutées avant et 0,35 s après la phrase, pour que la bouche démarre et se referme naturellement.
- Wav2Lip ne génère qu'un visage en 96x96. Pour garder une image nette, seule une **ellipse bouche/menton**
  du visage généré est recollée sur la vidéo d'origine (bords très adoucis, correction de couleur, léger
  renforcement de netteté). Les yeux, le nez, les cheveux et le fond restent les pixels d'origine.

## Temps de calcul (CPU, 4 cœurs)

- Détection du visage : ~182 s (S3FD sur des images pleine résolution, c'est l'étape lente)
- Wav2Lip : ~3,4 s
- Total : **~188 s pour 1,8 s de vidéo**
- Piste d'accélération évidente : détecter le visage une fois sur une image réduite, ou mettre en cache
  les cadres du visage de `talk.mp4` (même vidéo pour toutes les phrases) → on descendrait à quelques
  secondes par phrase.

## Limites constatées

- La vidéo source montre déjà Amara en train de parler et de sourire : Wav2Lip garde parfois ce sourire
  (dents visibles) même sur des sons qui demandent une bouche plus fermée.
- L'intérieur de la bouche (dents, langue) est un peu plus doux que le reste du visage (agrandissement
  du 96x96 vers ~145x230 px).
- La vidéo résultat dure la longueur de l'audio : seules les premières images de `talk.mp4` sont utilisées.
  Pour une phrase plus longue que 7,3 s, la vidéo reboucle (saut visible).
- La tête bouge beaucoup dans la vidéo d'origine ; le lissage des cadres suffit ici, mais un mouvement
  plus brusque pourrait faire « glisser » la bouche.
