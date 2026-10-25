"""Reconstruit un fichier depuis l'original de origin/main avec la coupe d'un enregistrement donné."""
import subprocess, os
REPO = '/home/user/ydou60.github.io'
CHAIN = "acompressor=threshold=0.06:ratio=3:attack=5:release=80:makeup=5,alimiter=limit=0.94:level=disabled"
def original(f, dst):
    open(dst, 'wb').write(subprocess.check_output(['git', '-C', REPO, 'show', 'origin/main:' + f]))
def refaire(r, dst, tmpdir):
    src = os.path.join(tmpdir, 'src.mp3'); original(r['file'], src)
    cut = r['fin_mot'] + 0.15
    af = f"atrim=0:{cut:.3f},afade=t=out:st={cut-0.08:.3f}:d=0.08,{CHAIN}"
    subprocess.run(['ffmpeg','-v','error','-y','-i',src,'-af',af,'-ar','24000','-ac','1','-b:a','56k',dst], check=True)
