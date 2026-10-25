# Détection du visage (s3fd de Wav2Lip) calculée une fois sur prof/talk.mp4 : python levres_visage.py prof/talk.mp4
import sys, numpy as np, cv2, torch
sys.path.insert(0, '/tmp/Wav2Lip')
import face_detection
src = sys.argv[1]
cap = cv2.VideoCapture(src); frames = []
while True:
    ok, f = cap.read()
    if not ok: break
    frames.append(f)
frames = np.array(frames); print(frames.shape)
det = face_detection.FaceAlignment(face_detection.LandmarksType._2D, flip_input=False, device='cpu')
rects = []
for i in range(0, len(frames), 8):
    rects.extend(det.get_detections_for_batch(frames[i:i+8]))
assert all(r is not None for r in rects)
np.save('/tmp/w2l/frames.npy', frames)
np.save('/tmp/w2l/rects.npy', np.array([r[:4] for r in rects], float))
print(np.array(rects)[:5], np.array(rects).min(0), np.array(rects).max(0))
