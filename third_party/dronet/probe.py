import cv2, os, glob

DRONET_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(DRONET_DIR))
base = os.environ.get("PILOTGURU_EXPORTS_DIR", os.path.join(PROJECT_ROOT, "data", "exports"))
total = 0
for d in sorted(os.listdir(base)):
    dd = os.path.join(base, d)
    if not os.path.isdir(dd):
        continue
    mp4 = glob.glob(os.path.join(dd, "*.mp4"))
    if not mp4:
        continue
    cap = cv2.VideoCapture(mp4[0])
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total += n
    print(f"{d}: {os.path.basename(mp4[0])}  {w}x{h}  fps={fps:.2f}  frames={n}")
    cap.release()
print("TOTAL frames:", total)
