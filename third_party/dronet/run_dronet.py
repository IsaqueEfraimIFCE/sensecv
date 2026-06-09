"""Run DroNet frame-by-frame over every exported clip."""
import os, glob, csv, json, math
import cv2, numpy as np, torch
from dronet_model import load_dronet, preprocess_bgr

DRONET_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(DRONET_DIR))
WEIGHTS = os.environ.get("DRONET_WEIGHTS", os.path.join(DRONET_DIR, "repo", "model", "model_weights.h5"))
BASE = os.environ.get("PILOTGURU_EXPORTS_DIR", os.path.join(PROJECT_ROOT, "data", "exports"))
OUT = os.environ.get("DRONET_RESULTS_DIR", os.path.join(PROJECT_ROOT, "data", "dronet_results"))
os.makedirs(OUT, exist_ok=True)

model = load_dronet(WEIGHTS)
videos = sorted(glob.glob(os.path.join(BASE, "**", "*.mp4"), recursive=True))
print(f"Found {len(videos)} mp4 files\n")

summary = []
combined_rows = []

for vpath in videos:
    folder = os.path.basename(os.path.dirname(vpath))
    stem = os.path.splitext(os.path.basename(vpath))[0]
    tag = f"{folder}__{stem}"
    cap = cv2.VideoCapture(vpath)
    if not cap.isOpened():
        print(f"[SKIP] {folder}: cannot open (broken file)")
        summary.append({"clip": folder, "video": stem, "status": "unreadable", "frames": 0})
        cap.release()
        continue

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # annotated output: scale longest side to ~720
    scale = 720.0 / max(src_w, src_h) if max(src_w, src_h) > 0 else 1.0
    ow, oh = max(1, int(src_w * scale)), max(1, int(src_h * scale))
    writer = cv2.VideoWriter(os.path.join(OUT, f"{tag}_annotated.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), fps, (ow, oh))

    rows = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t, _ = preprocess_bgr(frame)
        with torch.no_grad():
            s, c = model(t)
        steer = float(s.item())
        coll = float(c.item())
        yaw = steer * 90.0
        time_s = idx / fps
        rows.append([idx, round(time_s, 4), steer, yaw, coll])
        combined_rows.append([folder, stem, idx, round(time_s, 4), steer, yaw, coll])

        # ---- overlay ----
        vis = cv2.resize(frame, (ow, oh))
        cv2.rectangle(vis, (0, 0), (ow, 86), (0, 0, 0), -1)
        cv2.putText(vis, f"{folder}  frame {idx}", (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(vis, f"steer s={steer:+.2f}  yaw={yaw:+.0f}deg", (8, 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)
        ccol = (0, 0, 255) if coll >= 0.5 else (0, 255, 255)
        cv2.putText(vis, f"collision p={coll:.2f}", (8, 66),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, ccol, 1, cv2.LINE_AA)
        # steering needle
        cx, cy = ow // 2, oh - 28
        ex = int(cx + math.sin(math.radians(yaw)) * 60)
        ey = int(cy - math.cos(math.radians(yaw)) * 60)
        cv2.line(vis, (cx, cy), (ex, ey), (0, 255, 0), 3, cv2.LINE_AA)
        # collision bar
        bw = int((ow - 20) * coll)
        cv2.rectangle(vis, (10, oh - 10), (10 + bw, oh - 4), ccol, -1)
        writer.write(vis)
        idx += 1

    cap.release()
    writer.release()

    with open(os.path.join(OUT, f"{tag}.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "time_s", "steering", "yaw_deg", "collision_prob"])
        w.writerows(rows)

    if rows:
        steers = np.array([r[2] for r in rows])
        colls = np.array([r[4] for r in rows])
        rec = {
            "clip": folder, "video": stem, "status": "ok", "frames": len(rows),
            "fps": round(fps, 2), "resolution": f"{src_w}x{src_h}",
            "steering_mean": round(float(steers.mean()), 4),
            "steering_min": round(float(steers.min()), 4),
            "steering_max": round(float(steers.max()), 4),
            "collision_mean": round(float(colls.mean()), 4),
            "collision_max": round(float(colls.max()), 4),
            "frac_collision_ge_0.5": round(float((colls >= 0.5).mean()), 3),
        }
        summary.append(rec)
        print(f"[OK]  {folder}: {len(rows):3d} frames | steer mean {rec['steering_mean']:+.2f} "
              f"[{rec['steering_min']:+.2f},{rec['steering_max']:+.2f}] | "
              f"coll mean {rec['collision_mean']:.2f} max {rec['collision_max']:.2f}")
    else:
        summary.append({"clip": folder, "video": stem, "status": "no_frames", "frames": 0})
        print(f"[WARN] {folder}: opened but 0 frames decoded")

with open(os.path.join(OUT, "all_frames.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["clip", "video", "frame", "time_s", "steering", "yaw_deg", "collision_prob"])
    w.writerows(combined_rows)

with open(os.path.join(OUT, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

print(f"\nDone. {sum(1 for s in summary if s.get('status')=='ok')} clips processed, "
      f"{len(combined_rows)} total frames.\nResults in: {OUT}")
