"""Build the `dataset_endpoints` dataset: each clip's first 0.5 s and last 0.5 s.

For every discovered clip (all source roots, exports excluded) we cut TWO
self-contained SenseCV clip folders:

  <clip>_head   -> [0, SECONDS]                  (the first 0.5 s)
  <clip>_tail   -> [duration - SECONDS, duration] (the last 0.5 s)

Each output folder carries the WHOLE window: the trimmed video plus every sensor
sample inside it (frames.json + accelerations/rotations/external_sensors), the
source device metadata.json, and a cut_info.json describing the cut. So each
folder reloads as a normal SenseCV clip. A top-level sources.csv ledgers every
cut, and one review.mp4 plays every cut back-to-back with its info burned in.

Output: <SENSECV_DATA_DIR>/dataset_endpoints/   (default data/dataset_endpoints/)

  python scripts/make_dataset_endpoints.py              # all manifest clips, 0.5 s
  python scripts/make_dataset_endpoints.py --seconds 1  # first/last 1.0 s
  python scripts/make_dataset_endpoints.py --keep       # don't wipe existing build
"""
import argparse
import csv
import json
import os
import shutil
import sys

try:  # labels carry glyphs the Windows console (cp1252) can't encode
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
from sensecv import app  # noqa: E402

DATASET_NAME = 'dataset_endpoints'
MODE = 'deviation'  # only used to enumerate the manifest-matched clips

FIELDS = [
    'output_folder', 'source_display', 'source_path', 'segment',
    'start', 'end', 'duration', 'frames',
    'clip_duration', 'label', 'status',
]


def _clip_duration(idx):
    """Total clip length in seconds, from the source frames.json timestamps."""
    frames = app.load_json(app.clip_path(idx), 'frames.json')['frames']
    if not frames:
        return 0.0
    return (float(frames[-1]['time_usec']) - float(frames[0]['time_usec'])) / 1e6


def _export_cut(idx, display, start, end, segment, label, out_dir):
    """Cut one window into its own self-contained SenseCV clip folder. Returns a row."""
    name = app._safe_clip_name(display) + '_' + segment
    src_folder = app.CLIP_PATHS[display]
    out_folder = os.path.join(out_dir, name)
    out_video = os.path.join(out_folder, name + '.mp4')

    row = {k: '' for k in FIELDS}
    row.update(
        output_folder=name, source_display=display, source_path=src_folder,
        segment=segment, start=f'{start:.3f}', end=f'{end:.3f}',
        duration=f'{end - start:.3f}', label=label,
    )

    if os.path.isdir(out_folder):
        shutil.rmtree(out_folder, ignore_errors=True)
    os.makedirs(out_folder)

    try:
        app._ffmpeg_cut(app.clip_video_path(idx), start, end, out_video)
    except Exception as e:
        shutil.rmtree(out_folder, ignore_errors=True)
        row['status'] = f'ffmpeg_error: {e}'
        return row, None

    sensor_error = None
    try:
        app.save_sensor_data(idx, start, end, out_folder)
    except Exception as e:
        sensor_error = str(e)

    n_frames = ''
    try:
        n_frames = len(app.load_json(out_folder, 'frames.json')['frames'])
    except Exception:
        pass
    row['frames'] = n_frames

    meta_src = os.path.join(src_folder, 'metadata.json')
    if os.path.isfile(meta_src):
        try:
            shutil.copy2(meta_src, os.path.join(out_folder, 'metadata.json'))
        except Exception:
            pass

    with open(os.path.join(out_folder, 'cut_info.json'), 'w', encoding='utf-8') as f:
        json.dump({
            'source_display': display, 'segment': segment,
            'start': round(start, 3), 'end': round(end, 3),
            'duration': round(end - start, 3), 'label': label,
        }, f, ensure_ascii=False, indent=1)

    row['status'] = f'ok_no_sensor: {sensor_error}' if sensor_error else 'ok'
    return row, {'video_path': out_video, 'folder': name,
                 'segment': segment, 'label': label}


def _build_review(cuts, out_path, fps=None):
    """One MP4 of every cut back-to-back with folder/segment/label burned in."""
    import cv2
    fps = fps or app.SSIM_REVIEW_FPS
    cuts = [c for c in cuts if c and os.path.isfile(c['video_path'])]
    if not cuts:
        return 0

    import imageio_ffmpeg
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

    width = height = None
    for c in cuts:
        cap = cv2.VideoCapture(c['video_path'])
        if cap.isOpened():
            width, height = app._review_frame_size(
                cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            break
        cap.release()
    if width is None:
        return 0

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    seg_col = {'head': (120, 220, 120), 'tail': (120, 180, 250)}
    import subprocess
    cmd = [
        ffmpeg, '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo', '-pix_fmt', 'rgb24',
        '-s', f'{width}x{height}', '-r', f'{fps}', '-i', '-', '-an',
        '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '23',
        '-pix_fmt', 'yuv420p', '-movflags', '+faststart', out_path,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    written = 0
    try:
        for c in cuts:
            col = seg_col.get(c['segment'], (235, 235, 235))
            cap = cv2.VideoCapture(c['video_path'])
            if not cap.isOpened():
                cap.release()
                continue
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame.shape[1] != width or frame.shape[0] != height:
                    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
                app._draw_label_bar(frame, [
                    (c['folder'], (235, 235, 235)),
                    (f"{c['segment'].upper()}  |  {c['label']}", col),
                ])
                proc.stdin.write(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).tobytes())
                written += 1
            cap.release()
    finally:
        if proc.stdin:
            proc.stdin.close()
    stderr = proc.stderr.read().decode('utf-8', errors='replace')
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(stderr[-500:])
    return written


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--seconds', type=float, default=0.5,
                    help='Length of each endpoint cut in seconds (default 0.5).')
    ap.add_argument('--keep', action='store_true',
                    help="Don't wipe an existing build before writing.")
    ap.add_argument('--no-review', action='store_true',
                    help='Skip building the review.mp4.')
    args = ap.parse_args()

    span = float(args.seconds)
    out_dir = os.path.join(app.DATA_DIR, DATASET_NAME)

    # Every discovered clip except already-exported ones; carry a manifest label
    # when the clip's dataset happens to have one (else blank).
    label_by_idx = {it['clip_idx']: it.get('label', '')
                    for it in app.plan_manifest_export(MODE)}
    indices = [i for i, d in enumerate(app.CLIPS) if not d.startswith('exports/')]
    if not indices:
        print('No clips discovered.')
        return 1

    if os.path.isdir(out_dir) and not args.keep:
        shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir, exist_ok=True)

    print(f'{len(indices)} clip(s) -> {out_dir}  (first/last {span:.2f}s each)\n')
    rows = []
    review_cuts = []
    n_ok = n_skip = n_fail = 0

    for idx in indices:
        display = app.CLIPS[idx]
        label = label_by_idx.get(idx, '')
        dur = _clip_duration(idx)
        if dur < span:
            print(f'  [skip] {display}: too short ({dur:.2f}s < {span:.2f}s)')
            rows.append({**{k: '' for k in FIELDS}, 'source_display': display,
                         'clip_duration': f'{dur:.3f}', 'label': label,
                         'status': f'too_short: {dur:.3f}s'})
            n_skip += 1
            continue

        windows = [
            ('head', 0.0, span),
            ('tail', max(0.0, dur - span), dur),
        ]
        for segment, start, end in windows:
            row, cut = _export_cut(idx, display, start, end, segment, label, out_dir)
            row['clip_duration'] = f'{dur:.3f}'
            rows.append(row)
            if cut is None:
                n_fail += 1
                print(f'  [fail] {display} {segment}: {row["status"]}')
            else:
                review_cuts.append(cut)
                n_ok += 1
                print(f'  [ok]   {row["output_folder"]}: {start:.2f}-{end:.2f}s '
                      f'({row["frames"]} frames) [{label}]')

    with open(os.path.join(out_dir, 'sources.csv'), 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    review_frames = 0
    review_path = os.path.join(out_dir, 'review.mp4')
    if not args.no_review and review_cuts:
        try:
            review_frames = _build_review(review_cuts, review_path)
            print(f'\nreview.mp4: {review_frames} frames over {len(review_cuts)} cuts')
        except Exception as e:
            print(f'\n[warn] review video failed: {e}')

    print(f'\nDONE. ok={n_ok} skipped={n_skip} failed={n_fail} '
          f'({len(indices)} clips, {span:.2f}s endpoints)\n  dataset: {out_dir}\n'
          f'  ledger : {os.path.join(out_dir, "sources.csv")}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
