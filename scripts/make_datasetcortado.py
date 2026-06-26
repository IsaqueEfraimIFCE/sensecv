"""Build the `datasetcortado` dataset: every clip's suggested cut, full data.

For each manifest-matched clip we take `app.suggest_deviation_cut` (the same cut
the UI/Revisão use — desvio LEFT/RIGHT, parada, or caminhada livre, with the
class forced by the collection manifest) and export the WHOLE cut: the trimmed
video plus every sensor sample inside the window. There is deliberately NO frame
selector / SSIM step — all frames from the cut go into the dataset.

Each output clip is a self-contained SenseCV clip folder (video + frames.json +
accelerations/rotations/external_sensors + metadata.json), so the result reloads
as a normal dataset. A top-level sources.csv ledgers every cut.

Output: <SENSECV_DATA_DIR>/datasetcortado/   (default data/datasetcortado/)

  python scripts/make_datasetcortado.py            # all manifest-matched clips
  python scripts/make_datasetcortado.py --keep     # don't wipe an existing build
"""
import csv
import os
import shutil
import sys

try:  # labels carry glyphs the Windows console (cp1252) can't encode
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
from sensecv import app

MODE = 'deviation'
DATASET_NAME = 'datasetcortado'

FIELDS = [
    'output_folder', 'source_display', 'source_path',
    'start', 'end', 'duration', 'frames',
    'event_type', 'side', 'label', 'label_source', 'status',
]


def _label(event_type, side):
    try:
        return app._deviation_label(event_type, side)[0]
    except Exception:
        return ''


def main():
    keep = '--keep' in sys.argv[1:]
    out_dir = os.path.join(app.DATA_DIR, DATASET_NAME)

    items = app.plan_manifest_export(MODE)
    if not items:
        print('No manifest-matched clips found. Are the .xlsx files in '
              'data/labels/ and the clips under data/uploaded_datasets/?')
        return 1
    indices = sorted(it['clip_idx'] for it in items)

    if os.path.isdir(out_dir) and not keep:
        shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir, exist_ok=True)

    print(f'{len(indices)} clip(s) -> {out_dir}  (mode={MODE}, no frame selector)\n')
    rows = []
    n_ok = n_skip = n_fail = 0

    for idx in indices:
        display = app.CLIPS[idx]
        name = app._safe_clip_name(display)
        src_folder = app.CLIP_PATHS[display]
        out_folder = os.path.join(out_dir, name)
        out_video = os.path.join(out_folder, name + '.mp4')
        row = {k: '' for k in FIELDS}
        row.update(output_folder=name, source_display=display, source_path=src_folder)

        sug = app.suggest_deviation_cut(idx)
        if not sug.get('found'):
            row['status'] = 'no_segment: ' + str(sug.get('message', ''))
            rows.append(row); n_skip += 1
            print(f'  [skip] {display}: {sug.get("message", "")}')
            continue

        start, end = float(sug['start']), float(sug['end'])
        event_type = sug.get('event_type', '') or ''
        side = sug.get('side', '') or ''
        row.update(
            start=f'{start:.3f}', end=f'{end:.3f}', duration=f'{end - start:.3f}',
            event_type=event_type, side=side, label=_label(event_type, side),
            label_source=sug.get('label_source', 'imu'),
        )

        if os.path.isdir(out_folder):
            shutil.rmtree(out_folder, ignore_errors=True)
        os.makedirs(out_folder)

        try:
            app._ffmpeg_cut(app.clip_video_path(idx), start, end, out_video)
        except Exception as e:
            shutil.rmtree(out_folder, ignore_errors=True)
            row['status'] = f'ffmpeg_error: {e}'
            rows.append(row); n_fail += 1
            print(f'  [fail] {display}: {e}')
            continue

        # All sensor samples inside the cut window (no selection of any kind).
        sensor_error = None
        try:
            app.save_sensor_data(idx, start, end, out_folder)
        except Exception as e:
            sensor_error = str(e)

        # Count frames actually written; carry over the source device metadata.
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
        import json
        with open(os.path.join(out_folder, 'cut_info.json'), 'w', encoding='utf-8') as f:
            json.dump(
                {'source_display': display, 'start': round(start, 3),
                 'end': round(end, 3), 'duration': round(end - start, 3),
                 'event_type': event_type, 'side': side,
                 'label': _label(event_type, side),
                 'label_source': sug.get('label_source', 'imu')},
                f, ensure_ascii=False, indent=1)

        row['status'] = f'ok_no_sensor: {sensor_error}' if sensor_error else 'ok'
        rows.append(row); n_ok += 1
        print(f'  [ok]   {display}: {event_type}/{side} {start:.2f}-{end:.2f}s '
              f'({n_frames} frames)')

    with open(os.path.join(out_dir, 'sources.csv'), 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader(); w.writerows(rows)

    print(f'\nDONE. ok={n_ok} skipped={n_skip} failed={n_fail} '
          f'(total {len(indices)})\n  dataset: {out_dir}\n  ledger : '
          f'{os.path.join(out_dir, "sources.csv")}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
