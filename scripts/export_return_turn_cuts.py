"""CLI wrapper around `app.export_return_turn_set`: corridor re-alignment (retorno) cuts.

For each desvio clip, 0.5 s windows start at T1 + 0.5 / 0.75 / 1.0 s and are labelled
with the side opposite to the desvio (LEFT desvio -> turn RIGHT). Class and side come
from the Coleta sheet in data/labels when one exists, otherwise from the IMU.

  python scripts/export_return_turn_cuts.py                              # every clip the app sees
  python scripts/export_return_turn_cuts.py --root D:/SenseCV-12-09-2026-SUPERMERCADO
  python scripts/export_return_turn_cuts.py --filter SUPERMERCADO --out D:/retorno_cuts

Offsets/duration: SENSECV_RETURN_TURN_OFFSETS_SEC ("0.5,0.75,1.0"), SENSECV_RETURN_TURN_DURATION_SEC ("0.5").
"""
import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', action='append', default=[],
                    help='extra collection folder(s) holding clip subfolders')
    ap.add_argument('--filter', default='', help='only clips whose display name contains this')
    ap.add_argument('--out', default=None, help='output dir (default data/derived/return_turn_exports)')
    args = ap.parse_args()

    if args.root:
        extra = [os.environ.get('SENSECV_EXTRA_CLIP_ROOTS', '')] + args.root
        os.environ['SENSECV_EXTRA_CLIP_ROOTS'] = os.pathsep.join(p for p in extra if p)
    sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))
    from sensecv import app

    indices = [i for i, n in enumerate(app.CLIPS)
               if args.filter in n and not n.startswith('exports/')]
    print(f'{len(indices)} clips to process -> {args.out or app.RETURN_TURN_EXPORTS_DIR}')
    summary = app.export_return_turn_set(indices, args.out, verbose=True)
    print(f"\nDone. cuts={summary['ok']}  skipped_clips={summary['skipped']}  failed={summary['failed']}")
    print(f"Output: {summary['out_dir']}")
    print(f"Ledger: {summary['csv_path']}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
