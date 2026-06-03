"""CLI wrapper around `app.export_set` for batch walking / lateral cuts.

  python export_sensecv.py                       # sensecv    walking  (default)
  python export_sensecv.py supermarket           # supermarket walking
  python export_sensecv.py sensecv     lateral   # sensecv    lateral
  python export_sensecv.py supermarket lateral   # supermarket lateral

The Flask UI exposes the same operation via POST /api/batch-export.
"""
import sys
import app


def main():
    preset = sys.argv[1] if len(sys.argv) > 1 else 'sensecv'
    mode   = sys.argv[2] if len(sys.argv) > 2 else 'walking'
    if preset not in app.PRESETS or mode not in ('walking', 'lateral'):
        print(f'usage: python export_sensecv.py [{ "|".join(app.PRESETS) }] [walking|lateral]')
        return 1
    flt = app._preset_filter(preset)
    indices = [i for i, n in enumerate(app.CLIPS) if flt(n)]
    out_dir = app._preset_out_dir(preset, mode)
    print(f'{len(indices)} clips to process -> {out_dir} (mode={mode})')
    summary = app.export_set(indices, out_dir, mode=mode, verbose=True)
    print(f'\nDone. ok={summary["ok"]}  skipped={summary["skipped"]}  '
          f'failed={summary["failed"]}')
    print(f'Output: {summary["out_dir"]}')
    print(f'Ledger: {summary["csv_path"]}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
