"""Batch deviation cuts for every uploaded clip that has a manifest row.

Selection mirrors `app.plan_manifest_export()` — only clips whose dataset ships
an .xlsx manifest *and* whose folder ID appears in that manifest are queued.
Clips with no manifest entry are ignored (the user's "ignore the videos where
there is no info at the csvs").

Each queued clip is cut with `export_set(mode='deviation')`, i.e. the
`suggest_deviation_cut` chain: desvio (lateral) -> parada (stop) -> caminhada
livre (free walk). Output goes per group to
data/derived/manifest_exports/<group>/deviation/ (same layout the manifest
export already uses), each with its own sources.csv and review_all_frames.mp4.

  python scripts/cut_manifest_deviation.py            # all groups
  python scripts/cut_manifest_deviation.py SenseCV-06-06-2026-IFCE  # one group
"""
import os
import sys
from collections import defaultdict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
from sensecv import app

MODE = 'deviation'


def main():
    only = set(sys.argv[1:])
    items = app.plan_manifest_export(MODE)
    if not items:
        print('No manifest-matched clips found. Are the .xlsx files in the '
              'dataset folders under data/uploaded_datasets/?')
        return 1

    by_group = defaultdict(list)
    for it in items:
        by_group[it['group']].append(it['clip_idx'])

    groups = sorted(by_group)
    if only:
        groups = [g for g in groups if g in only]
        if not groups:
            print(f'No groups matched {sorted(only)}. Available: {sorted(by_group)}')
            return 1

    total_ok = total_skip = total_fail = 0
    print(f'{len(items)} manifest-matched clips across {len(by_group)} group(s); '
          f'cutting {len(groups)} group(s) in mode={MODE}\n')
    for group in groups:
        indices = sorted(by_group[group])
        out_dir = app._manifest_group_dir(group, MODE)
        print(f'=== {group}: {len(indices)} clips -> {out_dir}')
        summary = app.export_set(indices, out_dir, mode=MODE, verbose=True)
        total_ok += summary['ok']; total_skip += summary['skipped']
        total_fail += summary['failed']
        print(f'    ok={summary["ok"]} skipped={summary["skipped"]} '
              f'failed={summary["failed"]}  ledger={summary["csv_path"]}\n')

    print(f'DONE. ok={total_ok} skipped={total_skip} failed={total_fail} '
          f'(total queued {sum(len(v) for v in by_group.values())})')
    return 0


if __name__ == '__main__':
    sys.exit(main())
