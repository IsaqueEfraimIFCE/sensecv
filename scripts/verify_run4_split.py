"""Rebuild run4 MobileNetV1's exact split + per-head background reduction and
check every number against ss2ere/sibgrapi/run4/final_statistics1.txt."""
import csv, json, re, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

TMP = Path(r"C:\Users\Isaque\Desktop\sense_run4_experiments")
src = (TMP / "referencia" / "repro_split.py").read_text(encoding="utf-8").split("base = _pair_task_dirs")[0]
exec(src)

STATS = json.loads((TMP / "referencia" / "sibgrapi_run_stats" / "run4_1.txt").read_text(encoding="utf-8"))
br = STATS["background_reduction"]
VAL_CLIPS = set(br["validation"]["obstacle"]["clips"]) | set(br["validation"]["deviation"]["clips"])

base = _pair_task_dirs(BASE)
july_all = _pair_task_dirs(JULY)
known = {r["clip"] for r in base}
rows = base + [r for r in july_all if r["clip"] not in known]
auto, _ = _auto_trim_rows(rows)
allrows = rows + auto

test = [r for r in allrows if collection_day(r["clip"]) in set(TEST_DAYS)]
val = [r for r in allrows if r["clip"] in VAL_CLIPS]
train = [r for r in allrows if collection_day(r["clip"]) not in set(TEST_DAYS) and r["clip"] not in VAL_CLIPS]


def balance_background(rows, label_name, background):
    """Verbatim policy from scripts/mobilenetv1_two_head_all_frames.py."""
    groups, foreground = defaultdict(list), []
    for row in rows:
        (groups[row["clip"]] if row[label_name] == background else foreground).append(row)
    available = sum(map(len, groups.values()))
    target = min(available, len(foreground))
    if available <= target:
        return list(rows)
    quotas, fractions, allocated = {}, [], 0
    for clip, values in sorted(groups.items()):
        exact = len(values) * target / available
        quotas[clip] = int(np.floor(exact))
        allocated += quotas[clip]
        fractions.append((exact - quotas[clip], clip))
    for _, clip in sorted(fractions, key=lambda i: (-i[0], i[1]))[: target - allocated]:
        quotas[clip] += 1
    selected = set()
    for clip, values in groups.items():
        keep = quotas[clip]
        if keep:
            idx = np.floor((np.arange(keep) + .5) * len(values) / keep).astype(int)
            selected.update(id(values[i]) for i in idx)
    return [r for r in rows if r[label_name] != background or id(r) in selected]


def make_head_balanced_rows(rows):
    obstacle_rows = balance_background(rows, "obstacle", "free_path")
    deviation_rows = balance_background(rows, "deviation", "none")
    out = []
    for r in obstacle_rows:
        i = dict(r); i["obstacle_active"], i["deviation_active"] = True, False; out.append(i)
    for r in deviation_rows:
        i = dict(r); i["obstacle_active"], i["deviation_active"] = False, True; out.append(i)
    return out


def count(rows):
    return {"frames": len(rows), "clips": len({r["clip"] for r in rows}),
            "obstacle": dict(Counter(r["obstacle"] for r in rows)),
            "deviation": dict(Counter(r["deviation"] for r in rows))}


got = {
    "train_source": count(train),
    "train_reduced_per_head": count(make_head_balanced_rows(train)),
    "validation_source": count(val),
    "validation_reduced_per_head": count(make_head_balanced_rows(val)),
    "test": count(test),
}
ok = True
for key, want in STATS["splits"].items():
    if key not in got:
        continue
    g = got[key]
    same = (g["frames"] == want["frames"] and g["clips"] == want["clips"]
            and g["obstacle"] == want["obstacle"] and g["deviation"] == want["deviation"])
    ok &= same
    print(("MATCH " if same else "DIFF  ") + key)
    print("   got ", g)
    if not same:
        print("   want", {k: want[k] for k in ("frames", "clips", "obstacle", "deviation")})
print("\nALL MATCH" if ok else "\nMISMATCH")
