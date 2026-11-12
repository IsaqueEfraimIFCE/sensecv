"""Automatic cuts of the original dataset, shown as a starting point.

The SenseCV viewer cut every recording that has a row in its collection's
``.xlsx`` manifest (``suggest_deviation_cut``: desvio -> parada -> caminhada
livre) and logged each cut in a ``sources.csv`` ledger
(``data/derived/manifest_exports/Todos_cortes_auto_trim/deviation/sources.csv``).
The manifest gives the event type and the deviation direction; the cut gives
the window (seconds of the recording's timeline).

``build`` turns that ledger into ``data/auto_cuts.json`` (bundled with the app),
keyed by this app's clip id. The ledger names collections day-first
(``SenseCV-02-06-2026-IFCE/<sub>/01``) while the published clip ids are
month-first (``SenseCV-06-02-2026-IFCE__01``).

    python -m clip_annotator.auto_cuts --ledger <sources.csv> \
        [--clips clips.json] [--out clip_annotator/data/auto_cuts.json]

``--clips`` (the ``/api/clips`` answer) drops cuts of clips the index lacks and
cuts that end after the clip.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

from .events import is_safe_id

DEFAULT_PATH = Path(__file__).parent / "data" / "auto_cuts.json"
KINDS = ("desvio", "parada", "livre")
LEDGER_SIDES = {"LEFT": "esquerda", "RIGHT": "direita"}
SIDES = tuple(LEDGER_SIDES.values())
MAX_TIME_S = 3600.0
_DAY_FIRST = re.compile(r"^(SenseCV-)(\d{2})-(\d{2})(-\d{4}.*)$")


def clip_id_for(source_display: str) -> str:
    """``SenseCV-02-06-2026-IFCE/<sub>/07`` -> ``SenseCV-06-02-2026-IFCE__07``."""
    parts = [p for p in source_display.replace("\\", "/").split("/") if p]
    if len(parts) < 2:
        raise ValueError(f"source {source_display!r} has no collection/recording")
    match = _DAY_FIRST.match(parts[0])
    if not match:
        raise ValueError(f"collection {parts[0]!r} is not SenseCV-DD-MM-YYYY-...")
    prefix, day, month, rest = match.groups()
    clip_id = f"{prefix}{month}-{day}{rest}__{parts[-1]}"
    if not is_safe_id(clip_id):
        raise ValueError(f"clip id {clip_id!r} is not usable")
    return clip_id


def _cut(row: dict) -> dict:
    kind = (row.get("event_type") or "").strip().lower()
    if kind not in KINDS:
        raise ValueError(f"unknown event type {kind!r}")
    side = LEDGER_SIDES.get((row.get("side") or "").strip().upper())
    if kind == "desvio" and side is None:
        raise ValueError("a desvio needs side LEFT or RIGHT")
    start, end = round(float(row["start"]), 3), round(float(row["end"]), 3)
    if not 0 <= start < end <= MAX_TIME_S:
        raise ValueError(f"bad cut window {start}..{end}")
    return {"kind": kind, "side": side if kind == "desvio" else None,
            "start_s": start, "end_s": end}


def build(ledger_path, durations=None) -> tuple:
    """(cuts by clip id, list of skipped-row messages). ``durations`` maps the
    published clip ids to their duration; rows outside it are skipped."""
    cuts, skipped = {}, []
    with open(ledger_path, encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            source = row.get("source_display", "")
            if (row.get("status") or "ok").strip() != "ok":
                skipped.append(f"{source}: status {row.get('status')!r}")
                continue
            try:
                clip_id, cut = clip_id_for(source), _cut(row)
            except (KeyError, ValueError) as error:
                skipped.append(f"{source}: {error}")
                continue
            if durations is not None and clip_id not in durations:
                skipped.append(f"{source}: {clip_id} is not in the index")
                continue
            if durations is not None and cut["end_s"] > durations[clip_id] + 1e-3:
                skipped.append(f"{source}: cut ends after the clip ({durations[clip_id]} s)")
                continue
            if clip_id in cuts:
                skipped.append(f"{source}: second cut for {clip_id}")
                continue
            cuts[clip_id] = cut
    return dict(sorted(cuts.items())), skipped


def validate(raw) -> dict:
    """Checked copy of a loaded ``auto_cuts.json``; raises ValueError."""
    if not isinstance(raw, dict):
        raise ValueError("auto cuts must be a JSON object of clip id -> cut")
    checked = {}
    for clip_id, cut in raw.items():
        if not is_safe_id(clip_id) or not isinstance(cut, dict):
            raise ValueError(f"bad auto cut entry {clip_id!r}")
        kind, side = cut.get("kind"), cut.get("side")
        start, end = cut.get("start_s"), cut.get("end_s")
        if kind not in KINDS or side not in (*SIDES, None) or (kind == "desvio") != (side is not None):
            raise ValueError(f"{clip_id}: bad kind/side {kind!r}/{side!r}")
        numbers = all(isinstance(t, (int, float)) and not isinstance(t, bool) for t in (start, end))
        if not numbers or not 0 <= start < end <= MAX_TIME_S:
            raise ValueError(f"{clip_id}: bad cut window")
        checked[clip_id] = {"kind": kind, "side": side, "start_s": float(start), "end_s": float(end)}
    return checked


def load(path=DEFAULT_PATH) -> dict:
    """Cuts by clip id; an absent file means no suggestions (empty dict)."""
    path = Path(path)
    if not path.is_file():
        return {}
    return validate(json.loads(path.read_text(encoding="utf-8")))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ledger", required=True, type=Path)
    parser.add_argument("--clips", type=Path, help="/api/clips JSON, to check ids and durations")
    parser.add_argument("--out", type=Path, default=DEFAULT_PATH)
    args = parser.parse_args(argv)
    durations = None
    if args.clips:
        listed = json.loads(args.clips.read_text(encoding="utf-8"))["clips"]
        durations = {c["clip_id"]: float(c["duration_s"]) for c in listed}
    cuts, skipped = build(args.ledger, durations)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(cuts, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    kinds = {k: sum(c["kind"] == k for c in cuts.values()) for k in KINDS}
    print(f"{len(cuts)} cuts -> {args.out} {kinds}; skipped {len(skipped)}")
    for message in skipped:
        print("  skipped", message)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
