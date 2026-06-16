"""Batch IMU capture-validation and event-labeling report.

Implements the operational pipeline of the "criterios_imu_video_orientando"
PDF over the uploaded clips: per-clip capture quality verdict (section 1) and
detected physical events with T1/T2, label windows and confidence levels
(sections 2-5). Writes two CSVs under data/derived/imu_event_report/.
"""
import argparse
import csv
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sensecv import app  # noqa: E402


def selected_clips(uploads_only):
    app.refresh_clips()
    uploads_root = Path(app.UPLOADS_DIR).resolve()
    rows = []
    for idx, display in enumerate(app.CLIPS):
        if uploads_only:
            src = Path(app.CLIP_PATHS[display]).resolve()
            try:
                if os.path.commonpath([str(uploads_root), str(src)]) != str(uploads_root):
                    continue
            except ValueError:
                continue
        rows.append((idx, display))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true",
                        help="Include every discovered clip, not only uploads.")
    parser.add_argument("--delta", type=float, default=None,
                        help="Perception-action delay for the decision window.")
    parser.add_argument("--out-dir",
                        default=str(PROJECT_ROOT / "data" / "derived" / "imu_event_report"))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    quality_rows = []
    event_rows = []
    clips = selected_clips(uploads_only=not args.all)
    for ordinal, (idx, display) in enumerate(clips, start=1):
        try:
            result = app.detect_imu_events(idx, delta=args.delta)
        except Exception as e:
            quality_rows.append({"clip": display, "verdict": f"erro: {e}"})
            print(f"[{ordinal:03d}/{len(clips):03d}] erro {display}: {e}")
            continue
        quality = result["quality"]
        quality_rows.append({
            "clip": display,
            "verdict": quality["verdict"],
            "failed_checks": ";".join(quality["failed_checks"]),
            "acc_median_hz": quality["acc_rate"]["median_hz"],
            "acc_gap_count": quality["acc_rate"]["gap_count"],
            "acc_jitter_ratio": quality["acc_rate"]["jitter_ratio"],
            "acc_saturation_frac": quality["acc_saturation_frac"],
            "cadence_hz": quality["cadence_hz"],
            "gait_periodicity": quality["gait_periodicity"],
            "walking_fraction": quality["walking_fraction"],
            "duration_s": quality["duration_s"],
            "events": len(result["events"]),
        })
        for event in result["events"]:
            event_rows.append({
                "clip": display,
                "type": event["type"],
                "direction": event.get("direction") or "",
                "source": event.get("source") or "",
                "t1": event["t1"],
                "t2": event["t2"],
                "decisao_start": event["windows"]["decisao"][0],
                "decisao_end": event["windows"]["decisao"][1],
                "acao_start": event["windows"]["acao"][0],
                "acao_end": event["windows"]["acao"][1],
                "expandido_start": event["windows"]["expandido"][0],
                "expandido_end": event["windows"]["expandido"][1],
                "strength": event["strength"],
                "confidence": event["confidence"],
                "confidence_reasons": ";".join(event["confidence_reasons"]),
                "delta": result["delta"],
            })
        print(f"[{ordinal:03d}/{len(clips):03d}] {quality['verdict']} {display}: "
              f"{len(result['events'])} eventos")

    for name, rows in (("quality.csv", quality_rows), ("events.csv", event_rows)):
        path = out_dir / name
        if not rows:
            path.write_text("", encoding="utf-8")
            continue
        headers = max(rows, key=len).keys()
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(headers))
            writer.writeheader()
            writer.writerows(rows)

    print(f"Done: {len(quality_rows)} clips, {len(event_rows)} eventos -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
