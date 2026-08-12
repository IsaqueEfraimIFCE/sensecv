---
type: concept
tags: [magnetometer, compass, labeling, direction]
code_refs: [src/sensecv/app.py, src/sensecv/templates/index.html]
updated: 2026-07-21
---

# Compass turn labeling

`app.py › compass_turn_report()` implements the magnetometer method as an
optional annotation aid. The compass is not an input to either deployed image
model; it supplies a ground-truth-like direction choice in the viewer.

## Processing

The backend loads `magnetometer.json`, `accelerations.json`, and `frames.json`.
It denoises the XYZ streams with a centered 0.1 s mean, uses acceleration as the
up reference, projects magnetic north and the camera-forward axis onto the
horizontal plane, and resamples heading at 0.1 s intervals. For each time it
computes the signed, wraparound-safe change 0.5 s later:

- greater than +5 degrees → `RIGHT`;
- less than -5 degrees → `LEFT`;
- otherwise → `FRONT`.

Thus 358 to 3 degrees becomes +5 degrees, not -355 degrees. The report preserves
the complete 10 Hz label sequence and exposes the strongest absolute change as
`selection`, which is the clip-level choice used by the viewer.

## Viewer selection

The Anotar tab has a Bússola button that applies
`GET /api/compass-turn/<idx>`; `RIGHT` and `LEFT` activate the corresponding
deviation chip, while `FRONT` leaves the direction unset. The previous Normal
and Nenhum method buttons were retired on 2026-07-21.

An unset direction disables export for an obstacle classified as a deviation,
and `/api/crop` also rejects that incomplete classification. Exported cuts now
retain a rebased `magnetometer.json` when the source has one.

## Scope and limitation

Like the source method, this identifies the movement that was actually executed.
It cannot infer other paths that were available at an intersection; multi-label
availability still requires manual annotation. Magnetic disturbances indoors can
also make the compass result unavailable or less reliable; the direction chips
remain available for manual correction.

## Related

[[viewer-frontend]], [[api-routes]], [[imu-event-labeling]],
[[video-sensor-sync]], [[export-pipeline]].
