---
type: concept
tags: [sensor-fusion, imu, algorithm]
code_refs: [app.py]
updated: 2026-05-26
---

# Velocity estimation (IMU dead-reckoning)

`compute_velocities(acc_data, rot_data, ft_us)` in [[app-backend]] estimates a
per-frame velocity **without GPS** â€” essential for the indoor domain (see
[[overview]]). It fuses accelerometer + gyroscope into a world-frame velocity by
dead reckoning.

## Pipeline
1. **Initial gravity & orientation.** Average the first â‰¤200 accel samples to
   estimate the gravity vector in the phone frame; its norm `G` (fallback 9.81).
   Build a quaternion `q` that rotates the measured gravity direction onto world
   up `[0,0,1]`.
2. **Gyro integration.** For each gyro sample, form the incremental rotation
   quaternion from angle-axis (`Ï‰Â·dt`) and compose: `q = q âŠ— dq`. Samples with
   `dt â‰¤ 0` or `dt > 0.05 s` (gaps) are skipped â€” velocity is carried forward.
3. **Complementary correction.** Rotate the current accel into the world frame,
   compare its normalized direction to `[0,0,1]`, and nudge `q` back toward
   gravity with a small gain (`0.005`). This bleeds off gyro drift.
4. **Linear acceleration.** `a_world = qÂ·a_phone âˆ’ [0,0,G]` removes gravity.
5. **Deadband + integrate + decay.** Components with `|a| < 0.25` are zeroed
   (noise gate); `vel += a_worldÂ·dt`; then `vel *= 0.998` each step.
6. **Sample to frames.** The velocity track is sampled at each frame time â†’
   `{vx, vy, vz, speed=|v|}`.

## Why the decay and deadband
Pure double-integration of a noisy IMU drifts unbounded. The `0.998` leak pulls
velocity toward zero over ~seconds, and the `0.25` deadband suppresses
integrating sensor noise while stationary. The result is a **plausible relative
speed profile**, not a metrically exact velocity â€” good enough to spot
start/stop, walking, and the speed waveform on the timeline.

## Consumers
- `speed` drives the timeline waveform and the velocity chart in
  [[viewer-frontend]].
- The same accel magnitude feeds [[walking-detection]] (independently of this
  fused velocity).

## Caveats / open questions
- Absolute scale is unreliable; treat values as relative.
- Heading drift remains over long clips despite the complementary filter.
- *Could [[walking-detection]] use this fused `speed` instead of raw accel std?*
  â€” currently it does not; worth comparing during a [[log|lint]] pass.

