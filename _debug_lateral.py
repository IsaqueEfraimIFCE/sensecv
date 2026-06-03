"""Compare lateral-velocity-based scorers against the 5 GT deviation windows."""
import numpy as np, app

GT = [
    (179, 5.012, 5.947),
    (180, 4.679, 5.480),
    (181, 4.592, 6.114),
    (182, 4.672, 5.723),
    (197, 4.625, 5.625),
]


def features(idx):
    """Return (secs, fps, horiz_accel_2d, vel_2d_world) where world is the
    plane perpendicular to gravity (estimated frame-by-frame from rolling-mean
    accel, so it works for any phone orientation).
    """
    folder = app.clip_path(idx)
    frames = app.load_json(folder, 'frames.json')['frames']
    acc    = app.load_json(folder, 'accelerations.json')['accelerations']
    ft = np.array([f['time_usec'] for f in frames], dtype=np.float64)
    at = np.array([a['time_usec'] for a in acc], dtype=np.float64)
    av = np.array([[a['x'], a['y'], a['z']] for a in acc])
    secs = (ft - ft[0]) / 1e6
    fps  = float(1e6 / np.median(np.diff(ft)))
    a_at_f = np.array(app.interp_at_times(at, av, ft.tolist()))

    # Gravity = 1 s rolling mean
    W = max(1, int(fps * 1.0))
    kern = np.ones(W) / W
    g_vec = np.column_stack([np.convolve(a_at_f[:, k], kern, mode='same') for k in range(3)])
    g_norm = np.maximum(np.linalg.norm(g_vec, axis=1, keepdims=True), 1e-6)
    g_hat  = g_vec / g_norm

    # Horizontal linear acceleration (gravity removed, only the plane perp to g).
    linear = a_at_f - g_vec
    horiz3 = linear - np.sum(linear * g_hat, axis=1, keepdims=True) * g_hat  # (N,3)

    # Build an orthonormal 2D basis in the horizontal plane (any two orthogonal
    # axes will do; rotates with phone but that's fine since we'll later split
    # into "along walking" and "perpendicular" using a rolling-mean direction).
    e1 = np.cross(g_hat, np.array([1., 0., 0.]))
    bad = np.linalg.norm(e1, axis=1) < 1e-3
    if bad.any():
        e1[bad] = np.cross(g_hat[bad], np.array([0., 1., 0.]))
    e1 /= np.maximum(np.linalg.norm(e1, axis=1, keepdims=True), 1e-6)
    e2 = np.cross(g_hat, e1)
    e2 /= np.maximum(np.linalg.norm(e2, axis=1, keepdims=True), 1e-6)
    horiz2 = np.column_stack([np.sum(horiz3 * e1, axis=1),
                              np.sum(horiz3 * e2, axis=1)])  # (N,2)

    # Integrate to velocity (zero-mean over the clip to remove drift bias).
    dt = np.diff(np.concatenate([[secs[0]], secs]))
    vel2 = np.cumsum(horiz2 * dt[:, None], axis=0)
    vel2 -= vel2.mean(axis=0, keepdims=True)
    return secs, fps, horiz2, vel2


def split_along_lateral(vel2, fps, win_sec=2.0):
    """For each frame, decompose vel2 into the component along the local mean
    walking direction and the perpendicular (lateral) component."""
    W = max(1, int(fps * win_sec))
    k = np.ones(W) / W
    mean_v = np.column_stack([np.convolve(vel2[:, 0], k, mode='same'),
                              np.convolve(vel2[:, 1], k, mode='same')])
    mn = np.maximum(np.linalg.norm(mean_v, axis=1, keepdims=True), 1e-6)
    fwd = mean_v / mn  # local "forward" unit vector
    lat = np.column_stack([-fwd[:, 1], fwd[:, 0]])
    along = np.sum(vel2 * fwd, axis=1)
    perp  = np.sum(vel2 * lat, axis=1)
    return along, perp


def slide_best(signal, fps, window_sec=1.0):
    w = max(1, int(round(fps * window_sec)))
    if len(signal) <= w:
        return 0, len(signal) - 1
    means = np.convolve(np.abs(signal), np.ones(w), mode='valid') / w
    i = int(np.argmax(means))
    return i, i + w - 1


def report(label, picks):
    print(f'\n=== {label} ===')
    print(f'{"idx":>4} {"gt":>14} {"cut":>14} {"|s|+|e|":>8}')
    err = []
    for idx, gs, ge, cs, ce in picks:
        e = abs(cs - gs) + abs(ce - ge)
        err.append(e)
        print(f'{idx:>4} {f"{gs:.2f}-{ge:.2f}":>14} {f"{cs:.2f}-{ce:.2f}":>14} {e:>8.2f}')
    print(f'  mean abs |s|+|e| offset: {np.mean(err):.2f}')


# --- Approach 1: lateral velocity magnitude (perpendicular to mean walking dir)
picks = []
for idx, gs, ge in GT:
    secs, fps, horiz2, vel2 = features(idx)
    _, perp_vel = split_along_lateral(vel2, fps, win_sec=2.0)
    i0, i1 = slide_best(perp_vel, fps, window_sec=1.0)
    picks.append((idx, gs, ge, float(secs[i0]), float(secs[i1])))
report('lateral velocity (perp to mean walking direction, 1s window)', picks)

# --- Approach 2: lateral velocity peak excursion vs. mean — pick where the
# rolling std of perp_vel is highest (sudden change in lateral velocity).
picks = []
for idx, gs, ge in GT:
    secs, fps, horiz2, vel2 = features(idx)
    _, perp_vel = split_along_lateral(vel2, fps, win_sec=2.0)
    w = max(1, int(fps * 1.0))
    # Rolling std
    kn = np.ones(w) / w
    mu  = np.convolve(perp_vel, kn, mode='same')
    var = np.convolve((perp_vel - mu) ** 2, kn, mode='same')
    std = np.sqrt(np.maximum(var, 0))
    i0, i1 = slide_best(std, fps, window_sec=1.0)
    picks.append((idx, gs, ge, float(secs[i0]), float(secs[i1])))
report('rolling std of lateral velocity (1s window)', picks)

# --- Approach 3: horizontal velocity magnitude (any sideways acceleration shows here)
picks = []
for idx, gs, ge in GT:
    secs, fps, horiz2, vel2 = features(idx)
    speed = np.linalg.norm(vel2, axis=1)
    i0, i1 = slide_best(speed, fps, window_sec=1.0)
    picks.append((idx, gs, ge, float(secs[i0]), float(secs[i1])))
report('horizontal velocity magnitude (1s window)', picks)

# --- Approach 3b: yaw rate (the current detector, for comparison)
def yaw_rate_series(idx):
    folder = app.clip_path(idx)
    frames = app.load_json(folder, 'frames.json')['frames']
    acc = app.load_json(folder, 'accelerations.json')['accelerations']
    rot = app.load_json(folder, 'rotations.json')['rotations']
    ft = np.array([f['time_usec'] for f in frames], dtype=np.float64)
    at = np.array([a['time_usec'] for a in acc], dtype=np.float64)
    av = np.array([[a['x'], a['y'], a['z']] for a in acc])
    rt = np.array([r['time_usec'] for r in rot], dtype=np.float64)
    rv = np.array([[r['x'], r['y'], r['z']] for r in rot])
    secs = (ft - ft[0]) / 1e6
    fps  = float(1e6 / np.median(np.diff(ft)))
    a_at_f = np.array(app.interp_at_times(at, av, ft.tolist()))
    r_at_f = np.array(app.interp_at_times(rt, rv, ft.tolist()))
    W = max(1, int(fps * 1.0)); kg = np.ones(W) / W
    g_vec = np.column_stack([np.convolve(a_at_f[:, k], kg, mode='same') for k in range(3)])
    g_hat = g_vec / np.maximum(np.linalg.norm(g_vec, axis=1, keepdims=True), 1e-6)
    yaw = np.abs(np.sum(r_at_f * g_hat, axis=1))
    Ws = max(1, int(fps * 0.2))
    return secs, fps, np.convolve(yaw, np.ones(Ws)/Ws, mode='same')

# --- Approach 5: combine yaw + lateral velocity (z-scored sum)
picks = []
for idx, gs, ge in GT:
    secs, fps, horiz2, vel2 = features(idx)
    _, perp_vel = split_along_lateral(vel2, fps, win_sec=2.0)
    _, _, yaw_s = yaw_rate_series(idx)
    # z-score each then sum
    def zs(x):
        s = x.std() or 1.0
        return (x - x.mean()) / s
    score = zs(np.abs(perp_vel)) + zs(yaw_s)
    i0, i1 = slide_best(score, fps, window_sec=1.0)
    picks.append((idx, gs, ge, float(secs[i0]), float(secs[i1])))
report('combined: z(|lateral velocity|) + z(yaw rate), 1s mean', picks)

# --- Approach 4: peak |perp_vel| within window (max instead of mean)
picks = []
for idx, gs, ge in GT:
    secs, fps, horiz2, vel2 = features(idx)
    _, perp_vel = split_along_lateral(vel2, fps, win_sec=2.0)
    w = max(1, int(round(fps * 1.0)))
    n = len(perp_vel)
    best_score = -1; best_i = 0
    for i in range(n - w + 1):
        s = float(np.max(np.abs(perp_vel[i:i+w])))
        if s > best_score: best_score, best_i = s, i
    picks.append((idx, gs, ge, float(secs[best_i]), float(secs[best_i + w - 1])))
report('peak |lateral velocity| in 1s window', picks)
