from flask import Flask, request, Response, render_template, jsonify, send_from_directory
import json, os, re, subprocess, math, zipfile, shutil, sys
from datetime import datetime
from collections import OrderedDict
import numpy as np

APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(APP_DIR, os.pardir, os.pardir))

def load_local_env():
    env_path = os.path.join(PROJECT_ROOT, '.env')
    if not os.path.isfile(env_path):
        return
    with open(env_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            key, value = line.split('=', 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value

load_local_env()

DATA_DIR = os.environ.get('SENSECV_DATA_DIR', os.path.join(PROJECT_ROOT, 'data'))
CLIPS_DIR = os.environ.get('SENSECV_CLIPS_DIR', os.path.join(DATA_DIR, 'clips'))
DATASETS_DIR = os.environ.get('SENSECV_DATASETS_DIR', os.path.join(DATA_DIR, 'datasets'))
EXPORTS_DIR = os.environ.get('SENSECV_EXPORTS_DIR', os.path.join(DATA_DIR, 'exports'))
HISTORY_FILE = os.environ.get('SENSECV_HISTORY_FILE', os.path.join(DATA_DIR, 'history.json'))
UPLOADS_DIR = os.environ.get('SENSECV_UPLOADS_DIR', os.path.join(DATA_DIR, 'uploaded_datasets'))
DERIVED_DIR = os.environ.get('SENSECV_DERIVED_DIR', os.path.join(DATA_DIR, 'derived'))
os.makedirs(CLIPS_DIR, exist_ok=True)
os.makedirs(EXPORTS_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(DERIVED_DIR, exist_ok=True)

app = Flask(__name__, template_folder=os.path.join(APP_DIR, 'templates'))
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['MAX_CONTENT_LENGTH'] = int(os.environ.get('SENSECV_MAX_UPLOAD_BYTES', str(2 * 1024 * 1024 * 1024)))

@app.route('/health')
def health():
    refresh_clips()
    return jsonify({'status': 'ok', 'clips': len(CLIPS)})

@app.after_request
def no_cache(response):
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

# ─── Clips ───────────────────────────────────────────────────────────────────

# Extra input roots to scan for clip subfolders, in addition to CLIPS_DIR.
# Keep machine-specific dataset paths in SENSECV_EXTRA_CLIP_ROOTS instead of code.
EXTRA_CLIP_ROOTS = [DATASETS_DIR]
ENV_EXTRA_CLIP_ROOTS = [
    p.strip() for p in (
        os.environ.get('SENSECV_EXTRA_CLIP_ROOTS')
        or ''
    ).split(os.pathsep)
    if p.strip()
]
EXTRA_CLIP_ROOTS.extend(ENV_EXTRA_CLIP_ROOTS + [UPLOADS_DIR])

# Maps clip display name -> absolute folder path. Populated by find_clips().
CLIP_PATHS = {}
CLIP_VIDEO_PATHS = {}
CLIP_GROUPS = {}
# Display names that are filmed horizontally the whole time (the SenseCV roots).
# For these we skip vertical-orientation detection and only segment walking.
WALKING_ONLY = set()

def _clip_video_file(p):
    video = os.path.join(p, 'video.mp4')
    if os.path.isfile(video):
        return video
    if not os.path.isdir(p):
        return None
    mp4s = sorted(
        os.path.join(p, name) for name in os.listdir(p)
        if name.lower().endswith('.mp4') and os.path.isfile(os.path.join(p, name))
    )
    return mp4s[0] if mp4s else None

def _is_clip_dir(p):
    return os.path.isdir(p) and _clip_video_file(p) is not None \
        and os.path.isfile(os.path.join(p,'frames.json'))

def _safe_name(name):
    name = re.sub(r'[^\w.\-]+', '_', os.path.splitext(os.path.basename(name))[0]).strip('._')
    return name or 'dataset'

def _unique_dir(parent, name):
    base = _safe_name(name)
    target = os.path.join(parent, base)
    if not os.path.exists(target):
        return target
    i = 2
    while True:
        candidate = os.path.join(parent, f'{base}_{i}')
        if not os.path.exists(candidate):
            return candidate
        i += 1

def _safe_extract_zip(zip_file, target_dir):
    os.makedirs(target_dir, exist_ok=True)
    for info in zip_file.infolist():
        normalized = info.filename.replace('\\', '/')
        parts = [p for p in normalized.split('/') if p]
        if not parts or normalized.startswith('/') or any(p == '..' for p in parts):
            raise ValueError(f'unsafe zip path: {info.filename}')
        out_path = os.path.abspath(os.path.join(target_dir, *parts))
        target_abs = os.path.abspath(target_dir)
        if os.path.commonpath([target_abs, out_path]) != target_abs:
            raise ValueError(f'unsafe zip path: {info.filename}')
        if info.is_dir():
            os.makedirs(out_path, exist_ok=True)
            continue
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with zip_file.open(info) as src, open(out_path, 'wb') as dst:
            shutil.copyfileobj(src, dst)

def _discover_clip_dirs(root, recursive=False):
    if not os.path.isdir(root):
        return []
    found = []
    if not recursive:
        for name in sorted(os.listdir(root)):
            p = os.path.join(root, name)
            if _is_clip_dir(p):
                found.append((name, p, None))
        return found
    for current, dirs, _files in os.walk(root):
        if _is_clip_dir(current):
            rel = os.path.relpath(current, root).replace(os.sep, '/')
            first = rel.split('/', 1)[0] if rel != '.' else os.path.basename(os.path.normpath(root))
            found.append((rel, current, first))
            dirs[:] = []
    return sorted(found, key=lambda item: item[0])

def find_clips():
    """Scan CLIPS_DIR plus any EXTRA_CLIP_ROOTS for clip subfolders.

    Clips in the primary root keep their bare folder name (so existing history
    and exports stay valid); clips from extra roots are prefixed with the root
    folder name to keep display names unique across roots.
    """
    CLIP_PATHS.clear()
    CLIP_VIDEO_PATHS.clear()
    CLIP_GROUPS.clear()
    WALKING_ONLY.clear()
    clips = []
    # (root, label, walking_only). The primary root is the supermarket footage
    # (held vertical); the extra roots are filmed horizontally throughout.
    recursive_roots = {
        os.path.normcase(os.path.abspath(r))
        for r in [DATASETS_DIR] + ENV_EXTRA_CLIP_ROOTS + [UPLOADS_DIR]
    }
    roots = []
    seen_roots = set()

    def add_root(root, label, walking_only, recursive):
        root_abs = os.path.normcase(os.path.abspath(root))
        if root_abs in seen_roots:
            return
        seen_roots.add(root_abs)
        roots.append((root, label, walking_only, recursive))

    add_root(CLIPS_DIR, None, False, False)
    for r in EXTRA_CLIP_ROOTS:
        recursive = os.path.normcase(os.path.abspath(r)) in recursive_roots
        label = None if recursive else os.path.basename(os.path.normpath(r))
        add_root(r, label, True, recursive)
    if os.path.isdir(EXPORTS_DIR):
        add_root(EXPORTS_DIR, 'exports', True, False)
    for root, label, walking_only, recursive in roots:
        if not os.path.isdir(root):
            continue
        for display_name, clip_dir, discovered_group in _discover_clip_dirs(root, recursive):
            display = display_name if label is None else f"{label}/{display_name}"
            group = discovered_group if recursive and discovered_group else label or 'Supermercado Telefrango'
            CLIP_PATHS[display] = clip_dir
            CLIP_VIDEO_PATHS[display] = _clip_video_file(clip_dir)
            CLIP_GROUPS[display] = group
            if walking_only:
                WALKING_ONLY.add(display)
            clips.append(display)
    return clips

CLIPS = find_clips()

def refresh_clips():
    global CLIPS
    CLIPS = find_clips()
    return CLIPS

def clip_path(idx):
    return CLIP_PATHS[CLIPS[idx]]

def clip_video_path(idx):
    return CLIP_VIDEO_PATHS[CLIPS[idx]]

def clip_groups():
    return [CLIP_GROUPS.get(name, '') for name in CLIPS]

# ─── History ─────────────────────────────────────────────────────────────────

def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r', encoding='utf-8-sig') as f:
            history = json.load(f)
        pruned = prune_history_to_exports(history)
        if len(pruned) != len(history):
            save_history(pruned)
        return pruned
    return []

def save_history(h):
    with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(h, f, indent=2, ensure_ascii=False)

def list_export_folders():
    if not os.path.isdir(EXPORTS_DIR):
        return []
    return sorted(
        name for name in os.listdir(EXPORTS_DIR)
        if os.path.isdir(os.path.join(EXPORTS_DIR, name))
    )

def prune_history_to_exports(history):
    export_folders = set(list_export_folders())
    return [entry for entry in history if entry.get('folder') in export_folders]

def get_next_number():
    used = set()
    for name in list_export_folders():
        m = re.match(r'^(\d+)(?:_|$)', name)
        if m:
            used.add(int(m.group(1)))
    number = 1
    while number in used:
        number += 1
    return number

def name_exists(folder_name):
    return os.path.exists(os.path.join(EXPORTS_DIR, folder_name))

def learned_walking_window(idx):
    """Return the latest exported walking crop for this source clip, if any."""
    source = CLIPS[idx]
    for entry in reversed(load_history()):
        if entry.get('source_idx') != idx and entry.get('source_clip') != source:
            continue
        start = entry.get('start')
        end = entry.get('end')
        if start is None or end is None or end <= start:
            continue
        return round(float(start), 3), round(float(end), 3)
    return None

# ─── Data processing ─────────────────────────────────────────────────────────

def load_json(folder, name):
    with open(os.path.join(folder, name), 'r', encoding='utf-8-sig') as f:
        return json.load(f)

def interp_at_times(src_t, src_v, query_t):
    n = len(src_t); result = []
    for qt in query_t:
        idx = int(np.searchsorted(src_t, qt))
        if idx <= 0:            result.append(src_v[0].tolist())
        elif idx >= n:          result.append(src_v[-1].tolist())
        else:
            t0,t1 = src_t[idx-1],src_t[idx]; v0,v1 = src_v[idx-1],src_v[idx]
            f = (qt-t0)/(t1-t0) if t1!=t0 else 0.
            result.append((v0+f*(v1-v0)).tolist())
    return result

def external_input_at_times(folder, query_t):
    path = os.path.join(folder, 'external_sensors.json')
    if not os.path.isfile(path):
        return []
    samples = load_json(folder, 'external_sensors.json').get('external_sensors', [])
    if not samples:
        return []
    sensor_t = np.array([s.get('time_usec', 0) for s in samples], dtype=np.float64)
    buttons = np.array([1 if int(s.get('button', s.get('input', 0)) or 0) == 1 else 0
                        for s in samples], dtype=np.int8)
    result = []
    for qt in query_t:
        idx = int(np.searchsorted(sensor_t, qt, side='right')) - 1
        if idx < 0:
            result.append(0)
        elif idx >= len(buttons):
            result.append(int(buttons[-1]))
        else:
            result.append(int(buttons[idx]))
    return result

def compute_velocities(acc_data, rot_data, ft_us):
    acc_t = np.array([a['time_usec'] for a in acc_data], dtype=np.float64)
    acc_v = np.array([[a['x'],a['y'],a['z']] for a in acc_data])
    rot_t = np.array([r['time_usec'] for r in rot_data], dtype=np.float64)
    rot_v = np.array([[r['x'],r['y'],r['z']] for r in rot_data])
    ft    = np.array(ft_us, dtype=np.float64)

    n = min(200, len(acc_v)); g_phone = acc_v[:n].mean(0)
    G = float(np.linalg.norm(g_phone)); G = G if G>1e-3 else 9.81
    g_hat = g_phone/G; g_world = np.array([0.,0.,1.])
    cross = np.cross(g_hat, g_world); dot = float(np.dot(g_hat, g_world)); cn = float(np.linalg.norm(cross))
    if cn<1e-7: q = np.array([1.,0.,0.,0.]) if dot>0 else np.array([0.,1.,0.,0.])
    else:
        s = np.sqrt(max(0.,(1.+dot)*2.)); q = np.array([s/2,cross[0]/s,cross[1]/s,cross[2]/s]); q/=np.linalg.norm(q)

    def qmul(a,b):
        w1,x1,y1,z1=a; w2,x2,y2,z2=b
        return np.array([w1*w2-x1*x2-y1*y2-z1*z2, w1*x2+x1*w2+y1*z2-z1*y2,
                         w1*y2-x1*z2+y1*w2+z1*x2, w1*z2+x1*y2-y1*x2+z1*w2])
    def qrot(q,v):
        w,x,y,z=q
        return np.array([
            (1-2*(y*y+z*z))*v[0]+2*(x*y-z*w)*v[1]+2*(x*z+y*w)*v[2],
            2*(x*y+z*w)*v[0]+(1-2*(x*x+z*z))*v[1]+2*(y*z-x*w)*v[2],
            2*(x*z-y*w)*v[0]+2*(y*z+x*w)*v[1]+(1-2*(x*x+y*y))*v[2],
        ])

    vel=np.zeros(3); rec=[(rot_t[0],vel.copy())]; prev_t=rot_t[0]
    for i in range(1,len(rot_t)):
        t=rot_t[i]; dt=(t-prev_t)/1e6; prev_t=t
        if dt<=0 or dt>0.05: rec.append((t,vel.copy())); continue
        om=rot_v[i]; on=float(np.linalg.norm(om))
        if on>1e-8:
            ha=on*dt/2; ax=om/on
            dq=np.array([np.cos(ha),np.sin(ha)*ax[0],np.sin(ha)*ax[1],np.sin(ha)*ax[2]])
            q=qmul(q,dq); q/=np.linalg.norm(q)
        idx=min(max(int(np.searchsorted(acc_t,t)),0),len(acc_v)-1)
        aw=qrot(q,acc_v[idx]); an=aw/max(float(np.linalg.norm(aw)),1e-6)
        err=np.cross(an,np.array([0.,0.,1.])); cor=np.array([1.,err[0]*.005,err[1]*.005,err[2]*.005])
        q=qmul(q,cor/np.linalg.norm(cor)); q/=np.linalg.norm(q)
        aw=qrot(q,acc_v[idx]); al=aw-np.array([0.,0.,G])
        al=np.where(np.abs(al)<0.25,0.,al); vel+=al*dt; vel*=0.998
        rec.append((t,vel.copy()))

    vt=np.array([r[0] for r in rec]); vv=np.array([r[1] for r in rec])
    result=[]
    for f in ft:
        i=min(max(int(np.searchsorted(vt,f)),0),len(vv)-1); v=vv[i]
        result.append({'vx':float(v[0]),'vy':float(v[1]),'vz':float(v[2]),'speed':float(np.linalg.norm(v))})
    return result

_cache = OrderedDict()
_walking_template_cache = {'mtime': None, 'templates': None, 'durations': None}
_walking_classifier_cache = {'mtime': None, 'model': None}
_classifier_feature_cache = OrderedDict()
_dronet_runtime = {'model': None, 'preprocess_bgr': None, 'torch': None, 'error': None}
_dronet_cache = OrderedDict()
_ssim_selection_cache = OrderedDict()
DRONET_DIR = os.environ.get('DRONET_DIR', os.path.join(PROJECT_ROOT, "third_party", "dronet"))
DRONET_WEIGHTS = os.environ.get('DRONET_WEIGHTS', os.path.join(DRONET_DIR, "repo", "model", "model_weights.h5"))
DRONET_SAMPLE_FPS = 3.0
SSIM_THRESHOLD = float(os.environ.get('SENSECV_SSIM_THRESHOLD', '0.985'))
SSIM_MAX_GAP_SEC = float(os.environ.get('SENSECV_SSIM_MAX_GAP_SEC', '0.5'))
SSIM_REVIEW_FPS = float(os.environ.get('SENSECV_SSIM_REVIEW_FPS', '12'))
SSIM_REVIEW_MAX_WIDTH = int(os.environ.get('SENSECV_SSIM_REVIEW_MAX_WIDTH', '960'))

def get_clip_data(idx):
    key = CLIPS[idx]
    if key in _cache: _cache.move_to_end(key); return _cache[key]
    folder = clip_path(idx)
    frames   = load_json(folder,'frames.json')['frames']
    acc_data = load_json(folder,'accelerations.json')['accelerations']
    rot_data = load_json(folder,'rotations.json')['rotations']
    ft = np.array([f['time_usec'] for f in frames], dtype=np.float64)
    at = np.array([a['time_usec'] for a in acc_data], dtype=np.float64)
    av = np.array([[a['x'],a['y'],a['z']] for a in acc_data])
    rt = np.array([r['time_usec'] for r in rot_data], dtype=np.float64)
    rv = np.array([[r['x'],r['y'],r['z']] for r in rot_data])
    t0=float(ft[0]); secs=[(float(f)-t0)/1e6 for f in ft]
    fps=float(1e6/np.median(np.diff(ft)))
    data={'fps':round(fps,3),'duration':secs[-1],'times':secs,
          'accel':interp_at_times(at,av,ft),'rotation':interp_at_times(rt,rv,ft),
          'velocity':compute_velocities(acc_data,rot_data,ft.tolist()),
          'external_input':external_input_at_times(folder, ft),
          'name':key,'index':idx,'total':len(CLIPS)}
    _cache[key]=data
    if len(_cache)>5: _cache.popitem(last=False)
    return data

def _load_dronet_runtime():
    if _dronet_runtime['model'] is not None or _dronet_runtime['error'] is not None:
        return _dronet_runtime
    try:
        if DRONET_DIR not in sys.path:
            sys.path.insert(0, DRONET_DIR)
        import torch
        from dronet_model import load_dronet, preprocess_bgr
        _dronet_runtime['model'] = load_dronet(DRONET_WEIGHTS)
        _dronet_runtime['preprocess_bgr'] = preprocess_bgr
        _dronet_runtime['torch'] = torch
    except Exception as e:
        _dronet_runtime['error'] = str(e)
    return _dronet_runtime

def _dronet_direction(steering):
    if abs(steering) < 0.1:
        return 'STRAIGHT'
    return 'RIGHT' if steering > 0 else 'LEFT'

def dronet_frame_classification(idx, time_s, exact=False):
    runtime = _load_dronet_runtime()
    if runtime['error']:
        return {'available': False, 'error': runtime['error']}

    video_path = clip_video_path(idx)
    mtime = os.path.getmtime(video_path)
    try:
        cap = cv2.VideoCapture(video_path)
    except NameError:
        import cv2 as _cv2
        globals()['cv2'] = _cv2
        cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {'available': False, 'error': 'video unreadable'}

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if fps <= 0 or frame_count <= 0:
        cap.release()
        return {'available': False, 'error': 'invalid video metadata'}

    duration = frame_count / fps
    sample_time = max(0.0, min(float(time_s or 0.0), duration))
    if not exact:
        sample_time = math.floor(sample_time * DRONET_SAMPLE_FPS) / DRONET_SAMPLE_FPS
    frame_idx = min(frame_count - 1, max(0, int(round(sample_time * fps))))

    key = (CLIPS[idx], mtime, frame_idx)
    if key in _dronet_cache:
        _dronet_cache.move_to_end(key)
        cap.release()
        return _dronet_cache[key]

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return {'available': False, 'error': 'frame decode failed', 'frame': frame_idx}

    tensor, _crop = runtime['preprocess_bgr'](frame)
    with runtime['torch'].no_grad():
        steering_t, collision_t = runtime['model'](tensor)
    steering = float(steering_t.item())
    collision = float(collision_t.item())
    yaw = steering * 90.0
    result = {
        'available': True,
        'clip': CLIPS[idx],
        'frame': frame_idx,
        'time_s': round(frame_idx / fps, 4),
        'requested_time_s': round(float(time_s or 0.0), 4),
        'sample_fps': DRONET_SAMPLE_FPS,
        'exact': bool(exact),
        'source_fps': round(fps, 4),
        'steering': steering,
        'yaw_deg': yaw,
        'direction': _dronet_direction(steering),
        'collision_prob': collision,
        'collision_label': 'COLLISION' if collision >= 0.5 else 'CLEAR',
    }
    _dronet_cache[key] = result
    if len(_dronet_cache) > 300:
        _dronet_cache.popitem(last=False)
    return result

# ─── Sensor data save ────────────────────────────────────────────────────────

def save_sensor_data(clip_idx, start_sec, end_sec, export_folder):
    folder = clip_path(clip_idx)
    frames_raw = load_json(folder,'frames.json')
    acc_raw    = load_json(folder,'accelerations.json')
    rot_raw    = load_json(folder,'rotations.json')
    t0_usec = frames_raw['frames'][0]['time_usec']
    t_start = t0_usec + start_sec * 1e6
    t_end   = t0_usec + end_sec   * 1e6
    # re-base time_usec so t_start → 0
    def rebase(t): return t - t_start

    filtered_frames = [
        {**f, 'time_usec': int(rebase(f['time_usec'])),
               'sensor_timestamp': int(rebase(f['sensor_timestamp']))}
        for f in frames_raw['frames']
        if t_start <= f['time_usec'] <= t_end
    ]
    # reset frame_id
    for i, fr in enumerate(filtered_frames): fr['frame_id'] = i

    filtered_acc = [
        {**a, 'time_usec': int(rebase(a['time_usec']))}
        for a in acc_raw['accelerations'] if t_start <= a['time_usec'] <= t_end
    ]
    filtered_rot = [
        {**r, 'time_usec': int(rebase(r['time_usec']))}
        for r in rot_raw['rotations'] if t_start <= r['time_usec'] <= t_end
    ]
    external_raw = None
    external_path = os.path.join(folder, 'external_sensors.json')
    if os.path.isfile(external_path):
        external_raw = load_json(folder, 'external_sensors.json')
    filtered_external = []
    if external_raw:
        filtered_external = [
            {**s, 'time_usec': int(rebase(s['time_usec']))}
            for s in external_raw.get('external_sensors', [])
            if t_start <= s.get('time_usec', -1) <= t_end
        ]
    with open(os.path.join(export_folder,'frames.json'),'w') as f:
        json.dump({'frames': filtered_frames}, f)
    with open(os.path.join(export_folder,'accelerations.json'),'w') as f:
        json.dump({'accelerations': filtered_acc}, f)
    with open(os.path.join(export_folder,'rotations.json'),'w') as f:
        json.dump({'rotations': filtered_rot}, f)
    if external_raw is not None:
        with open(os.path.join(export_folder,'external_sensors.json'),'w') as f:
            json.dump({'external_sensors': filtered_external}, f)

# ─── Suggest crop ────────────────────────────────────────────────────────────

def _frame_records_in_window(clip_idx, start_sec, end_sec):
    frames = load_json(clip_path(clip_idx), 'frames.json')['frames']
    if not frames:
        return []
    t0 = float(frames[0]['time_usec'])
    records = []
    for source_index, frame in enumerate(frames):
        time_s = (float(frame['time_usec']) - t0) / 1e6
        if start_sec <= time_s <= end_sec:
            records.append((source_index, frame, time_s))
    return records


def _ssim_gray(a, b):
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    mu_a = float(a.mean())
    mu_b = float(b.mean())
    da = a - mu_a
    db = b - mu_b
    var_a = float((da * da).mean())
    var_b = float((db * db).mean())
    cov = float((da * db).mean())
    denom = (mu_a * mu_a + mu_b * mu_b + c1) * (var_a + var_b + c2)
    if denom <= 1e-12:
        return 1.0 if np.array_equal(a, b) else 0.0
    return ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / denom


def _decode_gray_frame(cap, frame_no):
    try:
        import cv2
    except Exception as e:
        raise RuntimeError(f'OpenCV indisponivel para SSIM: {e}')
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_no))
    ok, frame = cap.read()
    if not ok:
        return None
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (160, 90), interpolation=cv2.INTER_AREA)


def ssim_frame_selection(clip_idx, start_sec, end_sec,
                         threshold=SSIM_THRESHOLD,
                         max_gap_sec=SSIM_MAX_GAP_SEC):
    """Select visually distinct frames inside a proposed crop."""
    records = _frame_records_in_window(clip_idx, start_sec, end_sec)
    before = len(records)
    empty = {
        'frames_before': before,
        'frames_after': 0,
        'ssim_threshold': threshold,
        'ssim_max_gap_sec': max_gap_sec,
        'selected_frames': [],
    }
    if not records:
        return empty

    video_path = clip_video_path(clip_idx)
    key = (
        CLIPS[clip_idx],
        os.path.getmtime(video_path),
        round(float(start_sec), 3),
        round(float(end_sec), 3),
        float(threshold),
        float(max_gap_sec),
    )
    if key in _ssim_selection_cache:
        _ssim_selection_cache.move_to_end(key)
        return _ssim_selection_cache[key]

    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
    except Exception as e:
        return {**empty, 'error': f'OpenCV indisponivel para SSIM: {e}'}
    if not cap.isOpened():
        return {**empty, 'error': 'video unreadable for SSIM'}

    video_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if video_fps <= 0:
        cap.release()
        return {**empty, 'error': 'invalid video fps for SSIM'}

    selected = []
    prev_gray = None
    prev_time = None
    last_record_i = len(records) - 1
    for record_i, (source_index, frame, time_s) in enumerate(records):
        frame_no = max(0, int(round(time_s * video_fps)))
        gray = _decode_gray_frame(cap, frame_no)
        if gray is None:
            continue

        score = None if prev_gray is None else _ssim_gray(prev_gray, gray)
        forced_gap = prev_time is not None and (time_s - prev_time) >= max_gap_sec
        keep = (
            prev_gray is None
            or record_i == last_record_i
            or score is None
            or score < threshold
            or forced_gap
        )
        if keep:
            selected.append({
                'source_index': int(source_index),
                'frame_id': int(frame.get('frame_id', source_index)),
                'time_s': round(float(time_s), 3),
                'ssim_prev': None if score is None else round(float(score), 5),
            })
            prev_gray = gray
            prev_time = time_s
    cap.release()

    result = {
        'frames_before': before,
        'frames_after': len(selected),
        'ssim_threshold': threshold,
        'ssim_max_gap_sec': max_gap_sec,
        'selected_frames': selected,
    }
    _ssim_selection_cache[key] = result
    if len(_ssim_selection_cache) > 64:
        _ssim_selection_cache.popitem(last=False)
    return result


def _coerce_ssim_threshold(value):
    if value is None or value == '':
        return SSIM_THRESHOLD
    threshold = float(value)
    if not 0.0 < threshold < 1.0:
        raise ValueError('SSIM threshold must be between 0 and 1')
    return threshold


def _request_ssim_threshold():
    return _coerce_ssim_threshold(request.args.get('threshold'))


def _body_ssim_threshold(body):
    return _coerce_ssim_threshold((body or {}).get('ssim_threshold'))


def _selection_public(selection):
    return {
        k: v for k, v in selection.items()
        if k != 'selected_frames'
    }


def _suggestion_with_ssim(idx, suggestion, threshold=SSIM_THRESHOLD):
    if not suggestion.get('found'):
        return suggestion
    try:
        selection = ssim_frame_selection(
            idx,
            float(suggestion['start']),
            float(suggestion['end']),
            threshold=threshold,
        )
    except Exception as e:
        selection = {'frames_before': 0, 'frames_after': 0, 'error': str(e)}
    enriched = dict(suggestion)
    enriched['ssim'] = _selection_public(selection)
    return enriched


def save_ssim_selection(selection, export_folder):
    with open(os.path.join(export_folder, 'ssim_selection.json'), 'w', encoding='utf-8') as f:
        json.dump(selection, f, indent=2)


def _review_frame_size(width, height, max_width=SSIM_REVIEW_MAX_WIDTH):
    width = max(1, int(width or 640))
    height = max(1, int(height or 360))
    max_width = max(1, int(max_width or width))
    if width <= max_width:
        out_w, out_h = width, height
    else:
        ratio = max_width / float(width)
        out_w, out_h = max_width, max(1, int(round(height * ratio)))
    return max(2, out_w - (out_w % 2)), max(2, out_h - (out_h % 2))


def _write_ssim_review_video(src_video, frame_nos, out_video, fps=SSIM_REVIEW_FPS):
    try:
        import cv2
    except Exception as e:
        raise RuntimeError(f'OpenCV indisponivel para video SSIM: {e}')

    cap = cv2.VideoCapture(src_video)
    if not cap.isOpened():
        raise RuntimeError('video unreadable for SSIM review')

    width, height = _review_frame_size(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH),
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
    )
    try:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as e:
        cap.release()
        raise RuntimeError(f'ffmpeg indisponivel para video SSIM: {e}')

    cmd = [
        ffmpeg, '-y',
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-pix_fmt', 'rgb24',
        '-s', f'{width}x{height}',
        '-r', f'{max(1.0, float(fps)):.3f}',
        '-i', '-',
        '-an',
        '-c:v', 'libx264',
        '-preset', 'ultrafast',
        '-crf', '23',
        '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart',
        out_video,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    written = 0
    try:
        for frame_no in frame_nos:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_no))
            ok, frame = cap.read()
            if not ok:
                continue
            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            proc.stdin.write(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).tobytes())
            written += 1

        if written == 0:
            empty = np.zeros((height, width, 3), dtype=np.uint8)
            cv2.putText(empty, 'No frames', (24, max(42, height // 2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (220, 220, 220), 2, cv2.LINE_AA)
            proc.stdin.write(cv2.cvtColor(empty, cv2.COLOR_BGR2RGB).tobytes())
    finally:
        cap.release()
        if proc.stdin:
            proc.stdin.close()
            proc.stdin = None
    _stdout, stderr = proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError((stderr or b'').decode('utf-8', errors='ignore')[-500:])
    return written


def save_ssim_review_videos(clip_idx, start_sec, end_sec, selection, export_folder):
    """Write audit videos for all candidate, selected, and rejected SSIM frames."""
    records = _frame_records_in_window(clip_idx, start_sec, end_sec)
    video_path = clip_video_path(clip_idx)
    try:
        import cv2
    except Exception as e:
        raise RuntimeError(f'OpenCV indisponivel para video SSIM: {e}')

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError('video unreadable for SSIM review')
    video_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    cap.release()
    if video_fps <= 0:
        raise RuntimeError('invalid video fps for SSIM review')

    selected_sources = {
        int(frame['source_index'])
        for frame in selection.get('selected_frames', [])
        if 'source_index' in frame
    }
    all_items = [
        (source_index, max(0, int(round(time_s * video_fps))))
        for source_index, _frame, time_s in records
    ]
    chosen = [frame_no for source_index, frame_no in all_items
              if int(source_index) in selected_sources]
    not_chosen = [frame_no for source_index, frame_no in all_items
                  if int(source_index) not in selected_sources]

    review_dir = os.path.join(export_folder, 'ssim_review')
    os.makedirs(review_dir, exist_ok=True)
    counts = {
        'all_frames': _write_ssim_review_video(
            video_path, [frame_no for _source_index, frame_no in all_items],
            os.path.join(review_dir, 'all_frames.mp4'),
        ),
        'chosen_frames': _write_ssim_review_video(
            video_path, chosen, os.path.join(review_dir, 'chosen_frames.mp4'),
        ),
        'not_chosen_frames': _write_ssim_review_video(
            video_path, not_chosen, os.path.join(review_dir, 'not_chosen_frames.mp4'),
        ),
    }
    with open(os.path.join(review_dir, 'manifest.json'), 'w', encoding='utf-8') as f:
        json.dump({
            'source_video': video_path,
            'start_sec': round(float(start_sec), 3),
            'end_sec': round(float(end_sec), 3),
            'review_fps': SSIM_REVIEW_FPS,
            'counts': counts,
        }, f, indent=2)
    return counts


def ssim_selection_payload(clip_idx, start_sec, end_sec, threshold=SSIM_THRESHOLD):
    selection = ssim_frame_selection(clip_idx, start_sec, end_sec, threshold=threshold)
    payload = dict(selection)
    frames = []
    for frame in payload.get('selected_frames', []):
        item = dict(frame)
        item['image_url'] = f"/frame/{clip_idx}/{int(item['source_index'])}.jpg"
        frames.append(item)
    payload['selected_frames'] = frames
    return payload


def _orientation_walking_masks(idx):
    """
    Per-frame boolean masks for phone orientation and gait.

    vertical (portrait):
      ay/|a| > 0.95  →  gravity nearly along phone y-axis (portrait)
      az/|a| < 0.12  →  screen not facing up/down (camera pointing forward)
      Both smoothed over a 1 s window.
      Calibrated against 5 clips with known ground-truth start times
      (clip1→13.0s, clip2→11.5s, clip3→none, clip4→9.0s, clip5→none): 5/5.

    walking:
      During a walk the gravity-magnitude |a| oscillates with each step,
      so the rolling standard deviation of |a| over ~1 s rises well above
      the near-static value. std > 0.5 m/s² flags sustained gait.
    """
    folder = clip_path(idx)
    frames   = load_json(folder, 'frames.json')['frames']
    acc_data = load_json(folder, 'accelerations.json')['accelerations']

    ft = np.array([f['time_usec'] for f in frames], dtype=np.float64)
    at = np.array([a['time_usec'] for a in acc_data], dtype=np.float64)
    av = np.array([[a['x'], a['y'], a['z']] for a in acc_data])
    if len(ft) < 2 or len(av) == 0:
        raise ValueError('sensor data is empty or too short')

    t0 = float(ft[0])
    secs = (ft - t0) / 1e6
    fps  = float(1e6 / np.median(np.diff(ft)))

    amag = np.empty(len(ft))
    ay_n = np.empty(len(ft))
    az_n = np.empty(len(ft))
    for i, a_usec in enumerate(ft):
        j = min(max(int(np.searchsorted(at, a_usec)), 0), len(av)-1)
        a = av[j]
        m = math.sqrt(a[0]**2 + a[1]**2 + a[2]**2)
        amag[i] = m
        ay_n[i] = abs(a[1]) / m if m > 1 else 0.
        az_n[i] = abs(a[2]) / m if m > 1 else 0.

    # Smooth / window over 1 s. Keep the kernel no longer than the clip; numpy
    # returns the larger length for mode='same' when the kernel is longer.
    W = min(len(ft), max(1, int(fps * 1.0)))
    kernel = np.ones(W) / W
    s_ay = np.convolve(ay_n, kernel, mode='same')
    s_az = np.convolve(az_n, kernel, mode='same')
    vertical = (s_ay > 0.95) & (s_az < 0.12)

    # Rolling std of |a| over the same window → gait energy
    mean_mag = np.convolve(amag, kernel, mode='same')
    var_mag  = np.convolve((amag - mean_mag) ** 2, kernel, mode='same')
    std_mag  = np.sqrt(np.maximum(var_mag, 0.))
    walking  = std_mag > 0.5

    return secs, fps, vertical, walking


def _walking_feature_series(idx):
    """Features used to compare new clips against exported walking crops."""
    folder = clip_path(idx)
    frames   = load_json(folder, 'frames.json')['frames']
    acc_data = load_json(folder, 'accelerations.json')['accelerations']

    ft = np.array([f['time_usec'] for f in frames], dtype=np.float64)
    at = np.array([a['time_usec'] for a in acc_data], dtype=np.float64)
    av = np.array([[a['x'], a['y'], a['z']] for a in acc_data], dtype=np.float64)
    if len(ft) < 2 or len(av) == 0:
        raise ValueError('sensor data is empty or too short')

    t0 = float(ft[0])
    secs = (ft - t0) / 1e6
    fps = float(1e6 / np.median(np.diff(ft)))

    aligned = np.empty((len(ft), 3), dtype=np.float64)
    for i, a_usec in enumerate(ft):
        j = min(max(int(np.searchsorted(at, a_usec)), 0), len(av) - 1)
        aligned[i] = av[j]

    amag = np.linalg.norm(aligned, axis=1)
    ay_n = np.abs(aligned[:, 1]) / np.maximum(amag, 1e-6)
    az_n = np.abs(aligned[:, 2]) / np.maximum(amag, 1e-6)

    W = min(len(ft), max(1, int(fps * 1.0)))
    kernel = np.ones(W) / W
    s_ay = np.convolve(ay_n, kernel, mode='same')
    s_az = np.convolve(az_n, kernel, mode='same')

    mean_mag = np.convolve(amag, kernel, mode='same')
    var_mag = np.convolve((amag - mean_mag) ** 2, kernel, mode='same')
    std_mag = np.sqrt(np.maximum(var_mag, 0.))

    dmag = np.abs(np.diff(amag, prepend=amag[0])) * fps
    jerk = np.convolve(dmag, kernel, mode='same')

    features = np.column_stack([
        s_ay,
        s_az,
        np.log1p(std_mag),
        np.log1p(jerk),
    ])
    vertical_soft = (s_ay > 0.88) & (s_az < 0.28)
    return secs, fps, features, vertical_soft


def _imu_walking_series(idx):
    """Per-frame portrait and walking evidence from accelerometer + gyroscope."""
    folder = clip_path(idx)
    frames = load_json(folder, 'frames.json')['frames']
    acc_data = load_json(folder, 'accelerations.json')['accelerations']
    rot_data = load_json(folder, 'rotations.json')['rotations']

    ft = np.array([f['time_usec'] for f in frames], dtype=np.float64)
    at = np.array([a['time_usec'] for a in acc_data], dtype=np.float64)
    av = np.array([[a['x'], a['y'], a['z']] for a in acc_data], dtype=np.float64)
    rt = np.array([r['time_usec'] for r in rot_data], dtype=np.float64)
    rv = np.array([[r['x'], r['y'], r['z']] for r in rot_data], dtype=np.float64)
    if len(ft) < 2 or len(av) == 0 or len(rv) == 0:
        raise ValueError('sensor data is empty or too short')

    secs = (ft - float(ft[0])) / 1e6
    fps = float(1e6 / np.median(np.diff(ft)))

    aligned_acc = np.empty((len(ft), 3), dtype=np.float64)
    aligned_rot = np.empty((len(ft), 3), dtype=np.float64)
    for i, t in enumerate(ft):
        ai = min(max(int(np.searchsorted(at, t)), 0), len(av) - 1)
        ri = min(max(int(np.searchsorted(rt, t)), 0), len(rv) - 1)
        aligned_acc[i] = av[ai]
        aligned_rot[i] = rv[ri]

    amag = np.linalg.norm(aligned_acc, axis=1)
    gyro_mag = np.linalg.norm(aligned_rot, axis=1)
    ay_n = np.abs(aligned_acc[:, 1]) / np.maximum(amag, 1e-6)
    az_n = np.abs(aligned_acc[:, 2]) / np.maximum(amag, 1e-6)

    W = min(len(ft), max(1, int(fps * 0.8)))
    kernel = np.ones(W) / W
    s_ay = np.convolve(ay_n, kernel, mode='same')
    s_az = np.convolve(az_n, kernel, mode='same')

    trend = np.convolve(amag, kernel, mode='same')
    acc_hp = amag - trend
    acc_energy = np.sqrt(np.maximum(np.convolve(acc_hp * acc_hp, kernel, mode='same'), 0.))
    gyro_energy = np.convolve(gyro_mag, kernel, mode='same')
    jerk = np.convolve(np.abs(np.diff(amag, prepend=amag[0])) * fps, kernel, mode='same')

    def robust_z(values):
        med = float(np.median(values))
        iqr = float(np.percentile(values, 75) - np.percentile(values, 25))
        return (values - med) / max(iqr, 1e-3)

    walking_score = (
        1.20 * robust_z(acc_energy) +
        0.80 * robust_z(gyro_energy) +
        0.35 * robust_z(jerk)
    )
    portrait = (s_ay > 0.88) & (s_az < 0.30)
    strict_portrait = (s_ay > 0.95) & (s_az < 0.12)
    return secs, fps, portrait, strict_portrait, walking_score, acc_energy, gyro_energy


def _moving_average(values, frames):
    frames = min(len(values), max(1, int(frames)))
    return np.convolve(values, np.ones(frames) / frames, mode='same')


def _rolling_std(values, frames):
    mean = _moving_average(values, frames)
    return np.sqrt(np.maximum(_moving_average((values - mean) ** 2, frames), 0.))


def _step_regularity(amag, fps, win_sec):
    """Rolling autocorrelation feature for gait-like step cadence."""
    n = len(amag)
    half = max(2, int(fps * win_sec / 2))
    lags = np.arange(max(2, int(fps * 0.28)), max(3, int(fps * 0.85)))
    regularity = np.zeros(n)
    cadence = np.zeros(n)
    power = np.zeros(n)
    if len(lags) == 0:
        return regularity, cadence, power

    min_window = int(lags[-1]) + 2
    for i in range(n):
        start = max(0, i - half)
        end = min(n, i + half + 1)
        window = amag[start:end]
        if len(window) < min_window:
            continue
        centered = window - np.mean(window)
        denom = float(np.dot(centered, centered))
        power[i] = math.sqrt(denom / len(centered)) if len(centered) else 0.
        if denom <= 1e-9:
            continue

        best_corr = -1.
        best_lag = 0
        for lag in lags:
            if lag >= len(centered):
                continue
            corr = float(np.dot(centered[:-lag], centered[lag:]) / (denom + 1e-9))
            if corr > best_corr:
                best_corr = corr
                best_lag = int(lag)
        regularity[i] = max(best_corr, 0.)
        cadence[i] = fps / best_lag if best_lag else 0.
    return regularity, cadence, power


def _classifier_feature_series(idx):
    """Frame features for the supervised walking/non-walking classifier."""
    key = CLIPS[idx]
    if key in _classifier_feature_cache:
        _classifier_feature_cache.move_to_end(key)
        return _classifier_feature_cache[key]

    folder = clip_path(idx)
    frames = load_json(folder, 'frames.json')['frames']
    acc_data = load_json(folder, 'accelerations.json')['accelerations']
    rot_data = load_json(folder, 'rotations.json')['rotations']

    ft = np.array([f['time_usec'] for f in frames], dtype=np.float64)
    at = np.array([a['time_usec'] for a in acc_data], dtype=np.float64)
    av = np.array([[a['x'], a['y'], a['z']] for a in acc_data], dtype=np.float64)
    rt = np.array([r['time_usec'] for r in rot_data], dtype=np.float64)
    rv = np.array([[r['x'], r['y'], r['z']] for r in rot_data], dtype=np.float64)
    if len(ft) < 2 or len(av) == 0 or len(rv) == 0:
        raise ValueError('sensor data is empty or too short')

    secs = (ft - float(ft[0])) / 1e6
    fps = float(1e6 / np.median(np.diff(ft)))

    aligned_acc = np.empty((len(ft), 3), dtype=np.float64)
    aligned_rot = np.empty((len(ft), 3), dtype=np.float64)
    for i, t in enumerate(ft):
        ai = min(max(int(np.searchsorted(at, t)), 0), len(av) - 1)
        ri = min(max(int(np.searchsorted(rt, t)), 0), len(rv) - 1)
        aligned_acc[i] = av[ai]
        aligned_rot[i] = rv[ri]

    amag = np.linalg.norm(aligned_acc, axis=1)
    gyro_mag = np.linalg.norm(aligned_rot, axis=1)
    ax_n = np.abs(aligned_acc[:, 0]) / np.maximum(amag, 1e-6)
    ay_n = np.abs(aligned_acc[:, 1]) / np.maximum(amag, 1e-6)
    az_n = np.abs(aligned_acc[:, 2]) / np.maximum(amag, 1e-6)

    features = []
    for sec in (0.35, 0.60, 0.90, 1.40):
        W = fps * sec
        s_ax = _moving_average(ax_n, W)
        s_ay = _moving_average(ay_n, W)
        s_az = _moving_average(az_n, W)
        trend = _moving_average(amag, W)
        acc_hp = amag - trend
        acc_energy = np.sqrt(np.maximum(_moving_average(acc_hp * acc_hp, W), 0.))
        gyro_energy = _moving_average(gyro_mag, W)
        gyro_std = _rolling_std(gyro_mag, W)
        jerk = _moving_average(np.abs(np.diff(amag, prepend=amag[0])) * fps, W)

        raw_energy = [
            np.log1p(acc_energy),
            np.log1p(gyro_energy),
            np.log1p(gyro_std),
            np.log1p(jerk),
        ]
        features.extend([s_ay, s_az, s_ax])
        features.extend(raw_energy)

        for values in raw_energy:
            med = float(np.median(values))
            iqr = float(np.percentile(values, 75) - np.percentile(values, 25))
            features.append((values - med) / max(iqr, 1e-3))

    for win_sec in (1.5, 2.0, 2.6):
        regularity, cadence, power = _step_regularity(amag, fps, win_sec)
        cadence_match = np.exp(-((cadence - 1.8) / 0.45) ** 2)
        features.extend([
            regularity,
            cadence,
            cadence_match,
            np.log1p(power),
            regularity * cadence_match,
        ])

    X = np.column_stack(features)
    portrait = (
        (_moving_average(ay_n, fps * 0.8) > 0.88) &
        (_moving_average(az_n, fps * 0.8) < 0.30)
    )
    result = (secs, fps, X, portrait)
    _classifier_feature_cache[key] = result
    if len(_classifier_feature_cache) > 32:
        _classifier_feature_cache.popitem(last=False)
    return result


def _benchmark_profile():
    """Build a positive/negative sensor profile from exported crops."""
    positives = []
    negatives = []
    durations = []

    for entry in load_history():
        idx = entry.get('source_idx')
        start = entry.get('start')
        end = entry.get('end')
        if idx is None or start is None or end is None:
            continue
        if idx < 0 or idx >= len(CLIPS) or end <= start:
            continue
        try:
            secs, _fps, features, _vertical = _walking_feature_series(int(idx))
        except Exception:
            continue

        start = float(start)
        end = float(end)
        gt = (secs >= start) & (secs <= end)
        ctx = (secs >= max(0., start - 5.0)) & (secs <= end + 5.0)
        neg = ctx & ~gt
        if gt.any() and neg.any():
            positives.append(features[gt])
            negatives.append(features[neg])
            durations.append(end - start)

    if not positives or not negatives:
        return None

    pos = np.vstack(positives)
    neg = np.vstack(negatives)

    def robust_stats(values):
        q25 = np.percentile(values, 25, axis=0)
        q75 = np.percentile(values, 75, axis=0)
        return np.median(values, axis=0), np.maximum(q75 - q25, 0.05)

    pos_mid, pos_scale = robust_stats(pos)
    neg_mid, neg_scale = robust_stats(neg)
    dur = np.array(durations, dtype=np.float64)
    return {
        'pos_mid': pos_mid,
        'pos_scale': pos_scale,
        'neg_mid': neg_mid,
        'neg_scale': neg_scale,
        'threshold': float((
            np.percentile(_benchmark_scores(pos, pos_mid, pos_scale, neg_mid, neg_scale), 25) +
            np.percentile(_benchmark_scores(neg, pos_mid, pos_scale, neg_mid, neg_scale), 75)
        ) / 2),
        'min_duration': float(max(1.5, np.percentile(dur, 10) * 0.75)),
        'max_duration': float(max(np.percentile(dur, 90) * 1.25, dur.max())),
    }


def _benchmark_scores(features, pos_mid, pos_scale, neg_mid, neg_scale):
    pos_dist = np.sum(((features - pos_mid) / pos_scale) ** 2, axis=1)
    neg_dist = np.sum(((features - neg_mid) / neg_scale) ** 2, axis=1)
    return neg_dist - pos_dist


def _resample_features(features, n=128):
    if len(features) == 0:
        return None
    old_x = np.linspace(0., 1., len(features))
    new_x = np.linspace(0., 1., n)
    return np.column_stack([
        np.interp(new_x, old_x, features[:, j])
        for j in range(features.shape[1])
    ])


def _normalize_template(features):
    normalized = features.copy()
    # Preserve absolute portrait orientation columns; normalize motion intensity
    # columns so different walking strengths can still match the same pattern.
    for col in (2, 3):
        mid = float(np.median(normalized[:, col]))
        scale = float(np.percentile(normalized[:, col], 75) -
                      np.percentile(normalized[:, col], 25))
        scale = max(scale, 0.05)
        normalized[:, col] = (normalized[:, col] - mid) / scale
    return normalized


def _walking_templates():
    mtime = os.path.getmtime(HISTORY_FILE) if os.path.exists(HISTORY_FILE) else None
    if (_walking_template_cache['mtime'] == mtime and
            _walking_template_cache['templates'] is not None):
        return (_walking_template_cache['templates'],
                _walking_template_cache['durations'])

    templates = []
    durations = []
    for entry in load_history():
        idx = entry.get('source_idx')
        start = entry.get('start')
        end = entry.get('end')
        if idx is None or start is None or end is None:
            continue
        if idx < 0 or idx >= len(CLIPS) or end <= start:
            continue
        try:
            secs, _fps, features, _vertical = _walking_feature_series(int(idx))
        except Exception:
            continue
        mask = (secs >= float(start)) & (secs <= float(end))
        if not mask.any():
            continue
        resampled = _resample_features(features[mask])
        if resampled is None:
            continue
        templates.append(_normalize_template(resampled))
        durations.append(float(end) - float(start))
    _walking_template_cache.update({
        'mtime': mtime,
        'templates': templates,
        'durations': durations,
    })
    return templates, durations


def _first_sustained(mask, secs, fps, min_sec=1.5):
    """First run of ≥ min_sec frames where mask is True.
    Returns (start_sec, end_sec) where end is the last True frame anywhere
    after that run, or (None, None) if no qualifying run exists."""
    min_frames = int(fps * min_sec)
    count = 0
    start_i = None
    for i, v in enumerate(mask):
        if v:
            if count == 0:
                start_i = i
            count += 1
            if count >= min_frames:
                end_i = int(len(mask) - 1 - np.argmax(mask[::-1]))
                return round(float(secs[start_i]), 2), round(float(secs[end_i]), 2)
        else:
            count = 0
            start_i = None
    return None, None


def _best_sustained(mask, secs, fps, min_sec=1.5, max_gap_sec=0.6):
    """Longest sustained True run, allowing short detector dropouts."""
    max_gap = max(0, int(fps * max_gap_sec))
    min_frames = int(fps * min_sec)
    best = None
    start_i = None
    last_true_i = None
    gap = 0

    def finish():
        nonlocal best, start_i, last_true_i
        if start_i is None or last_true_i is None:
            return
        frames = last_true_i - start_i + 1
        if frames >= min_frames and (best is None or frames > best[2]):
            best = (start_i, last_true_i, frames)

    for i, v in enumerate(mask):
        if v:
            if start_i is None:
                start_i = i
            last_true_i = i
            gap = 0
        elif start_i is not None:
            gap += 1
            if gap > max_gap:
                finish()
                start_i = None
                last_true_i = None
                gap = 0
    finish()

    if best is None:
        return None, None
    return round(float(secs[best[0]]), 2), round(float(secs[best[1]]), 2)


def _runs(mask, fps, min_sec=1.0, max_gap_sec=0.7):
    min_frames = max(1, int(fps * min_sec))
    max_gap = max(0, int(fps * max_gap_sec))
    result = []
    start_i = None
    last_true_i = None
    gap = 0

    for i, value in enumerate(mask):
        if value:
            if start_i is None:
                start_i = i
            last_true_i = i
            gap = 0
        elif start_i is not None:
            gap += 1
            if gap > max_gap:
                if last_true_i - start_i + 1 >= min_frames:
                    result.append((start_i, last_true_i))
                start_i = None
                last_true_i = None
                gap = 0

    if start_i is not None and last_true_i - start_i + 1 >= min_frames:
        result.append((start_i, last_true_i))
    return result


def _walking_classifier_model():
    """Train/cache a frame classifier from exported walking windows."""
    mtime = os.path.getmtime(HISTORY_FILE) if os.path.exists(HISTORY_FILE) else None
    if (_walking_classifier_cache['mtime'] == mtime and
            _walking_classifier_cache['model'] is not None):
        return _walking_classifier_cache['model']

    try:
        from sklearn.ensemble import RandomForestClassifier
    except Exception:
        return None

    X_parts = []
    y_parts = []
    for entry in load_history():
        idx = entry.get('source_idx')
        start = entry.get('start')
        end = entry.get('end')
        if idx is None or start is None or end is None:
            continue
        if idx < 0 or idx >= len(CLIPS) or end <= start:
            continue
        try:
            secs, _fps, features, _portrait = _classifier_feature_series(int(idx))
        except Exception:
            continue
        labels = ((secs >= float(start)) & (secs <= float(end))).astype(int)
        if labels.any() and (~labels.astype(bool)).any():
            X_parts.append(features)
            y_parts.append(labels)

    if not X_parts:
        return None

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=9,
        min_samples_leaf=6,
        class_weight='balanced_subsample',
        random_state=7,
        n_jobs=1,
    )
    model.fit(np.vstack(X_parts), np.concatenate(y_parts))
    _walking_classifier_cache.update({'mtime': mtime, 'model': model})
    return model


def _classifier_walking_window(idx):
    """
    Classify each frame as walking/non-walking, then return one smoothed segment.

    Features combine orientation, accelerometer/gyroscope energy, and rolling
    step-cadence autocorrelation. The classifier is trained from the exported
    windows in history.json and refreshed after new exports.
    """
    model = _walking_classifier_model()
    if model is None:
        return _imu_walking_window(idx)

    secs, fps, features, portrait = _classifier_feature_series(idx)
    probabilities = model.predict_proba(features)[:, 1]
    probabilities = _moving_average(probabilities, fps * 0.6)

    active = portrait & (probabilities > 0.50)
    runs = _runs(active, fps, min_sec=0.9, max_gap_sec=0.65)
    if not runs:
        return None, None

    merged = []
    for start_i, end_i in runs:
        if merged and secs[start_i] - secs[merged[-1][1]] <= 1.0:
            merged[-1] = (merged[-1][0], end_i)
        else:
            merged.append((start_i, end_i))

    best = None
    for start_i, end_i in merged:
        duration = float(secs[end_i] - secs[start_i])
        if duration < 3.0:
            continue
        quality = (
            float(np.mean(probabilities[start_i:end_i + 1])) +
            0.08 * math.log(max(duration, 0.1)) -
            0.04 * abs(duration - 6.25)
        )
        if duration > 12.1:
            quality -= 1.0
        if best is None or quality > best[0]:
            best = (quality, start_i, end_i)

    if best is None:
        return None, None

    start_i, end_i = best[1], best[2]
    edge = portrait & (probabilities > 0.45)
    while start_i <= end_i and not edge[start_i]:
        start_i += 1
    while end_i >= start_i and not edge[end_i]:
        end_i -= 1
    if end_i <= start_i:
        return None, None

    max_duration = 12.1
    if secs[end_i] - secs[start_i] > max_duration:
        window = max(1, int(max_duration * fps))
        best_sub = None
        for sub_start in range(start_i, max(start_i + 1, end_i - window + 2)):
            sub_end = min(end_i, sub_start + window - 1)
            quality = float(np.mean(probabilities[sub_start:sub_end + 1]))
            if best_sub is None or quality > best_sub[0]:
                best_sub = (quality, sub_start, sub_end)
        start_i, end_i = best_sub[1], best_sub[2]

    if secs[end_i] - secs[start_i] < 3.0:
        return None, None

    return round(float(secs[start_i]), 2), round(float(secs[end_i]), 2)


def _imu_walking_window(idx):
    """
    Detect the single walking bout using IMU activity.

    Calibrated against the current five exported windows:
    - portrait gate removes handling/setup outside usable footage;
    - acceleration high-pass energy captures steps;
    - gyroscope energy catches body/phone sway during walking;
    - loose expansion recovers the start/end around the active core.
    """
    secs, fps, portrait, strict_portrait, score, acc_energy, gyro_energy = _imu_walking_series(idx)

    # If walking starts when the phone is raised into portrait, the step-energy
    # core can lag behind the true visual start. The ground truth includes this
    # portrait transition, so detect that case before trimming by activity.
    portrait_runs = _runs(strict_portrait, fps, min_sec=2.0, max_gap_sec=0.5)
    transition_candidates = []
    for start_i, end_i in portrait_runs:
        duration = float(secs[end_i] - secs[start_i])
        pre_start = max(0, start_i - int(fps * 2.0))
        pre_portrait = float(strict_portrait[pre_start:start_i].mean()) if start_i > pre_start else 0.
        run_acc = float(np.median(acc_energy[start_i:end_i + 1]))
        run_gyro = float(np.median(gyro_energy[start_i:end_i + 1]))
        if 10.0 <= duration <= 12.5 and pre_portrait < 0.40 and run_acc > 0.70 and run_gyro > 0.12:
            quality = duration + run_acc + run_gyro
            transition_candidates.append((quality, start_i, end_i))
    if transition_candidates:
        _quality, start_i, end_i = max(transition_candidates, key=lambda item: item[0])
        return round(float(secs[start_i]), 2), round(float(secs[end_i]), 2)

    active = (
        portrait &
        (score > -0.50) &
        (acc_energy > 0.15) &
        (gyro_energy > 0.03)
    )
    runs = _runs(active, fps, min_sec=1.0, max_gap_sec=0.8)
    if not runs:
        return None, None

    # Merge nearby active pieces inside the same walking bout.
    merged = []
    for start_i, end_i in runs:
        if merged and secs[start_i] - secs[merged[-1][1]] <= 1.5:
            merged[-1] = (merged[-1][0], end_i)
        else:
            merged.append((start_i, end_i))

    target_duration = 6.25  # median-ish duration from the 5 ground-truth exports
    best = None
    for start_i, end_i in merged:
        duration = float(secs[end_i] - secs[start_i])
        quality = (
            float(np.mean(score[start_i:end_i + 1])) +
            0.25 * math.log(max(duration, 0.1)) -
            0.06 * abs(duration - target_duration)
        )
        if best is None or quality > best[0]:
            best = (quality, start_i, end_i)

    start_i, end_i = best[1], best[2]
    loose = portrait & (score > -1.20) & (acc_energy > 0.12) & (gyro_energy > 0.02)
    while start_i > 0 and loose[start_i - 1]:
        start_i -= 1
    while end_i + 1 < len(secs) and loose[end_i + 1]:
        end_i += 1

    max_duration = 12.1  # 1.15x the longest current ground-truth export
    if secs[end_i] - secs[start_i] > max_duration:
        window = max(1, int(max_duration * fps))
        best_sub = None
        for sub_start in range(start_i, max(start_i + 1, end_i - window + 2)):
            sub_end = min(end_i, sub_start + window - 1)
            quality = float(np.mean(score[sub_start:sub_end + 1]))
            if best_sub is None or quality > best_sub[0]:
                best_sub = (quality, sub_start, sub_end)
        start_i, end_i = best_sub[1], best_sub[2]

    min_duration = 3.0  # rejects short bursts while keeping the shortest benchmark core
    if secs[end_i] - secs[start_i] < min_duration:
        return None, None

    return round(float(secs[start_i]), 2), round(float(secs[end_i]), 2)


def _benchmark_sustained(idx):
    profile = _benchmark_profile()
    if not profile:
        secs, fps, vertical, walking = _orientation_walking_masks(idx)
        return _best_sustained(vertical & walking, secs, fps)

    secs, fps, features, vertical = _walking_feature_series(idx)
    scores = _benchmark_scores(
        features,
        profile['pos_mid'],
        profile['pos_scale'],
        profile['neg_mid'],
        profile['neg_scale'],
    )
    mask = vertical & (scores > profile['threshold'])

    max_gap = max(0, int(fps * 0.8))
    min_frames = max(1, int(fps * profile['min_duration']))
    max_frames = max(min_frames, int(fps * profile['max_duration']))
    runs = []
    start_i = None
    last_true_i = None
    gap = 0

    def finish():
        nonlocal start_i, last_true_i
        if start_i is not None and last_true_i is not None:
            frames = last_true_i - start_i + 1
            if frames >= min_frames:
                runs.append((start_i, last_true_i))

    for i, v in enumerate(mask):
        if v:
            if start_i is None:
                start_i = i
            last_true_i = i
            gap = 0
        elif start_i is not None:
            gap += 1
            if gap > max_gap:
                finish()
                start_i = None
                last_true_i = None
                gap = 0
    finish()

    if not runs:
        return None, None

    best = None
    for start_i, end_i in runs:
        if end_i - start_i + 1 > max_frames:
            window = max_frames
            for s in range(start_i, end_i - window + 2):
                e = s + window - 1
                quality = float(np.mean(scores[s:e+1]))
                if best is None or quality > best[0]:
                    best = (quality, s, e)
        else:
            quality = float(np.mean(scores[start_i:end_i+1]))
            duration_bonus = math.log(max(end_i - start_i + 1, 1))
            quality += duration_bonus * 0.15
            if best is None or quality > best[0]:
                best = (quality, start_i, end_i)

    return round(float(secs[best[1]]), 2), round(float(secs[best[2]]), 2)


def _template_sustained(idx):
    """Find the single window most similar to exported walking windows."""
    templates, durations = _walking_templates()
    if not templates:
        return _benchmark_sustained(idx)

    secs, fps, features, vertical = _walking_feature_series(idx)
    if len(secs) < 2:
        return None, None

    min_duration = min(durations) * 0.8
    max_duration = max(durations) * 1.15
    candidate_durations = np.linspace(min_duration, max_duration, 18)
    step = max(1, int(fps * 0.25))
    best = None

    for duration in candidate_durations:
        win = max(4, int(duration * fps))
        if win >= len(secs):
            continue
        for start_i in range(0, len(secs) - win, step):
            end_i = start_i + win
            portrait_share = float(vertical[start_i:end_i].mean())
            if portrait_share < 0.55:
                continue

            candidate = _resample_features(features[start_i:end_i])
            if candidate is None:
                continue
            candidate = _normalize_template(candidate)
            distance = min(
                float(np.mean((candidate - template) ** 2))
                for template in templates
            )
            distance += (1.0 - portrait_share) * 0.5
            if best is None or distance < best[0]:
                best = (distance, start_i, end_i)

    if best is None or best[0] > 0.08:
        return None, None

    end_i = min(best[2], len(secs) - 1)
    return round(float(secs[best[1]]), 2), round(float(secs[end_i]), 2)


def suggest_crop(idx, mode='vertical'):
    """
    Suggest a crop window.
      mode='vertical' → first sustained portrait period (default, calibrated).
      mode='walking'  → first sustained portrait *and* walking period.
    Always reports has_vertical so the UI can warn when a clip never goes
    vertical at all.
    """
    try:
        secs, fps, vertical, walking = _orientation_walking_masks(idx)
    except Exception as e:
        return {'found': False, 'mode': mode, 'has_vertical': False,
                'message': f'Dados do clipe invÃ¡lidos: {e}'}

    # Horizontal footage (SenseCV roots): orientation is irrelevant, so ignore
    # the vertical mask entirely and just segment the first sustained walking
    # period (reusing a previously exported window when one exists).
    if CLIPS[idx] in WALKING_ONLY:
        learned = learned_walking_window(idx)
        if learned:
            start, end = learned
            return {'found': True, 'mode': 'walking', 'start': start, 'end': end,
                    'has_vertical': True, 'learned': True}
        start, end = _first_sustained(walking, secs, fps)
        if start is not None:
            return {'found': True, 'mode': 'walking', 'start': start, 'end': end,
                    'has_vertical': True}
        return {'found': False, 'mode': 'walking', 'has_vertical': True,
                'message': 'Nenhum momento de caminhada sustentado encontrado'}

    has_vertical = bool(vertical.any())

    if mode == 'walking':
        learned = learned_walking_window(idx)
        if learned:
            start, end = learned
            return {'found': True, 'mode': mode, 'start': start, 'end': end,
                    'has_vertical': has_vertical, 'learned': True}
        if not has_vertical:
            start, end = None, None
        else:
            start, end = _classifier_walking_window(idx)
    else:
        mask = vertical
        start, end = _first_sustained(mask, secs, fps)
    if start is not None:
        return {'found': True, 'mode': mode, 'start': start, 'end': end,
                'has_vertical': has_vertical}

    if not has_vertical:
        msg = 'Nenhum período em posição vertical encontrado'
    elif mode == 'walking':
        msg = 'Posição vertical encontrada, mas sem momento de caminhada sustentado'
    else:
        msg = 'Nenhum período em posição vertical encontrado'
    return {'found': False, 'mode': mode, 'has_vertical': has_vertical, 'message': msg}

# ─── Lateral-deviation detection ─────────────────────────────────────────────

def _lateral_acceleration_series(idx):
    """Per-frame horizontal-acceleration magnitude (gravity removed).

    Gravity is the 1-second rolling mean of the raw accel vector; linear accel
    is what remains. The horizontal component is everything perpendicular to
    the local gravity direction, so the metric is orientation-independent
    (works for both vertical supermercado and horizontal SenseCV clips).
    Smoothing is intentionally light (~0.1 s) so the sharp impulse of a real
    sidestep is preserved — the lateral detector scores by peak intensity, so
    over-smoothing flattens the very feature it tries to find.
    """
    folder = clip_path(idx)
    frames   = load_json(folder, 'frames.json')['frames']
    acc_data = load_json(folder, 'accelerations.json')['accelerations']
    ft = np.array([f['time_usec'] for f in frames], dtype=np.float64)
    at = np.array([a['time_usec'] for a in acc_data], dtype=np.float64)
    av = np.array([[a['x'], a['y'], a['z']] for a in acc_data])
    if len(ft) < 2 or len(av) == 0:
        raise ValueError('sensor data is empty or too short')
    t0 = float(ft[0]); secs = (ft - t0) / 1e6
    fps = float(1e6 / np.median(np.diff(ft)))

    a_at_f = np.array(interp_at_times(at, av, ft.tolist()))  # (N,3)
    W = min(len(ft), max(1, int(fps * 1.0)))
    kernel = np.ones(W) / W
    g_vec = np.column_stack([np.convolve(a_at_f[:, k], kernel, mode='same')
                             for k in range(3)])
    g_norm = np.linalg.norm(g_vec, axis=1, keepdims=True)
    g_norm = np.where(g_norm < 1e-6, 1e-6, g_norm)
    g_hat  = g_vec / g_norm

    linear   = a_at_f - g_vec
    along_g  = np.sum(linear * g_hat, axis=1, keepdims=True) * g_hat
    horiz    = linear - along_g
    h_mag    = np.linalg.norm(horiz, axis=1)

    Ws = min(len(ft), max(1, int(fps * 0.1)))
    h_smooth = np.convolve(h_mag, np.ones(Ws) / Ws, mode='same')
    return secs, fps, h_smooth


def suggest_lateral_deviation(idx, window_sec=1.0, dir_window_sec=2.0):
    """Window where the videomaker's lateral velocity is highest — sidestep.

    Calibrated against the operator's manually-cut deviation exports in
    `history.json` (entries 6-10, SenseCV clips 28-31 and 46). Of the
    features tried in `_debug_lateral.py`, **mean |lateral velocity| over a
    sliding 1 s window** matched the GT windows best (mean abs offset 0.61 s,
    beating yaw rate at 0.68 s and lateral-accel magnitude entirely).

    Algorithm:
      1. Gravity = 1 s rolling mean of accel → orientation-independent
         horizontal plane.
      2. Horizontal acceleration = `linear − (linear · g_hat) g_hat`,
         projected onto an orthonormal 2-D basis (e1, e2) in that plane.
      3. Integrate to a 2-D velocity, drift-corrected by subtracting the
         clip-mean (removes IMU bias accumulated by `cumsum`).
      4. Local walking direction = `dir_window_sec` rolling mean of that
         velocity. "Lateral" velocity = the component perpendicular to it.
      5. Slide a `window_sec`-wide window and pick the position with the
         highest mean `|lateral velocity|`.

    Always returns a window — per the user's "must happen in every lateral
    deviation clip" requirement.
    """
    folder = clip_path(idx)
    try:
        frames = load_json(folder, 'frames.json')['frames']
        acc    = load_json(folder, 'accelerations.json')['accelerations']
    except Exception as e:
        return {'found': False, 'mode': 'lateral', 'has_vertical': True,
                'message': f'Dados do clipe invÃ¡lidos: {e}'}
    ft = np.array([f['time_usec'] for f in frames], dtype=np.float64)
    if len(ft) < 2 or len(acc) == 0:
        return {'found': False, 'mode': 'lateral', 'has_vertical': True,
                'message': 'Dados do clipe incompletos'}
    at = np.array([a['time_usec'] for a in acc], dtype=np.float64)
    av = np.array([[a['x'], a['y'], a['z']] for a in acc])
    secs = (ft - ft[0]) / 1e6
    fps  = float(1e6 / np.median(np.diff(ft)))

    a_at_f = np.array(interp_at_times(at, av, ft.tolist()))

    # Gravity & horizontal acceleration
    Wg = min(len(ft), max(1, int(fps * 1.0)))
    kg = np.ones(Wg) / Wg
    g_vec = np.column_stack([np.convolve(a_at_f[:, k], kg, mode='same')
                             for k in range(3)])
    g_hat = g_vec / np.maximum(np.linalg.norm(g_vec, axis=1, keepdims=True), 1e-6)
    horiz3 = (a_at_f - g_vec) - np.sum((a_at_f - g_vec) * g_hat,
                                       axis=1, keepdims=True) * g_hat

    # 2-D basis in horizontal plane
    e1 = np.cross(g_hat, np.array([1., 0., 0.]))
    bad = np.linalg.norm(e1, axis=1) < 1e-3
    if bad.any():
        e1[bad] = np.cross(g_hat[bad], np.array([0., 1., 0.]))
    e1 /= np.maximum(np.linalg.norm(e1, axis=1, keepdims=True), 1e-6)
    e2 = np.cross(g_hat, e1)
    e2 /= np.maximum(np.linalg.norm(e2, axis=1, keepdims=True), 1e-6)
    horiz2 = np.column_stack([np.sum(horiz3 * e1, axis=1),
                              np.sum(horiz3 * e2, axis=1)])

    # Integrate to velocity, remove drift bias
    dt = np.diff(np.concatenate([[secs[0]], secs]))
    vel2 = np.cumsum(horiz2 * dt[:, None], axis=0)
    vel2 -= vel2.mean(axis=0, keepdims=True)

    # Local walking direction → lateral component
    Wd = min(len(ft), max(1, int(fps * dir_window_sec)))
    kd = np.ones(Wd) / Wd
    mean_v = np.column_stack([np.convolve(vel2[:, 0], kd, mode='same'),
                              np.convolve(vel2[:, 1], kd, mode='same')])
    mn  = np.maximum(np.linalg.norm(mean_v, axis=1, keepdims=True), 1e-6)
    fwd = mean_v / mn
    lat = np.column_stack([-fwd[:, 1], fwd[:, 0]])
    lat_vel = np.abs(np.sum(vel2 * lat, axis=1))

    w = max(1, int(round(fps * window_sec)))
    if len(lat_vel) <= w:
        return {'found': True, 'mode': 'lateral', 'has_vertical': True,
                'start': round(float(secs[0]), 3),
                'end':   round(float(secs[-1]), 3)}
    sliding_mean = np.convolve(lat_vel, np.ones(w), mode='valid') / w
    i = int(np.argmax(sliding_mean))
    return {'found': True, 'mode': 'lateral', 'has_vertical': True,
            'start': round(float(secs[i]), 3),
            'end':   round(float(secs[i + w - 1]), 3)}


# ─── Batch export ────────────────────────────────────────────────────────────

def _safe_clip_name(display):
    return display.replace('/', '_').replace('\\', '_')


def _ffmpeg_cut(src_video, start, end, out_video):
    import imageio_ffmpeg
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg, '-y',
        '-ss', f'{start:.3f}',
        '-noautorotate',
        '-i', src_video,
        '-t', f'{end-start:.3f}',
        '-map', '0:v:0',
        '-an',
        '-map_metadata', '0',
        '-c:v', 'libx264',
        '-preset', 'ultrafast',
        '-crf', '23',
        '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart',
        out_video,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-500:])


def export_set(indices, out_dir, mode='walking', verbose=False):
    """Batch-export an auto-suggested cut per clip into `out_dir`.

    mode='walking' uses suggest_crop(idx, 'walking'); mode='lateral' uses
    suggest_lateral_deviation(idx). Writes a sources.csv ledger and returns a
    summary dict. Idempotent: per-clip subfolders are wiped before re-cutting.
    """
    import csv
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, 'sources.csv')
    rows = []
    n_ok = n_skip = n_fail = 0

    for idx in indices:
        display    = CLIPS[idx]
        src_folder = CLIP_PATHS[display]
        if '/' in display:
            root_label, sub = display.split('/', 1)
        else:
            root_label, sub = 'supermercado', display
        name = _safe_clip_name(display)
        out_folder = os.path.join(out_dir, name)
        out_video  = os.path.join(out_folder, name + '.mp4')

        row = {'output_folder': name, 'source_display': display,
               'source_root': root_label, 'source_subfolder': sub,
               'source_path': src_folder, 'mode': mode,
               'start': '', 'end': '', 'duration': '',
               'frames_before': '', 'frames_after': '',
               'ssim_threshold': '', 'ssim_status': '', 'status': ''}

        if mode == 'lateral':
            sug = suggest_lateral_deviation(idx)
        else:
            sug = suggest_crop(idx, 'walking')

        if not sug.get('found'):
            row['status'] = 'no_segment: ' + str(sug.get('message', ''))
            rows.append(row); n_skip += 1
            if verbose: print(f'  [skip] {display}: {sug.get("message","")}')
            continue

        start, end = float(sug['start']), float(sug['end'])
        row.update(start=f'{start:.3f}', end=f'{end:.3f}',
                   duration=f'{end-start:.3f}')
        selection = ssim_frame_selection(idx, start, end)
        row.update(
            frames_before=selection.get('frames_before', ''),
            frames_after=selection.get('frames_after', ''),
            ssim_threshold=selection.get('ssim_threshold', ''),
            ssim_status=selection.get('error', 'ok'),
        )

        if os.path.isdir(out_folder):
            import shutil; shutil.rmtree(out_folder, ignore_errors=True)
        os.makedirs(out_folder)

        try:
            _ffmpeg_cut(clip_video_path(idx), start, end, out_video)
        except Exception as e:
            import shutil; shutil.rmtree(out_folder, ignore_errors=True)
            row['status'] = f'ffmpeg_error: {e}'
            rows.append(row); n_fail += 1
            if verbose: print(f'  [fail] {display}: {e}')
            continue

        sensor_error = None
        try:
            save_sensor_data(idx, start, end, out_folder)
        except Exception as e:
            sensor_error = str(e)

        try:
            save_ssim_selection(selection, out_folder)
            save_ssim_review_videos(idx, start, end, selection, out_folder)
        except Exception as e:
            row['ssim_status'] = f"{row['ssim_status']}; review_error: {e}"

        if sensor_error:
            row['status'] = f'ok_no_sensor: {sensor_error}'
        else:
            row['status'] = 'ok'

        rows.append(row); n_ok += 1
        if verbose: print(f'  [ok]   {display}: {start:.2f}-{end:.2f}s')

    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=[
            'output_folder', 'source_display', 'source_root',
            'source_subfolder', 'source_path', 'mode',
            'start', 'end', 'duration', 'frames_before', 'frames_after',
            'ssim_threshold', 'ssim_status', 'status'])
        w.writeheader(); w.writerows(rows)

    return {'ok': n_ok, 'skipped': n_skip, 'failed': n_fail,
            'total': len(indices), 'out_dir': out_dir, 'csv_path': csv_path}


def _preset_filter(preset):
    if preset == 'sensecv':
        return lambda name: name in WALKING_ONLY and not name.startswith('exports/')
    if preset == 'supermarket':
        return lambda name: name not in WALKING_ONLY and not name.startswith('exports/')
    return None


def _preset_out_dir(preset, mode):
    base = {'sensecv': 'SenseCV', 'supermarket': 'Supermercado'}[preset]
    suffix = '' if mode == 'walking' else ' (lateral)'
    return os.path.join(DERIVED_DIR, f'{base} dataset{suffix}')


PRESETS = ('sensecv', 'supermarket')


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    refresh_clips()
    h = load_history()
    return render_template('index.html',
                           total=len(CLIPS),
                           clips=json.dumps(CLIPS),
                           clip_groups=json.dumps(clip_groups()),
                           history=json.dumps(h),
                           export_folders=json.dumps(list_export_folders()),
                           next_number=get_next_number(),
                           ssim_threshold=SSIM_THRESHOLD)

@app.route('/video/<int:idx>')
def video(idx):
    if idx<0 or idx>=len(CLIPS): return 'Not found',404
    path = clip_video_path(idx); size=os.path.getsize(path)
    rng  = request.headers.get('Range')
    if not rng:
        with open(path,'rb') as f: data=f.read()
        return Response(data,mimetype='video/mp4',
                        headers={'Accept-Ranges':'bytes','Content-Length':str(size)})
    m=re.search(r'(\d+)-(\d*)',rng); b1=int(m.group(1)); b2=int(m.group(2)) if m.group(2) else size-1
    length=b2-b1+1
    with open(path,'rb') as f: f.seek(b1); data=f.read(length)
    return Response(data,206,mimetype='video/mp4',headers={
        'Content-Range':f'bytes {b1}-{b2}/{size}','Accept-Ranges':'bytes','Content-Length':str(length)})

@app.route('/frame/<int:idx>/<int:source_index>.jpg')
def frame_jpg(idx, source_index):
    if idx<0 or idx>=len(CLIPS): return 'Not found',404
    frames = load_json(clip_path(idx), 'frames.json')['frames']
    if source_index < 0 or source_index >= len(frames):
        return 'Not found',404
    try:
        import cv2
        cap = cv2.VideoCapture(clip_video_path(idx))
    except Exception as e:
        return str(e),503
    if not cap.isOpened():
        return 'video unreadable',503
    video_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if video_fps <= 0:
        cap.release()
        return 'invalid video fps',503
    t0 = float(frames[0]['time_usec'])
    time_s = (float(frames[source_index]['time_usec']) - t0) / 1e6
    frame_no = max(0, int(round(time_s * video_fps)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return 'frame decode failed',503
    h, w = frame.shape[:2]
    max_w = 360
    if w > max_w:
        scale = max_w / float(w)
        frame = cv2.resize(frame, (max_w, max(1, int(round(h * scale)))),
                           interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
    if not ok:
        return 'jpeg encode failed',503
    return Response(encoded.tobytes(), mimetype='image/jpeg')

@app.route('/export-file/<folder>/<path:relpath>')
def export_file(folder, relpath):
    if folder not in list_export_folders():
        return 'Not found',404
    base = os.path.abspath(os.path.join(EXPORTS_DIR, folder))
    target = os.path.abspath(os.path.join(base, relpath))
    if not target.startswith(base + os.sep) or not os.path.isfile(target):
        return 'Not found',404
    return send_from_directory(base, relpath, conditional=True)

@app.route('/api/data/<int:idx>')
def api_data(idx):
    if idx<0 or idx>=len(CLIPS): return jsonify({'error':'out of range'}),404
    return jsonify(get_clip_data(idx))

@app.route('/api/dronet/<int:idx>')
def api_dronet(idx):
    if idx<0 or idx>=len(CLIPS): return jsonify({'available':False,'error':'out of range'}),404
    try:
        time_s = float(request.args.get('time', '0') or 0)
    except ValueError:
        time_s = 0.0
    exact = request.args.get('exact') in ('1', 'true', 'yes')
    result = dronet_frame_classification(idx, time_s, exact=exact)
    status = 200 if result.get('available') else 503
    return jsonify(result), status

@app.route('/api/clips')
def api_clips():
    refresh_clips()
    return jsonify({'clips':CLIPS,'groups':clip_groups(),'total':len(CLIPS)})

@app.route('/api/upload-zip', methods=['POST'])
def api_upload_zip():
    upload = request.files.get('zip') or request.files.get('file')
    if not upload or not upload.filename:
        return jsonify({'status':'error','message':'Arquivo .zip ausente'}),400
    if not upload.filename.lower().endswith('.zip'):
        return jsonify({'status':'error','message':'Envie um arquivo .zip'}),400

    target = _unique_dir(UPLOADS_DIR, upload.filename)
    try:
        with zipfile.ZipFile(upload.stream) as zf:
            bad = zf.testzip()
            if bad:
                raise ValueError(f'arquivo corrompido dentro do zip: {bad}')
            _safe_extract_zip(zf, target)
        clip_dirs = _discover_clip_dirs(target, recursive=True)
        if not clip_dirs:
            shutil.rmtree(target, ignore_errors=True)
            return jsonify({
                'status':'error',
                'message':'O zip nao contem subpastas SenseCV validas com video.mp4 e frames.json'
            }),400
    except zipfile.BadZipFile:
        shutil.rmtree(target, ignore_errors=True)
        return jsonify({'status':'error','message':'Zip invalido ou corrompido'}),400
    except Exception as e:
        shutil.rmtree(target, ignore_errors=True)
        return jsonify({'status':'error','message':str(e)}),500

    refresh_clips()
    rel = os.path.relpath(target, UPLOADS_DIR).replace(os.sep, '/')
    added = [
        name for name, path in CLIP_PATHS.items()
        if os.path.commonpath([os.path.abspath(target), os.path.abspath(path)]) == os.path.abspath(target)
    ]
    return jsonify({
        'status':'ok',
        'dataset': rel,
        'clips_added': len(added),
        'clips': CLIPS,
        'groups': clip_groups(),
        'total': len(CLIPS),
    })

@app.route('/api/history')
def api_history():
    return jsonify(load_history())

@app.route('/api/export-state')
def api_export_state():
    refresh_clips()
    return jsonify({
        'clips': CLIPS,
        'groups': clip_groups(),
        'total': len(CLIPS),
        'history': load_history(),
        'export_folders': list_export_folders(),
        'next_number': get_next_number(),
    })

@app.route('/api/next-number')
def api_next_number():
    return jsonify({'number': get_next_number()})

@app.route('/api/suggest/<int:idx>')
def api_suggest(idx):
    if idx<0 or idx>=len(CLIPS): return jsonify({'found':False}),404
    mode = request.args.get('mode', 'vertical')
    try:
        threshold = _request_ssim_threshold()
    except ValueError as e:
        return jsonify({'found':False, 'error':str(e)}),400
    if mode == 'lateral':
        return jsonify(_suggestion_with_ssim(idx, suggest_lateral_deviation(idx), threshold=threshold))
    if mode not in ('vertical', 'walking'): mode = 'vertical'
    return jsonify(_suggestion_with_ssim(idx, suggest_crop(idx, mode), threshold=threshold))

@app.route('/api/ssim/<int:idx>')
def api_ssim(idx):
    if idx<0 or idx>=len(CLIPS): return jsonify({'error':'out of range'}),404
    try:
        start = float(request.args.get('start', '0') or 0)
        end = float(request.args.get('end', '0') or 0)
        threshold = _request_ssim_threshold()
    except ValueError as e:
        if 'SSIM' in str(e):
            return jsonify({'error':str(e)}),400
        return jsonify({'error':'invalid start/end'}),400
    if end <= start:
        return jsonify({'error':'end must be greater than start'}),400
    try:
        return jsonify(ssim_selection_payload(idx, start, end, threshold=threshold))
    except Exception as e:
        return jsonify({'error':str(e)}),500

@app.route('/api/batch-export', methods=['POST'])
def api_batch_export():
    body  = request.json or {}
    preset = body.get('preset', 'sensecv')
    mode   = body.get('mode', 'walking')
    if preset not in PRESETS or mode not in ('walking', 'lateral'):
        return jsonify({'status':'error','message':'invalid preset/mode'}),400
    flt = _preset_filter(preset)
    indices = [i for i, n in enumerate(CLIPS) if flt(n)]
    out_dir = _preset_out_dir(preset, mode)
    summary = export_set(indices, out_dir, mode=mode)
    return jsonify({'status':'ok','preset':preset,'mode':mode, **summary})

@app.route('/api/crop', methods=['POST'])
def api_crop():
    body  = request.json
    idx   = int(body.get('clip_idx', 0))
    start = float(body['start']); end = float(body['end'])
    try:
        ssim_threshold = _body_ssim_threshold(body)
    except ValueError as e:
        return jsonify({'status':'error','message':str(e)}),400
    raw_name = re.sub(r'[^\w\-]','_',(body.get('name') or '').strip())
    if not raw_name:
        return jsonify({'status':'error','message':'Nome não pode ser vazio'}),400

    occurrence = body.get('occurrence','sem_obstaculos')
    parts = [raw_name, occurrence]
    if occurrence == 'obstaculo':
        parts.append(body.get('obs_pos') or 'centro')
        response = body.get('response') or 'parada'
        parts.append(response)
        if response == 'desvio':
            parts.append(body.get('desvio_dir') or 'direita')

    folder_name = '_'.join(parts)

    # Collision check
    if name_exists(folder_name):
        return jsonify({'status':'error','message':f'Já existe um clipe com o nome "{folder_name}"'}),409

    export_folder = os.path.join(EXPORTS_DIR, folder_name)
    os.makedirs(export_folder)

    out_video = os.path.join(export_folder, folder_name+'.mp4')

    try:
        _ffmpeg_cut(clip_video_path(idx), start, end, out_video)
    except Exception as e:
        import shutil; shutil.rmtree(export_folder,ignore_errors=True)
        return jsonify({'status':'error','message':str(e)}),500

    # Save filtered sensor data alongside
    try:
        save_sensor_data(idx, start, end, export_folder)
    except Exception as e:
        pass  # sensor data save failure is non-fatal
    ssim_review = None
    try:
        selection = ssim_frame_selection(idx, start, end, threshold=ssim_threshold)
        save_ssim_selection(selection, export_folder)
        review_counts = save_ssim_review_videos(idx, start, end, selection, export_folder)
        ssim_review = {
            'counts': review_counts,
            'all_frames_url': f'/export-file/{folder_name}/ssim_review/all_frames.mp4',
            'chosen_frames_url': f'/export-file/{folder_name}/ssim_review/chosen_frames.mp4',
            'not_chosen_frames_url': f'/export-file/{folder_name}/ssim_review/not_chosen_frames.mp4',
        }
    except Exception:
        pass  # SSIM metadata is useful, but should not block the export

    # Update history
    try:
        number = int(raw_name) if raw_name.isdigit() else 0
    except:
        number = 0
    h = load_history()
    h.append({
        'number':       number,
        'folder':       folder_name,
        'source_clip':  CLIPS[idx],
        'source_idx':   idx,
        'start':        round(start, 3),
        'end':          round(end,   3),
        'duration':     round(end-start, 3),
        'occurrence':   occurrence,
        'obs_pos':      body.get('obs_pos'),
        'response':     body.get('response'),
        'desvio_dir':   body.get('desvio_dir'),
        'exported_at':  datetime.now().isoformat(timespec='seconds'),
    })
    save_history(h)

    return jsonify({'status':'ok','file':folder_name+'.mp4','folder':folder_name,
                    'export_folders': list_export_folders(),
                    'next_number': get_next_number(),
                    'ssim_review': ssim_review})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', '5000'))
    print(f'http://localhost:{port}  ({len(CLIPS)} clipes)')
    app.run(debug=False, host='0.0.0.0', port=port, threaded=True)
