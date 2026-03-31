import os
import pickle
import json
import random
import numpy as np
from collections import defaultdict
from PIL import Image, ImageDraw
from tqdm import tqdm
from typing import Optional, Dict, List, Tuple, Any


from src.config.constants import (
    TRAJ_INDEX, TRAJ_INDEX_EXPANDED, TRAJ_DIM, TRAJ_DIM_EXPANDED, PEN_CLASS
)
from src.data.data_utils import guarantee_stroke_endpoints

REP_PRIORITY = ["isolated", "connected_right", "connected_left", "connected_both", "unknown"]


REP_TARGET_SET = (
    [chr(c) for c in range(ord('a'), ord('z')+1)] +
    [chr(c) for c in range(ord('A'), ord('Z')+1)] +
    [str(d) for d in range(10)] +
    list("!\"#$%&'()*+,-./:;<=>?@[\\]^_{|}~`")
)


DSD_PROHIBITED_SAMPLES = {
    ('5', '118'),
    ('7', '14'),
    ('7', '101'),
    ('7', '58'),
    ('14', '20'),
    ('22', '45'),
    ('30', '45'),
    ('40', '85'),
    ('50', '45'),
    ('59', '29'),
    ('96', '120'),
    ('99', '134'),
    ('140', '35'),
    ('144', '55'),
    ('144', '91'),
    ('144', '28'),
    ('144', '69'),
}


def ensure_dir(path: str):
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def _save_style_rep_debug_images(style_root: str, writer_id: str, output_dir: str, prefix: str) -> bool:
    rep_pkl_path = os.path.join(style_root, f"{writer_id}_rep.pkl")
    if not os.path.exists(rep_pkl_path):
        print(f"[DEBUG] {prefix}: _rep.pkl not found: {rep_pkl_path}")
        return False

    try:
        with open(rep_pkl_path, "rb") as f:
            rep_map = pickle.load(f)
    except Exception as e:
        print(f"[DEBUG] {prefix}: failed to load {rep_pkl_path}: {e}")
        return False

    images = []
    labels = []
    for char, sample in sorted(rep_map.items()):
        if isinstance(sample, list) and len(sample) > 0:
            sample = sample[0]
        if not isinstance(sample, dict):
            continue
        img = sample.get("image")
        if img is None:
            continue
        arr = np.asarray(img)
        if arr.ndim == 3:
            arr = arr[:, :, 0]
        if arr.ndim != 2:
            continue
        images.append(arr.astype(np.uint8, copy=False))
        labels.append(str(char))

    if not images:
        print(f"[DEBUG] {prefix}: no representative images")
        return False

    cols = min(10, len(images))
    rows = (len(images) + cols - 1) // cols
    h, w = images[0].shape[:2]
    cell_h = h + 20
    cell_w = w + 4
    grid = np.ones((rows * cell_h, cols * cell_w), dtype=np.uint8) * 255

    for idx, img in enumerate(images):
        r = idx // cols
        c = idx % cols
        y = r * cell_h + 18
        x = c * cell_w + 2
        grid[y:y + h, x:x + w] = img

    grid_pil = Image.fromarray(grid)
    draw = ImageDraw.Draw(grid_pil)
    for idx, label in enumerate(labels):
        r = idx // cols
        c = idx % cols
        y = r * cell_h + 2
        x = c * cell_w + 2
        if label.isascii():
            draw.text((x, y), label, fill=0)

    ensure_dir(output_dir)
    output_path = os.path.join(output_dir, f"{prefix}_wid{writer_id}_style_rep.png")
    grid_pil.save(output_path)
    print(f"[DEBUG] style-rep saved: {output_path} ({len(images)} chars)")
    return True

def match_resample_type(fname: str, resample_type: str) -> bool:
    if resample_type == "original":
        return fname.isdigit()
    return fname.endswith(f"_{resample_type}")

def strip_suffix(fname: str, resample_type: str) -> str:
    return fname if resample_type == "original" else fname.replace(f"_{resample_type}", "")

def is_dsd_prohibited_sample(writer_id: str, fname: str, resample_type: str) -> bool:

    base_fname = strip_suffix(fname, resample_type)
    return (writer_id, base_fname) in DSD_PROHIBITED_SAMPLES

def filter_writers_by_split(
    writers: List[str],
    split_mode: str,
    split_train_max: int = 149,
    split_test_min: int = 150,
    split_test_max: int = 169,
) -> List[str]:
    numeric_writers: List[int] = []
    for w in writers:
        try:
            numeric_writers.append(int(w))
        except (TypeError, ValueError):
            continue

    if split_mode == "train":
        return [str(w) for w in numeric_writers if w <= int(split_train_max)]
    elif split_mode == "test":
        return [str(w) for w in numeric_writers if int(split_test_min) <= w <= int(split_test_max)]
    elif split_mode == "all":
        return [str(w) for w in numeric_writers]
    else:
        raise ValueError(f"Invalid split_mode: {split_mode}. Must be 'train', 'test', or 'all'.")


def _convert_old_to_new_format(traj_old: np.ndarray) -> np.ndarray:
    if traj_old.shape[1] != 5:
        raise ValueError(f"Expected old format [T,5], got {traj_old.shape}")

    pm_old = traj_old[:, 2] > 0.5
    eos_old = traj_old[:, 3] > 0.5
    eoc_old = traj_old[:, 4] > 0.5

    pen_class = np.zeros(len(traj_old), dtype=np.int64)


    pen_class[pm_old] = PEN_CLASS["PM"]
    pen_class[eos_old] = PEN_CLASS["PU"]
    pen_class[eoc_old] = PEN_CLASS["EOC"]

    traj_new = np.concatenate([
        traj_old[:, :2],
        pen_class.reshape(-1, 1).astype(np.float32)
    ], axis=1)

    return traj_new.astype(np.float32)

def _expand_pen_class_to_onehot(traj_compact: np.ndarray) -> np.ndarray:
    if traj_compact.shape[1] != 3:
        raise ValueError(f"Expected compact format [T,3], got {traj_compact.shape}")

    T = traj_compact.shape[0]
    pen_class = traj_compact[:, 2].astype(np.int64)

    expanded = np.zeros((T, 6), dtype=np.float32)
    expanded[:, :2] = traj_compact[:, :2]


    for i in range(T):
        pc = pen_class[i]
        if pc == PEN_CLASS["PM"]:
            expanded[i, TRAJ_INDEX_EXPANDED["PM"]] = 1.0
        elif pc == PEN_CLASS["PU"]:
            expanded[i, TRAJ_INDEX_EXPANDED["PU"]] = 1.0
        elif pc == PEN_CLASS["CURSIVE_EOC"]:
            expanded[i, TRAJ_INDEX_EXPANDED["CURSIVE_EOC"]] = 1.0
        elif pc == PEN_CLASS["EOC"]:
            expanded[i, TRAJ_INDEX_EXPANDED["EOC"]] = 1.0

    return expanded


def _assert_shape_and_dtype(traj: np.ndarray, where: str):
    assert traj.ndim == 2 and traj.shape[1] == TRAJ_DIM, f"[{where}] traj must be [T,3], got {traj.shape}"
    assert traj.dtype == np.float32, f"[{where}] traj dtype must be float32, got {traj.dtype}"

def _audit_and_enforce_pen_logic_char(
    rel: np.ndarray,
    where: str,
    sid: str,
    char_idx: Optional[int] = None,
    ch: Optional[str] = None,
    *,
    fix: bool = True,
    log_prefix: str = "[AUDIT]",
    is_cursive_connected: bool = False
) -> Tuple[np.ndarray, Dict[str,int]]:
    info = {"fixed_pen_class":0, "violations":0, "duplicated_single_pt":0}
    if rel.shape[0] == 0:
        return rel, info

    _assert_shape_and_dtype(rel, where)
    t = rel.copy()


    if t.shape[0] == 1:
        t = np.vstack([t, t])
        info["duplicated_single_pt"] = 1

    pen_class = t[:, TRAJ_INDEX["PEN_CLASS"]].astype(np.int64)


    expected_last_pen = PEN_CLASS["CURSIVE_EOC"] if is_cursive_connected else PEN_CLASS["EOC"]

    if t.shape[0] > 0:

        if pen_class[-1] not in (PEN_CLASS["EOC"], PEN_CLASS["CURSIVE_EOC"]):
            if fix:
                pen_class[-1] = expected_last_pen
                info["fixed_pen_class"] += 1
            else:
                info["violations"] += 1

        elif is_cursive_connected and pen_class[-1] == PEN_CLASS["EOC"]:
            if fix:
                pen_class[-1] = PEN_CLASS["CURSIVE_EOC"]
                info["fixed_pen_class"] += 1

        elif not is_cursive_connected and pen_class[-1] == PEN_CLASS["CURSIVE_EOC"]:
            if fix:
                pen_class[-1] = PEN_CLASS["EOC"]
                info["fixed_pen_class"] += 1


        if pen_class[0] != PEN_CLASS["PM"]:
            if fix:
                pen_class[0] = PEN_CLASS["PM"]
                info["fixed_pen_class"] += 1
            else:
                info["violations"] += 1


        for i in range(len(pen_class) - 1):
            if pen_class[i] < 0 or pen_class[i] > PEN_CLASS["EOC"]:
                if fix:
                    pen_class[i] = PEN_CLASS["PM"]
                    info["fixed_pen_class"] += 1
                else:
                    info["violations"] += 1

            elif pen_class[i] in (PEN_CLASS["EOC"], PEN_CLASS["CURSIVE_EOC"]):
                if fix:
                    pen_class[i] = PEN_CLASS["PM"]
                    info["fixed_pen_class"] += 1
                else:
                    info["violations"] += 1

    t[:, TRAJ_INDEX["PEN_CLASS"]] = pen_class.astype(np.float32)

    if (info["fixed_pen_class"] or info["duplicated_single_pt"]) and fix:
        where_info = f"{where} sid={sid} idx={char_idx} ch={repr(ch)}"


    return t, info

def _safe_pick_idxs(label: np.ndarray, i: int, sentence: str, sid: str) -> np.ndarray:
    if label.ndim != 2:
        print(f"[WARN][label-shape] sid={sid} label.ndim={label.ndim}, expect 2")
        return np.array([], dtype=np.int64)
    if i >= label.shape[1]:
        print(f"[WARN][label-len-mismatch] sid={sid} |len(sentence)|={len(sentence)} label.shape[1]={label.shape[1]}")
        return np.array([], dtype=np.int64)
    return np.where(label[:, i] == 1)[0]


def _rdp(points: np.ndarray, epsilon: float) -> np.ndarray:
    if points.shape[0] <= 2:
        return points
    start, end = points[0], points[-1]
    vec = end - start
    norm = np.linalg.norm(vec) + 1e-12
    dists = np.abs(np.cross(points[1:-1] - start, vec) / norm)
    idx = np.argmax(dists) + 1
    if dists[idx-1] > epsilon:
        left = _rdp(points[:idx+1], epsilon)
        right = _rdp(points[idx:], epsilon)
        return np.vstack([left[:-1], right])
    else:
        return np.vstack([start, end])

def _apply_rdp(traj: np.ndarray, epsilon: Optional[float]) -> np.ndarray:
    if traj.shape[0] == 0 or epsilon is None or epsilon <= 0:
        return traj


    pen_class = traj[:, TRAJ_INDEX["PEN_CLASS"]].astype(np.int64)
    strokes_xy, strokes_has_pu = [], []
    cur = []

    for i in range(traj.shape[0]):
        cur.append(traj[i, :2])
        if pen_class[i] == PEN_CLASS["PU"]:
            strokes_xy.append(np.array(cur, dtype=np.float32))
            strokes_has_pu.append(True)
            cur = []

    if len(cur) > 0:
        strokes_xy.append(np.array(cur, dtype=np.float32))
        strokes_has_pu.append(False)

    if not strokes_xy:
        return traj


    simp_xy = [_rdp(s, epsilon) if s.shape[0] > 2 else s for s in strokes_xy]

    xy = np.vstack(simp_xy).astype(np.float32)
    out = np.zeros((xy.shape[0], TRAJ_DIM), dtype=np.float32)
    out[:, TRAJ_INDEX["X"]], out[:, TRAJ_INDEX["Y"]] = xy[:, 0], xy[:, 1]


    out[:, TRAJ_INDEX["PEN_CLASS"]] = PEN_CLASS["PM"]

    cursor = 0
    for s, had_pu in zip(simp_xy, strokes_has_pu):
        n = s.shape[0]
        if n <= 0:
            continue
        if had_pu:
            out[cursor + n - 1, TRAJ_INDEX["PEN_CLASS"]] = PEN_CLASS["PU"]
        cursor += n


    return out


def resample_trajectory_to_fixed_length(
    traj: np.ndarray,
    target_len: int
) -> np.ndarray:
    if len(traj) <= target_len:
        return traj

    if target_len < 2:
        return traj[0:1].copy()

    def _split_strokes_by_pu(t: np.ndarray) -> Tuple[List[np.ndarray], List[bool]]:
        pen = t[:, TRAJ_INDEX["PEN_CLASS"]].astype(np.int64)
        strokes: List[np.ndarray] = []
        has_pu: List[bool] = []
        start = 0
        for i, p in enumerate(pen):
            if p == PEN_CLASS["PU"]:
                strokes.append(t[start:i + 1])
                has_pu.append(True)
                start = i + 1
        if start < len(t):
            strokes.append(t[start:])
            has_pu.append(False)
        return strokes, has_pu

    def _resample_stroke_to_len(stroke: np.ndarray, tlen: int) -> np.ndarray:
        if len(stroke) <= tlen:
            return stroke
        if tlen < 2:
            return stroke[0:1].copy()


        xy = stroke[:, :2]
        pen = stroke[:, 2]


        dists = np.sqrt(np.sum(np.diff(xy, axis=0) ** 2, axis=1))
        cumulative_dist = np.concatenate([[0], np.cumsum(dists)])
        total_length = cumulative_dist[-1]

        if total_length == 0:
            resampled = np.tile(stroke[0:1], (tlen, 1))
        else:

            sample_dists = np.linspace(0, total_length, tlen)
            resampled_xy = np.zeros((tlen, 2), dtype=stroke.dtype)
            resampled_pen = np.zeros(tlen, dtype=stroke.dtype)

            for i, s_dist in enumerate(sample_dists):
                idx = np.searchsorted(cumulative_dist, s_dist, side='right') - 1
                idx = np.clip(idx, 0, len(cumulative_dist) - 2)
                d0 = cumulative_dist[idx]
                d1 = cumulative_dist[idx + 1]
                t = (s_dist - d0) / (d1 - d0) if d1 - d0 > 0 else 0.0
                resampled_xy[i] = xy[idx] * (1 - t) + xy[idx + 1] * t
                resampled_pen[i] = pen[idx] if t < 0.5 else pen[idx + 1]

            resampled = np.column_stack([resampled_xy, resampled_pen])


        resampled[0, TRAJ_INDEX["PEN_CLASS"]] = stroke[0, TRAJ_INDEX["PEN_CLASS"]]
        resampled[-1, TRAJ_INDEX["PEN_CLASS"]] = stroke[-1, TRAJ_INDEX["PEN_CLASS"]]
        return resampled

    strokes, _ = _split_strokes_by_pu(traj)
    if not strokes:
        return traj


    min_per = [2 if len(s) >= 2 else 1 for s in strokes]
    min_total = sum(min_per)
    if target_len < min_total:
        return traj

    remaining = target_len - min_total
    if remaining == 0:
        targets = min_per
    else:
        capacity = [len(s) - m for s, m in zip(strokes, min_per)]
        total_cap = sum(capacity)
        if total_cap <= 0:
            targets = min_per
        else:
            raw = [c / total_cap * remaining for c in capacity]
            base = [int(np.floor(r)) for r in raw]
            targets = [m + b for m, b in zip(min_per, base)]

            rem = remaining - sum(base)
            if rem > 0:
                frac = [r - b for r, b in zip(raw, base)]
                order = np.argsort(frac)[::-1]
                for idx in order:
                    if rem == 0:
                        break
                    if targets[idx] < len(strokes[idx]):
                        targets[idx] += 1
                        rem -= 1

    resampled_strokes = [
        _resample_stroke_to_len(s, tlen)
        for s, tlen in zip(strokes, targets)
    ]

    return np.vstack(resampled_strokes)


def render_debug_comparison(
    traj_before: np.ndarray,
    traj_after: np.ndarray,
    img_size: Tuple[int, int],
    writer_id: str,
    sid: str,
    char: str,
    cursor: int,
    epsilon: float,
    save_dir: str
) -> None:
    import os
    from PIL import Image, ImageDraw, ImageFont

    H, W = img_size


    img_before = render_image_from_traj(traj_before, (H, W))
    img_after = render_image_from_traj(traj_after, (H, W))

    if img_before is None:
        img_before = np.ones((H, W), dtype=np.uint8) * 255
    if img_after is None:
        img_after = np.ones((H, W), dtype=np.uint8) * 255


    header_h = 30
    combined_w = W * 2 + 10
    combined_h = H + header_h

    combined = Image.new("RGB", (combined_w, combined_h), (255, 255, 255))
    draw = ImageDraw.Draw(combined)


    method_str = "Arc-length Resample" if epsilon == 0.0 else f"RDP eps={epsilon:.2f}"
    text = f"Writer:{writer_id} Sent:{sid} Char:'{char}'[{cursor}] | Before:{len(traj_before)}pts → After:{len(traj_after)}pts ({method_str})"
    try:
        font = ImageFont.load_default()
    except:
        font = None
    draw.text((5, 5), text, fill=(0, 0, 0), font=font)


    img_before_pil = Image.fromarray(img_before, mode='L')
    combined.paste(img_before_pil, (0, header_h))
    draw.text((5, header_h + 5), "Before", fill=(255, 0, 0), font=font)


    img_after_pil = Image.fromarray(img_after, mode='L')
    combined.paste(img_after_pil, (W + 10, header_h))
    draw.text((W + 15, header_h + 5), "After", fill=(0, 255, 0), font=font)


    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{writer_id}_{sid}_{char}_{cursor}.png")
    combined.save(save_path)


def render_image_from_traj(traj: np.ndarray, img_size: Tuple[int,int], line_width: int = 3,
                           use_antialiasing: bool = True) -> np.ndarray:
    H, W = img_size


    if use_antialiasing:
        scale_factor = 2
        H_render, W_render = H * scale_factor, W * scale_factor
        line_width_render = line_width * scale_factor
        margin_render = 4.0 * scale_factor
    else:
        H_render, W_render = H, W
        line_width_render = line_width
        margin_render = 4.0

    img = Image.new("L", (W_render, H_render), 255)
    if traj.shape[0] == 0:
        if use_antialiasing:
            img = img.resize((W, H), Image.LANCZOS)
        return np.array(img, dtype=np.uint8)

    draw = ImageDraw.Draw(img)


    traj_expanded = _expand_pen_class_to_onehot(traj)
    xy = traj_expanded[:, :2].copy()


    x_min, x_max = float(np.min(xy[:, 0])), float(np.max(xy[:, 0]))
    y_min, y_max = float(np.min(xy[:, 1])), float(np.max(xy[:, 1]))


    w_span = max(0.001, x_max - x_min)
    h_span = max(0.001, y_max - y_min)


    scale = min((W_render - 2 * margin_render) / w_span, (H_render - 2 * margin_render) / h_span)


    scaled_w = w_span * scale
    scaled_h = h_span * scale
    offset_x = (W_render - scaled_w) / 2.0
    offset_y = (H_render - scaled_h) / 2.0

    xy[:, 0] = (xy[:, 0] - x_min) * scale + offset_x
    xy[:, 1] = (xy[:, 1] - y_min) * scale + offset_y


    xy[:, 0] = np.clip(xy[:, 0], 0, W_render - 1)
    xy[:, 1] = np.clip(xy[:, 1], 0, H_render - 1)


    pm = traj_expanded[:, TRAJ_INDEX_EXPANDED["PM"]] > 0.5

    for i in range(len(traj_expanded) - 1):
        if pm[i]:
            draw.line([tuple(xy[i]), tuple(xy[i+1])], fill=0, width=line_width_render)


    dot_r = max(1, line_width_render // 2)
    for i in range(len(traj_expanded)):
        if not pm[i]:
            if i == 0 or not pm[i - 1]:
                cx, cy = float(xy[i, 0]), float(xy[i, 1])
                draw.ellipse(
                    [(cx - dot_r, cy - dot_r), (cx + dot_r, cy + dot_r)],
                    fill=0,
                )


    if use_antialiasing:
        img = img.resize((W, H), Image.LANCZOS)


    img_arr = np.array(img, dtype=np.uint8)
    if np.all(img_arr == 255):

        return None
    return img_arr


def get_connection_type(i: int, label: np.ndarray, drawing_old: np.ndarray) -> Optional[str]:
    this_idx = np.where(label[:, i] == 1)[0]
    if len(this_idx) == 0:
        return None
    prev_idx = np.where(label[:, i - 1] == 1)[0] if i > 0 else []
    next_idx = np.where(label[:, i + 1] == 1)[0] if i < label.shape[1] - 1 else []


    connected_left = (
        len(prev_idx) > 0 and prev_idx[-1] + 1 == this_idx[0]
        and int(drawing_old[prev_idx[-1], 3]) == 0
    )
    connected_right = (
        len(next_idx) > 0 and this_idx[-1] + 1 == next_idx[0]
        and int(drawing_old[this_idx[-1], 3]) == 0
    )
    if connected_left and connected_right: return "connected_both"
    if connected_left: return "connected_left"
    if connected_right: return "connected_right"
    return "isolated"


def _traj_len(sample: Dict[str, Any]) -> int:
    traj = sample.get("trajectory", None)
    return int(traj.shape[0]) if isinstance(traj, np.ndarray) else 0

def _is_quality_sample(sample: Dict[str, Any], min_points: int = 5, min_range: float = 0.05) -> bool:
    traj = sample.get("trajectory", None)
    if traj is None or not isinstance(traj, np.ndarray):
        return False

    n_points = traj.shape[0]
    if n_points < min_points:
        return False

    x_range = float(np.max(traj[:, 0]) - np.min(traj[:, 0]))
    y_range = float(np.max(traj[:, 1]) - np.min(traj[:, 1]))


    if x_range < min_range and y_range < min_range:
        return False

    return True

def _pick_median_len_first3(samples_for_char: list):
    if not samples_for_char: return None

    quality_samples = [s for s in samples_for_char if _is_quality_sample(s)]
    if not quality_samples:
        quality_samples = samples_for_char
    take = quality_samples[:3]
    lens = [int(s.get("trajectory", None).shape[0]) if s.get("trajectory", None) is not None else 0 for s in take]
    order = sorted(range(len(take)), key=lambda i: lens[i])
    mid = order[len(order)//2]
    return take[mid]

def _sample_quality_score(sample: Dict[str, Any]) -> float:
    traj = sample.get("trajectory", None)
    if traj is None or not isinstance(traj, np.ndarray) or traj.shape[0] == 0:
        return 0.0

    n_points = traj.shape[0]
    x_range = float(np.max(traj[:, 0]) - np.min(traj[:, 0]))
    y_range = float(np.max(traj[:, 1]) - np.min(traj[:, 1]))


    return n_points * (x_range + y_range + 0.01)

def _choose_representative(samples_for_char: List[Dict]) -> Tuple[Optional[Dict], Optional[str]]:
    if not samples_for_char:
        return None, None


    quality_samples = [s for s in samples_for_char if _is_quality_sample(s)]


    if not quality_samples:
        ch = samples_for_char[0].get('character', '?') if samples_for_char else '?'

        best = max(samples_for_char, key=_sample_quality_score)
        best_score = _sample_quality_score(best)
        print(f"[WARN][quality] No quality samples for '{ch}', using best fallback (score={best_score:.2f})")
        return best, best.get("connection", "unknown")

    by_conn: Dict[str, List[Dict]] = {}
    for s in quality_samples:
        conn = s.get("connection", "unknown")
        by_conn.setdefault(conn, []).append(s)

    for conn in REP_PRIORITY:
        cand = by_conn.get(conn, [])
        if not cand:
            continue
        lens = [_traj_len(s) for s in cand]
        if not lens:
            continue
        med = sorted(lens)[len(lens) // 2]
        rep = min(cand, key=lambda s: (abs(_traj_len(s) - med), _traj_len(s)))
        return rep, conn

    s0 = quality_samples[0]
    return s0, s0.get("connection", "unknown")

def _compute_conn_stats(samples_for_char: List[Dict]) -> Dict[str, Dict[str, float]]:
    acc: Dict[str, Dict[str, float]] = {}
    for s in samples_for_char:
        conn = s.get("connection", "unknown")
        L   = _traj_len(s)
        d = acc.setdefault(conn, {"count":0, "sum_len":0})
        d["count"]   += 1
        d["sum_len"] += L
    out = {}
    for conn, d in acc.items():
        c = max(1, d["count"])
        out[conn] = {"count": d["count"], "avg_len": d["sum_len"]/c}
    return out

def _left_align_sample(sample: Dict[str, Any], img_size: Tuple[int,int],
                       line_width: int = 3, use_antialiasing: bool = True) -> Dict[str, Any]:
    if not isinstance(sample, dict):
        return sample
    traj = sample.get("trajectory", None)
    if not isinstance(traj, np.ndarray) or traj.shape[0] == 0:
        new = dict(sample)
        new["coord_mode"] = "left_aligned"
        return new
    xmin = float(np.min(traj[:, TRAJ_INDEX["X"]]))
    new = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in sample.items()}
    new_traj = traj.copy().astype(np.float32)
    new_traj[:, TRAJ_INDEX["X"]] = new_traj[:, TRAJ_INDEX["X"]] - xmin
    new["trajectory"] = new_traj
    new["orig_min_x"] = xmin
    new["coord_mode"] = "left_aligned"


    rendered = render_image_from_traj(new_traj, img_size, line_width=line_width, use_antialiasing=use_antialiasing)
    new["image"] = rendered.astype(np.uint8) if rendered is not None else None
    return new


def _normalize_sentence_height(drawing: np.ndarray, target_height: float = 1.0) -> Tuple[float, float, float, float]:
    if drawing.size == 0:
        return 1.0, 0.0, 0.0, 0.0, 0.0


    x_min = float(np.min(drawing[:, 0]))
    x_max = float(np.max(drawing[:, 0]))
    y_min = float(np.min(drawing[:, 1]))
    y_max = float(np.max(drawing[:, 1]))

    original_width = x_max - x_min
    original_height = y_max - y_min


    if original_height > 0:
        scale = target_height / original_height
    else:

        if original_width > 0:
            scale = target_height / original_width
        else:
            scale = 1.0

    return float(scale), float(x_min), float(y_min), float(original_height), float(original_width)


def generate_all_pickles(
    data_root: str,
    save_char_root: str,
    save_sent_root: str,
    save_style_root: str,
    resample_type: str = "original",
    img_size: Tuple[int,int] = (64, 64),
    rdp_epsilon: Optional[float] = None,
    split_mode: str = "all",
    exclude_dsd_prohibits: bool = False,
    max_traj_len: Optional[int] = None,
    save_debug_vis: bool = False,
    debug_vis_writers: int = 3,
    split_train_max: int = 149,
    split_test_min: int = 150,
    split_test_max: int = 169,
):
    ensure_dir(save_char_root)
    ensure_dir(save_sent_root)
    ensure_dir(save_style_root)

    H, W = img_size
    global_conn_stats: Dict[str, Dict[str, float]] = {}
    global_pen_fix_count = 0
    global_pen_violation_count = 0
    all_writers_info: List[Dict[str, Any]] = []


    global_point_counts = []


    all_writers = sorted([d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))])
    writers = filter_writers_by_split(
        all_writers,
        split_mode,
        split_train_max=split_train_max,
        split_test_min=split_test_min,
        split_test_max=split_test_max,
    )

    print(f"\n{'='*80}")
    print(f"[BRUSH Dataset Generator]")
    print(f"Split Mode: {split_mode.upper()}")
    print(f"  Total writers in data_root: {len(all_writers)}")
    print(f"  Writers after split filter: {len(writers)}")
    if split_mode == "train":
        print(f"  Writer ID range: <= {split_train_max} (TRAIN)")
    elif split_mode == "test":
        print(f"  Writer ID range: {split_test_min}-{split_test_max} (TEST)")
    print(f"Exclude DSD prohibits: {exclude_dsd_prohibits}")
    if exclude_dsd_prohibits:
        print(f"  Will exclude {len(DSD_PROHIBITED_SAMPLES)} prohibited samples")
    print(f"RDP epsilon: {rdp_epsilon}")
    print(f"Max traj len: {max_traj_len} (Adaptive RDP if exceeded)")
    print(f"Save debug vis: {save_debug_vis}")
    print(f"Resample type: {resample_type}")
    print(f"{'='*80}\n")

    for writer_id in tqdm(writers, desc="Writers"):
        writer_path = os.path.join(data_root, writer_id)

        writer_chars: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        writer_sentences: Dict[str, List[Dict[str, Any]]] = {}
        alpha_rep_collect: Dict[str, list] = {}


        writer_pen_fix = 0
        writer_pen_violation = 0

        for fname in sorted(os.listdir(writer_path)):
            if not match_resample_type(fname, resample_type):
                continue


            if exclude_dsd_prohibits and is_dsd_prohibited_sample(writer_id, fname, resample_type):
                continue

            fpath = os.path.join(writer_path, fname)
            try:
                with open(fpath, "rb") as f:
                    sentence, drawing, label = pickle.load(f)
            except Exception as e:
                print(f"[ERROR] Failed to load {fpath}: {e}")
                continue


            drawing_old = None
            if drawing.shape[1] == TRAJ_DIM:
                xy  = drawing[:, :2].astype(np.float32)
                eos = (drawing[:, 2] > 0.5).astype(np.float32)
                pm = np.ones_like(eos, dtype=np.float32)
                eoc = np.zeros_like(eos, dtype=np.float32)
                pen = np.stack([pm, eos, eoc], axis=1)
                drawing_old = np.concatenate([xy, pen], axis=1).astype(np.float32)

                drawing_old[:, 2:5] = (drawing_old[:, 2:5] > 0.5).astype(np.float32)
            elif drawing.shape[1] == 5:
                drawing_old = drawing.copy().astype(np.float32)
                drawing_old[:, 2:5] = (drawing_old[:, 2:5] > 0.5).astype(np.float32)
            else:
                print(f"[ERROR] Unexpected drawing shape: {drawing.shape} for {fpath}")
                continue


            drawing_new = _convert_old_to_new_format(drawing_old)


            scale, x_min, y_min, original_height, original_width = _normalize_sentence_height(
                drawing_new, target_height=1.0
            )
            normed_xy = np.stack([
                (drawing_new[:, 0] - x_min) * scale,
                (drawing_new[:, 1] - y_min) * scale
            ], axis=1).astype(np.float32)

            normed_drawing = np.concatenate([normed_xy, drawing_new[:, 2:3]], axis=1).astype(np.float32)


            min_xy = np.array([x_min, y_min], dtype=np.float32)
            sid = strip_suffix(fname, resample_type)
            sent_items: List[Dict[str, Any]] = []

            if not isinstance(sentence, str):
                sentence = str(sentence)
            if label.ndim != 2 or label.shape[1] != len(sentence):
                print(f"[WARN][sid={fname}] label.shape[1]={label.shape[1] if hasattr(label,'shape') else '?'} "
                      f"!= len(sentence)={len(sentence)} → skip sentence")
                continue

            for i, ch in enumerate(sentence):
                if ch == ' ':
                    space_item = {
                        "trajectory": np.zeros((0, TRAJ_DIM), dtype=np.float32),
                        "image": None,
                        "connection": "space",
                        "character": ch,
                        "sentence_id": sid,
                        "cursor": i,
                        "scale": float(scale),
                        "min_xy": min_xy.astype(np.float32),
                        "orig_min_x": None,

                        "original_height": float(original_height),
                        "original_width": float(original_width),
                        "coord_mode": "normalized_height_1.0",

                        "_before_points": 0,
                        "_before_x_range": 0.0,
                        "_before_y_range": 0.0,
                        "_after_points": 0,
                        "_after_x_range": 0.0,
                        "_after_y_range": 0.0,
                    }
                    writer_chars[ch].append(space_item)
                    sent_items.append(space_item)
                    continue

                idxs = _safe_pick_idxs(label, i, sentence, sid)
                if len(idxs) == 0:
                    print(f"[WARN][empty-char-idxs] writer={writer_id} sid={sid} i={i} ch={repr(ch)}")
                    continue


                char_traj_orig = drawing_new[idxs].astype(np.float32)
                before_points = len(char_traj_orig)
                before_x_range = float(np.max(char_traj_orig[:, 0]) - np.min(char_traj_orig[:, 0])) if len(char_traj_orig) > 0 else 0.0
                before_y_range = float(np.max(char_traj_orig[:, 1]) - np.min(char_traj_orig[:, 1])) if len(char_traj_orig) > 0 else 0.0


                char_traj = normed_drawing[idxs].astype(np.float32)
                char_traj_original = char_traj.copy()


                traj_before_resample = char_traj.copy()
                applied_method = "none"


                if max_traj_len is not None and char_traj.shape[0] > max_traj_len:
                    char_traj = resample_trajectory_to_fixed_length(
                        char_traj,
                        target_len=max_traj_len
                    )
                    applied_method = f"arc_resample_to_{max_traj_len}"


                    if save_debug_vis and len(traj_before_resample) > max_traj_len:
                        render_debug_comparison(
                            traj_before=traj_before_resample,
                            traj_after=char_traj,
                            img_size=(H, W),
                            writer_id=writer_id,
                            sid=sid,
                            char=ch,
                            cursor=i,
                            epsilon=0.0,
                            save_dir=os.path.join(save_sent_root, "../debug_resample")
                        )


                if rdp_epsilon is not None and rdp_epsilon > 0 and char_traj.shape[0] > 0:
                    original_point_count = char_traj.shape[0]
                    char_traj = _apply_rdp(char_traj, rdp_epsilon)
                    applied_epsilon = rdp_epsilon


                    min_points = max(3, int(original_point_count * 0.05))
                    if char_traj.shape[0] < min_points and original_point_count >= min_points:

                        char_traj = normed_drawing[idxs].astype(np.float32)
                        applied_epsilon = 0.0


                char_traj = guarantee_stroke_endpoints(char_traj_original, char_traj)


                conn_type = get_connection_type(i, label, drawing_old) or "isolated"
                is_cursive_connected = conn_type in ("connected_right", "connected_both")


                fixed, info = _audit_and_enforce_pen_logic_char(
                    char_traj,
                    where="char_subseq",
                    sid=sid,
                    char_idx=i,
                    ch=ch,
                    fix=True,
                    is_cursive_connected=is_cursive_connected,
                )
                writer_pen_fix       += info["fixed_pen_class"]
                writer_pen_violation += info["violations"]
                char_traj = fixed


                after_points = len(char_traj)
                after_x_range = float(np.max(char_traj[:, 0]) - np.min(char_traj[:, 0])) if len(char_traj) > 0 else 0.0
                after_y_range = float(np.max(char_traj[:, 1]) - np.min(char_traj[:, 1])) if len(char_traj) > 0 else 0.0


                img = render_image_from_traj(char_traj, (H, W))


                if img is None:
                    print(f"[WARN][GENERATOR] Render failed (traj_len={len(char_traj)}): wid={writer_id}, sid={sid}, ch='{ch}', cursor={i}")

                char_item = {
                    "trajectory": char_traj.astype(np.float32),
                    "image": img.astype(np.uint8) if img is not None else None,
                    "connection": conn_type,
                    "character": ch,
                    "sentence_id": sid,
                    "cursor": i,
                    "scale": float(scale),
                    "min_xy": min_xy.astype(np.float32),
                    "orig_min_x": None,

                    "original_height": float(original_height),
                    "original_width": float(original_width),
                    "coord_mode": "normalized_height_1.0",

                    "_before_points": before_points,
                    "_before_x_range": before_x_range,
                    "_before_y_range": before_y_range,
                    "_after_points": after_points,
                    "_after_x_range": after_x_range,
                    "_after_y_range": after_y_range,
                }
                writer_chars[ch].append(char_item)
                sent_items.append(char_item)


            def _is_repetition_run_chars(chars: str) -> bool:
                return len(chars) >= 1 and (len(set(chars)) == 1) and (chars[0] in REP_TARGET_SET) and (chars[0] != ' ')
            def _is_repetition_sentence(sent: str) -> bool:
                toks = [t for t in (sent or "").split(" ") if t != ""]
                return len(toks) > 0 and all(_is_repetition_run_chars(tok) for tok in toks)

            if _is_repetition_sentence(sentence):
                i_char = 0
                while i_char < len(sentence):
                    ch0 = sentence[i_char]
                    if ch0 == ' ':
                        i_char += 1
                        continue
                    j = i_char
                    while j < len(sentence) and sentence[j] == ch0:
                        j += 1
                    if _is_repetition_run_chars(sentence[i_char:j]):
                        c = ch0
                        if c not in alpha_rep_collect:
                            alpha_rep_collect[c] = []
                        for _k in range(i_char, min(j, i_char + 3)):
                            if _k < len(sent_items):
                                alpha_rep_collect[c].append(sent_items[_k])
                    i_char = j

            else:

                non_space_chars = [item for item in sent_items if item['character'] != ' ']
                if len(non_space_chars) >= 2:
                    writer_sentences[sid] = sent_items
                elif len(sent_items) > 0:

                    print(f"[WARN] Skipping sentence with <2 non-space chars: wid={writer_id}, sid={sid}, chars={len(non_space_chars)}", flush=True)


        rep_map: Dict[str, Dict[str, Any]] = {}
        total_chars = 0
        writer_conn_counter = {"isolated":0, "connected_left":0, "connected_right":0, "connected_both":0, "unknown":0}

        for ch, samples in sorted(writer_chars.items()):
            if ch == ' ':
                continue
            total_chars += 1

            if ch in alpha_rep_collect and alpha_rep_collect[ch]:
                rep = _pick_median_len_first3(alpha_rep_collect[ch])
                if rep is not None:
                    rep_map[ch] = _left_align_sample(rep, (H, W))
            rep_already = (ch in rep_map)

            for s in samples:
                conn = s.get("connection", "unknown")
                if conn not in writer_conn_counter:
                    writer_conn_counter[conn] = 0
                writer_conn_counter[conn] += 1

            if not rep_already:
                rep, used_conn = _choose_representative(samples)
                if rep is not None:
                    rep_map[ch] = _left_align_sample(rep, (H, W))

            stats = _compute_conn_stats(samples)
            for conn, v in stats.items():
                d = global_conn_stats.setdefault(conn, {"count":0, "sum_len":0})
                d["count"]   += v["count"]
                d["sum_len"] += v["avg_len"] * v["count"]


        total_conn = sum(writer_conn_counter.values())


        global_pen_fix_count       += writer_pen_fix
        global_pen_violation_count += writer_pen_violation


        point_counts = []
        for char_samples in writer_chars.values():
            if isinstance(char_samples, list):
                for sample in char_samples:
                    if isinstance(sample, dict) and 'trajectory' in sample:
                        traj = sample['trajectory']
                        if traj is not None and len(traj) > 0:
                            point_counts.append(len(traj))

        avg_points = sum(point_counts) / len(point_counts) if point_counts else 0
        min_points = min(point_counts) if point_counts else 0
        max_points = max(point_counts) if point_counts else 0


        try:
            missing = sorted([c for c in REP_TARGET_SET if c not in rep_map])
            if missing:
                print(f"[WARN][rep-missing] writer {writer_id}: {len(missing)} chars missing. First 50: {''.join(missing[:50])}")
        except Exception as _e:
            print(f"[WARN] coverage check failed: {_e}")


        global_point_counts.extend(point_counts)
        print(f"[STAT][writer {writer_id}] avg_points={avg_points:.1f}, min={min_points}, max={max_points}, total_samples={len(point_counts)}")


        with open(os.path.join(save_char_root, f"{writer_id}.pkl"), "wb") as f:
            pickle.dump(dict(writer_chars), f)

        with open(os.path.join(save_char_root, f"{writer_id}_rep.pkl"), "wb") as f:
            pickle.dump(rep_map, f)


        def is_alphanumeric(ch: str) -> bool:
            return ch.isalpha() or ch.isdigit()


        writer_chars_filtered = {ch: samples for ch, samples in writer_chars.items() if is_alphanumeric(ch)}
        with open(os.path.join(save_style_root, f"{writer_id}.pkl"), "wb") as f:
            pickle.dump(writer_chars_filtered, f)


        style_rep_map = {}
        for ch, rep_sample in rep_map.items():
            if is_alphanumeric(ch) and rep_sample.get('image', None) is not None:

                style_rep_map[ch] = [rep_sample]

        with open(os.path.join(save_style_root, f"{writer_id}_rep.pkl"), "wb") as f:
            pickle.dump(style_rep_map, f)


        with open(os.path.join(save_sent_root, f"{writer_id}_sent.pkl"), "wb") as f:
            pickle.dump({"sentences": writer_sentences}, f)


        all_writers_info.append({
            "merged_id": int(writer_id),
            "original_id": str(writer_id),
            "source": "BRUSH",
            "num_chars": int(len(writer_chars)),
            "num_sentences": int(len(writer_sentences)),
        })


    print("\n=== Global connection stats (all writers) ===")
    for conn, d in global_conn_stats.items():
        c = max(1, d["count"])
        print(f"{conn:15s} : n={d['count']}, μlen={int(d['sum_len']/c)}")

    print(f"\n=== Global pen-state audit (One-Hot 4-class) ===")
    print(f" total_pen_class_fixes={global_pen_fix_count}, total_violations={global_pen_violation_count}")


    if global_point_counts:
        avg_global = sum(global_point_counts) / len(global_point_counts)
        min_global = min(global_point_counts)
        max_global = max(global_point_counts)
        print(f"\n=== Global point statistics ===")
        print(f" Total samples: {len(global_point_counts):,}")
        print(f" Points per character: avg={avg_global:.1f}, min={min_global}, max={max_global}")
        print(f" Distribution:")

        bins = [0, 5, 10, 15, 20, 30, 50, 100, 200, float('inf')]
        labels = ['0-5', '6-10', '11-15', '16-20', '21-30', '31-50', '51-100', '101-200', '200+']
        counts = [0] * len(labels)
        for p in global_point_counts:
            for i, (lower, upper) in enumerate(zip(bins[:-1], bins[1:])):
                if lower < p <= upper:
                    counts[i] += 1
                    break
        for label, count in zip(labels, counts):
            pct = count / len(global_point_counts) * 100
            print(f"     {label:8s}: {count:6,} ({pct:5.1f}%)")
    else:
        print(f"\n=== Global point statistics ===")
        print(f"  No point statistics collected")

    mapping_path = os.path.join(save_char_root, "writer_mapping.json")
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(all_writers_info, f, indent=2, ensure_ascii=False)
    print(f"[SAVE] Writer mapping: {mapping_path} ({len(all_writers_info)} writers)")


    if save_debug_vis and all_writers_info:
        debug_output_dir = os.path.join(save_style_root, "_debug_style_rep_images")
        ensure_dir(debug_output_dir)
        n_vis = max(1, int(debug_vis_writers))
        writer_pool = all_writers_info
        if len(writer_pool) > n_vis:
            writer_pool = random.sample(writer_pool, n_vis)
        print(f"[DEBUG] Saving style-rep visualization for {len(writer_pool)} writer(s)")
        for info in writer_pool:
            wid = str(info["merged_id"])
            orig = str(info.get("original_id", wid))
            _save_style_rep_debug_images(
                save_style_root,
                writer_id=wid,
                output_dir=debug_output_dir,
                prefix=f"BRUSH_{orig}",
            )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True, help="Path to refined_BRUSH dataset (writers/*/*.pkl)")
    parser.add_argument("--save_char_root", type=str, required=True, help="Output dir for per-writer character pickles")
    parser.add_argument("--save_sent_root", type=str, required=True, help="Output dir for per-writer sentence pickles")
    parser.add_argument("--save_style_root", type=str, required=True, help="Output dir for per-writer style pickles")
    parser.add_argument("--img_size", type=int, nargs=2, default=[64, 64], help="Image size (H W)")
    parser.add_argument("--resample_type", type=str, choices=["original","resample20","resample25"], default="original")
    parser.add_argument("--rdp_epsilon", type=float, default=None, help="RDP epsilon in pixels (None to disable)")
    parser.add_argument("--split_mode", type=str, choices=["all","train","test"], default="all",
                        help="DSD split: 'train' (0-149), 'test' (150-169), 'all' (0-169)")
    parser.add_argument("--split_train_max", type=int, default=149,
                        help="train split max writer id (inclusive)")
    parser.add_argument("--split_test_min", type=int, default=150,
                        help="test split min writer id (inclusive)")
    parser.add_argument("--split_test_max", type=int, default=169,
                        help="test split max writer id (inclusive)")
    parser.add_argument("--exclude_dsd_prohibits", action="store_true",
                        help="Exclude 17 prohibited samples from DSD paper")
    parser.add_argument("--max_traj_len", type=int, default=None,
                        help="Max trajectory length per character. If exceeded, apply adaptive RDP (None to disable)")
    parser.add_argument("--save_debug_vis", action="store_true",
                        help="Save debug visualization images (resample before/after + style rep grid)")
    parser.add_argument("--debug_vis_writers", type=int, default=3,
                        help="Number of random writers to export style-rep debug PNGs")
    args = parser.parse_args()

    generate_all_pickles(
        data_root=args.data_root,
        save_char_root=args.save_char_root,
        save_sent_root=args.save_sent_root,
        save_style_root=args.save_style_root,
        resample_type=args.resample_type,
        img_size=tuple(args.img_size),
        rdp_epsilon=args.rdp_epsilon,
        split_mode=args.split_mode,
        exclude_dsd_prohibits=args.exclude_dsd_prohibits,
        max_traj_len=args.max_traj_len,
        save_debug_vis=args.save_debug_vis,
        debug_vis_writers=args.debug_vis_writers,
        split_train_max=args.split_train_max,
        split_test_min=args.split_test_min,
        split_test_max=args.split_test_max,
    )
