from __future__ import annotations

import os
import pickle
import random
import json
import re
import sys
from glob import glob
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from collections import defaultdict
from src.utils.logger import print_once
from typing import Optional, List, Dict, Any, Tuple
from functools import lru_cache
from src.config.constants import TRAJ_INDEX_EXPANDED, TRAJ_INDEX, PEN_CLASS
import warnings


def _split_strokes_by_pu(traj: np.ndarray) -> List[np.ndarray]:
    if traj is None or len(traj) == 0:
        return []

    pen = traj[:, TRAJ_INDEX["PEN_CLASS"]].astype(np.int64)
    strokes = []
    start = 0

    for i, p in enumerate(pen):
        if p == PEN_CLASS["PU"]:

            strokes.append(traj[start:i + 1].copy())
            start = i + 1


    if start < len(traj):
        strokes.append(traj[start:].copy())

    return strokes


def get_last_stroke(traj: np.ndarray) -> Optional[np.ndarray]:
    if traj is None or len(traj) == 0:
        return None

    pen = traj[:, TRAJ_INDEX["PEN_CLASS"]].astype(np.int64)


    pu_indices = np.where(pen == PEN_CLASS["PU"])[0]

    if len(pu_indices) > 0:

        last_pu_idx = pu_indices[-1]
        if last_pu_idx + 1 < len(traj):
            return traj[last_pu_idx + 1:].copy()
        else:
            return None
    else:

        return traj.copy()


def get_first_stroke(traj: np.ndarray) -> Optional[np.ndarray]:
    if traj is None or len(traj) == 0:
        return None

    pen = traj[:, TRAJ_INDEX["PEN_CLASS"]].astype(np.int64)


    pu_indices = np.where(pen == PEN_CLASS["PU"])[0]

    if len(pu_indices) > 0:

        first_pu_idx = pu_indices[0]
        return traj[:first_pu_idx + 1].copy()
    else:

        return traj.copy()


def check_cursive_spatial_validity(
    prev_traj: np.ndarray,
    curr_traj: np.ndarray,
    tolerance: float = 0.1,
    max_bbox_overlap_ratio: float = 0.5
) -> Tuple[bool, str]:


    prev_bbox = (
        float(prev_traj[:, 0].min()), float(prev_traj[:, 0].max()),
        float(prev_traj[:, 1].min()), float(prev_traj[:, 1].max())
    )
    curr_bbox = (
        float(curr_traj[:, 0].min()), float(curr_traj[:, 0].max()),
        float(curr_traj[:, 1].min()), float(curr_traj[:, 1].max())
    )


    inter_x_min = max(prev_bbox[0], curr_bbox[0])
    inter_x_max = min(prev_bbox[1], curr_bbox[1])
    inter_y_min = max(prev_bbox[2], curr_bbox[2])
    inter_y_max = min(prev_bbox[3], curr_bbox[3])

    inter_w = max(0, inter_x_max - inter_x_min)
    inter_h = max(0, inter_y_max - inter_y_min)
    inter_area = inter_w * inter_h


    prev_area = max(1e-6, (prev_bbox[1] - prev_bbox[0]) * (prev_bbox[3] - prev_bbox[2]))
    curr_area = max(1e-6, (curr_bbox[1] - curr_bbox[0]) * (curr_bbox[3] - curr_bbox[2]))


    min_area = min(prev_area, curr_area)
    overlap_ratio = inter_area / min_area if min_area > 1e-6 else 0.0

    if overlap_ratio > max_bbox_overlap_ratio:
        reason = f"bbox_overlap_too_high: {overlap_ratio:.2f} > {max_bbox_overlap_ratio:.2f}"
        return False, reason


    prev_last_stroke = get_last_stroke(prev_traj)
    if prev_last_stroke is None or len(prev_last_stroke) == 0:
        return True, "prev_last_stroke empty"


    curr_first_stroke = get_first_stroke(curr_traj)
    if curr_first_stroke is None or len(curr_first_stroke) == 0:
        return True, "curr_first_stroke empty"


    prev_x_min = float(prev_last_stroke[:, 0].min())
    prev_x_max = float(prev_last_stroke[:, 0].max())

    curr_x_min = float(curr_first_stroke[:, 0].min())
    curr_x_max = float(curr_first_stroke[:, 0].max())


    if prev_x_min > curr_x_min + tolerance:
        reason = f"spatial_order_error: prev_left={prev_x_min:.3f} > curr_left={curr_x_min:.3f}"
        return False, reason


    if prev_x_max > curr_x_max + tolerance * 2:

        if prev_x_min < curr_x_min - tolerance:
            return True, "prev_starts_before_curr"

        reason = f"overlap_suspicious: prev_right={prev_x_max:.3f} >> curr_right={curr_x_max:.3f}"
        return False, reason

    return True, "valid"


def guarantee_stroke_endpoints(
    original_traj: np.ndarray,
    processed_traj: np.ndarray,
    distance_threshold: float = 0.05
) -> np.ndarray:
    if original_traj is None or len(original_traj) == 0:
        return processed_traj
    if processed_traj is None or len(processed_traj) == 0:
        return processed_traj


    original_strokes = _split_strokes_by_pu(original_traj)
    processed_strokes = _split_strokes_by_pu(processed_traj)


    if len(original_strokes) != len(processed_strokes):
        warnings.warn(
            f"Stroke count mismatch: original={len(original_strokes)}, "
            f"processed={len(processed_strokes)}. Returning processed as-is."
        )
        return processed_traj

    if len(original_strokes) == 0:
        return processed_traj


    result_strokes = []

    for orig_stroke, proc_stroke in zip(original_strokes, processed_strokes):
        if len(orig_stroke) == 0:

            result_strokes.append(proc_stroke)
            continue

        if len(proc_stroke) == 0:

            result_strokes.append(orig_stroke)
            continue


        first_point = orig_stroke[0:1].copy()
        last_point = orig_stroke[-1:].copy()

        if len(proc_stroke) == 1:


            result_stroke = last_point

        elif len(proc_stroke) == 2:

            result_stroke = np.vstack([first_point, last_point])

        else:

            middle_points = proc_stroke[1:-1].copy()
            result_stroke = np.vstack([first_point, middle_points, last_point])

        result_strokes.append(result_stroke)


    result = np.vstack(result_strokes)

    return result


def resample_trajectory_preserve_endpoints(
    traj: np.ndarray,
    target_len: int
) -> np.ndarray:
    if len(traj) <= target_len:
        return traj

    if target_len < 2:
        target_len = 2


    def _split_strokes_by_pu(t: np.ndarray) -> List[np.ndarray]:
        pen = t[:, TRAJ_INDEX["PEN_CLASS"]].astype(np.int64)
        strokes: List[np.ndarray] = []
        start = 0
        for i, p in enumerate(pen):
            if p == PEN_CLASS["PU"]:
                strokes.append(t[start:i + 1])
                start = i + 1
        if start < len(t):
            strokes.append(t[start:])
        return strokes


    def _resample_stroke(stroke: np.ndarray, tlen: int) -> np.ndarray:
        if len(stroke) <= tlen:
            return stroke

        dim = stroke.shape[1]


        if tlen == 1:
            return stroke[-1:].copy()

        first_point = stroke[0:1].copy()
        last_point = stroke[-1:].copy()


        if tlen == 2:
            return np.vstack([first_point, last_point])


        mid_target = tlen - 2
        mid_traj = stroke[1:-1]

        if len(mid_traj) == 0:

            return np.vstack([first_point, last_point])

        if len(mid_traj) <= mid_target:

            return stroke


        xy = mid_traj[:, :2]
        dists = np.sqrt(np.sum(np.diff(xy, axis=0) ** 2, axis=1))
        cumulative_dist = np.concatenate([[0], np.cumsum(dists)])
        total_length = cumulative_dist[-1]

        if total_length == 0:

            indices = np.linspace(0, len(mid_traj) - 1, mid_target).astype(int)
            resampled_mid = mid_traj[indices]
        else:
            sample_dists = np.linspace(0, total_length, mid_target)
            resampled_mid = np.zeros((mid_target, dim), dtype=np.float32)

            for i, s_dist in enumerate(sample_dists):
                idx = np.searchsorted(cumulative_dist, s_dist, side='right') - 1
                idx = np.clip(idx, 0, len(cumulative_dist) - 2)
                d0, d1 = cumulative_dist[idx], cumulative_dist[idx + 1]
                t = (s_dist - d0) / (d1 - d0) if d1 - d0 > 0 else 0.0


                resampled_mid[i, :2] = mid_traj[idx, :2] * (1 - t) + mid_traj[idx + 1, :2] * t

                resampled_mid[i, 2:] = mid_traj[idx, 2:] if t < 0.5 else mid_traj[idx + 1, 2:]

        resampled = np.vstack([first_point, resampled_mid, last_point])


        resampled[0, TRAJ_INDEX["PEN_CLASS"]] = stroke[0, TRAJ_INDEX["PEN_CLASS"]]
        resampled[-1, TRAJ_INDEX["PEN_CLASS"]] = stroke[-1, TRAJ_INDEX["PEN_CLASS"]]

        return resampled


    strokes = _split_strokes_by_pu(traj)
    n_strokes = len(strokes)

    if n_strokes == 0:
        return traj


    if target_len < n_strokes:
        warnings.warn(
            f"target_len ({target_len}) < n_strokes ({n_strokes}). "
            f"Adjusting to {n_strokes} to preserve all stroke endpoints."
        )
        target_len = n_strokes


    min_per = [1 for _ in strokes]
    remaining = target_len - n_strokes


    if remaining > 0:
        can_add_start = [(1 if len(s) >= 2 else 0) for s in strokes]
        total_can_add = sum(can_add_start)

        if remaining >= total_can_add:

            for i, can in enumerate(can_add_start):
                if can:
                    min_per[i] += 1
            remaining -= total_can_add
        else:

            indices_by_len = sorted(
                range(n_strokes),
                key=lambda i: len(strokes[i]) if can_add_start[i] else -1,
                reverse=True
            )
            for i in indices_by_len:
                if remaining == 0:
                    break
                if can_add_start[i]:
                    min_per[i] += 1
                    remaining -= 1


    if remaining > 0:

        capacity = [max(0, len(s) - m) for s, m in zip(strokes, min_per)]
        total_cap = sum(capacity)

        if total_cap > 0:

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
        else:
            targets = min_per
    else:
        targets = min_per


    resampled_strokes = [
        _resample_stroke(s, tlen) for s, tlen in zip(strokes, targets)
    ]

    result = np.vstack(resampled_strokes)


    result = guarantee_stroke_endpoints(traj, result, distance_threshold=0.05)

    return result


def _compute_skew_angle_lms(points: np.ndarray) -> float:
    if points.shape[0] < 2:
        return 0.0

    x = points[:, 0]
    y = points[:, 1]


    x_var = np.var(x)
    if x_var < 1e-8:
        return 0.0


    x_mean = np.mean(x)
    y_mean = np.mean(y)

    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sum((x - x_mean) ** 2)

    if abs(denominator) < 1e-8:
        return 0.0

    slope = numerator / denominator
    angle_rad = np.arctan(slope)
    angle_deg = np.degrees(angle_rad)

    return angle_deg


def _rotate_points_around_center(points: np.ndarray, angle_deg: float,
                                  center: Optional[Tuple[float, float]] = None) -> np.ndarray:
    if points.shape[0] == 0:
        return points.copy()


    if center is None:
        cx = np.mean(points[:, 0])
        cy = np.mean(points[:, 1])
    else:
        cx, cy = center


    angle_rad = np.radians(angle_deg)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)


    rotated = points.copy()
    x_centered = points[:, 0] - cx
    y_centered = points[:, 1] - cy

    rotated[:, 0] = cos_a * x_centered - sin_a * y_centered + cx
    rotated[:, 1] = sin_a * x_centered + cos_a * y_centered + cy

    return rotated


def _deskew_trajectory(drawing: np.ndarray,
                       angle_threshold: float = 1.0,
                       max_angle: float = 30.0) -> Tuple[np.ndarray, float, bool]:
    if drawing.shape[0] < 3:
        return drawing.copy(), 0.0, False


    angle = _compute_skew_angle_lms(drawing)


    if abs(angle) < angle_threshold or abs(angle) > max_angle:
        return drawing.copy(), angle, False


    deskewed = _rotate_points_around_center(drawing, -angle)

    return deskewed, angle, True


def traj_abs_to_delta(traj_abs: np.ndarray) -> np.ndarray:
    if traj_abs is None or len(traj_abs) == 0:
        return traj_abs
    traj_delta = traj_abs.copy()
    traj_delta[1:, :2] -= traj_delta[:-1, :2]
    return traj_delta


def delta_to_abs_norm(traj: torch.Tensor | np.ndarray, start_xy: Optional[torch.Tensor | np.ndarray] = None):
    if traj is None:
        return None
    if torch.is_tensor(traj):
        x = traj.clone()
        xy = x[..., :2]

        xy_abs = torch.cumsum(xy, dim=0)
        if start_xy is not None:
            if not torch.is_tensor(start_xy):
                start_xy = torch.tensor(start_xy, dtype=xy_abs.dtype, device=xy_abs.device)

            xy_abs[:, 0] += start_xy[0]
            xy_abs[:, 1] += start_xy[1]
        x[..., :2] = xy_abs
        return x
    else:
        arr = traj.copy()

        xy_abs = np.cumsum(arr[:, :2], axis=0)
        if start_xy is not None:
            start_xy = np.asarray(start_xy, dtype=np.float32).reshape(-1)

            xy_abs[:, 0] += start_xy[0]
            xy_abs[:, 1] += start_xy[1]
        arr[:, :2] = xy_abs
        return arr


def to_local_bigram(prev_abs, curr_abs, W_ref, H_ref, is_after_space=False):
    from src.config.constants import TRAJ_DIM, TRAJ_DIM_EXPANDED, PEN_CLASS, TRAJ_INDEX, TRAJ_INDEX_EXPANDED


    if prev_abs is not None and len(prev_abs) > 0:
        anchor = prev_abs[-1, :2].astype(np.float32)
    else:
        anchor = np.array([0.0, 0.0], np.float32)

    def _loc(tr, is_after_space_char=False):
        if tr is None or len(tr) == 0:
            return np.zeros((0, TRAJ_DIM_EXPANDED), np.float32)

        out = tr.astype(np.float32).copy()


        if out.shape[1] == TRAJ_DIM:
            T = out.shape[0]
            pen_class = out[:, TRAJ_INDEX["PEN_CLASS"]].astype(np.int64)
            expanded = np.zeros((T, TRAJ_DIM_EXPANDED), dtype=np.float32)
            expanded[:, :2] = out[:, :2]


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
            out = expanded


        out[:, TRAJ_INDEX_EXPANDED["X"]] -= anchor[0]
        out[:, TRAJ_INDEX_EXPANDED["Y"]] -= anchor[1]


        out[:, TRAJ_INDEX_EXPANDED["X"]] /= (float(W_ref) + 1e-6)
        out[:, TRAJ_INDEX_EXPANDED["Y"]] /= (float(H_ref) + 1e-6)


        out[1:, :2] -= out[:-1, :2]
        return out.astype(np.float32)

    return _loc(prev_abs, False), _loc(curr_abs, False), anchor


def normalize_xy_abs_symmetric(traj: np.ndarray, img_size: int | Tuple[int, int]) -> np.ndarray:
    if isinstance(img_size, int):
        H = W = img_size
    else:
        H, W = img_size
    out = traj.astype(np.float32).copy()
    out[:, TRAJ_INDEX_EXPANDED["X"]] = (out[:, TRAJ_INDEX_EXPANDED["X"]] / W) * 2 - 1
    out[:, TRAJ_INDEX_EXPANDED["Y"]] = (out[:, TRAJ_INDEX_EXPANDED["Y"]] / H) * 2 - 1
    return out

def denormalize_xy_abs_symmetric(norm_traj: np.ndarray, img_size: int | Tuple[int, int]) -> np.ndarray:
    if isinstance(img_size, int):
        H = W = img_size
    else:
        H, W = img_size
    out = norm_traj.astype(np.float32).copy()
    out[:, TRAJ_INDEX_EXPANDED["X"]] = (out[:, TRAJ_INDEX_EXPANDED["X"]] + 1) / 2 * W
    out[:, TRAJ_INDEX_EXPANDED["Y"]] = (out[:, TRAJ_INDEX_EXPANDED["Y"]] + 1) / 2 * H
    return out

def denormalize_height_based(norm_traj: np.ndarray, img_size: int | Tuple[int, int], keep_aspect=True) -> np.ndarray:
    if norm_traj is None or len(norm_traj) == 0:
        return norm_traj

    if isinstance(img_size, int):
        H = W = img_size
    else:
        H, W = img_size

    out = norm_traj.astype(np.float32).copy()

    if keep_aspect:


        out[:, TRAJ_INDEX_EXPANDED["Y"]] = out[:, TRAJ_INDEX_EXPANDED["Y"]] * H

        out[:, TRAJ_INDEX_EXPANDED["X"]] = out[:, TRAJ_INDEX_EXPANDED["X"]] * H
    else:

        out[:, TRAJ_INDEX_EXPANDED["X"]] = out[:, TRAJ_INDEX_EXPANDED["X"]] * W
        out[:, TRAJ_INDEX_EXPANDED["Y"]] = out[:, TRAJ_INDEX_EXPANDED["Y"]] * H

    return out


def load_pickle_cached(path: str):
    with open(path, "rb") as f:
        try:
            return pickle.load(f)
        except ModuleNotFoundError as e:


            if "numpy._core" not in str(e):
                raise
            import numpy.core as np_core
            sys.modules.setdefault("numpy._core", np_core)
            sys.modules.setdefault("numpy._core.multiarray", np_core.multiarray)
            f.seek(0)
            return pickle.load(f)


_GLOBAL_PRELOAD_CACHE: Dict[str, Any] = {}

def to_tensor_img_gray(x: np.ndarray, invert: bool = False) -> torch.Tensor:
    if x.dtype != np.float32 and x.max() > 1.0:
        x = x.astype(np.float32) / 255.0


    if invert:
        x = 1.0 - x

    t = torch.from_numpy(x).float()
    if t.dim() == 2:
        t = t.unsqueeze(0)
    return t

def pad_1d(seq, pad_value: float = 0.0):

    if isinstance(seq, (list, tuple)) and len(seq) > 0 and isinstance(seq[0], np.ndarray):
        seq = [torch.from_numpy(x) for x in seq]

    if not isinstance(seq, (list, tuple)) or len(seq) == 0:

        return torch.empty(0)


    ref = seq[0]
    if not isinstance(ref, torch.Tensor):
        ref = torch.as_tensor(ref)

    device = ref.device
    dtype  = ref.dtype


    tensors = []
    for t in seq:
        if not isinstance(t, torch.Tensor):
            t = torch.as_tensor(t, dtype=dtype, device=device)
        else:
            t = t.to(device=device, dtype=dtype, non_blocking=True)
        tensors.append(t)


    tail_shape = tensors[0].shape[1:] if tensors[0].ndim >= 1 else tuple()
    T_max = 0
    for t in tensors:

        T_max = max(T_max, int(t.shape[0]) if t.ndim >= 1 else 0)

    B = len(tensors)
    out_shape = (B, T_max) + tail_shape
    out = torch.full(out_shape, fill_value=pad_value, dtype=dtype, device=device)

    for i, t in enumerate(tensors):
        if t.ndim == 0 or t.numel() == 0:
            continue
        L = min(T_max, int(t.shape[0]))
        out[i, :L, ...] = t[:L, ...]
    return out

def sample_style_refs_from_writer(items: List[Dict[str,Any]], num_refs: int) -> List[Dict[str,Any]]:
    if not items or num_refs <= 0:
        return []


    char_to_indices = {}
    for idx, item in enumerate(items):
        ch = item.get('character', '?')
        if ch not in char_to_indices:
            char_to_indices[ch] = []
        char_to_indices[ch].append(idx)


    selected_indices = []
    for ch, indices in char_to_indices.items():
        chosen_idx = random.choice(indices)
        selected_indices.append(chosen_idx)


    if len(selected_indices) < num_refs:
        remaining = [i for i in range(len(items)) if i not in selected_indices]

        if remaining:
            need = min(num_refs - len(selected_indices), len(remaining))
            selected_indices.extend(random.sample(remaining, need))


    if len(selected_indices) < num_refs:
        deficit = num_refs - len(selected_indices)
        for dup_idx in range(deficit):
            src_idx = selected_indices[dup_idx % len(selected_indices)]
            selected_indices.append(src_idx)


    if len(selected_indices) > num_refs:
        random.shuffle(selected_indices)
        selected_indices = selected_indices[:num_refs]


    return [dict(items[idx], sample_idx=idx) for idx in selected_indices]

def _is_rep_file(path: str) -> bool:
    base = os.path.basename(path)
    return base.endswith("_rep.pkl")

def _is_bigram_file(path: str) -> bool:
    base = os.path.basename(path)
    return base.endswith("_bigram.pkl")

def _is_sent_file(path: str) -> bool:
    base = os.path.basename(path)
    return base.endswith("_sent.pkl")

def list_writer_pickles(pickle_root: str, include_rep: bool = False):
    all_pkls = sorted(glob(os.path.join(pickle_root, "*.pkl")))
    if include_rep:
        return [p for p in all_pkls if _is_rep_file(p)]
    else:
        return [p for p in all_pkls if not _is_rep_file(p) and not _is_bigram_file(p) and not _is_sent_file(p)]


def _cfg_get(obj, key: str, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _as_plain_dict(obj) -> Dict[str, Any]:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return dict(obj)
    try:
        return dict(obj)
    except Exception:
        return {}


def _is_int_like(v: Any) -> bool:
    try:
        int(v)
        return True
    except Exception:
        return False


def _expand_writer_id_spec(spec: Any) -> List[int]:
    ids = set()

    def _add(item: Any):
        if item is None:
            return
        if isinstance(item, (int, np.integer)):
            ids.add(int(item))
            return
        if isinstance(item, str):
            s = item.strip()
            if _is_int_like(s):
                ids.add(int(s))
                return
            raise ValueError(f"Invalid writer id string: {item}")
        if isinstance(item, (list, tuple)):
            if len(item) == 2 and _is_int_like(item[0]) and _is_int_like(item[1]):
                lo, hi = int(item[0]), int(item[1])
                if lo > hi:
                    lo, hi = hi, lo
                ids.update(range(lo, hi + 1))
                return
            for sub in item:
                _add(sub)
            return
        raise ValueError(f"Unsupported writer id spec element: {type(item)} ({item})")

    _add(spec)
    return sorted(ids)


def get_sample_split_test_list_from_cfg(env_cfg) -> Optional[str]:
    sample_cfg = _cfg_get(env_cfg, "SAMPLE_SPLIT", None)

    if sample_cfg is None:
        return None

    if isinstance(sample_cfg, bool):
        if not sample_cfg:
            return None
        raise ValueError(
            "ENV.SAMPLE_SPLIT=True requires dict form with TEST_LIST path "
            "(e.g., SAMPLE_SPLIT: {ENABLED: true, TEST_LIST: ...})."
        )

    sample_dict = _as_plain_dict(sample_cfg)
    enabled = bool(sample_dict.get("ENABLED", True))
    if not enabled:
        return None

    path = sample_dict.get("TEST_LIST", None)
    if not path:
        raise ValueError("ENV.SAMPLE_SPLIT.ENABLED=true requires SAMPLE_SPLIT.TEST_LIST")
    return str(path)


def get_sample_split_train_filter_options_from_cfg(
    env_cfg,
    default_max_sentence_len: int = 35,
) -> Tuple[bool, int]:
    sample_cfg = _cfg_get(env_cfg, "SAMPLE_SPLIT", None)
    if sample_cfg is None:
        return False, int(default_max_sentence_len)

    if isinstance(sample_cfg, bool):
        return False, int(default_max_sentence_len)

    sample_dict = _as_plain_dict(sample_cfg)
    enabled = bool(sample_dict.get("ENABLED", True))
    if not enabled:
        return False, int(default_max_sentence_len)

    apply_to_train = bool(sample_dict.get("APPLY_TO_TRAIN", False))
    max_sentence_len = int(sample_dict.get("MAX_SENTENCE_LEN", default_max_sentence_len))
    if max_sentence_len <= 0:
        max_sentence_len = int(default_max_sentence_len)
    return apply_to_train, max_sentence_len


def truncate_text_word_boundary(text: str, max_len: int = 35) -> str:
    s = str(text)
    if len(s) <= max_len:
        return s
    truncated = s[:max_len]
    last_space_idx = truncated.rfind(" ")
    if last_space_idx > 0:
        return truncated[:last_space_idx]
    return truncated


def normalize_text_for_sample_split(text: str, max_sentence_len: int = 35) -> str:
    t = truncate_text_word_boundary(str(text), max_sentence_len)
    return re.sub(r"[^\w\s]", "", t).lower().strip()


def load_sample_split_pair_quota(
    test_list_path: str,
    *,
    max_sentence_len: int = 35,
) -> Dict[Tuple[int, str], int]:
    with open(test_list_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    samples = data.get("samples", [])
    pair_quota: Dict[Tuple[int, str], int] = {}

    for sample in samples:
        wid_raw = sample.get("writer_id_cashg", sample.get("writer_id"))
        if wid_raw is None:
            continue
        try:
            wid = int(wid_raw)
        except Exception:
            continue

        text_norm = normalize_text_for_sample_split(
            str(sample.get("text", "")),
            max_sentence_len=max_sentence_len,
        )
        key = (wid, text_norm)
        pair_quota[key] = pair_quota.get(key, 0) + 1

    return pair_quota


def get_writer_split_policy_from_cfg(env_cfg, default_seed: int = 42) -> Tuple[str, Dict[str, Any]]:
    split_cfg = _cfg_get(env_cfg, "WRITER_SPLIT", None)

    if split_cfg is None:
        return "random", {"SEED": int(default_seed)}

    if isinstance(split_cfg, bool):
        if not split_cfg:
            return "random", {"SEED": int(default_seed)}


        train_list = _cfg_get(env_cfg, "TRAIN_LIST", None)
        test_list = _cfg_get(env_cfg, "TEST_LIST", None)
        if train_list is None or test_list is None:
            raise ValueError(
                "WRITER_SPLIT=true requires TRAIN_LIST and TEST_LIST "
                "(or use dict form WRITER_SPLIT: {ENABLED: true, TRAIN_LIST: ..., TEST_LIST: ...})."
            )
        return "explicit_lists", {"TRAIN_LIST": train_list, "TEST_LIST": test_list}

    split_dict = _as_plain_dict(split_cfg)
    enabled = bool(split_dict.get("ENABLED", True))
    seed = int(split_dict.get("SEED", default_seed))

    if not enabled:
        return "random", {
            "SEED": seed,
            "TRAIN_COUNT": split_dict.get("TRAIN_COUNT", None),
            "TEST_COUNT": split_dict.get("TEST_COUNT", None),
            "TRAIN_RATIO": split_dict.get("TRAIN_RATIO", None),
        }


    if ("TRAIN_LIST" in split_dict) and ("TEST_LIST" in split_dict):
        return "explicit_lists", {
            "TRAIN_LIST": split_dict.get("TRAIN_LIST"),
            "TEST_LIST": split_dict.get("TEST_LIST"),
            "SEED": seed,
        }


    mode = str(split_dict.get("MODE", "random")).strip().lower()
    if mode == "random":
        return "random", {
            "SEED": seed,
            "TRAIN_COUNT": split_dict.get("TRAIN_COUNT", None),
            "TEST_COUNT": split_dict.get("TEST_COUNT", None),
            "TRAIN_RATIO": split_dict.get("TRAIN_RATIO", None),
        }
    if mode in ("explicit_ranges", "ranges"):
        if "TRAIN_RANGE" not in split_dict or "TEST_RANGE" not in split_dict:
            raise ValueError("WRITER_SPLIT MODE=explicit_ranges requires TRAIN_RANGE and TEST_RANGE")
        return "explicit_ranges", {
            "TRAIN_RANGE": split_dict.get("TRAIN_RANGE"),
            "TEST_RANGE": split_dict.get("TEST_RANGE"),
            "SEED": seed,
        }
    if mode in ("id_threshold", "threshold"):
        if "TEST_ID_START" not in split_dict:
            raise ValueError("WRITER_SPLIT MODE=id_threshold requires TEST_ID_START")
        return "id_threshold", {
            "TEST_ID_START": int(split_dict.get("TEST_ID_START")),
            "SEED": seed,
        }

    raise ValueError(
        f"Unsupported WRITER_SPLIT mode: {mode}. "
        "Use TRAIN_LIST/TEST_LIST (recommended) or MODE in {random, explicit_ranges, id_threshold}."
    )


def _normalize_range_tuple(v, default: Tuple[int, int]) -> Tuple[int, int]:
    if v is None:
        return default
    if isinstance(v, (list, tuple)) and len(v) >= 2:
        a, b = int(v[0]), int(v[1])
        if a > b:
            a, b = b, a
        return (a, b)
    return default


def _select_ids_by_range(all_ids: List[int], r: Tuple[int, int]) -> List[int]:
    lo, hi = r
    return sorted([wid for wid in all_ids if lo <= wid <= hi])

def get_writer_ids_from_dir(
    pickle_root,
    seed=42,
    split_mode: Optional[str] = None,
    split_config: Optional[Dict[str, Any]] = None,
):
    import random

    writer_pkls = list_writer_pickles(pickle_root, include_rep=False)
    all_writer_ids = []
    for f in writer_pkls:
        basename = os.path.splitext(os.path.basename(f))[0]
        try:
            all_writer_ids.append(int(basename))
        except ValueError:

            continue
    all_writer_ids = sorted(all_writer_ids)


    mode = str(split_mode).strip().lower() if split_mode is not None else None
    cfg = dict(split_config or {})

    if mode is None:
        mode = "random"

    split_seed = int(cfg.get("SEED", seed))

    if mode in ("id_threshold", "casia_cn"):
        test_id_start = int(cfg.get("TEST_ID_START", 42000))
        train_writer_ids = [wid for wid in all_writer_ids if wid < test_id_start]
        test_writer_ids = [wid for wid in all_writer_ids if wid >= test_id_start]
        print(f"[INFO] Split Mode: id_threshold (test_id_start={test_id_start})")
        print(f"  Train: {len(train_writer_ids)} writers (< {test_id_start})")
        print(f"  Test: {len(test_writer_ids)} writers (>= {test_id_start})")
        return sorted(train_writer_ids), sorted(test_writer_ids)

    if mode in ("explicit_ranges", "ranges"):
        train_range = _normalize_range_tuple(cfg.get("TRAIN_RANGE"), (0, -1))
        test_range = _normalize_range_tuple(cfg.get("TEST_RANGE"), (0, -1))
        if train_range[1] < train_range[0] or test_range[1] < test_range[0]:
            raise ValueError("explicit_ranges requires valid TRAIN_RANGE and TEST_RANGE")
        train_writer_ids = _select_ids_by_range(all_writer_ids, train_range)
        test_writer_ids = _select_ids_by_range(all_writer_ids, test_range)
        print(f"[INFO] Split Mode: explicit_ranges (train={train_range}, test={test_range})")
        print(f"  Train: {len(train_writer_ids)} writers")
        print(f"  Test: {len(test_writer_ids)} writers")
        return sorted(train_writer_ids), sorted(test_writer_ids)

    if mode in ("explicit_lists", "writer_lists", "lists"):
        if "TRAIN_LIST" not in cfg or "TEST_LIST" not in cfg:
            raise ValueError("explicit_lists mode requires TRAIN_LIST and TEST_LIST")

        train_requested = _expand_writer_id_spec(cfg.get("TRAIN_LIST"))
        test_requested = _expand_writer_id_spec(cfg.get("TEST_LIST"))

        available_set = set(all_writer_ids)
        train_writer_ids = sorted([wid for wid in train_requested if wid in available_set])
        test_writer_ids = sorted([wid for wid in test_requested if wid in available_set])

        overlap = sorted(set(train_writer_ids).intersection(test_writer_ids))
        if overlap:
            print(f"[WARNING] explicit_lists overlap detected ({len(overlap)} ids).")
            print("[WARNING] Overlapped IDs are removed from TEST split.")
            overlap_set = set(overlap)
            test_writer_ids = [wid for wid in test_writer_ids if wid not in overlap_set]

        print("[INFO] Split Mode: explicit_lists")
        print(f"  Requested train IDs: {len(train_requested)} -> matched: {len(train_writer_ids)}")
        print(f"  Requested test IDs: {len(test_requested)} -> matched: {len(test_writer_ids)}")
        return train_writer_ids, test_writer_ids

    if mode != "random":
        print(f"[WARNING] Unknown split mode '{mode}', falling back to random.")


    cfg_train_count = cfg.get("TRAIN_COUNT", None)
    cfg_test_count = cfg.get("TEST_COUNT", None)
    cfg_train_ratio = cfg.get("TRAIN_RATIO", None)

    if cfg_train_count is None and cfg_test_count is None:

        train_ratio = float(0.8 if cfg_train_ratio is None else cfg_train_ratio)
        if not (0.0 < train_ratio < 1.0):
            raise ValueError(f"Invalid TRAIN_RATIO: {train_ratio}. Must be in (0,1).")
        cfg_train_count = int(len(all_writer_ids) * train_ratio)
        cfg_test_count = len(all_writer_ids) - cfg_train_count
    elif cfg_train_count is None:
        cfg_test_count = int(cfg_test_count)
        cfg_train_count = len(all_writer_ids) - cfg_test_count
    elif cfg_test_count is None:
        cfg_train_count = int(cfg_train_count)
        cfg_test_count = len(all_writer_ids) - cfg_train_count
    else:
        cfg_train_count = int(cfg_train_count)
        cfg_test_count = int(cfg_test_count)

    if cfg_train_count < 0 or cfg_test_count < 0:
        raise ValueError(f"Invalid random split counts: train={cfg_train_count}, test={cfg_test_count}")
    if cfg_train_count + cfg_test_count > len(all_writer_ids):
        raise ValueError(
            f"Not enough writer files for random split: train({cfg_train_count})+test({cfg_test_count}) > total({len(all_writer_ids)})."
        )

    random.seed(split_seed)
    shuffled_ids = all_writer_ids.copy()
    random.shuffle(shuffled_ids)

    train_writer_ids = sorted(shuffled_ids[:cfg_train_count])
    test_writer_ids = sorted(shuffled_ids[cfg_train_count:cfg_train_count + cfg_test_count])

    print(f"[INFO] Split Mode: random (seed={split_seed})")
    print(f"  Train: {len(train_writer_ids)} writers")
    print(f"  Test: {len(test_writer_ids)} writers")
    return train_writer_ids, test_writer_ids
