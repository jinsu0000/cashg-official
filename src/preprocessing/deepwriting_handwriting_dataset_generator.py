#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

from src.config.constants import PEN_CLASS, TRAJ_DIM
from src.data.data_utils import (
    guarantee_stroke_endpoints,
    resample_trajectory_preserve_endpoints,
    _deskew_trajectory,
)
from src.preprocessing.brush_handwriting_dataset_generator import (
    _apply_rdp,
    _audit_and_enforce_pen_logic_char,
    _choose_representative,
    _left_align_sample,
    _normalize_sentence_height,
    ensure_dir,
    render_image_from_traj,
)
from src.preprocessing.olhwd_cn_sent_dataset_generator import (
    visualize_cursive_eoc_pair,
)


@dataclass
class LoadedSplit:
    name: str
    data: Any
    classes: List[str]
    preprocessing_tokens: List[str]
    mean_xy: Optional[np.ndarray]
    std_xy: Optional[np.ndarray]


def _load_split(npz_path: str, split_name: str) -> LoadedSplit:
    data = np.load(npz_path, allow_pickle=True)
    alphabet = [str(c) for c in data["alphabet"]]

    classes = sorted(alphabet)
    preprocessing_tokens = [str(t) for t in data.get("preprocessing", [])]

    mean_xy = None
    std_xy = None
    if "mean" in data.files and "std" in data.files:
        mean_raw = np.asarray(data["mean"], dtype=np.float32).reshape(-1)
        std_raw = np.asarray(data["std"], dtype=np.float32).reshape(-1)
        if mean_raw.size >= 2 and std_raw.size >= 2:
            mean_xy = mean_raw[:2]
            std_xy = std_raw[:2]

    return LoadedSplit(
        name=split_name,
        data=data,
        classes=classes,
        preprocessing_tokens=preprocessing_tokens,
        mean_xy=mean_xy,
        std_xy=std_xy,
    )


def _to_abs_visual_space(
    stroke: np.ndarray,
    preprocessing_tokens: List[str],
    mean_xy: Optional[np.ndarray],
    std_xy: Optional[np.ndarray],
) -> Optional[np.ndarray]:
    if stroke is None:
        return None
    arr = np.asarray(stroke, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] < 3:
        return None

    out = arr[:, :3].copy()
    tokens = set(preprocessing_tokens)

    if "normalization" in tokens and mean_xy is not None and std_xy is not None:
        out[:, :2] = out[:, :2] * std_xy[None, :] + mean_xy[None, :]

    if "relative_representation" in tokens:
        out[:, :2] = np.cumsum(out[:, :2], axis=0)

    return out.astype(np.float32, copy=False)


def _majority_char(seg_labels: np.ndarray, classes: List[str]) -> str:
    if seg_labels.size == 0:
        return ""
    labels = np.asarray(seg_labels, dtype=np.int64).reshape(-1)
    labels = labels[(labels >= 0) & (labels < len(classes))]
    if labels.size == 0:
        return ""
    uniq, cnt = np.unique(labels, return_counts=True)
    idx = int(uniq[int(np.argmax(cnt))])
    return classes[idx]


def _build_segments(
    stroke_norm: np.ndarray,
    char_labels: np.ndarray,
    eoc_labels: np.ndarray,
    eow_labels: Optional[np.ndarray],
    classes: List[str],
    eoc_threshold: float,
    eow_threshold: float,
) -> List[Dict[str, Any]]:
    T = int(stroke_norm.shape[0])
    if T == 0:
        return []

    char_labels = np.asarray(char_labels).reshape(-1)
    eoc = np.asarray(eoc_labels).reshape(-1)
    if eow_labels is None:
        eow = np.zeros_like(eoc, dtype=np.float32)
    else:
        eow = np.asarray(eow_labels).reshape(-1)

    T = min(T, len(char_labels), len(eoc), len(eow))
    if T <= 0:
        return []

    stroke_norm = stroke_norm[:T]
    char_labels = char_labels[:T]
    eoc = eoc[:T]
    eow = eow[:T]

    boundaries = np.where(eoc > eoc_threshold)[0].tolist()
    if not boundaries:
        return []

    segments: List[Dict[str, Any]] = []
    start = 0
    for end in boundaries:
        if end < start:
            continue
        seg_traj = stroke_norm[start : end + 1]
        seg_labels = char_labels[start : end + 1]
        ch = _majority_char(seg_labels, classes)
        if ch:


            valid_mask = seg_labels != 0
            if valid_mask.any():
                seg_traj = seg_traj[valid_mask]
            else:
                start = end + 1
                continue

            word_end = bool(np.any(eow[start : end + 1] > eow_threshold))
            segments.append(
                {
                    "char": ch,
                    "traj_raw": seg_traj.astype(np.float32, copy=False),
                    "word_end": word_end,
                }
            )
        start = end + 1

    return segments


def _insert_spaces_by_word_end(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for i, seg in enumerate(segments):
        out.append(seg)
        if seg["word_end"] and i < len(segments) - 1:
            out.append(
                {
                    "char": " ",
                    "traj_raw": np.zeros((0, 3), dtype=np.float32),
                    "word_end": False,
                }
            )
    return out


def _mark_cursive_connections(
    tokens: List[Dict[str, Any]],
    pen_up_threshold: float,
    cursive_distance_threshold: float,
) -> None:
    for i, tok in enumerate(tokens):
        tok["is_cursive_connected"] = False
        if tok["char"] == " ":
            continue
        if i + 1 >= len(tokens):
            continue

        nxt = tokens[i + 1]
        if nxt["char"] == " ":
            continue

        curr_traj = tok["traj_raw"]
        next_traj = nxt["traj_raw"]
        if curr_traj.shape[0] == 0 or next_traj.shape[0] == 0:
            continue


        curr_pm_idx = np.where(curr_traj[:, 2] < pen_up_threshold)[0]
        next_pm_idx = np.where(next_traj[:, 2] < pen_up_threshold)[0]
        if curr_pm_idx.size == 0 or next_pm_idx.size == 0:
            continue

        curr_end_xy = curr_traj[int(curr_pm_idx[-1]), :2]
        next_start_xy = next_traj[int(next_pm_idx[0]), :2]
        dist = float(np.linalg.norm(curr_end_xy - next_start_xy))
        tok["is_cursive_connected"] = dist <= cursive_distance_threshold


def _pen_to_compact_pen_class(
    traj_raw: np.ndarray,
    pen_up_threshold: float,
    is_cursive_connected: bool,
) -> np.ndarray:
    pen_down = traj_raw[:, 2] < pen_up_threshold
    pen_class = np.where(
        pen_down,
        PEN_CLASS["PM"],
        PEN_CLASS["PU"],
    ).astype(np.float32)

    traj = np.concatenate(
        [traj_raw[:, :2].astype(np.float32), pen_class[:, None]],
        axis=1,
    ).astype(np.float32)

    if traj.shape[0] > 0:
        traj[-1, 2] = (
            PEN_CLASS["CURSIVE_EOC"] if is_cursive_connected else PEN_CLASS["EOC"]
        )
    return traj


def _build_space_item(
    cursor: int,
    sentence_id: str,
    scale: float,
    min_xy: np.ndarray,
    original_height: float,
    original_width: float,
) -> Dict[str, Any]:
    return {
        "trajectory": np.zeros((0, TRAJ_DIM), dtype=np.float32),
        "image": None,
        "connection": "space",
        "character": " ",
        "sentence_id": sentence_id,
        "cursor": int(cursor),
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


def _to_char_item(
    token: Dict[str, Any],
    *,
    cursor: int,
    sentence_id: str,
    scale: float,
    min_xy: np.ndarray,
    original_height: float,
    original_width: float,
    img_size: Tuple[int, int],
    max_traj_len: Optional[int],
    rdp_epsilon_normalized: Optional[float],
    pen_up_threshold: float,
    connection_type: str,
) -> Dict[str, Any]:
    traj_raw = token["traj_raw"].astype(np.float32, copy=True)
    ch = token["char"]
    is_cursive_connected = bool(token.get("is_cursive_connected", False))


    if traj_raw.shape[0] > 1:
        pen_down = traj_raw[:, 2] < pen_up_threshold
        if pen_down.any():
            first_pm_idx = int(np.argmax(pen_down))
            if first_pm_idx > 0:
                traj_raw = traj_raw[first_pm_idx:]

    before_points = int(traj_raw.shape[0])
    if before_points > 0:
        before_x_range = float(traj_raw[:, 0].max() - traj_raw[:, 0].min())
        before_y_range = float(traj_raw[:, 1].max() - traj_raw[:, 1].min())
    else:
        before_x_range = 0.0
        before_y_range = 0.0


    traj_compact = _pen_to_compact_pen_class(
        traj_raw,
        pen_up_threshold=pen_up_threshold,
        is_cursive_connected=is_cursive_connected,
    )


    traj_original = traj_compact.copy()


    if max_traj_len is not None and traj_compact.shape[0] > max_traj_len:
        traj_compact = resample_trajectory_preserve_endpoints(
            traj_compact, max_traj_len
        )


    MIN_POINTS_FOR_RDP = 10
    MAX_REDUCTION_RATIO = 0.5

    if rdp_epsilon_normalized is not None and rdp_epsilon_normalized > 0 and traj_compact.shape[0] > 0:
        original_point_count = traj_compact.shape[0]

        if original_point_count <= MIN_POINTS_FOR_RDP:
            pass
        else:
            min_target_points = int(original_point_count * (1.0 - MAX_REDUCTION_RATIO))

            traj_rdp = _apply_rdp(traj_compact, rdp_epsilon_normalized)
            rdp_point_count = traj_rdp.shape[0]


            if rdp_point_count < min_target_points:
                current_eps = rdp_epsilon_normalized
                for _ in range(5):
                    current_eps *= 0.5
                    traj_rdp = _apply_rdp(traj_compact, current_eps)
                    if traj_rdp.shape[0] >= min_target_points:
                        break

                if traj_rdp.shape[0] < min_target_points:
                    traj_rdp = traj_compact.copy()

            traj_compact = traj_rdp


    traj_compact = guarantee_stroke_endpoints(traj_original, traj_compact)


    traj_compact, _ = _audit_and_enforce_pen_logic_char(
        traj_compact,
        where="deepwriting_char",
        sid=sentence_id,
        char_idx=cursor,
        ch=ch,
        fix=True,
        is_cursive_connected=is_cursive_connected,
    )


    for i in range(traj_compact.shape[0]):
        if int(traj_compact[i, 2]) == PEN_CLASS["PU"] and i > 0:
            traj_compact[i, 0] = traj_compact[i - 1, 0]
            traj_compact[i, 1] = traj_compact[i - 1, 1]

    after_points = int(traj_compact.shape[0])
    if after_points > 0:
        after_x_range = float(traj_compact[:, 0].max() - traj_compact[:, 0].min())
        after_y_range = float(traj_compact[:, 1].max() - traj_compact[:, 1].min())
    else:
        after_x_range = 0.0
        after_y_range = 0.0

    img = render_image_from_traj(traj_compact, img_size)

    return {
        "trajectory": traj_compact.astype(np.float32, copy=False),
        "image": img.astype(np.uint8) if img is not None else None,
        "connection": connection_type,
        "character": ch,
        "sentence_id": sentence_id,
        "cursor": int(cursor),
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


def _compute_connection_type(tokens: List[Dict[str, Any]], idx: int) -> str:
    tok = tokens[idx]
    if tok.get("char") == " ":
        return "space"

    connected_right = bool(tok.get("is_cursive_connected", False))
    connected_left = False

    if idx > 0:
        prev = tokens[idx - 1]
        if prev.get("char") != " " and bool(prev.get("is_cursive_connected", False)):
            connected_left = True

    if connected_left and connected_right:
        return "connected_both"
    if connected_left:
        return "connected_left"
    if connected_right:
        return "connected_right"
    return "isolated"


def _select_alnum_style_chars(writer_chars: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[Dict[str, Any]]]:
    return {
        ch: samples
        for ch, samples in writer_chars.items()
        if ch.isalpha() or ch.isdigit()
    }


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


def generate_deepwriting_pickles(
    *,
    train_npz: str,
    valid_npz: str,
    source_split: str,
    save_char_root: str,
    save_sent_root: str,
    save_style_root: str,
    img_size: Tuple[int, int],
    writer_id_offset: int,
    compact_writer_ids: bool,
    rdp_epsilon_normalized: Optional[float],
    max_traj_len: Optional[int],
    eoc_threshold: float,
    eow_threshold: float,
    pen_up_threshold: float,
    cursive_distance_threshold: float,
    max_samples: Optional[int],
    max_writers: Optional[int],
    max_coord_after_norm: float,
    min_original_height: float,
    max_scale_after_norm: Optional[float],
    scale_overflow_mode: str,
    save_debug_vis: bool,
    visualize_cursive: bool = False,
    deskew_visualize: bool = False,
    deskew_angle_threshold: float = 0.0,
    deskew_max_angle: float = 0.0,
) -> None:
    ensure_dir(save_char_root)
    ensure_dir(save_sent_root)
    ensure_dir(save_style_root)

    cursive_visualize_dir: Optional[str] = None
    if visualize_cursive and save_style_root:
        cursive_visualize_dir = os.path.join(save_style_root, "_debug_cursive_eoc")
        ensure_dir(cursive_visualize_dir)


    deskew_visualize_dir: Optional[str] = None
    if deskew_visualize and save_style_root:
        deskew_visualize_dir = os.path.join(save_style_root, "_debug_deskew")
        ensure_dir(deskew_visualize_dir)

    splits: Dict[str, LoadedSplit] = {}
    if source_split in ("train", "all"):
        splits["train"] = _load_split(train_npz, "train")
    if source_split in ("valid", "all"):
        splits["valid"] = _load_split(valid_npz, "valid")
    if not splits:
        raise ValueError("No split selected.")


    raw_writer_ids = set()
    split_writer_indices: Dict[str, Dict[int, List[int]]] = {}
    for split_name, split in splits.items():
        writer_to_indices: Dict[int, List[int]] = defaultdict(list)
        subjects = np.asarray(split.data["subject_labels"]).reshape(-1)
        for i, wid in enumerate(subjects):
            w = int(wid)
            raw_writer_ids.add(w)
            writer_to_indices[w].append(i)
        split_writer_indices[split_name] = writer_to_indices

    selected_raw_writers = sorted(raw_writer_ids)
    if max_writers is not None:
        selected_raw_writers = selected_raw_writers[: max(0, int(max_writers))]

    if compact_writer_ids:
        writer_id_map = {
            raw_wid: int(writer_id_offset + idx)
            for idx, raw_wid in enumerate(selected_raw_writers)
        }
    else:
        writer_id_map = {
            raw_wid: int(raw_wid + writer_id_offset)
            for raw_wid in selected_raw_writers
        }

    print("=" * 80)
    print("[DeepWriting Dataset Generator]")
    print(f"Source split: {source_split}")
    print(f"Writers selected: {len(selected_raw_writers)}")
    print(f"Writer ID mode: {'compact+offset' if compact_writer_ids else 'raw+offset'}")
    print(f"Writer ID offset: {writer_id_offset}")
    print(f"RDP epsilon (normalized): {rdp_epsilon_normalized}")
    print(f"Max trajectory len: {max_traj_len}")
    clip_enabled = max_coord_after_norm > 0.0
    if clip_enabled:
        print(f"Max coord after norm: {max_coord_after_norm} (clipping enabled)")
    else:
        print("Max coord after norm: disabled (shape fidelity mode)")
    print(f"Min original height: {min_original_height}")
    print(f"Max scale after norm: {max_scale_after_norm}")
    print(f"Scale overflow mode: {scale_overflow_mode}")
    deskew_enabled = deskew_angle_threshold > 0.0 and deskew_max_angle > 0.0
    if deskew_enabled:
        print(
            f"Deskew: enabled (angle_threshold={deskew_angle_threshold}, max_angle={deskew_max_angle})"
        )
    else:
        print("Deskew: disabled")
    print("=" * 80)

    all_writers_info: List[Dict[str, Any]] = []
    total_sentences_saved = 0
    total_char_samples = 0
    processed_samples = 0
    skipped_tiny_height = 0
    skipped_scale_overflow = 0
    clamped_scale_count = 0
    norm_samples = 0
    clipped_point_count = 0
    total_point_count = 0
    scale_values: List[float] = []
    original_heights: List[float] = []

    stop_early = False
    for raw_wid in tqdm(selected_raw_writers, desc="Writers"):
        mapped_wid = writer_id_map[raw_wid]

        writer_chars: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        writer_sentences: Dict[str, List[Dict[str, Any]]] = {}
        split_counts = {"train": 0, "valid": 0}

        for split_name, split in splits.items():
            idx_list = split_writer_indices[split_name].get(raw_wid, [])
            if not idx_list:
                continue

            strokes = split.data["strokes"] if "strokes" in split.data.files else split.data["samples"]
            char_labels_all = split.data["char_labels"]
            eoc_labels_all = split.data["eoc_labels"]
            eow_labels_all = split.data["eow_labels"] if "eow_labels" in split.data.files else None

            for sample_idx in idx_list:
                if max_samples is not None and processed_samples >= max_samples:
                    stop_early = True
                    break

                stroke = np.asarray(strokes[sample_idx], dtype=np.float32)
                stroke_abs = _to_abs_visual_space(
                    stroke,
                    preprocessing_tokens=split.preprocessing_tokens,
                    mean_xy=split.mean_xy,
                    std_xy=split.std_xy,
                )
                if stroke_abs is None or stroke_abs.shape[0] == 0:
                    continue


                stroke_before_deskew = stroke_abs.copy() if deskew_visualize else None
                if deskew_enabled:
                    stroke_abs, skew_angle, was_deskewed = _deskew_trajectory(
                        stroke_abs,
                        angle_threshold=deskew_angle_threshold,
                        max_angle=deskew_max_angle
                    )
                else:
                    skew_angle = 0.0
                    was_deskewed = False


                if was_deskewed and deskew_visualize_dir and stroke_before_deskew is not None:
                    from src.preprocessing.merging_handwriting_dataset_generator import (
                        _visualize_deskew_before_after,
                    )
                    sentence_id = f"{split_name}_{sample_idx:07d}"
                    _visualize_deskew_before_after(
                        stroke_before_deskew,
                        stroke_abs,
                        skew_angle,
                        sentence_id,
                        deskew_visualize_dir,
                    )

                scale, x_min, y_min, original_height, original_width = _normalize_sentence_height(
                    stroke_abs,
                    target_height=1.0,
                )

                if min_original_height > 0.0 and original_height < min_original_height:
                    skipped_tiny_height += 1
                    continue

                if max_scale_after_norm is not None and max_scale_after_norm > 0.0 and scale > max_scale_after_norm:
                    if scale_overflow_mode == "skip":
                        skipped_scale_overflow += 1
                        continue
                    scale = float(max_scale_after_norm)
                    clamped_scale_count += 1

                raw_xy = np.stack(
                    [
                        (stroke_abs[:, 0] - x_min) * scale,
                        (stroke_abs[:, 1] - y_min) * scale,
                    ],
                    axis=1,
                ).astype(np.float32)

                if clip_enabled:
                    coord_limit = float(max_coord_after_norm)
                    xy_norm = np.stack(
                        [
                            np.clip(raw_xy[:, 0], -coord_limit, coord_limit),
                            np.clip(raw_xy[:, 1], -coord_limit, coord_limit),
                        ],
                        axis=1,
                    ).astype(np.float32)
                else:
                    xy_norm = raw_xy

                stroke_norm = np.concatenate(
                    [
                        xy_norm,
                        stroke_abs[:, 2:3].astype(np.float32),
                    ],
                    axis=1,
                ).astype(np.float32)
                norm_samples += 1
                scale_values.append(float(scale))
                original_heights.append(float(original_height))
                total_point_count += int(raw_xy.shape[0])
                if clip_enabled:
                    clip_mask = (
                        (raw_xy[:, 0] < -coord_limit)
                        | (raw_xy[:, 0] > coord_limit)
                        | (raw_xy[:, 1] < -coord_limit)
                        | (raw_xy[:, 1] > coord_limit)
                    )
                    clipped_point_count += int(np.count_nonzero(clip_mask))
                min_xy = np.array([x_min, y_min], dtype=np.float32)

                eow_labels = None if eow_labels_all is None else eow_labels_all[sample_idx]
                segments = _build_segments(
                    stroke_norm=stroke_norm,
                    char_labels=char_labels_all[sample_idx],
                    eoc_labels=eoc_labels_all[sample_idx],
                    eow_labels=eow_labels,
                    classes=split.classes,
                    eoc_threshold=eoc_threshold,
                    eow_threshold=eow_threshold,
                )
                if not segments:
                    continue

                tokens = _insert_spaces_by_word_end(segments)
                _mark_cursive_connections(
                    tokens,
                    pen_up_threshold=pen_up_threshold,
                    cursive_distance_threshold=cursive_distance_threshold,
                )

                sentence_id = f"{split_name}_{sample_idx:07d}"
                sentence_items: List[Dict[str, Any]] = []
                prev_char_item: Optional[Dict[str, Any]] = None

                for cursor, tok in enumerate(tokens):
                    if tok["char"] == " ":
                        item = _build_space_item(
                            cursor=cursor,
                            sentence_id=sentence_id,
                            scale=scale,
                            min_xy=min_xy,
                            original_height=original_height,
                            original_width=original_width,
                        )
                        prev_char_item = None
                    else:
                        connection_type = _compute_connection_type(tokens, cursor)
                        item = _to_char_item(
                            tok,
                            cursor=cursor,
                            sentence_id=sentence_id,
                            scale=scale,
                            min_xy=min_xy,
                            original_height=original_height,
                            original_width=original_width,
                            img_size=img_size,
                            max_traj_len=max_traj_len,
                            rdp_epsilon_normalized=rdp_epsilon_normalized,
                            pen_up_threshold=pen_up_threshold,
                            connection_type=connection_type,
                        )


                        if (
                            cursive_visualize_dir
                            and prev_char_item is not None
                            and prev_char_item["trajectory"].shape[0] > 0
                            and item["trajectory"].shape[0] > 0
                        ):
                            prev_traj = prev_char_item["trajectory"]
                            if int(prev_traj[-1, 2]) == PEN_CLASS["CURSIVE_EOC"]:
                                dist = float(
                                    np.linalg.norm(
                                        prev_traj[-1, :2] - item["trajectory"][0, :2]
                                    )
                                )
                                visualize_cursive_eoc_pair(
                                    prev_traj,
                                    item["trajectory"],
                                    prev_char_item["character"],
                                    item["character"],
                                    dist,
                                    sentence_id,
                                    cursor,
                                    cursive_visualize_dir,
                                )

                        prev_char_item = item

                    sentence_items.append(item)
                    writer_chars[item["character"]].append(item)
                    if item["character"] != " ":
                        total_char_samples += 1

                non_space_count = sum(1 for it in sentence_items if it["character"] != " ")
                if non_space_count >= 2:
                    writer_sentences[sentence_id] = sentence_items
                    total_sentences_saved += 1
                    split_counts[split_name] += 1

                processed_samples += 1

            if stop_early:
                break


        rep_map: Dict[str, Dict[str, Any]] = {}
        for ch, samples in writer_chars.items():
            if ch == " ":
                continue
            rep, _ = _choose_representative(samples)
            if rep is not None:
                rep_map[ch] = _left_align_sample(rep, img_size)

        with open(os.path.join(save_char_root, f"{mapped_wid}.pkl"), "wb") as f:
            pickle.dump(dict(writer_chars), f)
        with open(os.path.join(save_char_root, f"{mapped_wid}_rep.pkl"), "wb") as f:
            pickle.dump(rep_map, f)

        style_all = _select_alnum_style_chars(writer_chars)
        with open(os.path.join(save_style_root, f"{mapped_wid}.pkl"), "wb") as f:
            pickle.dump(style_all, f)

        style_rep = {
            ch: [rep]
            for ch, rep in rep_map.items()
            if (ch.isalpha() or ch.isdigit()) and rep.get("image") is not None
        }
        with open(os.path.join(save_style_root, f"{mapped_wid}_rep.pkl"), "wb") as f:
            pickle.dump(style_rep, f)

        with open(os.path.join(save_sent_root, f"{mapped_wid}_sent.pkl"), "wb") as f:
            pickle.dump({"sentences": writer_sentences}, f)

        all_writers_info.append(
            {
                "merged_id": int(mapped_wid),
                "original_id": str(raw_wid),
                "source": "DEEPWRITING",
                "num_chars": int(len(writer_chars)),
                "num_sentences": int(len(writer_sentences)),
                "num_train_sentences": int(split_counts["train"]),
                "num_valid_sentences": int(split_counts["valid"]),
            }
        )

        if stop_early:
            break

    mapping_path = os.path.join(save_char_root, "writer_mapping.json")
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(all_writers_info, f, indent=2, ensure_ascii=False)

    if save_debug_vis and all_writers_info:
        debug_output_dir = os.path.join(save_style_root, "_debug_style_rep_images")
        ensure_dir(debug_output_dir)

        writer_pool = all_writers_info
        print(f"[DEBUG] Saving style-rep visualization for {len(writer_pool)} writer(s)")
        for info in writer_pool:
            wid = str(info["merged_id"])
            orig = str(info.get("original_id", wid))
            _save_style_rep_debug_images(
                save_style_root,
                writer_id=wid,
                output_dir=debug_output_dir,
                prefix=f"DEEPWRITING_{orig}",
            )

    print("\n" + "=" * 80)
    print("[DeepWriting Dataset Generator] Done")
    print(f"Writers saved: {len(all_writers_info)}")
    print(f"Sentences saved: {total_sentences_saved}")
    print(f"Character samples saved: {total_char_samples}")
    print(f"Processed source samples: {processed_samples}")
    print(f"Skipped (tiny height): {skipped_tiny_height}")
    print(f"Skipped (scale overflow): {skipped_scale_overflow}")
    print(f"Scale clamped: {clamped_scale_count}")
    if norm_samples > 0:
        clip_ratio = (clipped_point_count / max(1, total_point_count)) * 100.0
        scales = np.asarray(scale_values, dtype=np.float32)
        heights = np.asarray(original_heights, dtype=np.float32)
        p50_scale = float(np.percentile(scales, 50))
        p95_scale = float(np.percentile(scales, 95))
        p99_scale = float(np.percentile(scales, 99))
        min_h = float(np.min(heights))
        p1_h = float(np.percentile(heights, 1))
        print(f"Normalized samples: {norm_samples}")
        print(f"Scale stats: p50={p50_scale:.3f} p95={p95_scale:.3f} p99={p99_scale:.3f}")
        print(f"Original height stats: min={min_h:.6f} p1={p1_h:.6f}")
        print(f"Clipped points: {clipped_point_count}/{total_point_count} ({clip_ratio:.3f}%)")
    print(f"Writer mapping: {mapping_path}")
    print("=" * 80)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate CASHG-format pickles from DeepWriting npz")
    parser.add_argument(
        "--train_npz",
        type=str,
        default="/root/0_DB/DeepWriting/data/deepwriting_training.npz",
        help="Path to deepwriting_training.npz",
    )
    parser.add_argument(
        "--valid_npz",
        type=str,
        default="/root/0_DB/DeepWriting/data/deepwriting_validation.npz",
        help="Path to deepwriting_validation.npz",
    )
    parser.add_argument(
        "--source_split",
        type=str,
        choices=["train", "valid", "all"],
        default="train",
        help="Which DeepWriting split(s) to convert",
    )
    parser.add_argument("--save_char_root", type=str, required=True, help="Output dir for char pickles")
    parser.add_argument("--save_sent_root", type=str, required=True, help="Output dir for sentence pickles")
    parser.add_argument("--save_style_root", type=str, required=True, help="Output dir for style pickles")
    parser.add_argument("--img_size", type=int, nargs=2, default=[64, 64], help="Image size (H W)")
    parser.add_argument("--writer_id_offset", type=int, default=0, help="Writer ID offset after mapping")
    parser.add_argument(
        "--compact_writer_ids",
        action="store_true",
        help="Remap writer IDs to contiguous [offset, offset+N)",
    )
    parser.add_argument(
        "--rdp_epsilon_normalized",
        type=float,
        default=0.0,
        help="RDP epsilon in normalized coordinate space (0.0 recommended for fidelity)",
    )
    parser.add_argument(
        "--max_traj_len",
        type=int,
        default=99,
        help="If >0 and char trajectory is longer, apply arc-length resampling (<=0 to disable)",
    )
    parser.add_argument("--eoc_threshold", type=float, default=0.5, help="EOC threshold")
    parser.add_argument("--eow_threshold", type=float, default=0.5, help="EOW threshold")
    parser.add_argument("--pen_up_threshold", type=float, default=0.5, help="Pen-up threshold")
    parser.add_argument(
        "--cursive_distance_threshold",
        type=float,
        default=0.05,
        help="Distance threshold for CURSIVE_EOC in normalized coordinates (0.005 too tight for DW)",
    )
    parser.add_argument("--max_samples", type=int, default=None, help="Optional cap on processed source samples")
    parser.add_argument("--max_writers", type=int, default=None, help="Optional cap on number of writers")
    parser.add_argument(
        "--max_coord_after_norm",
        type=float,
        default=0.0,
        help="Clip normalized x/y coordinates to [-v, v] (disabled by default for shape fidelity)",
    )
    parser.add_argument(
        "--min_original_height",
        type=float,
        default=0.0,
        help="Skip samples whose original sentence height is below this threshold (0 to disable)",
    )
    parser.add_argument(
        "--max_scale_after_norm",
        type=float,
        default=0.0,
        help="If >0, control excessive normalization scale with --scale_overflow_mode",
    )
    parser.add_argument(
        "--scale_overflow_mode",
        type=str,
        choices=["clamp", "skip"],
        default="clamp",
        help="How to handle samples above --max_scale_after_norm",
    )
    parser.add_argument("--save_debug_vis", action="store_true",
                        help="Save style representative grid PNGs to {save_style_root}/_debug_style_rep_images")
    parser.add_argument("--visualize_cursive", action="store_true",
                        help="Save CURSIVE_EOC pair debug images to {save_style_root}/_debug_cursive_eoc")
    parser.add_argument("--deskew_visualize", action="store_true",
                        help="Save deskew before/after debug images to {save_style_root}/_debug_deskew")
    parser.add_argument("--deskew_angle_threshold", type=float, default=0.0,
                        help="Minimum angle (degrees) to apply deskew correction (<=0 disables deskew)")
    parser.add_argument("--deskew_max_angle", type=float, default=0.0,
                        help="Maximum angle (degrees) to apply deskew correction (<=0 disables deskew)")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    generate_deepwriting_pickles(
        train_npz=args.train_npz,
        valid_npz=args.valid_npz,
        source_split=args.source_split,
        save_char_root=args.save_char_root,
        save_sent_root=args.save_sent_root,
        save_style_root=args.save_style_root,
        img_size=tuple(args.img_size),
        writer_id_offset=int(args.writer_id_offset),
        compact_writer_ids=bool(args.compact_writer_ids),
        rdp_epsilon_normalized=args.rdp_epsilon_normalized,
        max_traj_len=(None if (args.max_traj_len is not None and int(args.max_traj_len) <= 0) else args.max_traj_len),
        eoc_threshold=float(args.eoc_threshold),
        eow_threshold=float(args.eow_threshold),
        pen_up_threshold=float(args.pen_up_threshold),
        cursive_distance_threshold=float(args.cursive_distance_threshold),
        max_samples=args.max_samples,
        max_writers=args.max_writers,
        max_coord_after_norm=float(args.max_coord_after_norm),
        min_original_height=float(args.min_original_height),
        max_scale_after_norm=(
            None
            if (args.max_scale_after_norm is None or float(args.max_scale_after_norm) <= 0.0)
            else float(args.max_scale_after_norm)
        ),
        scale_overflow_mode=str(args.scale_overflow_mode),
        save_debug_vis=bool(args.save_debug_vis),
        visualize_cursive=bool(args.visualize_cursive),
        deskew_visualize=bool(args.deskew_visualize),
        deskew_angle_threshold=float(args.deskew_angle_threshold),
        deskew_max_angle=float(args.deskew_max_angle),
    )


if __name__ == "__main__":
    main()
