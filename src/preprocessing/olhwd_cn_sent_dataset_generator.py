import os
import pickle
import json
import numpy as np
from collections import defaultdict
from PIL import Image, ImageDraw
from tqdm import tqdm
from typing import Optional, Dict, List, Tuple, Any
import argparse
import random


from src.config.constants import (
    TRAJ_INDEX, TRAJ_DIM, PEN_CLASS
)
from src.data.data_utils import (
    resample_trajectory_preserve_endpoints,
    guarantee_stroke_endpoints,
    get_first_stroke,
    get_last_stroke,
    check_cursive_spatial_validity,
)


from src.preprocessing.brush_handwriting_dataset_generator import (
    _audit_and_enforce_pen_logic_char,
    _apply_rdp,
    render_image_from_traj,
    _choose_representative,
    _left_align_sample,
    ensure_dir,
    _normalize_sentence_height,
)
from src.data.data_utils import (
    _deskew_trajectory,
)


DEFAULT_POINTS_PER_CHAR = 160
MIN_CHAR_HEIGHT = 0.8
DEFAULT_BOUNDARY_TOLERANCE = 3


def _normalize_char_tag(ch: Any) -> str:
    if ch is None:
        return ""
    if isinstance(ch, str):
        return ch
    return str(ch)


def _is_valid_char_tag(ch: str) -> bool:
    if ch == "":
        return False
    if ch == " ":
        return True
    return len(ch) == 1


def _scale_up_small_char(sample: Dict[str, Any], min_height: float = MIN_CHAR_HEIGHT) -> Dict[str, Any]:
    if not isinstance(sample, dict):
        return sample

    traj = sample.get('trajectory')
    if traj is None or not isinstance(traj, np.ndarray) or len(traj) == 0:
        return sample


    y_min = float(traj[:, 1].min())
    y_max = float(traj[:, 1].max())
    curr_height = y_max - y_min


    if curr_height >= min_height or curr_height <= 0:
        return sample


    scale = min_height / curr_height


    new_sample = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in sample.items()}
    new_traj = traj.copy().astype(np.float32)


    x_min = float(traj[:, 0].min())
    new_traj[:, 0] = (new_traj[:, 0] - x_min) * scale
    new_traj[:, 1] = (new_traj[:, 1] - y_min) * scale

    new_sample['trajectory'] = new_traj
    new_sample['scale'] = sample.get('scale', 1.0) * scale
    new_sample['_scaled_up'] = True
    new_sample['_original_height'] = curr_height

    return new_sample


def _normalize_char_height(sample: Dict[str, Any], target_height: float = 1.0) -> Dict[str, Any]:
    if not isinstance(sample, dict):
        return sample
    traj = sample.get('trajectory')
    if traj is None or not isinstance(traj, np.ndarray) or len(traj) == 0:
        return sample

    new_sample = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in sample.items()}
    new_traj = traj.copy().astype(np.float32)
    scale, cx_min, cy_min, orig_h, orig_w = _normalize_sentence_height(new_traj, target_height=target_height)
    new_traj[:, 0] = (new_traj[:, 0] - cx_min) * scale
    new_traj[:, 1] = (new_traj[:, 1] - cy_min) * scale
    new_sample['trajectory'] = new_traj
    new_sample['scale'] = scale
    new_sample['_rep_char_normalized'] = True
    return new_sample


def _check_boundary_stroke_alignment(
    trajectory: np.ndarray,
    durations: List[int],
    tolerance: int = DEFAULT_BOUNDARY_TOLERANCE
) -> Tuple[bool, List[int]]:
    pen_state = trajectory[:, 2].astype(int)


    stroke_ends = set(np.where(pen_state == -1)[0])


    boundary_indices = []
    cumsum = 0
    for d in durations[:-1]:
        cumsum += d
        boundary_indices.append(cumsum - 1)


    misaligned = []
    for idx, boundary in enumerate(boundary_indices):

        found = False
        for offset in range(-tolerance, tolerance + 1):
            check_pos = boundary + offset
            if 0 <= check_pos < len(pen_state):
                if check_pos in stroke_ends or pen_state[check_pos] == -1:
                    found = True
                    break
        if not found:
            misaligned.append(idx)

    is_valid = len(misaligned) == 0
    return is_valid, misaligned


_preprocessing_stats = {
    'resample': {
        'total_checked': 0,
        'applied': 0,
        'before_lengths': [],
        'after_lengths': [],
    },
    'rdp': {
        'total_checked': 0,
        'applied': 0,
        'before_lengths': [],
        'after_lengths': [],
    },
    'cursive_eoc': {
        'total_char_boundaries': 0,
        'cursive_detected': 0,
        'cursive_rejected_spatial': 0,
        'distances': [],
        'rejection_reasons': [],
    },
    'boundary_check': {
        'total_sentences': 0,
        'dropped_boundary_mismatch': 0,
        'dropped_duration_mismatch': 0,
    },
    'deskew': {
        'applied': 0,
        'angles': [],
    },
    'char_points': [],
    'sentence_points': [],
    'sentence_chars': [],
}

def reset_preprocessing_stats():
    global _preprocessing_stats
    _preprocessing_stats = {
        'resample': {'total_checked': 0, 'applied': 0, 'before_lengths': [], 'after_lengths': []},
        'rdp': {'total_checked': 0, 'applied': 0, 'before_lengths': [], 'after_lengths': []},
        'cursive_eoc': {'total_char_boundaries': 0, 'cursive_detected': 0, 'cursive_rejected_spatial': 0, 'distances': [], 'rejection_reasons': []},
        'boundary_check': {'total_sentences': 0, 'dropped_boundary_mismatch': 0, 'dropped_duration_mismatch': 0},
        'deskew': {'applied': 0, 'angles': []},
        'char_points': [],
        'sentence_points': [],
        'sentence_chars': [],
    }


def print_preprocessing_stats(points_per_char: int):
    global _preprocessing_stats

    print("\n" + "="*80)
    print(" 전처리 통계 요약 (Preprocessing Statistics)")
    print("="*80)


    rs = _preprocessing_stats['resample']
    print(f"\n[1] Resample (max_points_per_char={points_per_char} 기준)")
    print(f"    - 체크한 글자: {rs['total_checked']:,}개")
    print(f"    - 리샘플링된 글자: {rs['applied']:,}개 ({100*rs['applied']/max(1,rs['total_checked']):.2f}%)")
    if rs['before_lengths']:
        before = np.array(rs['before_lengths'])
        after = np.array(rs['after_lengths'])
        print(f"    - 리샘플링 전: min={before.min()}, max={before.max()}, mean={before.mean():.1f}")
        print(f"    - 리샘플링 후: min={after.min()}, max={after.max()}, mean={after.mean():.1f}")


    rdp = _preprocessing_stats['rdp']
    print(f"\n[2] RDP (Douglas-Peucker)")
    print(f"    - 체크한 글자: {rdp['total_checked']:,}개")
    print(f"    - 적용된 글자: {rdp['applied']:,}개 ({100*rdp['applied']/max(1,rdp['total_checked']):.2f}%)")


    ce = _preprocessing_stats['cursive_eoc']
    print(f"\n[3] CURSIVE_EOC Detection")
    print(f"    - 총 글자 경계: {ce['total_char_boundaries']:,}개")
    print(f"    - CURSIVE_EOC 감지: {ce['cursive_detected']:,}개 ({100*ce['cursive_detected']/max(1,ce['total_char_boundaries']):.2f}%)")
    print(f"    - 공간적 검증 거부: {ce['cursive_rejected_spatial']:,}개 (segmentation 오류 의심)")
    if ce['distances']:
        dists = np.array(ce['distances'])
        print(f"    - 글자간 거리: min={dists.min():.4f}, max={dists.max():.4f}, mean={dists.mean():.4f}")


    if ce['rejection_reasons']:
        from collections import Counter
        reason_counts = Counter(ce['rejection_reasons'])
        print(f"    - 거부 사유 (상위 5개):")
        for reason, count in reason_counts.most_common(5):
            print(f"        {reason}: {count}개")


    bc = _preprocessing_stats['boundary_check']
    total_dropped = bc['dropped_boundary_mismatch'] + bc['dropped_duration_mismatch']
    print(f"\n[4] Boundary-Stroke Alignment Check")
    print(f"    - 총 문장: {bc['total_sentences']:,}개")
    print(f"    - 경계 불일치로 drop: {bc['dropped_boundary_mismatch']:,}개 ({100*bc['dropped_boundary_mismatch']/max(1,bc['total_sentences']):.2f}%)")
    print(f"    - duration 합 불일치로 drop: {bc['dropped_duration_mismatch']:,}개 ({100*bc['dropped_duration_mismatch']/max(1,bc['total_sentences']):.2f}%)")
    print(f"    - 유효 문장: {bc['total_sentences'] - total_dropped:,}개")


    ds = _preprocessing_stats.get('deskew', {'applied': 0, 'angles': []})
    valid_sentences = bc['total_sentences'] - total_dropped
    print(f"\n[5] Deskew (기울기 보정)")
    print(f"    - 보정된 문장: {ds['applied']:,}개 ({100*ds['applied']/max(1,valid_sentences):.2f}%)")
    if ds['angles']:
        angles = np.array(ds['angles'])
        print(f"    - 보정 각도: min={angles.min():.2f}°, max={angles.max():.2f}°, mean={abs(angles).mean():.2f}°")


    print(f"\n[6] 글자별 포인트 통계 (AFTER)")
    if _preprocessing_stats['char_points']:
        char_pts = np.array(_preprocessing_stats['char_points'])
        print(f"    - 샘플 수: {len(char_pts):,}개 글자")
        print(f"    - 포인트/글자: 평균={char_pts.mean():.1f}, 최소={char_pts.min()}, 최대={char_pts.max()}")


        bins = [0, 10, 20, 50, 80, 100, 150, 200, float('inf')]
        labels = ['1-10', '11-20', '21-50', '51-80', '81-100', '101-150', '151-200', '200+']
        dist_counts = [0] * len(labels)
        for p in char_pts:
            for i, (lower, upper) in enumerate(zip(bins[:-1], bins[1:])):
                if lower < p <= upper:
                    dist_counts[i] += 1
                    break
        print(f"    - 분포:")
        for label, count in zip(labels, dist_counts):
            pct = count / len(char_pts) * 100
            if count > 0:
                print(f"        {label:8s}: {count:6,} ({pct:5.1f}%)")


    print(f"\n[7] 문장별 통계 (AFTER)")
    if _preprocessing_stats['sentence_points']:
        sent_pts = np.array(_preprocessing_stats['sentence_points'])
        sent_chars = np.array(_preprocessing_stats['sentence_chars'])
        print(f"    - 샘플 수: {len(sent_pts):,}개 문장")
        print(f"    - 포인트/문장: 평균={sent_pts.mean():.1f}, 최소={sent_pts.min()}, 최대={sent_pts.max()}")
        print(f"    - 글자/문장: 평균={sent_chars.mean():.1f}, 최소={sent_chars.min()}, 최대={sent_chars.max()}")

    print("\n" + "="*80)


def visualize_cursive_eoc_pair(
    prev_char_traj: np.ndarray,
    curr_char_traj: np.ndarray,
    prev_char: str,
    curr_char: str,
    distance: float,
    sentence_id: str,
    char_idx: int,
    output_dir: str,
    img_size: Tuple[int, int] = (200, 200)
) -> str:
    H, W = img_size
    combined_traj = np.vstack([prev_char_traj, curr_char_traj])
    x, y = combined_traj[:, 0], combined_traj[:, 1]

    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    x_range = max(x_max - x_min, 0.1)
    y_range = max(y_max - y_min, 0.1)

    margin = 15
    scale = min((W - 2 * margin) / x_range, (H - 2 * margin) / y_range)
    x_offset = margin + (W - 2 * margin - x_range * scale) / 2
    y_offset = margin + (H - 2 * margin - y_range * scale) / 2

    canvas = np.ones((H, W, 3), dtype=np.uint8) * 255
    img_pil = Image.fromarray(canvas)
    draw = ImageDraw.Draw(img_pil)

    prev_color, curr_color = (50, 50, 200), (200, 50, 50)

    def draw_traj(traj, color):
        if len(traj) < 2:
            return
        for i in range(len(traj) - 1):
            if traj[i, 2] in [PEN_CLASS['PM'], PEN_CLASS['CURSIVE_EOC']]:
                x1 = int((traj[i, 0] - x_min) * scale + x_offset)
                y1 = int((traj[i, 1] - y_min) * scale + y_offset)
                x2 = int((traj[i+1, 0] - x_min) * scale + x_offset)
                y2 = int((traj[i+1, 1] - y_min) * scale + y_offset)
                draw.line([(x1, y1), (x2, y2)], fill=color, width=2)

    draw_traj(prev_char_traj, prev_color)
    draw_traj(curr_char_traj, curr_color)


    px = int((prev_char_traj[-1, 0] - x_min) * scale + x_offset)
    py = int((prev_char_traj[-1, 1] - y_min) * scale + y_offset)
    cx = int((curr_char_traj[0, 0] - x_min) * scale + x_offset)
    cy = int((curr_char_traj[0, 1] - y_min) * scale + y_offset)
    draw.line([(px, py), (cx, cy)], fill=(0, 180, 0), width=1)


    try:
        from PIL import ImageFont
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except:
        font = None

    draw.text((5, 3), f"'{prev_char}' → '{curr_char}'", fill=(0, 0, 0), font=font)
    draw.text((5, 25), f"dist: {distance:.4f}", fill=(0, 120, 0), font=font)

    ensure_dir(output_dir)
    safe_sid = sentence_id.replace("/", "_").replace("\\", "_").replace(":", "_")[:30]
    output_path = os.path.join(output_dir, f"cursive_pair_{safe_sid}_char{char_idx}.png")
    img_pil.save(output_path)
    return output_path


def visualize_sentence_with_colors(
    char_items: List[Dict],
    sentence_id: str,
    output_dir: str,
    img_size: Tuple[int, int] = (150, 800)
) -> str:
    H, W = img_size


    all_points = []
    for item in char_items:
        traj = item.get('trajectory')
        if traj is not None and len(traj) > 0:
            all_points.extend(traj.tolist())

    if not all_points:
        return None

    all_traj = np.array(all_points, dtype=np.float32)
    x, y = all_traj[:, 0], all_traj[:, 1]

    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    x_range = max(x_max - x_min, 0.1)
    y_range = max(y_max - y_min, 0.1)

    margin = 20
    scale = min((W - 2 * margin) / x_range, (H - 2 * margin) / y_range)
    x_offset = margin + (W - 2 * margin - x_range * scale) / 2
    y_offset = margin + (H - 2 * margin - y_range * scale) / 2

    canvas = np.ones((H, W, 3), dtype=np.uint8) * 255
    img_pil = Image.fromarray(canvas)
    draw = ImageDraw.Draw(img_pil)


    colors = [
        (50, 50, 50),
        (200, 50, 50),
        (50, 150, 50),
        (50, 50, 200),
        (150, 100, 50),
        (150, 50, 150),
        (50, 150, 150),
        (200, 100, 50),
    ]


    for char_idx, item in enumerate(char_items):
        traj = item.get('trajectory')
        if traj is None or len(traj) < 2:
            continue

        color = colors[char_idx % len(colors)]
        pen_class = traj[:, 2]

        for i in range(len(traj) - 1):
            if pen_class[i] in [PEN_CLASS['PM'], PEN_CLASS['CURSIVE_EOC']]:
                x1 = int((traj[i, 0] - x_min) * scale + x_offset)
                y1 = int((traj[i, 1] - y_min) * scale + y_offset)
                x2 = int((traj[i+1, 0] - x_min) * scale + x_offset)
                y2 = int((traj[i+1, 1] - y_min) * scale + y_offset)
                draw.line([(x1, y1), (x2, y2)], fill=color, width=1)


    text = ''.join([item['character'] for item in char_items])[:40]


    try:
        from PIL import ImageFont
        font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except:
        font_large = None
        font_small = None


    bbox = draw.textbbox((5, 3), text, font=font_large) if font_large else (5, 3, 200, 25)
    draw.rectangle(bbox, fill=(255, 255, 255, 200))

    draw.text((5, 3), text, fill=(0, 0, 0), font=font_large)
    draw.text((5, H-30), f"{sentence_id[:30]}", fill=(100, 100, 100), font=font_small)
    draw.text((5, H-15), f"chars:{len(char_items)} pts:{len(all_traj)}", fill=(100, 100, 100), font=font_small)

    ensure_dir(output_dir)
    safe_sid = sentence_id.replace("/", "_").replace("\\", "_").replace(":", "_")[:40]
    output_path = os.path.join(output_dir, f"sent_{safe_sid}.png")
    img_pil.save(output_path)
    return output_path


def load_olhwd_char_data(olhwd_dir: str) -> Dict[int, List[Tuple[str, np.ndarray]]]:
    datas_path = os.path.join(olhwd_dir, "datas.npy")
    writer_char_data = {}

    if os.path.exists(datas_path):
        print(f"[OLHWD-CHAR] 낱자 데이터 로드 중: {datas_path}")
        data = np.load(datas_path, allow_pickle=True)

        for writer_idx, writer_samples in enumerate(data):
            char_list = []
            for char, coords in writer_samples:
                if isinstance(coords, np.ndarray) and coords.ndim == 2 and coords.shape[1] >= 3:

                    coords_abs = coords.copy()
                    coords_abs[:, 0:2] = np.cumsum(coords_abs[:, 0:2], axis=0)


                    char_list.append((char, coords_abs))
            if char_list:
                writer_char_data[writer_idx] = char_list

        print(f"[OLHWD-CHAR] {len(writer_char_data)}명 작가 로드 완료")

    return writer_char_data


def process_olhwd_writer_sentences(
    sample: Dict,
    writer_id: int,
    img_size: Tuple[int, int],
    points_per_char: int,
    rdp_epsilon: Optional[float],
    cursive_threshold: float,
    cursive_visualize_dir: Optional[str],
    sentence_visualize_dir: Optional[str],
    boundary_tolerance: int = DEFAULT_BOUNDARY_TOLERANCE,
) -> Tuple[Dict[str, List], Dict[str, List[Dict]]]:
    global _preprocessing_stats

    H, W = img_size
    writer_chars = defaultdict(list)
    writer_sentences = {}

    line_num = sample.get('line_num', 0)

    for line_id in range(line_num + 1):
        line_key = str(line_id)
        line_tag_key = f'{line_id}tag'
        line_duration_key = f'{line_id}duration'

        if line_key not in sample:
            continue

        trajectory = sample[line_key]
        if trajectory is None or len(trajectory) == 0:
            continue
        if not isinstance(trajectory, np.ndarray) or trajectory.ndim != 2 or trajectory.shape[1] < 3:
            continue


        tags = []
        if line_tag_key in sample:
            tag = sample[line_tag_key]
            if isinstance(tag, (list, np.ndarray)):
                tags = [str(t) if isinstance(t, (str, np.str_)) else '?' for t in tag]

        durations = []
        if line_duration_key in sample:
            duration = sample[line_duration_key]
            if isinstance(duration, (list, np.ndarray)):
                durations = [int(d) for d in duration]

        if len(tags) != len(durations) or len(tags) == 0:
            continue

        sentence_id = f"writer_{writer_id}_line_{line_id}"

        _preprocessing_stats['boundary_check']['total_sentences'] += 1


        duration_sum = sum(durations)
        traj_len = len(trajectory)
        if duration_sum != traj_len:

            _preprocessing_stats['boundary_check']['dropped_duration_mismatch'] += 1
            continue


        is_valid, misaligned = _check_boundary_stroke_alignment(
            trajectory, durations, tolerance=boundary_tolerance
        )
        if not is_valid:

            _preprocessing_stats['boundary_check']['dropped_boundary_mismatch'] += 1
            continue


        trajectory_abs = trajectory.copy()
        trajectory_abs[:, 0:2] = np.cumsum(trajectory_abs[:, 0:2], axis=0)


        trajectory_abs, skew_angle, was_deskewed = _deskew_trajectory(
            trajectory_abs, angle_threshold=1.0, max_angle=30.0
        )
        if was_deskewed:
            _preprocessing_stats['deskew'] = _preprocessing_stats.get('deskew', {'applied': 0, 'angles': []})
            _preprocessing_stats['deskew']['applied'] += 1
            _preprocessing_stats['deskew']['angles'].append(skew_angle)


        x_min = float(trajectory_abs[:, 0].min())
        x_max = float(trajectory_abs[:, 0].max())
        y_min = float(trajectory_abs[:, 1].min())
        y_max = float(trajectory_abs[:, 1].max())

        sent_height = y_max - y_min
        if sent_height > 0:
            sent_scale = 1.0 / sent_height
        else:
            sent_width = x_max - x_min
            sent_scale = 1.0 / sent_width if sent_width > 0 else 1.0


        trajectory_abs[:, 0] = (trajectory_abs[:, 0] - x_min) * sent_scale
        trajectory_abs[:, 1] = (trajectory_abs[:, 1] - y_min) * sent_scale


        char_items = []
        prev_char_item = None
        traj_start = 0

        for char_idx, (char, char_duration) in enumerate(zip(tags, durations)):
            if traj_start >= len(trajectory_abs):
                break

            traj_end = min(traj_start + char_duration, len(trajectory_abs))
            char_traj_raw = trajectory_abs[traj_start:traj_end].copy()

            if len(char_traj_raw) == 0:
                traj_start = traj_end
                continue


            char = _normalize_char_tag(char)
            if not _is_valid_char_tag(char):
                traj_start = traj_end

                prev_char_item = None
                continue


            char_points = []
            for pt_idx, pt in enumerate(char_traj_raw):
                x, y, pen_raw = pt[0], pt[1], int(pt[2]) if not np.isnan(pt[2]) else 1
                is_char_end = (pt_idx == len(char_traj_raw) - 1)

                if is_char_end:
                    pen_class_val = PEN_CLASS['EOC']
                elif pen_raw == -1:
                    pen_class_val = PEN_CLASS['PU']
                else:
                    pen_class_val = PEN_CLASS['PM']

                char_points.append([x, y, pen_class_val])

            char_traj = np.array(char_points, dtype=np.float32)


            char_traj_original = char_traj.copy()


            is_cursive = False
            distance = float('inf')

            if prev_char_item is not None:
                prev_traj = prev_char_item['trajectory']
                prev_end = prev_traj[-1, :2]
                curr_start = char_traj[0, :2]
                distance = np.sqrt(np.sum((prev_end - curr_start) ** 2))

                _preprocessing_stats['cursive_eoc']['total_char_boundaries'] += 1
                _preprocessing_stats['cursive_eoc']['distances'].append(distance)

                if distance <= cursive_threshold:


                    is_spatial_valid, reason = check_cursive_spatial_validity(
                        prev_traj, char_traj, tolerance=0.1
                    )

                    if is_spatial_valid:

                        is_cursive = True
                        _preprocessing_stats['cursive_eoc']['cursive_detected'] += 1


                        prev_char_item['trajectory'][-1, 2] = PEN_CLASS['CURSIVE_EOC']


                        if cursive_visualize_dir:
                            visualize_cursive_eoc_pair(
                                prev_char_item['trajectory'], char_traj,
                                prev_char_item['character'], char,
                                distance, sentence_id, char_idx,
                                cursive_visualize_dir
                            )
                    else:

                        _preprocessing_stats['cursive_eoc']['cursive_rejected_spatial'] += 1
                        _preprocessing_stats['cursive_eoc']['rejection_reasons'].append(reason)


                        if cursive_visualize_dir:
                            rejected_dir = os.path.join(cursive_visualize_dir, "_rejected")
                            ensure_dir(rejected_dir)
                            visualize_cursive_eoc_pair(
                                prev_char_item['trajectory'], char_traj,
                                prev_char_item['character'], char,
                                distance, sentence_id, char_idx,
                                rejected_dir
                            )


            _preprocessing_stats['resample']['total_checked'] += 1
            if len(char_traj) > points_per_char:
                _preprocessing_stats['resample']['applied'] += 1
                _preprocessing_stats['resample']['before_lengths'].append(len(char_traj))
                char_traj = resample_trajectory_preserve_endpoints(char_traj, points_per_char)
                _preprocessing_stats['resample']['after_lengths'].append(len(char_traj))


            if rdp_epsilon is not None and rdp_epsilon > 0:
                _preprocessing_stats['rdp']['total_checked'] += 1
                before_len = len(char_traj)
                char_traj = _apply_rdp(char_traj, rdp_epsilon)
                if len(char_traj) < before_len:
                    _preprocessing_stats['rdp']['applied'] += 1


            char_traj = guarantee_stroke_endpoints(char_traj_original, char_traj)


            conn_type = "connected_left" if is_cursive else "isolated"
            is_cursive_connected = is_cursive

            char_traj, _ = _audit_and_enforce_pen_logic_char(
                char_traj, where="olhwd_char", sid=sentence_id, char_idx=char_idx,
                ch=char, fix=True, is_cursive_connected=False
            )


            img = render_image_from_traj(char_traj, (H, W), line_width=1, use_antialiasing=False)


            _preprocessing_stats['char_points'].append(len(char_traj))

            char_item = {
                "trajectory": char_traj.astype(np.float32),
                "image": img.astype(np.uint8) if img is not None else None,
                "connection": conn_type,
                "character": char,
                "sentence_id": sentence_id,
                "cursor": char_idx,
                "scale": sent_scale,
                "min_xy": np.array([0.0, 0.0], dtype=np.float32),
                "source_dataset": "OLHWD",
                "_before_points": len(char_traj_raw),
                "_after_points": len(char_traj),
            }


            writer_chars[char].append(char_item)
            char_items.append(char_item)
            prev_char_item = char_item
            traj_start = traj_end

        if len(char_items) >= 2:

            non_space = [it for it in char_items if it.get("character") not in (" ", "")]
            if len(non_space) == 0:
                continue
            writer_sentences[sentence_id] = char_items


            total_pts = sum(len(item['trajectory']) for item in char_items)
            _preprocessing_stats['sentence_points'].append(total_pts)
            _preprocessing_stats['sentence_chars'].append(len(char_items))


            if sentence_visualize_dir and line_id < 3:
                visualize_sentence_with_colors(char_items, sentence_id, sentence_visualize_dir)

    return dict(writer_chars), writer_sentences


def _save_style_rep_debug_images(style_root: str, writer_id: int, output_dir: str):
    rep_pkl_path = os.path.join(style_root, f"{writer_id}_rep.pkl")

    if not os.path.exists(rep_pkl_path):
        print(f"   rep 파일 없음: {rep_pkl_path}")
        return

    with open(rep_pkl_path, 'rb') as f:
        rep_map = pickle.load(f)

    images = []
    labels = []

    for char, samples in sorted(rep_map.items()):
        sample = samples[0] if isinstance(samples, list) else samples
        if isinstance(sample, dict):
            img = sample.get('image')
            if img is not None:
                images.append(np.array(img))
                labels.append(char)

    if not images:
        print(f"   이미지 없음")
        return


    if len(images) > 50:
        indices = random.sample(range(len(images)), 50)
        images = [images[i] for i in indices]
        labels = [labels[i] for i in indices]

    n = len(images)
    cols = min(10, n)
    rows = (n + cols - 1) // cols

    h, w = images[0].shape[:2]
    cell_h, cell_w = h + 20, w + 4

    grid = np.ones((rows * cell_h, cols * cell_w), dtype=np.uint8) * 255

    for idx, img in enumerate(images):
        row, col = idx // cols, idx % cols
        y, x = row * cell_h + 18, col * cell_w + 2
        if len(img.shape) == 3:
            img = img[:, :, 0]
        grid[y:y+h, x:x+w] = img

    grid_pil = Image.fromarray(grid)
    output_path = os.path.join(output_dir, f"OLHWD_wid{writer_id}_style_rep.png")
    grid_pil.save(output_path)
    print(f"   Style Rep 저장: {output_path} ({len(images)}개 글자)")


def generate_olhwd_dataset(
    olhwd_root: str,
    save_char_root: Optional[str] = None,
    save_sent_root: Optional[str] = None,
    save_style_root: Optional[str] = None,
    img_size: Tuple[int, int] = (64, 64),
    points_per_char: int = DEFAULT_POINTS_PER_CHAR,
    rdp_epsilon: Optional[float] = None,
    cursive_threshold: float = 0.03,
    boundary_tolerance: int = DEFAULT_BOUNDARY_TOLERANCE,
    max_writers: Optional[int] = None,
    visualize_cursive: bool = False,
    save_debug_vis: bool = False,
):

    if save_char_root:
        ensure_dir(save_char_root)
    if save_sent_root:
        ensure_dir(save_sent_root)
    if save_style_root:
        ensure_dir(save_style_root)

    if not any([save_char_root, save_sent_root, save_style_root]):
        raise ValueError("최소 하나의 save_*_root 경로를 지정해야 합니다.")

    reset_preprocessing_stats()

    H, W = img_size

    cursive_visualize_dir = None
    if visualize_cursive and save_style_root:
        cursive_visualize_dir = os.path.join(save_style_root, "_debug_cursive_eoc")
        ensure_dir(cursive_visualize_dir)

    sentence_visualize_dir = None
    if save_debug_vis and save_style_root:
        sentence_visualize_dir = os.path.join(save_style_root, "_debug_sentence_vis")
        ensure_dir(sentence_visualize_dir)

    print("=" * 80)
    print(" OLHWD 중국어 손글씨 데이터셋 생성")
    print(f"    Char:  {'생성' if save_char_root else '스킵'}")
    print(f"    Sent:  {'생성' if save_sent_root else '스킵'}")
    print(f"    Style: {'생성' if save_style_root else '스킵'}")
    print("=" * 80)
    print(f"OLHWD 루트: {olhwd_root}")
    print(f"이미지 크기: {img_size}")
    print(f"글자당 최대 포인트: {points_per_char} (resample 기준)")
    print(f"RDP epsilon: {rdp_epsilon if rdp_epsilon else '미적용'}")
    print(f"CURSIVE_EOC threshold: {cursive_threshold}")
    print(f"Boundary tolerance: ±{boundary_tolerance} (duration 경계 검사)")
    print(f"시각화 (Cursive): {'활성화' if visualize_cursive else '비활성화'}")
    print(f"시각화 (Debug): {'활성화' if save_debug_vis else '비활성화'}")
    print("=" * 80)
    print()


    print("[1] 문장 데이터 로드 중...")
    all_datas_path = os.path.join(olhwd_root, "all_datas.npy")

    if not os.path.exists(all_datas_path):
        print(f"[ERROR] 파일 없음: {all_datas_path}")
        return

    all_datas = np.load(all_datas_path, allow_pickle=True)
    total_writers = len(all_datas)
    print(f"    총 작가 수: {total_writers}")

    if max_writers:
        all_datas = all_datas[:max_writers]
        print(f"    제한: {max_writers}명만 처리")


    print("\n[2] 낱자 데이터 로드 중 (Style 보완용)...")
    writer_char_data = load_olhwd_char_data(olhwd_root)


    print("\n[3] 작가별 데이터 처리 중...")
    all_results = []

    for writer_idx, sample in enumerate(tqdm(all_datas, desc="작가 처리")):
        writer_id = writer_idx
        writer_id_str = str(writer_id)


        writer_chars_char = defaultdict(list)
        char_data = writer_char_data.get(writer_id, [])
        for char, coords in char_data:
            char = _normalize_char_tag(char)
            if not _is_valid_char_tag(char):
                continue
            if len(coords) == 0:
                continue


            char_points = []
            for i, pt in enumerate(coords):
                x, y = pt[0], pt[1]
                pen_raw = int(pt[2]) if len(pt) > 2 and not np.isnan(pt[2]) else 1
                is_end = (i == len(coords) - 1)

                if is_end:
                    pen_class_val = PEN_CLASS['EOC']
                elif pen_raw == -1:
                    pen_class_val = PEN_CLASS['PU']
                else:
                    pen_class_val = PEN_CLASS['PM']

                char_points.append([x, y, pen_class_val])

            char_traj = np.array(char_points, dtype=np.float32)


            scale, cx_min, cy_min, orig_h, orig_w = _normalize_sentence_height(char_traj, target_height=1.0)
            char_traj[:, 0] = (char_traj[:, 0] - cx_min) * scale
            char_traj[:, 1] = (char_traj[:, 1] - cy_min) * scale


            img = render_image_from_traj(char_traj, (H, W), line_width=1, use_antialiasing=False)

            writer_chars_char[char].append({
                "trajectory": char_traj.astype(np.float32),
                "image": img.astype(np.uint8) if img is not None else None,
                "connection": "isolated",
                "character": char,
                "sentence_id": f"char_writer_{writer_id}",
                "cursor": 0,
                "scale": scale,
                "min_xy": np.array([0.0, 0.0], dtype=np.float32),
                "source_dataset": "OLHWD_CHAR",
            })


        writer_chars_sent, writer_sentences = process_olhwd_writer_sentences(
            sample, writer_id, img_size, points_per_char,
            rdp_epsilon, cursive_threshold, cursive_visualize_dir, sentence_visualize_dir,
            boundary_tolerance=boundary_tolerance
        )


        writer_chars_all = defaultdict(list)

        for ch, samples in writer_chars_char.items():
            writer_chars_all[ch].extend(samples)

        for ch, samples in writer_chars_sent.items():
            if ch not in writer_chars_char:
                writer_chars_all[ch].extend(samples)

        if not writer_chars_all:
            continue


        rep_map = {}
        for ch in writer_chars_all.keys():

            prefer = writer_chars_char.get(ch, [])
            samples = prefer if prefer else writer_chars_all.get(ch, [])
            rep, _ = _choose_representative(samples)
            if rep is not None:

                rep = _normalize_char_height(rep, target_height=1.0)

                rep = _scale_up_small_char(rep, min_height=MIN_CHAR_HEIGHT)

                rep_map[ch] = _left_align_sample(rep, (H, W), line_width=1, use_antialiasing=False)


        rep_map = {ch: rep for ch, rep in rep_map.items()
                   if isinstance(rep, dict) and rep.get("trajectory") is not None and len(rep.get("trajectory")) > 0}
        if not rep_map:
            continue


        if save_char_root:
            with open(os.path.join(save_char_root, f"{writer_id_str}_rep.pkl"), "wb") as f:
                pickle.dump(rep_map, f)


        if save_style_root:
            with open(os.path.join(save_style_root, f"{writer_id_str}.pkl"), "wb") as f:
                pickle.dump(dict(writer_chars_all), f)


            style_rep_map = {ch: [rep] for ch, rep in rep_map.items() if rep.get('image') is not None}
            with open(os.path.join(save_style_root, f"{writer_id_str}_rep.pkl"), "wb") as f:
                pickle.dump(style_rep_map, f)


        if save_sent_root:
            with open(os.path.join(save_sent_root, f"{writer_id_str}_sent.pkl"), "wb") as f:
                pickle.dump({"sentences": writer_sentences}, f)

        all_results.append({
            'writer_id': writer_id,
            'num_chars': len(rep_map),
            'num_sentences': len(writer_sentences),
            'num_char_samples': sum(len(s) for s in writer_chars_all.values()),
        })


    print("\n" + "=" * 80)
    print(" 생성 결과 요약")
    print("=" * 80)
    print(f"처리된 작가 수: {len(all_results)}")

    if all_results:
        total_chars = sum(r['num_chars'] for r in all_results)
        total_sents = sum(r['num_sentences'] for r in all_results)
        total_samples = sum(r['num_char_samples'] for r in all_results)

        print(f"\n[요약]")
        print(f"  OLHWD: {len(all_results)}명 작가, {total_sents:,}개 문장, {total_samples:,}개 글자 샘플")
        print(f"         → 작가당 평균 {total_sents/len(all_results):.1f}개 문장, {total_chars/len(all_results):.1f}개 글자 타입")

    print_preprocessing_stats(points_per_char)


    if save_debug_vis and all_results:
        print("\n" + "=" * 80)
        print(" 디버그 시각화 저장")
        print("=" * 80)

        debug_output_dir = os.path.join(save_style_root, "_debug_style_rep")
        ensure_dir(debug_output_dir)

        random.seed(42)
        sample_writers = random.sample(all_results, min(3, len(all_results)))

        for wr in sample_writers:
            _save_style_rep_debug_images(save_style_root, wr['writer_id'], debug_output_dir)

        print(f"\n 디버그 이미지 저장 완료")


    mapping_file = os.path.join(save_style_root, "writer_mapping.json")
    with open(mapping_file, 'w', encoding='utf-8') as f:
        json.dump({
            'source': 'OLHWD',
            'total_writers': len(all_results),
            'writers': [r['writer_id'] for r in all_results],
            'config': {
                'points_per_char': points_per_char,
                'rdp_epsilon': rdp_epsilon,
                'cursive_threshold': cursive_threshold,
            }
        }, f, indent=2)

    print(f"\n 완료! Writer mapping: {mapping_file}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="OLHWD 중국어 손글씨 데이터셋 생성기",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  python -m src.preprocessing.olhwd_cn_sent_dataset_generator \\
    --olhwd_root /root/0_DB/OLHWD \\
    --save_char_root dataset/CN/OLHWD_pickles/char_pickles \\
    --save_sent_root dataset/CN/OLHWD_pickles/sent_pickles \\
    --save_style_root dataset/CN/OLHWD_pickles/style_pickles \\
    --img_size 64 64 \\
    --points_per_char 160 \\
    --cursive_threshold 0.03 \\
    --save_debug_vis
        """
    )

    parser.add_argument('--olhwd_root', type=str, required=True,
                        help='OLHWD 디렉토리 경로')
    parser.add_argument('--save_char_root', type=str, default=None,
                        help='글자 pickle 저장 경로 (생략 시 스킵)')
    parser.add_argument('--save_sent_root', type=str, default=None,
                        help='문장 pickle 저장 경로 (생략 시 스킵)')
    parser.add_argument('--save_style_root', type=str, default=None,
                        help='스타일 pickle 저장 경로 (생략 시 스킵)')
    parser.add_argument('--img_size', type=int, nargs=2, default=[64, 64],
                        help='이미지 크기 (H, W)')
    parser.add_argument('--points_per_char', type=int, default=DEFAULT_POINTS_PER_CHAR,
                        help=f'글자당 최대 포인트 수 (기본: {DEFAULT_POINTS_PER_CHAR})')
    parser.add_argument('--rdp_epsilon', type=float, default=0.0,
                        help='RDP epsilon (0이면 미적용)')
    parser.add_argument('--cursive_threshold', type=float, default=0.03,
                        help='CURSIVE_EOC 판정 거리 threshold')
    parser.add_argument('--boundary_tolerance', type=int, default=DEFAULT_BOUNDARY_TOLERANCE,
                        help=f'duration 경계와 stroke end 허용 오차 (기본: {DEFAULT_BOUNDARY_TOLERANCE})')
    parser.add_argument('--max_writers', type=int, default=None,
                        help='최대 작가 수 (디버깅용)')
    parser.add_argument('--visualize_cursive', action='store_true',
                        help='CURSIVE_EOC 시각화 저장')
    parser.add_argument('--save_debug_vis', action='store_true',
                        help='디버그 시각화 저장')

    args = parser.parse_args()

    generate_olhwd_dataset(
        olhwd_root=args.olhwd_root,
        save_char_root=args.save_char_root,
        save_sent_root=args.save_sent_root,
        save_style_root=args.save_style_root,
        img_size=tuple(args.img_size),
        points_per_char=args.points_per_char,
        rdp_epsilon=args.rdp_epsilon if args.rdp_epsilon > 0 else None,
        cursive_threshold=args.cursive_threshold,
        boundary_tolerance=args.boundary_tolerance,
        max_writers=args.max_writers,
        visualize_cursive=args.visualize_cursive,
        save_debug_vis=args.save_debug_vis,
    )


if __name__ == "__main__":
    main()
