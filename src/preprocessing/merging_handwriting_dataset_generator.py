import os
import pickle
import json
import subprocess
import sys
import numpy as np
from collections import defaultdict
from PIL import Image, ImageDraw
from tqdm import tqdm
from typing import Optional, Dict, List, Tuple, Any


from src.config.constants import (
    TRAJ_INDEX, TRAJ_INDEX_EXPANDED, TRAJ_DIM, TRAJ_DIM_EXPANDED, PEN_CLASS
)
from src.data.data_utils import (
    resample_trajectory_preserve_endpoints,
    guarantee_stroke_endpoints,
    _deskew_trajectory,
)


def _visualize_deskew_before_after(
    before: np.ndarray,
    after: np.ndarray,
    angle: float,
    sentence_id: str,
    output_dir: str,
    img_size: Tuple[int, int] = (200, 800)
):
    H, W = img_size

    def _draw_trajectory(traj: np.ndarray, canvas_size: Tuple[int, int]) -> np.ndarray:
        h, w = canvas_size
        canvas = np.ones((h, w), dtype=np.uint8) * 255

        if traj.shape[0] < 2:
            return canvas


        x = traj[:, 0]
        y = traj[:, 1]

        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()

        x_range = x_max - x_min if x_max > x_min else 1.0
        y_range = y_max - y_min if y_max > y_min else 1.0


        margin = 20
        scale_x = (w - 2 * margin) / x_range
        scale_y = (h - 2 * margin) / y_range
        scale = min(scale_x, scale_y)


        x_scaled = (x - x_min) * scale + margin + (w - 2 * margin - x_range * scale) / 2
        y_scaled = (y - y_min) * scale + margin + (h - 2 * margin - y_range * scale) / 2


        img_pil = Image.fromarray(canvas)
        draw = ImageDraw.Draw(img_pil)


        pen_class = traj[:, 2] if traj.shape[1] > 2 else np.zeros(len(traj))

        for i in range(len(x_scaled) - 1):
            x1, y1 = int(x_scaled[i]), int(y_scaled[i])
            x2, y2 = int(x_scaled[i + 1]), int(y_scaled[i + 1])


            if pen_class[i] < 1.5:
                draw.line([(x1, y1), (x2, y2)], fill=0, width=2)


        if len(x_scaled) > 0:

            draw.ellipse([int(x_scaled[0])-4, int(y_scaled[0])-4,
                         int(x_scaled[0])+4, int(y_scaled[0])+4], fill=0, outline=0)

            draw.rectangle([int(x_scaled[-1])-3, int(y_scaled[-1])-3,
                           int(x_scaled[-1])+3, int(y_scaled[-1])+3], fill=0, outline=0)

        return np.array(img_pil)


    before_img = _draw_trajectory(before, (H, W))
    after_img = _draw_trajectory(after, (H, W))


    if before.shape[0] >= 2:
        before_pil = Image.fromarray(before_img)
        draw = ImageDraw.Draw(before_pil)

        x = before[:, 0]
        y = before[:, 1]
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()
        x_range = x_max - x_min if x_max > x_min else 1.0
        y_range = y_max - y_min if y_max > y_min else 1.0

        margin = 20
        scale_x = (W - 2 * margin) / x_range
        scale_y = (H - 2 * margin) / y_range
        scale = min(scale_x, scale_y)


        x_mean = np.mean(x)
        y_mean = np.mean(y)
        slope = np.tan(np.radians(angle))


        line_x1 = x_min
        line_y1 = y_mean + slope * (line_x1 - x_mean)
        line_x2 = x_max
        line_y2 = y_mean + slope * (line_x2 - x_mean)


        lx1 = int((line_x1 - x_min) * scale + margin + (W - 2 * margin - x_range * scale) / 2)
        ly1 = int((line_y1 - y_min) * scale + margin + (H - 2 * margin - y_range * scale) / 2)
        lx2 = int((line_x2 - x_min) * scale + margin + (W - 2 * margin - x_range * scale) / 2)
        ly2 = int((line_y2 - y_min) * scale + margin + (H - 2 * margin - y_range * scale) / 2)


        num_dashes = 20
        for i in range(num_dashes):
            if i % 2 == 0:
                t1 = i / num_dashes
                t2 = (i + 1) / num_dashes
                dx1 = int(lx1 + (lx2 - lx1) * t1)
                dy1 = int(ly1 + (ly2 - ly1) * t1)
                dx2 = int(lx1 + (lx2 - lx1) * t2)
                dy2 = int(ly1 + (ly2 - ly1) * t2)
                draw.line([(dx1, dy1), (dx2, dy2)], fill=128, width=1)

        before_img = np.array(before_pil)


    combined = np.vstack([before_img, after_img])


    combined_pil = Image.fromarray(combined)
    draw = ImageDraw.Draw(combined_pil)


    draw.text((10, 5), f"BEFORE (angle={angle:.2f}°)", fill=0)
    draw.text((10, H + 5), f"AFTER (deskewed)", fill=0)


    ensure_dir(output_dir)
    safe_sid = sentence_id.replace("/", "_").replace("\\", "_")
    output_path = os.path.join(output_dir, f"deskew_{safe_sid}_angle{angle:.1f}.png")
    combined_pil.save(output_path)

    return output_path


def _visualize_rdp_before_after(
    before: np.ndarray,
    after: np.ndarray,
    epsilon: float,
    char: str,
    sentence_id: str,
    char_idx: int,
    output_dir: str,
    source_dataset: str = "IAM",
    img_size: Tuple[int, int] = (150, 150)
):
    H, W = img_size

    def _draw_char_trajectory(traj: np.ndarray, canvas_size: Tuple[int, int],
                               show_points: bool = True) -> np.ndarray:
        h, w = canvas_size
        canvas = np.ones((h, w), dtype=np.uint8) * 255

        if traj.shape[0] < 1:
            return canvas


        x = traj[:, 0]
        y = traj[:, 1]

        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()

        x_range = x_max - x_min if x_max > x_min else 0.1
        y_range = y_max - y_min if y_max > y_min else 0.1


        margin = 15
        scale_x = (w - 2 * margin) / x_range
        scale_y = (h - 2 * margin) / y_range
        scale = min(scale_x, scale_y)


        x_scaled = (x - x_min) * scale + margin + (w - 2 * margin - x_range * scale) / 2
        y_scaled = (y - y_min) * scale + margin + (h - 2 * margin - y_range * scale) / 2


        img_pil = Image.fromarray(canvas)
        draw = ImageDraw.Draw(img_pil)


        pen_class = traj[:, 2] if traj.shape[1] > 2 else np.zeros(len(traj))


        for i in range(len(x_scaled) - 1):
            x1, y1 = int(x_scaled[i]), int(y_scaled[i])
            x2, y2 = int(x_scaled[i + 1]), int(y_scaled[i + 1])


            if pen_class[i] < 1.5:
                draw.line([(x1, y1), (x2, y2)], fill=0, width=2)


        if show_points:
            for i in range(len(x_scaled)):
                px, py = int(x_scaled[i]), int(y_scaled[i])

                draw.ellipse([px-2, py-2, px+2, py+2], fill=100, outline=50)

        return np.array(img_pil)


    before_img = _draw_char_trajectory(before, (H, W), show_points=True)
    after_img = _draw_char_trajectory(after, (H, W), show_points=True)


    before_pts = before.shape[0]
    after_pts = after.shape[0]
    reduction = (1 - after_pts / before_pts) * 100 if before_pts > 0 else 0


    combined = np.hstack([before_img, after_img])


    combined_pil = Image.fromarray(combined)
    draw = ImageDraw.Draw(combined_pil)


    draw.text((5, 3), f"BEFORE ({before_pts}pts)", fill=0)
    draw.text((W + 5, 3), f"AFTER ({after_pts}pts, -{reduction:.0f}%)", fill=0)


    ensure_dir(output_dir)
    safe_sid = sentence_id.replace("/", "_").replace("\\", "_")[:30]
    safe_char = char if char.isalnum() else f"ord{ord(char)}"
    output_path = os.path.join(
        output_dir,
        f"rdp_{source_dataset}_{safe_sid}_idx{char_idx}_{safe_char}_eps{epsilon:.3f}.png"
    )
    combined_pil.save(output_path)

    return output_path


_rdp_visualize_samples: Dict[str, List[Dict]] = {}


def _collect_rdp_visualize_sample(
    before: np.ndarray,
    after: np.ndarray,
    epsilon: float,
    char: str,
    sentence_id: str,
    char_idx: int,
    source_dataset: str,
    max_samples_per_char: int = 3
):
    global _rdp_visualize_samples

    key = f"{source_dataset}_{char}"
    if key not in _rdp_visualize_samples:
        _rdp_visualize_samples[key] = []

    if len(_rdp_visualize_samples[key]) < max_samples_per_char:
        _rdp_visualize_samples[key].append({
            'before': before.copy(),
            'after': after.copy(),
            'epsilon': epsilon,
            'char': char,
            'sentence_id': sentence_id,
            'char_idx': char_idx,
            'source_dataset': source_dataset,
        })


def _save_rdp_visualize_samples(output_dir: str, img_size: Tuple[int, int] = (150, 150)):
    global _rdp_visualize_samples

    ensure_dir(output_dir)
    saved_count = 0

    for key, samples in _rdp_visualize_samples.items():
        for sample in samples:
            _visualize_rdp_before_after(
                before=sample['before'],
                after=sample['after'],
                epsilon=sample['epsilon'],
                char=sample['char'],
                sentence_id=sample['sentence_id'],
                char_idx=sample['char_idx'],
                output_dir=output_dir,
                source_dataset=sample['source_dataset'],
                img_size=img_size
            )
            saved_count += 1


    _rdp_visualize_samples = {}

    return saved_count


from src.preprocessing.brush_handwriting_dataset_generator import (
    _convert_old_to_new_format,
    _audit_and_enforce_pen_logic_char,
    _normalize_sentence_height,
    _apply_rdp,
    render_image_from_traj,
    get_connection_type,
    _choose_representative,
    _compute_conn_stats,
    _left_align_sample,
    _traj_len,
    _pick_median_len_first3,
    REP_PRIORITY,
    REP_TARGET_SET,
    ensure_dir,
)


DEFAULT_MAX_TRAJ_LEN = 100


_runtime_max_traj_len = DEFAULT_MAX_TRAJ_LEN


_preprocessing_stats = {
    'deskew': {
        'total_checked': 0,
        'applied': 0,
        'angles': [],
    },
    'resample': {
        'total_checked': 0,
        'applied': 0,
        'before_lens': [],
        'after_lens': [],
    },
    'rdp': {
        'total_checked': 0,
        'applied': 0,
    }
}

def reset_preprocessing_stats():
    global _preprocessing_stats
    _preprocessing_stats = {
        'deskew': {'total_checked': 0, 'applied': 0, 'angles': []},
        'resample': {'total_checked': 0, 'applied': 0, 'before_lens': [], 'after_lens': []},
        'rdp': {'total_checked': 0, 'applied': 0}
    }

def print_preprocessing_stats():
    print("\n" + "="*70)
    print(" 전처리 통계 요약 (Preprocessing Statistics)")
    print("="*70)


    ds = _preprocessing_stats['deskew']
    print(f"\n[1] Deskew (기울기 보정)")
    print(f"    - 체크한 문장: {ds['total_checked']:,}개")
    print(f"    - 적용된 문장: {ds['applied']:,}개 ({100*ds['applied']/max(1,ds['total_checked']):.2f}%)")
    if ds['angles']:
        import numpy as np
        angles = np.array(ds['angles'])
        print(f"    - 적용 각도: min={angles.min():.2f}°, max={angles.max():.2f}°, mean={angles.mean():.2f}°")


    rs = _preprocessing_stats['resample']
    print(f"\n[2] Resample (max_traj_len={_runtime_max_traj_len} 초과 시)")
    print(f"    - 체크한 글자: {rs['total_checked']:,}개")
    print(f"    - 리샘플링된 글자: {rs['applied']:,}개 ({100*rs['applied']/max(1,rs['total_checked']):.2f}%)")
    if rs['before_lens']:
        import numpy as np
        before = np.array(rs['before_lens'])
        print(f"    - 리샘플링 전 길이: min={before.min()}, max={before.max()}, mean={before.mean():.1f}")
        print(f"    - 리샘플링 후 길이: {_runtime_max_traj_len} (고정)")


    rdp = _preprocessing_stats['rdp']
    print(f"\n[3] RDP (Douglas-Peucker)")
    print(f"    - 체크한 글자: {rdp['total_checked']:,}개")
    print(f"    - 적용된 글자: {rdp['applied']:,}개 ({100*rdp['applied']/max(1,rdp['total_checked']):.2f}%)")

    print("\n" + "="*70)


from src.test_script.dataset_analyzer import (
    _load_iam_writer_mapping,
    _extract_form_id_from_key,
)


def _check_and_run_convert_gt(iam_root: str) -> bool:
    import glob


    if iam_root.endswith("converted"):
        converted_dir = iam_root
        iam_base = os.path.dirname(iam_root)
    else:
        iam_base = iam_root
        converted_dir = os.path.join(iam_root, "converted")


    json_files = glob.glob(os.path.join(converted_dir, "**", "*.xml.json"), recursive=True)
    if json_files:
        print(f"[IAM]  변환된 JSON 파일 발견: {len(json_files)}개")
        return True

    print(f"[IAM]   변환된 JSON 파일이 없습니다. convert_gt.py 실행 시도...")


    linestrokes_tar = os.path.join(iam_base, "lineStrokes-all.tar.gz")
    if not os.path.exists(linestrokes_tar):
        print(f"[ERROR] lineStrokes-all.tar.gz 파일을 찾을 수 없습니다: {linestrokes_tar}")
        return False


    script_dir = os.path.dirname(os.path.abspath(__file__))
    gt_dir = os.path.join(script_dir, "IAM_segmentation_GT")

    gt_files = [
        os.path.join(gt_dir, "trainset_segmented.json"),
        os.path.join(gt_dir, "testset_v_segmented.json"),
        os.path.join(gt_dir, "testset_t_segmented.json"),
        os.path.join(gt_dir, "testset_f_segmented.json"),
    ]


    existing_gt_files = [f for f in gt_files if os.path.exists(f)]
    if not existing_gt_files:
        print(f"[ERROR] GT segmentation 파일을 찾을 수 없습니다.")
        print(f"        예상 위치: {gt_dir}/")
        return False

    print(f"[IAM] GT 파일 발견: {len(existing_gt_files)}개")


    convert_script = os.path.join(gt_dir, "convert_gt.py")
    if not os.path.exists(convert_script):
        print(f"[ERROR] convert_gt.py를 찾을 수 없습니다: {convert_script}")
        return False


    cmd = [
        sys.executable,
        convert_script,
        "-d", linestrokes_tar,
        "-s", *existing_gt_files,
        "-o", converted_dir,
    ]

    print(f"[IAM] 실행 명령:")
    print(f"  python convert_gt.py \\")
    print(f"    -d {linestrokes_tar} \\")
    print(f"    -s {' '.join(existing_gt_files)} \\")
    print(f"    -o {converted_dir}")
    print()

    try:

        result = subprocess.run(
            cmd,
            cwd=gt_dir,
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONPATH": gt_dir + ":" + os.environ.get("PYTHONPATH", "")}
        )

        if result.returncode != 0:
            print(f"[ERROR] convert_gt.py 실행 실패:")
            print(result.stderr)


            if "No module named 'data_preparation'" in result.stderr:
                print()
                print("=" * 70)
                print(" 'data_preparation' 모듈이 필요합니다!")
                print("=" * 70)
                print()
                print("해결 방법:")
                print("  1. IAM Segmentation GT 저장소 클론:")
                print("     git clone https://github.com/jungomi/character-segmentation.git")
                print()
                print("  2. data_preparation 폴더를 복사:")
                print(f"     cp -r character-segmentation/data_preparation {gt_dir}/")
                print()
                print("  3. 또는 수동으로 convert_gt.py 실행:")
                print(f"     cd {gt_dir}")
                print(f"     python convert_gt.py -d {linestrokes_tar} \\")
                print(f"       -s trainset_segmented.json testset_v_segmented.json \\")
                print(f"          testset_t_segmented.json testset_f_segmented.json \\")
                print(f"       -o {converted_dir}")
                print()
            return False

        print(result.stdout)
        print(f"[IAM]  convert_gt.py 실행 완료!")
        return True

    except Exception as e:
        print(f"[ERROR] convert_gt.py 실행 중 오류: {e}")
        return False


def _load_iam_json_file(json_file: str, writer_id: str, _unused_mapping: Dict) -> List[Dict]:
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)


    key = data.get('key', '')
    form_id = _extract_form_id_from_key(key)


    actual_writer_id = writer_id


    sentence_text = data.get('text', '')
    if not sentence_text:
        return []

    sentence_id = f"{form_id}_{key}"


    points_list = data.get('points', [])
    if not points_list:
        return []


    strokes_dict = defaultdict(list)
    for pt in points_list:
        stroke_idx = pt.get('stroke', -1)
        if stroke_idx >= 0:
            strokes_dict[stroke_idx].append(pt)


    strokes = [strokes_dict[i] for i in sorted(strokes_dict.keys())]


    gt_segmentation = data.get('gt_segmentation', {})
    segments = gt_segmentation.get('segments', [])

    sentences = []
    characters = []

    for char_idx, seg in enumerate(segments):
        char = seg.get('substring', '')
        if not char:
            continue


        ink_ranges = seg.get('inkRanges', [])
        if not ink_ranges:
            continue


        char_points = []
        for ir in ink_ranges:
            start_stroke = ir.get('startStroke', ir.get('stroke', -1))
            end_stroke = ir.get('endStroke', start_stroke)
            start_point = ir.get('startPoint', 0)
            end_point = ir.get('endPoint', -1)

            if start_stroke < 0:
                continue


            for s_idx in range(start_stroke, end_stroke + 1):
                if s_idx < 0 or s_idx >= len(strokes):
                    continue

                stroke_points = strokes[s_idx]


                if s_idx == start_stroke:
                    sp = start_point
                else:
                    sp = 0

                if s_idx == end_stroke:
                    ep = end_point if end_point >= 0 else len(stroke_points) - 1
                else:
                    ep = len(stroke_points) - 1


                for pt_idx in range(sp, min(ep + 1, len(stroke_points))):
                    pt = stroke_points[pt_idx]
                    x = pt.get('x', 0)
                    y = pt.get('y', 0)
                    is_stroke_end = (pt_idx == ep)
                    char_points.append((x, y, s_idx, is_stroke_end))

        if char_points:

            first_stroke = ink_ranges[0].get('startStroke', ink_ranges[0].get('stroke', -1)) if ink_ranges else -1
            last_stroke = ink_ranges[-1].get('endStroke', ink_ranges[-1].get('startStroke', -1)) if ink_ranges else -1


            stroke_ends = [pt[3] for pt in char_points]
            points_xy = np.array([[pt[0], pt[1]] for pt in char_points], dtype=np.float32)

            characters.append({
                'char': char,
                'char_idx': char_idx,
                'points': points_xy,
                'stroke_ends': stroke_ends,
                'first_stroke': first_stroke,
                'last_stroke': last_stroke,
            })


    for i in range(len(characters)):
        curr = characters[i]
        prev_char = characters[i - 1] if i > 0 else None
        next_char = characters[i + 1] if i + 1 < len(characters) else None

        connected_left = False
        connected_right = False


        if prev_char and prev_char['last_stroke'] == curr['first_stroke']:
            connected_left = True


        if next_char and curr['last_stroke'] == next_char['first_stroke']:
            connected_right = True


        if connected_left and connected_right:
            conn_type = 'connected_both'
        elif connected_left:
            conn_type = 'connected_left'
        elif connected_right:
            conn_type = 'connected_right'
        else:
            conn_type = 'isolated'

        curr['connection'] = conn_type

    if characters:
        sentences.append({
            'sentence_id': sentence_id,
            'sentence_text': sentence_text,
            'writer_id': actual_writer_id,
            'characters': characters,
            'segments': segments,
        })

    return sentences


def _iam_sentence_to_brush_format(sent_dict: Dict, img_size: Tuple[int, int], rdp_epsilon: Optional[float],
                                   deskew_enabled: bool = True,
                                   deskew_visualize_dir: Optional[str] = None,
                                   deskew_angle_threshold: float = 1.0,
                                   deskew_max_angle: float = 30.0,
                                   rdp_visualize: bool = False) -> Tuple[List[Dict], str, Optional[float]]:
    H, W = img_size
    characters = sent_dict['characters']
    sentence_text = sent_dict['sentence_text']
    sentence_id = sent_dict['sentence_id']
    writer_id = sent_dict['writer_id']
    segments = sent_dict.get('segments', [])


    deskew_angle = None
    deskew_applied = False


    char_by_seg_idx = {char_data['char_idx']: char_data for char_data in characters}


    all_points = []
    all_pen_states = []
    char_labels = []

    for char_data in characters:
        points = char_data['points']
        stroke_ends = char_data.get('stroke_ends', [False] * len(points))
        char_idx = char_data['char_idx']
        conn_type = char_data.get('connection', 'isolated')
        is_connected_right = conn_type in ('connected_right', 'connected_both')

        n_points = len(points)
        for i, pt in enumerate(points):
            all_points.append(pt)
            char_labels.append(char_idx)

            is_last_point = (i == n_points - 1)
            is_stroke_end = stroke_ends[i] if i < len(stroke_ends) else False


            if is_last_point:

                if is_connected_right:


                    pen_state = PEN_CLASS["CURSIVE_EOC"]
                else:

                    pen_state = PEN_CLASS["EOC"]
            elif is_stroke_end:

                pen_state = PEN_CLASS["PU"]
            else:

                pen_state = PEN_CLASS["PM"]

            all_pen_states.append(pen_state)


    has_points = len(all_points) > 0
    if not has_points:

        scale = 1.0
        x_min, y_min = 0.0, 0.0
        original_height, original_width = 0.0, 0.0
        normed_drawing = np.zeros((0, 3), dtype=np.float32)
        drawing_new = np.zeros((0, 3), dtype=np.float32)
    else:

        all_points = np.array(all_points, dtype=np.float32)
        char_labels = np.array(char_labels, dtype=np.int32)
        pen_class = np.array(all_pen_states, dtype=np.float32)


        drawing_new = np.concatenate([all_points, pen_class.reshape(-1, 1)], axis=1).astype(np.float32)


        drawing_before_deskew = drawing_new.copy() if deskew_visualize_dir else None

        if deskew_enabled and drawing_new.shape[0] >= 3:
            drawing_new, deskew_angle, deskew_applied = _deskew_trajectory(
                drawing_new,
                angle_threshold=deskew_angle_threshold,
                max_angle=deskew_max_angle
            )


            _preprocessing_stats['deskew']['total_checked'] += 1
            if deskew_applied:
                _preprocessing_stats['deskew']['applied'] += 1
                _preprocessing_stats['deskew']['angles'].append(abs(deskew_angle))


            if deskew_visualize_dir and deskew_applied:
                _visualize_deskew_before_after(
                    drawing_before_deskew,
                    drawing_new,
                    deskew_angle,
                    sentence_id,
                    deskew_visualize_dir
                )


        scale, x_min, y_min, original_height, original_width = _normalize_sentence_height(
            drawing_new, target_height=1.0
        )
        normed_xy = np.stack([
            (drawing_new[:, 0] - x_min) * scale,
            (drawing_new[:, 1] - y_min) * scale
        ], axis=1).astype(np.float32)
        normed_drawing = np.concatenate([normed_xy, drawing_new[:, 2:3]], axis=1).astype(np.float32)

    min_xy = np.array([x_min, y_min], dtype=np.float32)


    char_items = []
    cursor_pos = 0


    for seg_idx, seg in enumerate(segments):
        ch = seg.get('substring', '')
        if not ch:
            continue


        ink_ranges = seg.get('inkRanges', [])
        if not ink_ranges or ch == ' ':

            space_item = {
                "trajectory": np.zeros((0, TRAJ_DIM), dtype=np.float32),
                "image": None,
                "connection": "space",
                "character": ch,
                "sentence_id": sentence_id,
                "cursor": cursor_pos,
                "scale": float(scale) if has_points else 1.0,
                "min_xy": min_xy.astype(np.float32),
                "orig_min_x": None,
                "original_height": float(original_height),
                "original_width": float(original_width),
                "coord_mode": "normalized_height_1.0",
                "source_dataset": "IAM",
                "_before_points": 0,
                "_before_x_range": 0.0,
                "_before_y_range": 0.0,
                "_after_points": 0,
                "_after_x_range": 0.0,
                "_after_y_range": 0.0,
            }
            char_items.append(space_item)
            cursor_pos += 1
            continue


        if seg_idx not in char_by_seg_idx:

            cursor_pos += 1
            continue

        char_data = char_by_seg_idx[seg_idx]


        conn_type = char_data.get('connection', 'isolated')
        is_cursive_connected = conn_type in ('connected_right', 'connected_both')


        if len(all_points) > 0:
            idxs = np.where(char_labels == seg_idx)[0]
            if len(idxs) == 0:

                cursor_pos += 1
                continue


            char_traj_orig = drawing_new[idxs].astype(np.float32)
            before_points = len(char_traj_orig)
            before_x_range = float(np.max(char_traj_orig[:, 0]) - np.min(char_traj_orig[:, 0])) if len(char_traj_orig) > 0 else 0.0
            before_y_range = float(np.max(char_traj_orig[:, 1]) - np.min(char_traj_orig[:, 1])) if len(char_traj_orig) > 0 else 0.0


            char_traj = normed_drawing[idxs].astype(np.float32)


            char_traj_original = char_traj.copy()


            _preprocessing_stats['resample']['total_checked'] += 1
            if char_traj.shape[0] > _runtime_max_traj_len:
                orig_len = char_traj.shape[0]
                char_traj = resample_trajectory_preserve_endpoints(char_traj, _runtime_max_traj_len)
                _preprocessing_stats['resample']['applied'] += 1
                _preprocessing_stats['resample']['before_lens'].append(orig_len)
                _preprocessing_stats['resample']['after_lens'].append(char_traj.shape[0])


            MIN_POINTS_FOR_RDP = 10
            MAX_REDUCTION_RATIO = 0.5

            char_traj_before_rdp = char_traj.copy() if rdp_visualize else None
            _preprocessing_stats['rdp']['total_checked'] += 1

            if rdp_epsilon is not None and rdp_epsilon > 0 and char_traj.shape[0] > 0:
                original_point_count = char_traj.shape[0]


                if original_point_count <= MIN_POINTS_FOR_RDP:

                    pass
                else:

                    min_target_points = int(original_point_count * (1.0 - MAX_REDUCTION_RATIO))

                    char_traj_rdp = _apply_rdp(char_traj, rdp_epsilon)
                    rdp_point_count = char_traj_rdp.shape[0]


                    if rdp_point_count < min_target_points:

                        current_eps = rdp_epsilon
                        for _ in range(5):
                            current_eps *= 0.5
                            char_traj_rdp = _apply_rdp(char_traj, current_eps)
                            if char_traj_rdp.shape[0] >= min_target_points:
                                break

                        if char_traj_rdp.shape[0] < min_target_points:
                            char_traj_rdp = char_traj.copy()

                    char_traj = char_traj_rdp
                    _preprocessing_stats['rdp']['applied'] += 1


                if rdp_visualize and char_traj_before_rdp is not None:
                    _collect_rdp_visualize_sample(
                        before=char_traj_before_rdp,
                        after=char_traj,
                        epsilon=rdp_epsilon,
                        char=ch,
                        sentence_id=sentence_id,
                        char_idx=cursor_pos,
                        source_dataset="IAM"
                    )


            char_traj = guarantee_stroke_endpoints(char_traj_original, char_traj)


            char_traj, info = _audit_and_enforce_pen_logic_char(
                char_traj, where="iam_char", sid=sentence_id, char_idx=cursor_pos, ch=ch,
                fix=True, is_cursive_connected=is_cursive_connected
            )


            after_points = len(char_traj)
            after_x_range = float(np.max(char_traj[:, 0]) - np.min(char_traj[:, 0])) if len(char_traj) > 0 else 0.0
            after_y_range = float(np.max(char_traj[:, 1]) - np.min(char_traj[:, 1])) if len(char_traj) > 0 else 0.0


            img = render_image_from_traj(char_traj, (H, W))
        else:

            char_traj = np.zeros((0, TRAJ_DIM), dtype=np.float32)
            before_points = 0
            before_x_range = 0.0
            before_y_range = 0.0
            after_points = 0
            after_x_range = 0.0
            after_y_range = 0.0
            img = None


        char_item = {
            "trajectory": char_traj.astype(np.float32),
            "image": img.astype(np.uint8) if img is not None else None,
            "connection": conn_type,
            "character": ch,
            "sentence_id": sentence_id,
            "cursor": cursor_pos,
            "scale": float(scale) if has_points else 1.0,
            "min_xy": min_xy.astype(np.float32),
            "orig_min_x": None,
            "original_height": float(original_height),
            "original_width": float(original_width),
            "coord_mode": "normalized_height_1.0",
            "source_dataset": "IAM",

            "_before_points": before_points,
            "_before_x_range": before_x_range,
            "_before_y_range": before_y_range,
            "_after_points": after_points,
            "_after_x_range": after_x_range,
            "_after_y_range": after_y_range,
        }
        char_items.append(char_item)
        cursor_pos += 1


    reconstructed_text = ''.join([item['character'] for item in char_items])

    return char_items, reconstructed_text, deskew_angle


def load_iam_writer_data(iam_root: str, writer_id: str, json_files: List[str],
                         img_size: Tuple[int, int], rdp_epsilon: Optional[float],
                         verbose_per_writer: bool = False,
                         deskew_enabled: bool = True,
                         deskew_visualize_dir: Optional[str] = None,
                         deskew_angle_threshold: float = 1.0,
                         deskew_max_angle: float = 30.0,
                         rdp_visualize: bool = False) -> Tuple[Dict, Dict, Dict]:
    writer_chars = defaultdict(list)
    writer_sentences = {}
    deskew_stats = {'corrected': 0, 'angles': []}

    if not json_files:
        print(f"[DEBUG] writer {writer_id}: json_files가 비어있음")
        return writer_chars, writer_sentences, deskew_stats
    if verbose_per_writer:
        print(f"[DEBUG] writer {writer_id}: {len(json_files)}개 JSON 파일 처리 중...")


    total_sentences = 0
    total_chars = 0
    for json_file in json_files:
        sentences = _load_iam_json_file(json_file, writer_id, {})
        total_sentences += len(sentences)

        for sent_dict in sentences:
            char_items, sentence_text, deskew_angle = _iam_sentence_to_brush_format(
                sent_dict, img_size, rdp_epsilon,
                deskew_enabled=deskew_enabled,
                deskew_visualize_dir=deskew_visualize_dir,
                deskew_angle_threshold=deskew_angle_threshold,
                deskew_max_angle=deskew_max_angle,
                rdp_visualize=rdp_visualize
            )
            total_chars += len(char_items)


            if deskew_angle is not None:
                deskew_stats['angles'].append(deskew_angle)
                if abs(deskew_angle) >= deskew_angle_threshold:
                    deskew_stats['corrected'] += 1

            sentence_id = sent_dict['sentence_id']


            for item in char_items:
                ch = item['character']
                writer_chars[ch].append(item)


            non_space_chars = [item for item in char_items if item['character'] != ' ']
            if len(non_space_chars) >= 2:
                writer_sentences[sentence_id] = char_items
            elif len(char_items) > 0:

                print(f"[WARN] Skipping sentence with <2 non-space chars: wid={writer_id}, sid={sentence_id}, chars={len(non_space_chars)}", flush=True)

    if verbose_per_writer:
        print(f"[DEBUG] writer {writer_id}: {total_sentences}개 문장, {total_chars}개 글자 로드")
        print(f"[DEBUG] writer {writer_id}: writer_chars keys: {len(writer_chars)}, writer_sentences keys: {len(writer_sentences)}")
        if deskew_stats['corrected'] > 0:
            print(f"[DEBUG] writer {writer_id}: Deskew 보정 {deskew_stats['corrected']}개 문장")

    return dict(writer_chars), writer_sentences, deskew_stats


def load_brush_writer_data(brush_root: str, writer_id: str, img_size: Tuple[int, int],
                          resample_type: str, rdp_epsilon: Optional[float],
                          rdp_visualize: bool = False) -> Tuple[Dict, Dict, Dict]:
    H, W = img_size
    writer_path = os.path.join(brush_root, writer_id)

    if not os.path.isdir(writer_path):
        return {}, {}, {}

    writer_chars = defaultdict(list)
    writer_sentences = {}
    alpha_rep_collect: Dict[str, list] = {}


    def _is_repetition_run_chars(chars: str) -> bool:
        return len(chars) >= 1 and (len(set(chars)) == 1) and (chars[0] in REP_TARGET_SET) and (chars[0] != ' ')

    def _is_repetition_sentence(sent: str) -> bool:
        toks = [t for t in (sent or "").split(" ") if t != ""]
        return len(toks) > 0 and all(_is_repetition_run_chars(tok) for tok in toks)

    for fname in sorted(os.listdir(writer_path)):

        if resample_type == "original":
            if not fname.isdigit():
                continue
        else:
            if not fname.endswith(f"_{resample_type}"):
                continue

        fpath = os.path.join(writer_path, fname)
        try:
            with open(fpath, "rb") as f:
                sentence, drawing, label = pickle.load(f)
        except:
            continue

        if drawing.shape[0] == 0:
            continue


        drawing_old = None
        if drawing.shape[1] == TRAJ_DIM:
            xy = drawing[:, :2].astype(np.float32)
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
        sid = fname if resample_type == "original" else fname.replace(f"_{resample_type}", "")
        sent_items = []

        if not isinstance(sentence, str):
            sentence = str(sentence)
        if label.ndim != 2 or label.shape[1] != len(sentence):
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
                    "source_dataset": "BRUSH",
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

            idxs = np.where(label[:, i] == 1)[0]
            if len(idxs) == 0:
                continue


            char_traj_orig = drawing_new[idxs].astype(np.float32)
            before_points = len(char_traj_orig)
            before_x_range = float(np.max(char_traj_orig[:, 0]) - np.min(char_traj_orig[:, 0])) if len(char_traj_orig) > 0 else 0.0
            before_y_range = float(np.max(char_traj_orig[:, 1]) - np.min(char_traj_orig[:, 1])) if len(char_traj_orig) > 0 else 0.0


            char_traj = normed_drawing[idxs].astype(np.float32)


            char_traj_original = char_traj.copy()


            _preprocessing_stats['resample']['total_checked'] += 1
            if char_traj.shape[0] > _runtime_max_traj_len:
                orig_len = char_traj.shape[0]
                char_traj = resample_trajectory_preserve_endpoints(char_traj, _runtime_max_traj_len)
                _preprocessing_stats['resample']['applied'] += 1
                _preprocessing_stats['resample']['before_lens'].append(orig_len)
                _preprocessing_stats['resample']['after_lens'].append(char_traj.shape[0])


            MIN_POINTS_FOR_RDP = 10
            MAX_REDUCTION_RATIO = 0.5

            char_traj_before_rdp = char_traj.copy() if rdp_visualize else None
            _preprocessing_stats['rdp']['total_checked'] += 1


            if rdp_epsilon is not None and rdp_epsilon > 0 and char_traj.shape[0] > 0:
                original_point_count = char_traj.shape[0]


                if original_point_count <= MIN_POINTS_FOR_RDP:

                    pass
                else:

                    min_target_points = int(original_point_count * (1.0 - MAX_REDUCTION_RATIO))

                    char_traj_rdp = _apply_rdp(char_traj, rdp_epsilon)
                    rdp_point_count = char_traj_rdp.shape[0]


                    if rdp_point_count < min_target_points:

                        current_eps = rdp_epsilon
                        for _ in range(5):
                            current_eps *= 0.5
                            char_traj_rdp = _apply_rdp(char_traj, current_eps)
                            if char_traj_rdp.shape[0] >= min_target_points:
                                break

                        if char_traj_rdp.shape[0] < min_target_points:
                            char_traj_rdp = char_traj.copy()

                    char_traj = char_traj_rdp
                    _preprocessing_stats['rdp']['applied'] += 1


                if rdp_visualize and char_traj_before_rdp is not None:
                    _collect_rdp_visualize_sample(
                        before=char_traj_before_rdp,
                        after=char_traj,
                        epsilon=rdp_epsilon,
                        char=ch,
                        sentence_id=sid,
                        char_idx=i,
                        source_dataset="BRUSH"
                    )


            char_traj = guarantee_stroke_endpoints(char_traj_original, char_traj)


            conn_type = get_connection_type(i, label, drawing_old) or "isolated"
            is_cursive_connected = conn_type in ('connected_right', 'connected_both')


            char_traj, info = _audit_and_enforce_pen_logic_char(
                char_traj, where="brush_char", sid=sid, char_idx=i, ch=ch,
                fix=True, is_cursive_connected=is_cursive_connected
            )


            after_points = len(char_traj)
            after_x_range = float(np.max(char_traj[:, 0]) - np.min(char_traj[:, 0])) if len(char_traj) > 0 else 0.0
            after_y_range = float(np.max(char_traj[:, 1]) - np.min(char_traj[:, 1])) if len(char_traj) > 0 else 0.0

            img = render_image_from_traj(char_traj, (H, W))


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
                "source_dataset": "BRUSH",
                "_before_points": before_points,
                "_before_x_range": before_x_range,
                "_before_y_range": before_y_range,
                "_after_points": after_points,
                "_after_x_range": after_x_range,
                "_after_y_range": after_y_range,
            }
            writer_chars[ch].append(char_item)
            sent_items.append(char_item)


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

    return dict(writer_chars), writer_sentences, alpha_rep_collect


def _save_style_rep_debug_images(style_root: str, writer_id: int, output_dir: str, prefix: str):
    rep_pkl_path = os.path.join(style_root, f"{writer_id}_rep.pkl")

    if not os.path.exists(rep_pkl_path):
        print(f"[DEBUG] {prefix}: _rep.pkl not found: {rep_pkl_path}")
        return

    with open(rep_pkl_path, 'rb') as f:
        rep_map = pickle.load(f)

    print(f"\n[DEBUG] Style Rep Images: {prefix} (wid={writer_id})")
    print(f"  글자 수: {len(rep_map)}")


    images = []
    labels = []

    for char, sample in sorted(rep_map.items()):

        if isinstance(sample, list) and len(sample) > 0:
            sample = sample[0]
        if isinstance(sample, dict):
            img = sample.get('image')
            if img is not None:
                images.append(np.array(img))
                labels.append(char)

    if not images:
        print(f"   이미지 없음")
        return


    n = len(images)
    cols = min(10, n)
    rows = (n + cols - 1) // cols

    h, w = images[0].shape[:2]
    cell_h = h + 20
    cell_w = w + 4

    grid = np.ones((rows * cell_h, cols * cell_w), dtype=np.uint8) * 255

    for idx, (img, label) in enumerate(zip(images, labels)):
        row = idx // cols
        col = idx % cols
        y = row * cell_h + 18
        x = col * cell_w + 2

        if len(img.shape) == 3:
            img = img[:, :, 0]

        grid[y:y+h, x:x+w] = img


    grid_pil = Image.fromarray(grid)
    draw = ImageDraw.Draw(grid_pil)

    for idx, label in enumerate(labels):
        row = idx // cols
        col = idx % cols
        y = row * cell_h + 2
        x = col * cell_w + 2


        if label.isascii():
            draw.text((x, y), label, fill=0)

    output_path = os.path.join(output_dir, f"{prefix}_wid{writer_id}_style_rep.png")
    grid_pil.save(output_path)
    print(f"   저장: {output_path}")
    print(f"     {len(images)}개 글자 이미지")


def generate_merged_pickles(
    iam_root: str,
    brush_root: str,
    save_char_root: str,
    save_sent_root: str,
    save_style_root: str,
    img_size: Tuple[int, int] = (64, 64),
    resample_type: str = "original",
    rdp_epsilon_normalized: Optional[float] = None,
    max_iam_writers: Optional[int] = None,
    max_brush_writers: Optional[int] = None,
    verbose_per_writer: bool = False,

    deskew_enabled: bool = True,
    deskew_visualize: bool = False,
    deskew_angle_threshold: float = 1.0,
    deskew_max_angle: float = 30.0,

    rdp_visualize: bool = False,

    max_traj_len: int = DEFAULT_MAX_TRAJ_LEN,
    iam_writer_offset: int = 10000,
    brush_writer_offset: int = 20000,
):
    global _runtime_max_traj_len
    _runtime_max_traj_len = max_traj_len

    ensure_dir(save_char_root)
    ensure_dir(save_sent_root)
    ensure_dir(save_style_root)


    reset_preprocessing_stats()

    H, W = img_size


    deskew_visualize_dir = None
    if deskew_visualize:
        deskew_visualize_dir = os.path.join(save_style_root, "_debug_deskew_images")
        ensure_dir(deskew_visualize_dir)


    rdp_visualize_dir = None
    if rdp_visualize:
        rdp_visualize_dir = os.path.join(save_style_root, "_debug_rdp_images")
        ensure_dir(rdp_visualize_dir)

        global _rdp_visualize_samples
        _rdp_visualize_samples = {}

    print("=" * 80)
    print("IAM + BRUSH 통합 데이터셋 생성")
    print("=" * 80)
    print(f"IAM 루트: {iam_root}")
    print(f"BRUSH 루트: {brush_root}")
    print(f"정규화: 높이=1.0 기준")
    print(f"RDP Epsilon: {rdp_epsilon_normalized} (정규화 좌표 기준, 높이의 {rdp_epsilon_normalized*100:.1f}%)" if rdp_epsilon_normalized else "RDP Epsilon: None (미적용)")
    print(f" Deskew: {'활성화' if deskew_enabled else '비활성화'} (threshold={deskew_angle_threshold}°, max={deskew_max_angle}°)")
    if deskew_visualize:
        print(f"   시각화 저장: {deskew_visualize_dir}")
    print(f" RDP 시각화: {'활성화' if rdp_visualize else '비활성화'}")
    if rdp_visualize:
        print(f"   시각화 저장: {rdp_visualize_dir}")
    print(f"출력 디렉토리: merging_* 시리즈")
    print("=" * 80)
    print()


    IAM_WRITER_OFFSET = int(iam_writer_offset)
    BRUSH_WRITER_OFFSET = int(brush_writer_offset)

    all_writers_info = []


    global_stats = {
        'IAM': {
            'before': {'point_counts': [], 'x_ranges': [], 'y_ranges': []},
            'after': {'point_counts': [], 'x_ranges': [], 'y_ranges': []},
            'deskew': {'corrected': 0, 'angles': []}
        },
        'BRUSH': {
            'before': {'point_counts': [], 'x_ranges': [], 'y_ranges': []},
            'after': {'point_counts': [], 'x_ranges': [], 'y_ranges': []}
        }
    }


    iam_conversion_ok = _check_and_run_convert_gt(iam_root)

    if not iam_conversion_ok:
        print("[ERROR] IAM GT 변환 실패. IAM 데이터는 스킵됩니다.")
        print("        위의 안내를 따라 data_preparation 모듈을 설치하거나")
        print("        convert_gt.py를 수동으로 실행해주세요.")
        print()
        iam_writers = []
        writer_to_json_files = {}
    else:
        print("[IAM] 작가 목록 수집 중...")


        if iam_root.endswith("converted"):
            iam_original_base = os.path.dirname(iam_root)
        else:
            iam_original_base = iam_root


        form_to_writer = _load_iam_writer_mapping(iam_original_base)


        import glob
        json_files = glob.glob(os.path.join(iam_root, "**", "*.xml.json"), recursive=True)
        print(f"[IAM] JSON 파일 스캔 완료: {len(json_files)}개 파일")

        writer_to_json_files = defaultdict(list)
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                key = data.get('key', '')
                form_id = _extract_form_id_from_key(key)
                writer_id = form_to_writer.get(form_id)
                if writer_id:
                    writer_to_json_files[writer_id].append(json_file)
            except:
                continue

        print(f"[IAM] Writer 매핑 완료: {len(writer_to_json_files)}명 작가 (고유 writerID)")
        print(f"      총 {sum(len(files) for files in writer_to_json_files.values())}개 JSON 파일")

        iam_writers = sorted(writer_to_json_files.keys())
        if max_iam_writers is not None:
            iam_writers = iam_writers[:max_iam_writers]

        print(f"[IAM] 처리할 작가: {len(iam_writers)}명")
        print()

    for iam_idx, iam_writer_id in enumerate(tqdm(iam_writers, desc="[IAM] 작가 처리")):
        merged_writer_id = IAM_WRITER_OFFSET + iam_idx
        merged_writer_id_str = str(merged_writer_id)


        char_file = os.path.join(save_char_root, f"{merged_writer_id_str}_rep.pkl")
        sent_file = os.path.join(save_sent_root, f"{merged_writer_id_str}_sent.pkl")
        style_file = os.path.join(save_style_root, f"{merged_writer_id_str}.pkl")
        style_rep_file = os.path.join(save_style_root, f"{merged_writer_id_str}_rep.pkl")

        if os.path.exists(char_file) and os.path.exists(sent_file) and os.path.exists(style_file) and os.path.exists(style_rep_file):
            if verbose_per_writer:
                print(f"[SKIP] IAM writer {merged_writer_id_str}: 이미 생성됨")
            continue

        json_file_list = writer_to_json_files.get(iam_writer_id, [])
        if not json_file_list:
            continue


        writer_chars, writer_sentences, deskew_stats = load_iam_writer_data(
            iam_root, iam_writer_id, json_file_list, img_size, rdp_epsilon_normalized,
            verbose_per_writer=verbose_per_writer,
            deskew_enabled=deskew_enabled,
            deskew_visualize_dir=deskew_visualize_dir,
            deskew_angle_threshold=deskew_angle_threshold,
            deskew_max_angle=deskew_max_angle,
            rdp_visualize=rdp_visualize
        )


        global_stats['IAM']['deskew']['corrected'] += deskew_stats['corrected']
        global_stats['IAM']['deskew']['angles'].extend(deskew_stats['angles'])

        if not writer_chars:
            continue


        for ch, samples in writer_chars.items():
            for sample in samples:

                if '_before_points' in sample:
                    global_stats['IAM']['before']['point_counts'].append(sample['_before_points'])
                    global_stats['IAM']['before']['x_ranges'].append(sample.get('_before_x_range', 0.0))
                    global_stats['IAM']['before']['y_ranges'].append(sample.get('_before_y_range', 0.0))


                if '_after_points' in sample:
                    global_stats['IAM']['after']['point_counts'].append(sample['_after_points'])
                    global_stats['IAM']['after']['x_ranges'].append(sample.get('_after_x_range', 0.0))
                    global_stats['IAM']['after']['y_ranges'].append(sample.get('_after_y_range', 0.0))


        if verbose_per_writer:
            point_counts = []
            for ch, samples in writer_chars.items():
                for sample in samples:
                    if '_after_points' in sample:
                        point_counts.append(sample['_after_points'])

            if point_counts:
                avg_points = sum(point_counts) / len(point_counts)
                min_points = min(point_counts)
                max_points = max(point_counts)

                print(f"[IAM] writer {merged_writer_id_str}: {len(writer_chars)}개 글자 타입, {len(point_counts)}개 샘플")
                print(f"  포인트/글자: 평균={avg_points:.1f}, 최소={min_points}, 최대={max_points}")

            print(f"   [IAM] {merged_writer_id_str}")


        rep_map = {}
        for ch, samples in writer_chars.items():
            if ch == ' ':
                continue
            rep, used_conn = _choose_representative(samples)
            if rep is not None:
                rep_map[ch] = _left_align_sample(rep, (H, W))


        with open(os.path.join(save_char_root, f"{merged_writer_id_str}_rep.pkl"), "wb") as f:
            pickle.dump(rep_map, f)


        writer_chars_filtered = {ch: samples for ch, samples in writer_chars.items()
                                if ch.isalpha() or ch.isdigit()}
        with open(os.path.join(save_style_root, f"{merged_writer_id_str}.pkl"), "wb") as f:
            pickle.dump(writer_chars_filtered, f)


        style_rep_map = {ch: [rep] for ch, rep in rep_map.items()
                        if (ch.isalpha() or ch.isdigit()) and rep.get('image') is not None}
        with open(os.path.join(save_style_root, f"{merged_writer_id_str}_rep.pkl"), "wb") as f:
            pickle.dump(style_rep_map, f)


        with open(os.path.join(save_sent_root, f"{merged_writer_id_str}_sent.pkl"), "wb") as f:
            pickle.dump({"sentences": writer_sentences}, f)

        all_writers_info.append({
            'merged_id': merged_writer_id,
            'original_id': iam_writer_id,
            'source': 'IAM',
            'num_chars': len(writer_chars),
            'num_sentences': len(writer_sentences),
        })

    print("-" * 80)
    print(f" [IAM] 처리 완료: {len([w for w in all_writers_info if w['source'] == 'IAM'])}명 작가")


    if deskew_enabled and global_stats['IAM']['deskew']['angles']:
        angles = global_stats['IAM']['deskew']['angles']
        corrected = global_stats['IAM']['deskew']['corrected']
        total_sentences = len(angles)


        abs_angles = [abs(a) for a in angles]
        avg_angle = np.mean(abs_angles)
        max_angle_val = max(abs_angles)


        angle_bins = [0, 1, 2, 5, 10, 20, 30]
        angle_dist = []
        for i in range(len(angle_bins) - 1):
            count = sum(1 for a in abs_angles if angle_bins[i] <= a < angle_bins[i+1])
            angle_dist.append((f"{angle_bins[i]}-{angle_bins[i+1]}°", count))
        over_30 = sum(1 for a in abs_angles if a >= 30)
        angle_dist.append((f"30°+", over_30))

        print(f"\n [IAM Deskew 통계]")
        print(f"   총 문장: {total_sentences}개")
        print(f"   보정 적용: {corrected}개 ({corrected/total_sentences*100:.1f}%)")
        print(f"   평균 기울기: {avg_angle:.2f}° (최대: {max_angle_val:.2f}°)")
        print(f"   분포:")
        for label, count in angle_dist:
            if count > 0:
                print(f"     {label:8s}: {count:5d} ({count/total_sentences*100:5.1f}%)")

        if deskew_visualize_dir:
            visualized_count = len([f for f in os.listdir(deskew_visualize_dir) if f.startswith("deskew_")])
            print(f"   시각화 저장: {visualized_count}개 → {deskew_visualize_dir}")

    print()
    print()


    print("[BRUSH] 작가 목록 수집 중...")
    brush_writers = sorted([d for d in os.listdir(brush_root) if os.path.isdir(os.path.join(brush_root, d))])
    if max_brush_writers is not None:
        brush_writers = brush_writers[:max_brush_writers]

    print(f"[BRUSH] 처리할 작가: {len(brush_writers)}명")
    print()

    for brush_idx, brush_writer_id in enumerate(tqdm(brush_writers, desc="[BRUSH] 작가 처리")):
        merged_writer_id = BRUSH_WRITER_OFFSET + brush_idx
        merged_writer_id_str = str(merged_writer_id)


        char_file = os.path.join(save_char_root, f"{merged_writer_id_str}_rep.pkl")
        sent_file = os.path.join(save_sent_root, f"{merged_writer_id_str}_sent.pkl")
        style_file = os.path.join(save_style_root, f"{merged_writer_id_str}.pkl")
        style_rep_file = os.path.join(save_style_root, f"{merged_writer_id_str}_rep.pkl")

        if os.path.exists(char_file) and os.path.exists(sent_file) and os.path.exists(style_file) and os.path.exists(style_rep_file):
            if verbose_per_writer:
                print(f"[SKIP] BRUSH writer {merged_writer_id_str}: 이미 생성됨")
            continue


        writer_chars, writer_sentences, alpha_rep_collect = load_brush_writer_data(
            brush_root, brush_writer_id, img_size, resample_type, rdp_epsilon_normalized,
            rdp_visualize=rdp_visualize
        )

        if not writer_chars:
            continue


        for ch, samples in writer_chars.items():
            for sample in samples:

                if '_before_points' in sample:
                    global_stats['BRUSH']['before']['point_counts'].append(sample['_before_points'])
                    global_stats['BRUSH']['before']['x_ranges'].append(sample.get('_before_x_range', 0.0))
                    global_stats['BRUSH']['before']['y_ranges'].append(sample.get('_before_y_range', 0.0))


                if '_after_points' in sample:
                    global_stats['BRUSH']['after']['point_counts'].append(sample['_after_points'])
                    global_stats['BRUSH']['after']['x_ranges'].append(sample.get('_after_x_range', 0.0))
                    global_stats['BRUSH']['after']['y_ranges'].append(sample.get('_after_y_range', 0.0))


        if verbose_per_writer:
            point_counts = []
            for ch, samples in writer_chars.items():
                for sample in samples:
                    if '_after_points' in sample:
                        point_counts.append(sample['_after_points'])

            if point_counts:
                avg_points = sum(point_counts) / len(point_counts)
                min_points = min(point_counts)
                max_points = max(point_counts)

                print(f"[BRUSH] writer {merged_writer_id_str}: {len(writer_chars)}개 글자 타입, {len(point_counts)}개 샘플")
                print(f"  포인트/글자: 평균={avg_points:.1f}, 최소={min_points}, 최대={max_points}")

            print(f"   [BRUSH] {merged_writer_id_str}")


        rep_map = {}
        for ch, samples in writer_chars.items():
            if ch == ' ':
                continue


            if ch in alpha_rep_collect and alpha_rep_collect[ch]:
                rep = _pick_median_len_first3(alpha_rep_collect[ch])
                if rep is not None:
                    rep_map[ch] = _left_align_sample(rep, (H, W))


            if ch not in rep_map:
                rep, used_conn = _choose_representative(samples)
                if rep is not None:
                    rep_map[ch] = _left_align_sample(rep, (H, W))


        with open(os.path.join(save_char_root, f"{merged_writer_id_str}_rep.pkl"), "wb") as f:
            pickle.dump(rep_map, f)


        writer_chars_filtered = {ch: samples for ch, samples in writer_chars.items()
                                if ch.isalpha() or ch.isdigit()}
        with open(os.path.join(save_style_root, f"{merged_writer_id_str}.pkl"), "wb") as f:
            pickle.dump(writer_chars_filtered, f)


        style_rep_map = {ch: [rep] for ch, rep in rep_map.items()
                        if (ch.isalpha() or ch.isdigit()) and rep.get('image') is not None}
        with open(os.path.join(save_style_root, f"{merged_writer_id_str}_rep.pkl"), "wb") as f:
            pickle.dump(style_rep_map, f)


        with open(os.path.join(save_sent_root, f"{merged_writer_id_str}_sent.pkl"), "wb") as f:
            pickle.dump({"sentences": writer_sentences}, f)

        all_writers_info.append({
            'merged_id': merged_writer_id,
            'original_id': brush_writer_id,
            'source': 'BRUSH',
            'num_chars': len(writer_chars),
            'num_sentences': len(writer_sentences),
        })

    print("-" * 80)
    print(f" [BRUSH] 처리 완료: {len([w for w in all_writers_info if w['source'] == 'BRUSH'])}명 작가")
    print()


    print("=" * 80)
    print(" 통합 데이터셋 통계")
    print("=" * 80)
    print()


    iam_writers_count = len([w for w in all_writers_info if w['source'] == 'IAM'])
    brush_writers_count = len([w for w in all_writers_info if w['source'] == 'BRUSH'])
    iam_total_samples = len(global_stats['IAM']['after']['point_counts'])
    brush_total_samples = len(global_stats['BRUSH']['after']['point_counts'])


    iam_total_sentences = sum(w['num_sentences'] for w in all_writers_info if w['source'] == 'IAM')
    brush_total_sentences = sum(w['num_sentences'] for w in all_writers_info if w['source'] == 'BRUSH')


    iam_avg_chars_per_sent = iam_total_samples / iam_total_sentences if iam_total_sentences > 0 else 0
    brush_avg_chars_per_sent = brush_total_samples / brush_total_sentences if brush_total_sentences > 0 else 0
    iam_avg_sents_per_writer = iam_total_sentences / iam_writers_count if iam_writers_count > 0 else 0
    brush_avg_sents_per_writer = brush_total_sentences / brush_writers_count if brush_writers_count > 0 else 0

    print(f"[요약]")
    print(f"  IAM:   {iam_writers_count}명 작가, {iam_total_sentences:,}개 문장, {iam_total_samples:,}개 샘플 (글자)")
    if iam_writers_count > 0:
        print(f"        → 문장당 평균 {iam_avg_chars_per_sent:.1f}개 글자, 작가당 평균 {iam_avg_sents_per_writer:.1f}개 문장")
    print(f"  BRUSH: {brush_writers_count}명 작가, {brush_total_sentences:,}개 문장, {brush_total_samples:,}개 샘플 (글자)")
    if brush_writers_count > 0:
        print(f"        → 문장당 평균 {brush_avg_chars_per_sent:.1f}개 글자, 작가당 평균 {brush_avg_sents_per_writer:.1f}개 문장")
    print(f"  전체:  {iam_writers_count + brush_writers_count}명 작가, {iam_total_sentences + brush_total_sentences:,}개 문장, {iam_total_samples + brush_total_samples:,}개 샘플")
    print()


    for source in ['IAM', 'BRUSH']:
        print("=" * 80)
        print(f" [{source}] Before/After 통계")
        print("=" * 80)

        for stage in ['before', 'after']:
            stage_data = global_stats[source][stage]
            counts = stage_data['point_counts']
            x_ranges = stage_data['x_ranges']
            y_ranges = stage_data['y_ranges']

            if not counts:
                continue

            stage_label = "BEFORE (정규화 전, 원본 픽셀)" if stage == 'before' else "AFTER (정규화 + RDP 적용 후)"

            avg_points = sum(counts) / len(counts)
            min_points = min(counts)
            max_points = max(counts)

            avg_x = sum(x_ranges) / len(x_ranges) if x_ranges else 0
            avg_y = sum(y_ranges) / len(y_ranges) if y_ranges else 0
            min_x = min(x_ranges) if x_ranges else 0
            max_x = max(x_ranges) if x_ranges else 0
            min_y = min(y_ranges) if y_ranges else 0
            max_y = max(y_ranges) if y_ranges else 0

            print(f"\n[{stage_label}]")
            print(f"  샘플 수: {len(counts):,}개")
            print(f"  포인트/글자: 평균={avg_points:.1f}, 최소={min_points}, 최대={max_points}")
            print(f"  좌표 범위:")
            print(f"    X: 평균={avg_x:.3f}, 최소={min_x:.3f}, 최대={max_x:.3f}")
            print(f"    Y: 평균={avg_y:.3f}, 최소={min_y:.3f}, 최대={max_y:.3f}")


            bins = [0, 5, 10, 15, 20, 30, 50, 100, 200, float('inf')]
            labels = ['1-5', '6-10', '11-15', '16-20', '21-30', '31-50', '51-100', '101-200', '200+']
            dist_counts = [0] * len(labels)
            for p in counts:
                for i, (lower, upper) in enumerate(zip(bins[:-1], bins[1:])):
                    if lower < p <= upper:
                        dist_counts[i] += 1
                        break

            print(f"  분포:")
            for label, count in zip(labels, dist_counts):
                pct = count / len(counts) * 100
                if count > 0:
                    print(f"    {label:8s}: {count:6,} ({pct:5.1f}%)")
        print()


    if global_stats['IAM']['after']['point_counts'] and global_stats['BRUSH']['after']['point_counts']:
        print("=" * 80)
        print(" 데이터셋 간 비교 (AFTER)")
        print("=" * 80)

        iam_before_avg = sum(global_stats['IAM']['before']['point_counts']) / len(global_stats['IAM']['before']['point_counts'])
        iam_after_avg = sum(global_stats['IAM']['after']['point_counts']) / len(global_stats['IAM']['after']['point_counts'])
        brush_before_avg = sum(global_stats['BRUSH']['before']['point_counts']) / len(global_stats['BRUSH']['before']['point_counts'])
        brush_after_avg = sum(global_stats['BRUSH']['after']['point_counts']) / len(global_stats['BRUSH']['after']['point_counts'])

        iam_reduction = (1 - iam_after_avg / iam_before_avg) * 100 if iam_before_avg > 0 else 0
        brush_reduction = (1 - brush_after_avg / brush_before_avg) * 100 if brush_before_avg > 0 else 0


        iam_before_x = sum(global_stats['IAM']['before']['x_ranges']) / len(global_stats['IAM']['before']['x_ranges']) if global_stats['IAM']['before']['x_ranges'] else 0
        iam_before_y = sum(global_stats['IAM']['before']['y_ranges']) / len(global_stats['IAM']['before']['y_ranges']) if global_stats['IAM']['before']['y_ranges'] else 0
        iam_after_x = sum(global_stats['IAM']['after']['x_ranges']) / len(global_stats['IAM']['after']['x_ranges']) if global_stats['IAM']['after']['x_ranges'] else 0
        iam_after_y = sum(global_stats['IAM']['after']['y_ranges']) / len(global_stats['IAM']['after']['y_ranges']) if global_stats['IAM']['after']['y_ranges'] else 0

        brush_before_x = sum(global_stats['BRUSH']['before']['x_ranges']) / len(global_stats['BRUSH']['before']['x_ranges']) if global_stats['BRUSH']['before']['x_ranges'] else 0
        brush_before_y = sum(global_stats['BRUSH']['before']['y_ranges']) / len(global_stats['BRUSH']['before']['y_ranges']) if global_stats['BRUSH']['before']['y_ranges'] else 0
        brush_after_x = sum(global_stats['BRUSH']['after']['x_ranges']) / len(global_stats['BRUSH']['after']['x_ranges']) if global_stats['BRUSH']['after']['x_ranges'] else 0
        brush_after_y = sum(global_stats['BRUSH']['after']['y_ranges']) / len(global_stats['BRUSH']['after']['y_ranges']) if global_stats['BRUSH']['after']['y_ranges'] else 0

        diff_after = abs(iam_after_avg - brush_after_avg)
        ratio_after = max(iam_after_avg, brush_after_avg) / min(iam_after_avg, brush_after_avg) if min(iam_after_avg, brush_after_avg) > 0 else 0

        print(f"\n[포인트 개수: BEFORE → AFTER 변화]")
        print(f"  IAM:   {iam_before_avg:.1f} → {iam_after_avg:.1f}개/글자 ({iam_reduction:.1f}% 감소)")
        print(f"  BRUSH: {brush_before_avg:.1f} → {brush_after_avg:.1f}개/글자 ({brush_reduction:.1f}% 감소)")

        print(f"\n[좌표 범위: BEFORE → AFTER 변화]")
        print(f"  IAM   X: {iam_before_x:.3f} → {iam_after_x:.3f}")
        print(f"  IAM   Y: {iam_before_y:.3f} → {iam_after_y:.3f}")
        print(f"  BRUSH X: {brush_before_x:.3f} → {brush_after_x:.3f}")
        print(f"  BRUSH Y: {brush_before_y:.3f} → {brush_after_y:.3f}")

        print(f"\n[데이터셋 간 차이 (AFTER)]")
        print(f"  IAM 평균:   {iam_after_avg:.1f}개/글자")
        print(f"  BRUSH 평균: {brush_after_avg:.1f}개/글자")
        print(f"  차이:       {diff_after:.1f}개")
        print(f"  비율:       {ratio_after:.2f}배")
        print()
        if diff_after < 5:
            print("   포인트 밀도 균등화 성공! (차이 < 5개)")
        elif diff_after < 8:
            print("    포인트 밀도 차이 있지만 허용 범위 (5~8개)")
        else:
            print(f"   포인트 밀도 차이 큼 (차이 {diff_after:.1f}개)")
            print(f"     → RDP epsilon을 {rdp_epsilon_normalized * 1.2:.3f} ~ {rdp_epsilon_normalized * 1.5:.3f}로 증가 권장")
        print()


    mapping_file = os.path.join(save_char_root, "writer_mapping.json")
    with open(mapping_file, 'w', encoding='utf-8') as f:
        json.dump(all_writers_info, f, indent=2, ensure_ascii=False)

    print(f" Writer 매핑 정보 저장: {mapping_file}")
    print()


    import random as _random
    _random.seed(None)

    debug_output_dir = os.path.join(save_style_root, "_debug_style_rep_images")
    ensure_dir(debug_output_dir)


    iam_writers_processed = [w for w in all_writers_info if w['source'] == 'IAM']
    if iam_writers_processed:
        iam_sample = _random.choice(iam_writers_processed)
        _save_style_rep_debug_images(
            save_style_root,
            iam_sample['merged_id'],
            debug_output_dir,
            f"IAM_{iam_sample['original_id']}"
        )


    brush_writers_processed = [w for w in all_writers_info if w['source'] == 'BRUSH']
    if brush_writers_processed:
        brush_sample = _random.choice(brush_writers_processed)
        _save_style_rep_debug_images(
            save_style_root,
            brush_sample['merged_id'],
            debug_output_dir,
            f"BRUSH_{brush_sample['original_id']}"
        )


    if rdp_visualize and rdp_visualize_dir:
        print()
        print("=" * 80)
        print(" RDP Before/After 시각화 저장")
        print("=" * 80)
        saved_count = _save_rdp_visualize_samples(rdp_visualize_dir, img_size=(150, 150))
        print(f"  저장 완료: {saved_count}개 이미지 → {rdp_visualize_dir}")


        if saved_count > 0:
            files = [f for f in os.listdir(rdp_visualize_dir) if f.startswith("rdp_")]
            iam_count = len([f for f in files if "IAM" in f])
            brush_count = len([f for f in files if "BRUSH" in f])
            print(f"  IAM: {iam_count}개, BRUSH: {brush_count}개")


    print_preprocessing_stats()

    print()
    print("=" * 80)
    print(" 통합 데이터셋 생성 완료!")
    print("=" * 80)
    print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="IAM + BRUSH 통합 데이터셋 생성")
    parser.add_argument("--iam_root", type=str, required=True, help="IAM 데이터 루트 (converted 디렉토리)")
    parser.add_argument("--brush_root", type=str, required=True, help="BRUSH 데이터 루트")
    parser.add_argument("--save_char_root", type=str, required=True, help="문자 pickle 저장 경로")
    parser.add_argument("--save_sent_root", type=str, required=True, help="문장 pickle 저장 경로")
    parser.add_argument("--save_style_root", type=str, required=True, help="스타일 pickle 저장 경로")
    parser.add_argument("--img_size", type=int, nargs=2, default=[64, 64], help="이미지 크기 (H W)")
    parser.add_argument(
        "--resample_type",
        type=str,
        default="original",
        choices=["original", "resample", "resample20", "resample25"],
        help="BRUSH 리샘플링 타입",
    )
    parser.add_argument("--rdp_epsilon_normalized", type=float, default=None, help="RDP epsilon (정규화 좌표 기준)")
    parser.add_argument("--max_iam_writers", type=int, default=None, help="최대 IAM 작가 수 (테스트용)")
    parser.add_argument("--max_brush_writers", type=int, default=None, help="최대 BRUSH 작가 수 (테스트용)")
    parser.add_argument("--iam_writer_offset", type=int, default=10000, help="IAM merged writer id offset")
    parser.add_argument("--brush_writer_offset", type=int, default=20000, help="BRUSH merged writer id offset")
    parser.add_argument("--verbose_per_writer", action="store_true", default=False, help="작가별 상세 로그 출력")

    parser.add_argument("--deskew_enabled", action="store_true", default=True, help="IAM Skew 보정 활성화 (기본: True)")
    parser.add_argument("--no_deskew", action="store_true", default=False, help="IAM Skew 보정 비활성화")
    parser.add_argument("--deskew_visualize", action="store_true", default=False, help="Skew 보정 Before/After 시각화")
    parser.add_argument("--deskew_angle_threshold", type=float, default=1.0, help="Skew 보정 최소 각도 (기본: 1.0도)")
    parser.add_argument("--deskew_max_angle", type=float, default=30.0, help="Skew 보정 최대 각도 (기본: 30.0도)")

    parser.add_argument("--rdp_visualize", action="store_true", default=False, help="RDP 전후 시각화 (글자당 최대 3개 샘플)")

    parser.add_argument("--max_traj_len", type=int, default=DEFAULT_MAX_TRAJ_LEN,
                        help=f"글자별 최대 trajectory 포인트 수 (기본: {DEFAULT_MAX_TRAJ_LEN})")
    args = parser.parse_args()


    deskew_enabled = not args.no_deskew

    generate_merged_pickles(
        iam_root=args.iam_root,
        brush_root=args.brush_root,
        save_char_root=args.save_char_root,
        save_sent_root=args.save_sent_root,
        save_style_root=args.save_style_root,
        img_size=tuple(args.img_size),
        resample_type=args.resample_type,
        rdp_epsilon_normalized=args.rdp_epsilon_normalized,
        max_iam_writers=args.max_iam_writers,
        max_brush_writers=args.max_brush_writers,
        verbose_per_writer=args.verbose_per_writer,

        deskew_enabled=deskew_enabled,
        deskew_visualize=args.deskew_visualize,
        deskew_angle_threshold=args.deskew_angle_threshold,
        deskew_max_angle=args.deskew_max_angle,

        rdp_visualize=args.rdp_visualize,

        max_traj_len=args.max_traj_len,
        iam_writer_offset=args.iam_writer_offset,
        brush_writer_offset=args.brush_writer_offset,
    )
