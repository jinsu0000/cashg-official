import os
import pickle
import json
import numpy as np
import lmdb
from collections import defaultdict
from PIL import Image, ImageDraw
from tqdm import tqdm
from typing import Optional, Dict, List, Tuple, Any


from src.config.constants import (
    TRAJ_INDEX, TRAJ_DIM, PEN_CLASS
)

from src.preprocessing.brush_handwriting_dataset_generator import (
    _normalize_sentence_height,
    render_image_from_traj,
    _choose_representative,
    _left_align_sample,
    _traj_len,
    ensure_dir,
)


def _convert_sdt_to_cashg_coords(sdt_coords: np.ndarray) -> np.ndarray:
    T = sdt_coords.shape[0]
    cashg_traj = np.zeros((T, 3), dtype=np.float32)


    cashg_traj[:, 0] = sdt_coords[:, 0]
    cashg_traj[:, 1] = sdt_coords[:, 1]


    for i in range(T):
        p2 = sdt_coords[i, 3]
        is_last_point = (i == T - 1)

        if is_last_point:

            cashg_traj[i, 2] = PEN_CLASS["EOC"]
        elif p2 == 1:

            cashg_traj[i, 2] = PEN_CLASS["PU"]
        else:

            cashg_traj[i, 2] = PEN_CLASS["PM"]

    return cashg_traj


def _normalize_char_trajectory(traj: np.ndarray, target_height: float = 1.0) -> Tuple[np.ndarray, float, float, float, float, float]:
    if traj.shape[0] == 0:
        return traj, 1.0, 0.0, 0.0, 0.0, 0.0

    x_min, y_min = traj[:, 0].min(), traj[:, 1].min()
    x_max, y_max = traj[:, 0].max(), traj[:, 1].max()

    original_height = y_max - y_min
    original_width = x_max - x_min

    if original_height < 1e-6:

        scale = target_height / max(original_width, 1e-6)
    else:
        scale = target_height / original_height

    normed_traj = traj.copy()
    normed_traj[:, 0] = (traj[:, 0] - x_min) * scale
    normed_traj[:, 1] = (traj[:, 1] - y_min) * scale

    return normed_traj, scale, x_min, y_min, original_height, original_width


def _process_single_char(
    sdt_coords: np.ndarray,
    char: str,
    writer_id: str,
    sample_idx: int,
    img_size: Tuple[int, int],
) -> Dict:
    H, W = img_size


    cashg_traj = _convert_sdt_to_cashg_coords(sdt_coords)


    before_points = len(cashg_traj)
    before_x_range = float(np.max(cashg_traj[:, 0]) - np.min(cashg_traj[:, 0])) if len(cashg_traj) > 0 else 0.0
    before_y_range = float(np.max(cashg_traj[:, 1]) - np.min(cashg_traj[:, 1])) if len(cashg_traj) > 0 else 0.0


    normed_traj, scale, x_min, y_min, original_height, original_width = _normalize_char_trajectory(cashg_traj)


    after_points = len(normed_traj)
    after_x_range = float(np.max(normed_traj[:, 0]) - np.min(normed_traj[:, 0])) if len(normed_traj) > 0 else 0.0
    after_y_range = float(np.max(normed_traj[:, 1]) - np.min(normed_traj[:, 1])) if len(normed_traj) > 0 else 0.0


    img = render_image_from_traj(normed_traj, (H, W))


    sentence_id = f"casia_{writer_id}_{sample_idx}"

    char_item = {
        "trajectory": normed_traj.astype(np.float32),
        "image": img.astype(np.uint8) if img is not None else None,
        "connection": "isolated",
        "character": char,
        "sentence_id": sentence_id,
        "cursor": 0,
        "scale": float(scale),
        "min_xy": np.array([x_min, y_min], dtype=np.float32),
        "orig_min_x": None,
        "original_height": float(original_height),
        "original_width": float(original_width),
        "coord_mode": "normalized_height_1.0",
        "source_dataset": "CASIA_ENGLISH",

        "_before_points": before_points,
        "_before_x_range": before_x_range,
        "_before_y_range": before_y_range,
        "_after_points": after_points,
        "_after_x_range": after_x_range,
        "_after_y_range": after_y_range,
    }

    return char_item


def load_casia_lmdb_data(
    lmdb_path: str,
    img_size: Tuple[int, int],
    max_len: int = 150,
) -> Dict[str, Dict[str, List[Dict]]]:
    env = lmdb.open(lmdb_path, max_readers=8, readonly=True, lock=False, readahead=False, meminit=False)

    writer_data = defaultdict(lambda: defaultdict(list))
    skipped_count = 0

    with env.begin(write=False) as txn:
        num_sample = int(txn.get('num_sample'.encode('utf-8')).decode())

        print(f"  총 샘플 수: {num_sample:,}")

        for i in tqdm(range(num_sample), desc="  LMDB 로딩"):
            data = pickle.loads(txn.get(str(i).encode('utf-8')))

            tag_char = data['tag_char']
            coords = data['coordinates']
            fname = data['fname']


            writer_id = fname.split('.')[0]


            if max_len > 0 and len(coords) >= max_len:
                skipped_count += 1
                continue


            char_item = _process_single_char(
                sdt_coords=coords,
                char=tag_char,
                writer_id=writer_id,
                sample_idx=i,
                img_size=img_size,
            )

            writer_data[writer_id][tag_char].append(char_item)

    env.close()

    print(f"  스킵된 샘플 (포인트 > {max_len}): {skipped_count:,}")
    print(f"  로드된 작가 수: {len(writer_data)}")

    return dict(writer_data)


def load_style_samples(style_path: str) -> Dict[str, np.ndarray]:
    style_images = {}

    for fname in os.listdir(style_path):
        if not fname.endswith('.pkl'):
            continue

        pkl_path = os.path.join(style_path, fname)
        with open(pkl_path, 'rb') as f:
            samples = pickle.load(f)


        for sample in samples:
            char = sample['label']
            img = sample['img']
            if char not in style_images:
                style_images[char] = img

    return style_images


def generate_casia_pickles(
    casia_root: str,
    save_char_root: str,
    save_sent_root: str,
    save_style_root: str,
    img_size: Tuple[int, int] = (64, 64),
    max_len: int = 150,
    process_train: bool = True,
    process_test: bool = True,
    max_writers: Optional[int] = None,
    verbose_per_writer: bool = False,
):
    ensure_dir(save_char_root)
    ensure_dir(save_sent_root)
    ensure_dir(save_style_root)

    H, W = img_size

    print("=" * 80)
    print("CASIA_ENGLISH → CASHG 데이터셋 변환")
    print("=" * 80)
    print(f"CASIA 루트: {casia_root}")
    print(f"정규화: 높이=1.0 기준")
    print(f"RDP: 미적용 (GT 비교용 원본 유지)")
    print(f"최대 포인트 수: {max_len}")
    print(f"출력 디렉토리:")
    print(f"  - char: {save_char_root}")
    print(f"  - sent: {save_sent_root}")
    print(f"  - style: {save_style_root}")
    print("=" * 80)
    print()


    TRAIN_WRITER_OFFSET = 30000
    TEST_WRITER_OFFSET = 31000

    all_writers_info = []


    global_stats = {
        'train': {
            'before': {'point_counts': [], 'x_ranges': [], 'y_ranges': []},
            'after': {'point_counts': [], 'x_ranges': [], 'y_ranges': []}
        },
        'test': {
            'before': {'point_counts': [], 'x_ranges': [], 'y_ranges': []},
            'after': {'point_counts': [], 'x_ranges': [], 'y_ranges': []}
        }
    }


    if process_train:
        print("[TRAIN] 데이터 로딩 중...")
        train_lmdb_path = os.path.join(casia_root, 'train')
        train_style_path = os.path.join(casia_root, 'train_style_samples')

        train_writer_data = load_casia_lmdb_data(
            train_lmdb_path, img_size, max_len
        )

        train_writers = sorted(train_writer_data.keys())
        if max_writers is not None:
            train_writers = train_writers[:max_writers]

        print(f"\n[TRAIN] 처리할 작가: {len(train_writers)}명")

        for writer_idx, writer_id in enumerate(tqdm(train_writers, desc="[TRAIN] 작가 처리")):
            merged_writer_id = TRAIN_WRITER_OFFSET + writer_idx
            merged_writer_id_str = str(merged_writer_id)

            writer_chars = train_writer_data[writer_id]

            if not writer_chars:
                continue


            for ch, samples in writer_chars.items():
                for sample in samples:
                    if '_before_points' in sample:
                        global_stats['train']['before']['point_counts'].append(sample['_before_points'])
                        global_stats['train']['before']['x_ranges'].append(sample.get('_before_x_range', 0.0))
                        global_stats['train']['before']['y_ranges'].append(sample.get('_before_y_range', 0.0))
                    if '_after_points' in sample:
                        global_stats['train']['after']['point_counts'].append(sample['_after_points'])
                        global_stats['train']['after']['x_ranges'].append(sample.get('_after_x_range', 0.0))
                        global_stats['train']['after']['y_ranges'].append(sample.get('_after_y_range', 0.0))


            rep_map = {}
            for ch, samples in writer_chars.items():
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


            sentences = {}
            for ch, samples in writer_chars.items():
                for sample in samples:
                    sid = sample['sentence_id']
                    sentences[sid] = [sample]

            with open(os.path.join(save_sent_root, f"{merged_writer_id_str}_sent.pkl"), "wb") as f:
                pickle.dump({"sentences": sentences}, f)

            all_writers_info.append({
                'merged_id': merged_writer_id,
                'original_id': writer_id,
                'source': 'CASIA_ENGLISH_TRAIN',
                'num_chars': len(writer_chars),
                'num_sentences': len(sentences),
            })

            if verbose_per_writer:
                print(f"   [TRAIN] {merged_writer_id_str} (orig: {writer_id}): {len(writer_chars)}개 글자 타입")

        print(f"\n [TRAIN] 처리 완료: {len([w for w in all_writers_info if 'TRAIN' in w['source']])}명 작가")
        print()


    if process_test:
        print("[TEST] 데이터 로딩 중...")
        test_lmdb_path = os.path.join(casia_root, 'test')
        test_style_path = os.path.join(casia_root, 'test_style_samples')

        test_writer_data = load_casia_lmdb_data(
            test_lmdb_path, img_size, max_len
        )

        test_writers = sorted(test_writer_data.keys())
        if max_writers is not None:
            test_writers = test_writers[:max_writers]

        print(f"\n[TEST] 처리할 작가: {len(test_writers)}명")

        for writer_idx, writer_id in enumerate(tqdm(test_writers, desc="[TEST] 작가 처리")):
            merged_writer_id = TEST_WRITER_OFFSET + writer_idx
            merged_writer_id_str = str(merged_writer_id)

            writer_chars = test_writer_data[writer_id]

            if not writer_chars:
                continue


            for ch, samples in writer_chars.items():
                for sample in samples:
                    if '_before_points' in sample:
                        global_stats['test']['before']['point_counts'].append(sample['_before_points'])
                        global_stats['test']['before']['x_ranges'].append(sample.get('_before_x_range', 0.0))
                        global_stats['test']['before']['y_ranges'].append(sample.get('_before_y_range', 0.0))
                    if '_after_points' in sample:
                        global_stats['test']['after']['point_counts'].append(sample['_after_points'])
                        global_stats['test']['after']['x_ranges'].append(sample.get('_after_x_range', 0.0))
                        global_stats['test']['after']['y_ranges'].append(sample.get('_after_y_range', 0.0))


            rep_map = {}
            for ch, samples in writer_chars.items():
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


            sentences = {}
            for ch, samples in writer_chars.items():
                for sample in samples:
                    sid = sample['sentence_id']
                    sentences[sid] = [sample]

            with open(os.path.join(save_sent_root, f"{merged_writer_id_str}_sent.pkl"), "wb") as f:
                pickle.dump({"sentences": sentences}, f)

            all_writers_info.append({
                'merged_id': merged_writer_id,
                'original_id': writer_id,
                'source': 'CASIA_ENGLISH_TEST',
                'num_chars': len(writer_chars),
                'num_sentences': len(sentences),
            })

            if verbose_per_writer:
                print(f"   [TEST] {merged_writer_id_str} (orig: {writer_id}): {len(writer_chars)}개 글자 타입")

        print(f"\n [TEST] 처리 완료: {len([w for w in all_writers_info if 'TEST' in w['source']])}명 작가")
        print()


    print("=" * 80)
    print(" 데이터셋 통계")
    print("=" * 80)
    print()


    train_writers_count = len([w for w in all_writers_info if 'TRAIN' in w['source']])
    test_writers_count = len([w for w in all_writers_info if 'TEST' in w['source']])
    train_total_samples = len(global_stats['train']['after']['point_counts'])
    test_total_samples = len(global_stats['test']['after']['point_counts'])

    print(f"[요약]")
    print(f"  TRAIN: {train_writers_count}명 작가, {train_total_samples:,}개 샘플 (글자)")
    print(f"  TEST:  {test_writers_count}명 작가, {test_total_samples:,}개 샘플 (글자)")
    print(f"  전체:  {train_writers_count + test_writers_count}명 작가, {train_total_samples + test_total_samples:,}개 샘플")
    print()


    for split in ['train', 'test']:
        if not global_stats[split]['after']['point_counts']:
            continue

        print(f"\n[{split.upper()}] Before/After 통계")
        print("-" * 40)

        for stage in ['before', 'after']:
            stage_data = global_stats[split][stage]
            counts = stage_data['point_counts']

            if not counts:
                continue

            stage_label = "BEFORE (정규화 전)" if stage == 'before' else "AFTER (정규화 후)"

            avg_points = sum(counts) / len(counts)
            min_points = min(counts)
            max_points = max(counts)

            print(f"\n  [{stage_label}]")
            print(f"    샘플 수: {len(counts):,}개")
            print(f"    포인트/글자: 평균={avg_points:.1f}, 최소={min_points}, 최대={max_points}")

    print()


    mapping_file = os.path.join(save_char_root, "writer_mapping.json")
    with open(mapping_file, 'w', encoding='utf-8') as f:
        json.dump(all_writers_info, f, indent=2, ensure_ascii=False)

    print(f" Writer 매핑 정보 저장: {mapping_file}")
    print()
    print("=" * 80)
    print(" CASIA_ENGLISH 데이터셋 변환 완료!")
    print("=" * 80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CASIA_ENGLISH → CASHG 데이터셋 변환")
    parser.add_argument("--casia_root", type=str, required=True, help="CASIA_ENGLISH 루트 디렉토리")
    parser.add_argument("--save_char_root", type=str, required=True, help="문자 pickle 저장 경로")
    parser.add_argument("--save_sent_root", type=str, required=True, help="문장 pickle 저장 경로")
    parser.add_argument("--save_style_root", type=str, required=True, help="스타일 pickle 저장 경로")
    parser.add_argument("--img_size", type=int, nargs=2, default=[64, 64], help="이미지 크기 (H W)")
    parser.add_argument("--max_len", type=int, default=150, help="최대 포인트 수 (초과 시 스킵)")
    parser.add_argument("--process_train", action="store_true", help="train 데이터 처리")
    parser.add_argument("--process_test", action="store_true", help="test 데이터 처리")
    parser.add_argument("--max_writers", type=int, default=None, help="최대 작가 수 (테스트용)")
    parser.add_argument("--verbose_per_writer", action="store_true", default=False, help="작가별 상세 로그 출력")
    args = parser.parse_args()


    if not args.process_train and not args.process_test:
        args.process_train = True
        args.process_test = True

    generate_casia_pickles(
        casia_root=args.casia_root,
        save_char_root=args.save_char_root,
        save_sent_root=args.save_sent_root,
        save_style_root=args.save_style_root,
        img_size=tuple(args.img_size),
        max_len=args.max_len,
        process_train=args.process_train,
        process_test=args.process_test,
        max_writers=args.max_writers,
        verbose_per_writer=args.verbose_per_writer,
    )
