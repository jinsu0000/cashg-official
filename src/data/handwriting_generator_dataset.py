import os
import glob
import random
import threading
from collections import defaultdict, OrderedDict
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.data_utils import (
    load_pickle_cached,
    to_tensor_img_gray,
    pad_1d,
    traj_abs_to_delta,
    to_local_bigram,
    sample_style_refs_from_writer,
)
from src.utils.logger import print_once


class StyleCache:

    def __init__(self, max_tensor_cache_items: Optional[int] = None):
        self._file_cache: Dict[str, Dict] = {}
        self._valid_samples_cache: Dict[Tuple[int, str], List[Tuple[str, Dict]]] = {}
        self._tensor_cache: "OrderedDict[Tuple[int, int], torch.Tensor]" = OrderedDict()
        self._max_tensor_cache_items = int(max_tensor_cache_items) if max_tensor_cache_items is not None else -1
        self._lock = threading.Lock()

    def set_tensor_cache_limit(self, max_items: Optional[int]):
        with self._lock:
            self._max_tensor_cache_items = int(max_items) if max_items is not None else -1
            if self._max_tensor_cache_items <= 0:
                self._max_tensor_cache_items = -1
                return
            while len(self._tensor_cache) > self._max_tensor_cache_items:
                self._tensor_cache.popitem(last=False)

    def load_pickle(self, path: str) -> Dict:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Style pickle not found: {path}")

        with self._lock:
            if path in self._file_cache:
                return self._file_cache[path]


        data = load_pickle_cached(path)

        with self._lock:
            self._file_cache[path] = data

        return data

    def get_style_refs(
        self,
        wid: int,
        style_path: str,
        num_refs: int,
        img_size: Tuple[int, int]
    ) -> torch.Tensor:
        H, W = img_size
        N_total = num_refs * 2


        cache_key_samples = (wid, style_path)


        with self._lock:
            if cache_key_samples in self._valid_samples_cache:
                valid_samples = self._valid_samples_cache[cache_key_samples]
            else:
                valid_samples = None


        if valid_samples is None:
            try:
                style_map = self.load_pickle(style_path)
            except Exception as e:
                print(f"[CRITICAL] Failed to load style pickle for wid={wid}: {e}", flush=True)
                raise RuntimeError(f"Style pickle load failed: {style_path}") from e


            valid_samples = []
            skipped_empty = 0
            for char, samples in style_map.items():
                if char == ' ':
                    continue

                if isinstance(samples, list):
                    for item in samples:
                        if isinstance(item, dict) and item.get('image') is not None:

                            img_arr = np.asarray(item['image'])
                            if img_arr.mean() > 253:
                                skipped_empty += 1
                                continue
                            valid_samples.append((char, item))
                elif isinstance(samples, dict) and samples.get('image') is not None:
                    img_arr = np.asarray(samples['image'])
                    if img_arr.mean() > 253:
                        skipped_empty += 1
                        continue
                    valid_samples.append((char, samples))

            if skipped_empty > 0:
                print(f"[WARN] wid={wid}: Skipped {skipped_empty} empty images (mean>253)", flush=True)

            if len(valid_samples) == 0:
                raise RuntimeError(f"No valid style samples for wid={wid}")


            with self._lock:
                if cache_key_samples not in self._valid_samples_cache:
                    self._valid_samples_cache[cache_key_samples] = valid_samples
                else:

                    valid_samples = self._valid_samples_cache[cache_key_samples]

        if len(valid_samples) < N_total:
            print_once(f"[WARN] wid={wid} has only {len(valid_samples)} samples (need {N_total}), using duplicates", flush=True)


        sampling_pool = [
            {"character": char, "sample_idx": idx}
            for idx, (char, _) in enumerate(valid_samples)
        ]
        sampled_items = sample_style_refs_from_writer(sampling_pool, N_total)


        sampled_indices = [item['sample_idx'] for item in sampled_items]


        tensors = []
        for sample_idx in sampled_indices:
            cache_key_tensor = (wid, sample_idx)


            with self._lock:
                cached = self._tensor_cache.get(cache_key_tensor)
                if cached is not None:

                    self._tensor_cache.move_to_end(cache_key_tensor)
                    tensors.append(cached)
                    continue


            char, item = valid_samples[sample_idx]
            img = item['image']

            if img is None:
                raise RuntimeError(f"Image is None: wid={wid}, char={char}, sample_idx={sample_idx}")

            t = to_tensor_img_gray(np.asarray(img), invert=False)
            if t.dim() == 2:
                t = t.unsqueeze(0).unsqueeze(0)
            elif t.dim() == 3:
                t = t.unsqueeze(0)


            with self._lock:
                cached = self._tensor_cache.get(cache_key_tensor)
                if cached is None:
                    self._tensor_cache[cache_key_tensor] = t
                    self._tensor_cache.move_to_end(cache_key_tensor)
                    if self._max_tensor_cache_items > 0:
                        while len(self._tensor_cache) > self._max_tensor_cache_items:
                            self._tensor_cache.popitem(last=False)
                else:

                    t = cached
                    self._tensor_cache.move_to_end(cache_key_tensor)

            tensors.append(t)

        return torch.cat(tensors, dim=0)

    def preload_files(self, paths: List[str]):
        print(f"[StyleCache] Preloading {len(paths)} pickle files...", flush=True)
        import time
        start = time.perf_counter()

        for path in paths:
            try:
                self.load_pickle(path)
            except Exception as e:
                print(f"[WARN] Preload failed: {path}: {e}", flush=True)

        elapsed = time.perf_counter() - start
        print(f"[StyleCache] Preload complete in {elapsed:.1f}s", flush=True)

    def clear(self):
        with self._lock:
            self._file_cache.clear()
            self._valid_samples_cache.clear()
            self._tensor_cache.clear()

    def get_cache_stats(self):
        with self._lock:
            file_count = len(self._file_cache)
            valid_count = len(self._valid_samples_cache)
            tensor_count = len(self._tensor_cache)

        print(f"[StyleCache Stats]", flush=True)
        print(f"  File cache: {file_count} pickles", flush=True)
        print(f"  Valid samples cache: {valid_count} writers", flush=True)
        print(f"  Tensor cache: {tensor_count} images", flush=True)
        if self._max_tensor_cache_items > 0:
            print(f"  Tensor cache limit: {self._max_tensor_cache_items} (LRU)", flush=True)
        else:
            print(f"  Tensor cache limit: unlimited", flush=True)

        if valid_count > 0:
            with self._lock:
                total_samples = sum(len(v) for v in self._valid_samples_cache.values())
            avg_samples = total_samples / valid_count
            print(f"  Avg samples per writer: {avg_samples:.1f}", flush=True)


        if tensor_count > 0:
            mb_per_tensor = (64 * 64 * 4) / (1024 * 1024)
            total_mb = tensor_count * mb_per_tensor
            print(f"  Estimated tensor memory: {total_mb:.1f} MB", flush=True)


class BaseHWGenDataset(Dataset):


    _global_style_cache = StyleCache()

    def __init__(
        self,
        *,
        num_style_refs: int = 8,
        img_size: Tuple[int, int] = (64, 64),
        seed: int = 1234,
        content_font_img_pkl: Optional[str] = None,
        character_dict_pkl: Optional[str] = None,
        style_pickles_root: Optional[str] = None,
        writer_ids: Optional[List[int]] = None,
        preload: bool = False,
        style_ref_mode: str = "representative",
        skip_font_reference: bool = False,
    ):
        super().__init__()
        self.num_style_refs = int(num_style_refs)
        self.img_size = tuple(img_size)
        self.seed = seed
        self.style_ref_mode = style_ref_mode
        self.skip_font_reference = skip_font_reference
        random.seed(seed)


        self.content_font_img: Dict[str, np.ndarray] = {}
        self.character_dict: Dict[str, Any] = {}


        if content_font_img_pkl and not skip_font_reference:
            self.content_font_img = load_pickle_cached(content_font_img_pkl)
            print(f"[Base] Loaded content_font_img: {len(self.content_font_img)} chars", flush=True)
        elif skip_font_reference:
            print(f"[Base]  skip_font_reference=True: Font image loading SKIPPED (memory saving)", flush=True)


        if character_dict_pkl and not skip_font_reference:
            self.character_dict = load_pickle_cached(character_dict_pkl)
            print(f"[Base] Loaded character_dict: {len(self.character_dict)} entries", flush=True)
        elif skip_font_reference:
            print(f"[Base]  skip_font_reference=True: Character dict loading SKIPPED", flush=True)


        self.style_paths: Dict[int, str] = {}
        if style_pickles_root:
            self._init_style_paths(style_pickles_root, writer_ids)


        if preload and self.style_paths:
            paths = list(self.style_paths.values())
            self._global_style_cache.preload_files(paths)

    def _init_style_paths(self, root: str, writer_ids: Optional[List[int]]):

        all_wids = set()
        for p in sorted(glob.glob(os.path.join(root, "*.pkl"))):
            name = os.path.basename(p).replace('.pkl', '')

            if '_' in name:
                name = name.split('_')[0]
            try:
                wid = int(name)
                all_wids.add(wid)
            except ValueError:
                continue


        if writer_ids is not None:
            all_wids = all_wids.intersection(set(writer_ids))


        rep_count = 0
        base_count = 0
        missing_count = 0

        for wid in sorted(all_wids):
            rep_path = os.path.join(root, f"{wid}_rep.pkl")
            base_path = os.path.join(root, f"{wid}.pkl")

            if self.style_ref_mode == "representative":

                if os.path.exists(rep_path):
                    self.style_paths[wid] = rep_path
                    rep_count += 1
                elif os.path.exists(base_path):

                    print(f"[WARN] Writer {wid}: _rep.pkl not found, falling back to .pkl", flush=True)
                    self.style_paths[wid] = base_path
                    base_count += 1
                else:
                    missing_count += 1
            else:

                if os.path.exists(base_path):
                    self.style_paths[wid] = base_path
                    base_count += 1
                elif os.path.exists(rep_path):

                    print(f"[WARN] Writer {wid}: .pkl not found, falling back to _rep.pkl", flush=True)
                    self.style_paths[wid] = rep_path
                    rep_count += 1
                else:
                    missing_count += 1


        print(f"[Base] Style ref mode: {self.style_ref_mode}", flush=True)
        print(f"[Base] Initialized style_paths: {len(self.style_paths)} writers", flush=True)
        print(f"  - Using _rep.pkl (representative): {rep_count} writers", flush=True)
        print(f"  - Using .pkl (all samples): {base_count} writers", flush=True)
        if missing_count > 0:
            print(f"  - Missing: {missing_count} writers", flush=True)

        if len(self.style_paths) == 0:
            raise RuntimeError(f"No valid style pickles found in {root}")

    def get_style_reference(self, wid: int) -> torch.Tensor:
        if wid not in self.style_paths:
            raise ValueError(f"Writer {wid} not in style_paths")

        style_path = self.style_paths[wid]

        return self._global_style_cache.get_style_refs(
            wid=wid,
            style_path=style_path,
            num_refs=self.num_style_refs,
            img_size=self.img_size
        )

    @staticmethod
    def _chars_to_utf32_tensor(chars_rows: List[List[str]], pad_code: int = 0) -> torch.Tensor:
        B = len(chars_rows)
        S = max((len(r) for r in chars_rows), default=1)
        out = torch.full((B, S), fill_value=pad_code, dtype=torch.int32)

        for b, row in enumerate(chars_rows):
            for s, ch in enumerate(row[:S]):
                if isinstance(ch, str) and len(ch) > 0:
                    code_point = ord(ch[0])

                    if 0 <= code_point <= 0x10FFFF:
                        out[b, s] = code_point
                    else:
                        out[b, s] = pad_code
                else:
                    out[b, s] = pad_code

        return out


class CharGenDataset(BaseHWGenDataset):

    def __init__(
        self,
        content_font_img_pkl: str,
        character_dict_pkl: str,
        style_pickles_root: str,
        rep_pickle_root: str,
        writer_ids: Optional[List[int]] = None,
        num_style_refs: int = 8,
        seed: int = 1234,
        img_size=(64, 64),
        preload_data: bool = False,
        style_ref_mode: str = "representative",
        skip_font_reference: bool = False,
        n_gram_window: int = 2,
    ):
        super().__init__(
            num_style_refs=num_style_refs,
            img_size=img_size,
            seed=seed,
            content_font_img_pkl=content_font_img_pkl,
            character_dict_pkl=character_dict_pkl,
            style_pickles_root=style_pickles_root,
            writer_ids=writer_ids,
            preload=preload_data,
            style_ref_mode=style_ref_mode,
            skip_font_reference=skip_font_reference,
        )

        self.n_gram_window = n_gram_window
        self.rep_root = rep_pickle_root


        self.rep_paths: Dict[int, str] = {}
        for wid in self.style_paths.keys():
            rep_path = os.path.join(self.rep_root, f"{wid}_rep.pkl")
            if os.path.exists(rep_path):
                self.rep_paths[wid] = rep_path

        print(f"[CharDataset] Found {len(self.rep_paths)} rep pickles", flush=True)


        if preload_data and self.rep_paths:
            rep_paths = list(self.rep_paths.values())
            self._global_style_cache.preload_files(rep_paths)


        self.index: List[Tuple[int, str]] = []

        for wid, style_path in self.style_paths.items():
            try:
                style_map = self._global_style_cache.load_pickle(style_path)
            except Exception as e:
                print(f"[ERROR] Failed to load style for wid={wid}: {e}", flush=True)
                continue

            for char in style_map.keys():
                if not isinstance(char, str):
                    char = str(char)

                if char == "" or (char != " " and len(char) != 1):
                    continue

                if self.skip_font_reference:
                    if char != ' ':
                        self.index.append((wid, char))
                else:
                    if char in self.content_font_img and char != ' ':
                        self.index.append((wid, char))

        print(f"[CharDataset] Index size: {len(self.index)}", flush=True)

        if len(self.index) == 0:
            raise RuntimeError("CharGenDataset index is empty")

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        wid, char = self.index[idx]
        return {
            "writer_id": wid,
            "char": char,
        }

    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        H, W = self.img_size

        style_refs = []
        style_labels = []
        font_imgs = []
        target_trajs = []
        target_pens = []
        font_chars = []

        for item in batch:
            wid = item["writer_id"]
            char = item["char"]


            try:
                style_ref = self.get_style_reference(wid)
                style_refs.append(style_ref)
                style_labels.append(wid)
            except Exception as e:
                print(f"[CRITICAL] Style ref failed: wid={wid}, {e}", flush=True)
                raise


            if self.skip_font_reference:
                ft = torch.zeros((1, H, W), dtype=torch.float32)
            else:
                font_np = self.content_font_img.get(char)
                if font_np is None:
                    raise RuntimeError(f"Font not found: char={char}")
                ft = to_tensor_img_gray(np.asarray(font_np), invert=False)
                if ft.dim() == 2:
                    ft = ft.unsqueeze(0)
            font_imgs.append(ft)
            font_chars.append(char)


            rep_path = self.rep_paths.get(wid)
            traj_abs = None

            if rep_path:
                try:
                    rep_map = self._global_style_cache.load_pickle(rep_path)
                    char_samples = rep_map.get(char, [])


                    if isinstance(char_samples, list):
                        valid = [s for s in char_samples
                                if isinstance(s, dict) and s.get('trajectory') is not None]
                        if valid:
                            traj_abs = random.choice(valid)['trajectory']
                    elif isinstance(char_samples, dict):
                        traj_abs = char_samples.get('trajectory')
                except Exception as e:
                    print(f"[WARN] Rep load failed: wid={wid}, char={char}: {e}", flush=True)

            if traj_abs is None:

                try:
                    style_map = self._global_style_cache.load_pickle(self.style_paths[wid])
                    char_samples = style_map.get(char, [])

                    if isinstance(char_samples, list):
                        valid = [s for s in char_samples
                                if isinstance(s, dict) and s.get('trajectory') is not None]
                        if valid:
                            traj_abs = random.choice(valid)['trajectory']
                    elif isinstance(char_samples, dict):
                        traj_abs = char_samples.get('trajectory')
                except Exception as e:
                    print(f"[ERROR] No trajectory: wid={wid}, char={char}: {e}", flush=True)

            if traj_abs is None or len(traj_abs) == 0:
                raise RuntimeError(f"[CRITICAL] No trajectory found: wid={wid}, char={char}")


            _, traj_delta, _ = to_local_bigram(
                prev_abs=None,
                curr_abs=traj_abs,
                W_ref=1.0,
                H_ref=1.0
            )


            target_trajs.append(torch.from_numpy(traj_delta).float())
            target_pens.append(torch.from_numpy(traj_delta[:, 2:6]).float())


        style_reference = torch.stack(style_refs, dim=0)
        style_labels_t = torch.tensor(style_labels, dtype=torch.long)
        font_reference_1 = torch.stack(font_imgs, dim=0)
        font_reference = font_reference_1.unsqueeze(1)

        traj_pad = pad_1d(target_trajs, pad_value=0.0)
        pen_pad = pad_1d(target_pens, pad_value=0.0)
        target_len = torch.tensor([t.size(0) for t in target_trajs], dtype=torch.long)

        seq_chars = self._chars_to_utf32_tensor([[ch] for ch in font_chars])
        seq_loss_mask = torch.ones(seq_chars.size(0), seq_chars.size(1), dtype=torch.bool)

        return {
            "style_reference": style_reference,
            "style_labels": style_labels_t,
            "font_reference": font_reference,
            "target_traj": traj_pad,
            "target_len": target_len,
            "target_pen": pen_pad,
            "seq_chars": seq_chars,
            "seq_loss_mask": seq_loss_mask,
            "seq_chars_text": [[ch] for ch in font_chars],
        }


class BigramCharGenDataset(BaseHWGenDataset):

    def __init__(
        self,
        sent_pickles_root: str,
        style_pickles_root: str,
        content_font_img_pkl: str,
        character_dict_pkl: Optional[str] = None,
        writer_ids: Optional[List[int]] = None,
        num_style_refs: int = 8,
        seed: int = 1234,
        img_size: Tuple[int, int] = (64, 64),
        min_prev_T: int = 1,
        min_curr_T: int = 1,
        preload_data: bool = False,
        style_ref_mode: str = "representative",
        skip_font_reference: bool = False,
        n_gram_window: int = 2,
    ):
        super().__init__(
            num_style_refs=num_style_refs,
            img_size=img_size,
            seed=seed,
            content_font_img_pkl=content_font_img_pkl,
            character_dict_pkl=character_dict_pkl,
            style_pickles_root=style_pickles_root,
            writer_ids=writer_ids,
            preload=preload_data,
            style_ref_mode=style_ref_mode,
            skip_font_reference=skip_font_reference,
        )

        self.sent_root = sent_pickles_root
        self.min_prev_T = min_prev_T
        self.min_curr_T = min_curr_T
        self.rng = random.Random(seed)
        self.n_gram_window = n_gram_window


        self.sent_paths: Dict[int, str] = {}
        pattern = os.path.join(sent_pickles_root, "*_sent.pkl")

        for path in sorted(glob.glob(pattern)):
            basename = os.path.basename(path)
            try:
                wid = int(basename.split('_')[0])
            except ValueError:
                continue


            if writer_ids is not None and wid not in writer_ids:
                continue


            if wid not in self.style_paths:
                continue

            self.sent_paths[wid] = path

        print(f"[BigramDataset] Found {len(self.sent_paths)} sent pickles", flush=True)


        if preload_data:
            paths = list(self.sent_paths.values())
            self._global_style_cache.preload_files(paths)


        self.index: List[Tuple[int, str, str]] = []

        for wid, sent_path in self.sent_paths.items():
            try:
                data = self._global_style_cache.load_pickle(sent_path)
                sentences = data.get("sentences", {})

                for sid, items in sentences.items():
                    if len(items) >= 2:
                        self.index.append((wid, str(sid), sent_path))
            except Exception as e:
                print(f"[ERROR] Failed to load sent for wid={wid}: {e}", flush=True)
                continue

        print(f"[BigramDataset] Index size: {len(self.index)}", flush=True)

        if len(self.index) == 0:
            raise RuntimeError("BigramCharGenDataset index is empty")

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Tuple[int, str, str]:
        return self.index[idx]

    def collate_fn(self, batch: List[Tuple[int, str, str]]) -> Dict[str, Any]:
        H, W = self.img_size

        style_refs = []
        style_labels = []
        pair_prev_trajs = []
        pair_curr_trajs = []
        pair_prev_fonts = []
        pair_curr_fonts = []
        seq_chars_list = []
        curr_indices = []
        after_space_flags = []
        _dummy_indices = []

        for wid, sid, sent_path in batch:

            try:
                style_ref = self.get_style_reference(wid)
                style_refs.append(style_ref)
                style_labels.append(wid)
            except Exception as e:
                print(f"[CRITICAL] Style ref failed: wid={wid}, {e}", flush=True)
                raise


            try:
                data = self._global_style_cache.load_pickle(sent_path)
                sent_items = data.get("sentences", {}).get(sid, [])
                sent_items = sorted(sent_items, key=lambda x: int(x.get("cursor", 0)))
            except Exception as e:
                print(f"[ERROR] Sent load failed: wid={wid}, sid={sid}: {e}", flush=True)
                raise

            chars = [it.get("character", "") for it in sent_items]
            seq_chars_list.append(chars)


            valid_k = []
            for k in range(1, len(sent_items)):
                prev_it, curr_it = sent_items[k-1], sent_items[k]
                prev_ch = prev_it.get("character", ' ')
                curr_ch = curr_it.get("character", ' ')


                if prev_ch == ' ' or curr_ch == ' ' or prev_ch == '':
                    continue


                prev_traj = prev_it.get("trajectory")
                if prev_traj is None or len(prev_traj) < self.min_prev_T:
                    continue


                curr_traj = curr_it.get("trajectory")
                if curr_traj is None or len(curr_traj) < self.min_curr_T:
                    continue

                valid_k.append(k)

            if not valid_k:


                dummy_traj = torch.zeros(1, 6, dtype=torch.float32)
                pair_prev_trajs.append(dummy_traj)
                pair_curr_trajs.append(dummy_traj)
                pair_prev_fonts.append(torch.zeros((1, H, W), dtype=torch.float32))
                pair_curr_fonts.append(torch.zeros((1, H, W), dtype=torch.float32))
                curr_indices.append(0)
                after_space_flags.append(False)
                _dummy_indices.append(len(pair_prev_trajs) - 1)
                continue


            k = self.rng.choice(valid_k)
            prev_it = sent_items[k-1]
            curr_it = sent_items[k]
            prev_ch = prev_it["character"]
            curr_ch = curr_it["character"]


            prev_abs = prev_it.get("trajectory")
            curr_abs = curr_it.get("trajectory")


            prev_d, curr_d, _ = to_local_bigram(
                prev_abs, curr_abs, W_ref=1.0, H_ref=1.0, is_after_space=False
            )

            pair_prev_trajs.append(torch.from_numpy(prev_d).float() if isinstance(prev_d, np.ndarray) else prev_d.float())
            pair_curr_trajs.append(torch.from_numpy(curr_d).float() if isinstance(curr_d, np.ndarray) else curr_d.float())
            curr_indices.append(k)
            after_space_flags.append(False)


            def _font(ch: str):
                np_img = self.content_font_img.get(ch)
                if np_img is None:
                    return torch.zeros((1, H, W), dtype=torch.float32)
                t = to_tensor_img_gray(np.asarray(np_img), invert=False)
                if t.dim() == 2:
                    t = t.unsqueeze(0)
                return t

            pair_prev_fonts.append(_font(prev_ch))
            pair_curr_fonts.append(_font(curr_ch))


        if len(style_refs) == 0:
            print(f"[ERROR] All samples in batch were invalid, returning dummy batch", flush=True)

            B = 1
            return {
                "style_reference": torch.zeros(B, self.num_style_refs * 2, 1, H, W),
                "style_labels": torch.zeros(B, dtype=torch.long),
                "font_reference": torch.zeros(B, 2, 1, H, W),
                "seq_trajs": torch.zeros(B, 2, 1, 6),
                "seq_traj_lens": torch.zeros(B, 2, dtype=torch.long),
                "seq_loss_mask": torch.zeros(B, 2, dtype=torch.bool),
                "seq_lengths": torch.zeros(B, dtype=torch.long),
                "seq_chars": torch.zeros(B, 1, dtype=torch.long),
                "curr_indices": torch.zeros(B, dtype=torch.long),
                "is_after_space": torch.zeros(B, dtype=torch.bool),
            }


        B = len(style_refs)
        style_reference = torch.stack(style_refs, dim=0)
        style_labels_t = torch.tensor(style_labels, dtype=torch.long)

        T_max = max([1] + [int(t.size(0)) for t in pair_prev_trajs + pair_curr_trajs])

        seq_trajs = torch.zeros(B, 2, T_max, 6, dtype=torch.float32)
        seq_traj_lens = torch.zeros(B, 2, dtype=torch.long)
        font_reference = torch.zeros(B, 2, 1, H, W, dtype=torch.float32)

        for i in range(B):
            pt = pair_prev_trajs[i]
            if pt.numel() > 0:
                L = min(T_max, pt.size(0))
                seq_trajs[i, 0, :L, :] = pt[:L]
                seq_traj_lens[i, 0] = L
            font_reference[i, 0] = pair_prev_fonts[i]

            ct = pair_curr_trajs[i]
            if ct.numel() > 0:
                L = min(T_max, ct.size(0))
                seq_trajs[i, 1, :L, :] = ct[:L]
                seq_traj_lens[i, 1] = L
            font_reference[i, 1] = pair_curr_fonts[i]

        seq_loss_mask = torch.zeros(B, 2, dtype=torch.bool)
        seq_loss_mask[:, 1] = True

        for di in _dummy_indices:
            seq_loss_mask[di, :] = False

        seq_chars = self._chars_to_utf32_tensor(seq_chars_list)
        curr_indices_t = torch.tensor(curr_indices, dtype=torch.long)
        is_after_space = torch.tensor(after_space_flags, dtype=torch.bool)
        seq_lengths = torch.full((B,), 2, dtype=torch.long)

        return {
            "style_reference": style_reference,
            "style_labels": style_labels_t,
            "font_reference": font_reference,
            "seq_trajs": seq_trajs,
            "seq_traj_lens": seq_traj_lens,
            "seq_loss_mask": seq_loss_mask,
            "seq_lengths": seq_lengths,
            "seq_chars": seq_chars,
            "curr_indices": curr_indices_t,
            "is_after_space": is_after_space,
        }


class SentenceGenDataset(BaseHWGenDataset):

    def __init__(
        self,
        sent_pickles_root: str,
        style_pickles_root: str,
        content_font_img_pkl: str,
        character_dict_pkl: str,
        writer_ids: Optional[List[int]] = None,
        num_style_refs: int = 8,
        seed: int = 1234,
        img_size=(64, 64),
        use_rep_pickle_root: Optional[str] = None,
        preload_data: bool = False,
        style_ref_mode: str = "representative",
        skip_font_reference: bool = False,
        n_gram_window: int = 2,
    ):
        super().__init__(
            num_style_refs=num_style_refs,
            img_size=img_size,
            seed=seed,
            content_font_img_pkl=content_font_img_pkl,
            character_dict_pkl=character_dict_pkl,
            style_pickles_root=style_pickles_root,
            writer_ids=writer_ids,
            preload=preload_data,
            style_ref_mode=style_ref_mode,
            skip_font_reference=skip_font_reference,
        )

        self.sent_root = sent_pickles_root
        self.rep_root = use_rep_pickle_root
        self.n_gram_window = n_gram_window


        self.sent_paths: Dict[int, str] = {}

        if writer_ids is not None:
            for wid in writer_ids:
                sent_path = os.path.join(self.sent_root, f"{wid}_sent.pkl")
                if os.path.exists(sent_path) and wid in self.style_paths:
                    self.sent_paths[wid] = sent_path
        else:
            pattern = os.path.join(self.sent_root, "*_sent.pkl")
            for path in sorted(glob.glob(pattern)):
                try:
                    wid = int(os.path.basename(path).split("_")[0])
                    if wid in self.style_paths:
                        self.sent_paths[wid] = path
                except ValueError:
                    continue

        print(f"[SentenceDataset] Found {len(self.sent_paths)} sent pickles", flush=True)


        self.rep_paths: Dict[int, str] = {}
        if self.rep_root:
            for wid in self.sent_paths.keys():
                rep_path = os.path.join(self.rep_root, f"{wid}_rep.pkl")
                if os.path.exists(rep_path):
                    self.rep_paths[wid] = rep_path

        print(f"[SentenceDataset] Found {len(self.rep_paths)} rep pickles", flush=True)


        if preload_data:
            paths = list(self.sent_paths.values())
            self._global_style_cache.preload_files(paths)


        self.index: List[Tuple[int, Any]] = []

        for wid, sent_path in self.sent_paths.items():
            try:
                data = self._global_style_cache.load_pickle(sent_path)


                if "sentences" in data and isinstance(data["sentences"], dict):
                    for sid in data["sentences"].keys():
                        self.index.append((wid, sid))
                else:
                    print(f"[WARN] Invalid sent structure: wid={wid}", flush=True)
            except Exception as e:
                print(f"[ERROR] Failed to load sent for wid={wid}: {e}", flush=True)
                continue

        print(f"[SentenceDataset] Index size: {len(self.index)}", flush=True)

        if len(self.index) == 0:
            raise RuntimeError("SentenceGenDataset index is empty")

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        wid, sid = self.index[idx]
        return {
            "writer_id": wid,
            "sentence_id": sid,
        }

    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        H, W = self.img_size

        style_refs = []
        style_labels = []
        sent_chars_all = []
        sent_prev_all = []
        sent_curr_all = []
        sent_fonts_all = []

        for item in batch:
            wid = item["writer_id"]
            sid = item["sentence_id"]


            try:
                style_ref = self.get_style_reference(wid)
                style_refs.append(style_ref)
                style_labels.append(wid)
            except Exception as e:
                print(f"[CRITICAL] Style ref failed: wid={wid}, {e}", flush=True)
                raise


            sent_path = self.sent_paths[wid]

            try:
                data = self._global_style_cache.load_pickle(sent_path)
                sent_items = data.get("sentences", {}).get(sid, [])
                sent_items = sorted(sent_items, key=lambda x: int(x.get("cursor", 0)))
            except Exception as e:
                print(f"[ERROR] Sent load failed: wid={wid}, sid={sid}: {e}", flush=True)
                raise

            chars = [it.get("character", "") for it in sent_items]
            trajs_abs = [it.get("trajectory") for it in sent_items]

            sent_chars_all.append(chars)


            per_prev = []
            per_curr = []

            for s_idx, ch in enumerate(chars):
                curr_abs = trajs_abs[s_idx]


                if ch == " ":

                    prev_end = None
                    if s_idx > 0:
                        prev_traj = trajs_abs[s_idx - 1]
                        if prev_traj is not None and len(prev_traj) > 0:
                            prev_end = prev_traj[-1, :2]


                    next_start = None
                    next_char = None
                    for k in range(s_idx + 1, len(chars)):
                        if chars[k] != " ":
                            next_traj = trajs_abs[k]
                            next_char = chars[k]
                            if next_traj is not None and len(next_traj) > 0:
                                next_start = next_traj[0, :2]
                            break


                    if prev_end is not None and next_start is not None:

                        space_traj = np.array([
                            [prev_end[0], prev_end[1], 1, 0, 0, 0],
                            [next_start[0], prev_end[1], 0, 0, 0, 1]
                        ], dtype=np.float32)

                        curr_abs = space_traj
                        trajs_abs[s_idx] = space_traj
                    else:

                        default_width = 0.3
                        if prev_end is not None:

                            space_traj = np.array([
                                [prev_end[0], prev_end[1], 1, 0, 0, 0],
                                [prev_end[0] + default_width, prev_end[1], 0, 0, 0, 1]
                            ], dtype=np.float32)
                        else:

                            space_traj = np.array([
                                [0.0, 0.5, 1, 0, 0, 0],
                                [default_width, 0.5, 0, 0, 0, 1]
                            ], dtype=np.float32)
                        curr_abs = space_traj
                        trajs_abs[s_idx] = space_traj


                if curr_abs is None or len(curr_abs) == 0:
                    per_curr.append(torch.zeros(0, 6, dtype=torch.float32))
                    per_prev.append(torch.zeros(0, 6, dtype=torch.float32))
                    continue


                prev_abs = None
                if s_idx > 0:
                    prev_abs = trajs_abs[s_idx - 1]


                prev_d, curr_d, _ = to_local_bigram(
                    prev_abs, curr_abs, W_ref=1.0, H_ref=1.0, is_after_space=False
                )

                per_prev.append(torch.from_numpy(prev_d).float() if isinstance(prev_d, np.ndarray) else prev_d.float())
                per_curr.append(torch.from_numpy(curr_d).float() if isinstance(curr_d, np.ndarray) else curr_d.float())

            sent_prev_all.append(per_prev)
            sent_curr_all.append(per_curr)


            per_char_fonts = []
            for ch in chars:
                np_img = self.content_font_img.get(ch)
                if np_img is None:
                    ft = torch.zeros((1, H, W), dtype=torch.float32)
                else:
                    ft = to_tensor_img_gray(np.asarray(np_img), invert=False)
                if ft.dim() == 2:
                    ft = ft.unsqueeze(0)
                per_char_fonts.append(ft)

            sent_fonts_all.append(per_char_fonts)


        B = len(batch)
        style_reference = torch.stack(style_refs, dim=0)
        style_labels_t = torch.tensor(style_labels, dtype=torch.long)


        MAX_SENTENCE_LENGTH = 35

        truncated_sent_chars = []
        truncated_sent_prev = []
        truncated_sent_curr = []
        truncated_sent_fonts = []

        for idx, chars in enumerate(sent_chars_all):
            if len(chars) > MAX_SENTENCE_LENGTH:

                truncated_chars = []
                truncated_prev = []
                truncated_curr = []
                truncated_fonts = []

                current_len = 0
                for i, ch in enumerate(chars):
                    if current_len >= MAX_SENTENCE_LENGTH:


                        last_space_idx = -1
                        for j in range(len(truncated_chars) - 1, -1, -1):
                            if truncated_chars[j] == ' ':
                                last_space_idx = j
                                break

                        if last_space_idx > 0:

                            truncated_chars = truncated_chars[:last_space_idx]
                            truncated_prev = truncated_prev[:last_space_idx]
                            truncated_curr = truncated_curr[:last_space_idx]
                            truncated_fonts = truncated_fonts[:last_space_idx]

                        print(f"[WARN] Sentence {idx} truncated from {len(chars)} to {len(truncated_chars)} chars (word boundary)", flush=True)
                        break

                    truncated_chars.append(ch)
                    truncated_prev.append(sent_prev_all[idx][i])
                    truncated_curr.append(sent_curr_all[idx][i])
                    truncated_fonts.append(sent_fonts_all[idx][i])
                    current_len += 1

                truncated_sent_chars.append(truncated_chars)
                truncated_sent_prev.append(truncated_prev)
                truncated_sent_curr.append(truncated_curr)
                truncated_sent_fonts.append(truncated_fonts)
            else:

                truncated_sent_chars.append(chars)
                truncated_sent_prev.append(sent_prev_all[idx])
                truncated_sent_curr.append(sent_curr_all[idx])
                truncated_sent_fonts.append(sent_fonts_all[idx])

        sent_chars_all = truncated_sent_chars
        sent_prev_all = truncated_sent_prev
        sent_curr_all = truncated_sent_curr
        sent_fonts_all = truncated_sent_fonts

        S_list = [len(c) for c in sent_chars_all]
        S_max = max(S_list) if S_list else 0

        T_max = 1
        for pv_list, cu_list in zip(sent_prev_all, sent_curr_all):
            for t in pv_list + cu_list:
                if t is not None and t.numel() > 0:
                    T_max = max(T_max, t.shape[0])


        font_reference = torch.zeros((B, S_max, 1, H, W), dtype=torch.float32)
        sentence_trajs = torch.zeros((B, S_max, T_max, 6), dtype=torch.float32)
        sentence_prev_trajs = torch.zeros((B, S_max, T_max, 6), dtype=torch.float32)
        sentence_traj_lens = torch.zeros((B, S_max), dtype=torch.long)
        sentence_prev_lens = torch.zeros((B, S_max), dtype=torch.long)
        sentence_valid_mask = torch.zeros((B, S_max), dtype=torch.bool)
        sentence_after_space = torch.zeros((B, S_max), dtype=torch.bool)
        sentence_loss_mask = torch.zeros((B, S_max), dtype=torch.bool)


        for b_idx in range(B):
            chars = sent_chars_all[b_idx]
            S = len(chars)
            sentence_valid_mask[b_idx, :S] = True

            is_space_b = torch.zeros(S, dtype=torch.bool)
            is_oov_b = torch.zeros(S, dtype=torch.bool)
            lens_pos_b = torch.zeros(S, dtype=torch.bool)

            for s_idx, ch in enumerate(chars):
                is_space_b[s_idx] = (ch == " ")


                if self.skip_font_reference:
                    is_oov_b[s_idx] = False
                else:
                    is_oov_b[s_idx] = (ch not in self.content_font_img) and (ch != " ")

                if s_idx > 0 and chars[s_idx-1] == " ":
                    sentence_after_space[b_idx, s_idx] = True

            pv_list = sent_prev_all[b_idx]
            cu_list = sent_curr_all[b_idx]

            for s_idx in range(S):
                pt = pv_list[s_idx]
                ct = cu_list[s_idx]

                if pt is not None and pt.numel() > 0:
                    Lp = min(T_max, pt.shape[0])
                    sentence_prev_trajs[b_idx, s_idx, :Lp, :] = pt[:Lp]
                    sentence_prev_lens[b_idx, s_idx] = Lp

                if ct is not None and ct.numel() > 0:
                    Lc = min(T_max, ct.shape[0])
                    sentence_trajs[b_idx, s_idx, :Lc, :] = ct[:Lc]
                    sentence_traj_lens[b_idx, s_idx] = Lc
                    lens_pos_b[s_idx] = True


            sentence_loss_mask[b_idx, :S] = (~is_oov_b) & lens_pos_b

            fonts = sent_fonts_all[b_idx]
            for s_idx, ft in enumerate(fonts[:S]):
                if ft is not None:
                    font_reference[b_idx, s_idx] = ft

        sentence_lengths = torch.tensor(S_list, dtype=torch.long)
        seq_chars = self._chars_to_utf32_tensor(sent_chars_all)
        seq_loss_mask = sentence_loss_mask.clone()

        return {
            "style_reference": style_reference,
            "style_labels": style_labels_t,
            "seq_chars_text": sent_chars_all,
            "seq_chars": seq_chars,
            "seq_lengths": sentence_lengths,
            "seq_loss_mask": seq_loss_mask,
            "font_reference": font_reference,
            "sentence_prev_trajs": sentence_prev_trajs,
            "sentence_prev_mask": (sentence_prev_lens > 0),

            "seq_trajs": sentence_trajs,
            "seq_traj_lens": sentence_traj_lens,
        }
