import os
import time
import random
import gc
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor
import numpy as np, torch
import time

import torch
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter


torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
try:
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)
except Exception:
    pass


from src.utils.logger import print_once, print_trace
from src.config.constants import PEN_STATE_RANGE, TRAJ_INDEX, TRAJ_INDEX_EXPANDED, PEN_CLASS


from src.utils.train_util import (
    visualize_snapshot_chars,
    visualize_snapshot_sentence,
    get_mixture_coef,
    get_seq_from_gmm,
    convert_unigram_gmm_to_bigram,
    save_checkpoint,
    load_checkpoint,
    load_latest_checkpoint,
    extract_model_config,
)

from src.utils.tb_util import (
    visualize_style_images_tb,
)


from src.loss.pen_loss import get_pen_loss
from src.loss.contrastive_loss import WriterGlyphNCELoss
from src.loss.vq_loss import RVQLoss


from src.data.handwriting_generator_dataset import (
    CharGenDataset,
    BigramCharGenDataset,
    SentenceGenDataset,
)
from src.data.data_utils import (
    get_writer_ids_from_dir,
    get_writer_split_policy_from_cfg,
    get_sample_split_test_list_from_cfg,
    get_sample_split_train_filter_options_from_cfg,
    load_sample_split_pair_quota,
    normalize_text_for_sample_split,
)


from src.model.style_identifier import StyleIdentifier
from src.model.font_encoder import FontEncoder
from src.model.context_encoder import ContextEncoder
from src.model.handwriting_generator import HandwritingGenerator
from src.model.full_model import FullModel
from src.model.residual_vq import FontVQBranch, ResidualVQBranch
from src.utils.embedding_monitor import visualize_content_vs_context, visualize_content_vs_context_v2, visualize_content_vs_context_v3


def _normalize_cfg_path(path_val: Optional[Any]) -> Optional[str]:
    if path_val is None:
        return None
    if isinstance(path_val, str):
        s = path_val.strip()
        if s == "" or s.lower() in {"none", "null", "~"}:
            return None
        return s
    return str(path_val)


def _sentence_text_key_from_items(sent_items: List[Dict[str, Any]], max_sentence_len: int) -> str:
    text = "".join(str(it.get("character", "")) for it in sent_items)
    return normalize_text_for_sample_split(text, max_sentence_len=max_sentence_len)


def _build_excluded_sentence_ids_from_quota(
    sent_dataset: SentenceGenDataset,
    pair_quota: Dict[Tuple[int, str], int],
    max_sentence_len: int,
) -> Tuple[set, Dict[Tuple[int, str], int]]:
    remaining_quota = dict(pair_quota)
    excluded_sentence_ids = set()

    for wid, sid in sent_dataset.index:
        sent_path = sent_dataset.sent_paths.get(wid)
        if sent_path is None:
            continue
        try:
            data = sent_dataset._global_style_cache.load_pickle(sent_path)
            sent_items = data.get("sentences", {}).get(sid, [])
        except Exception:
            continue

        key = (int(wid), _sentence_text_key_from_items(sent_items, max_sentence_len))
        if remaining_quota.get(key, 0) > 0:
            remaining_quota[key] -= 1
            excluded_sentence_ids.add((int(wid), str(sid)))

    return excluded_sentence_ids, remaining_quota


def _filter_sentence_dataset_index(
    dataset: SentenceGenDataset,
    excluded_sentence_ids: set,
) -> int:
    kept = []
    removed = 0
    for wid, sid in dataset.index:
        if (int(wid), str(sid)) in excluded_sentence_ids:
            removed += 1
        else:
            kept.append((wid, sid))
    dataset.index = kept
    return removed


def _filter_bigram_dataset_index(
    dataset: BigramCharGenDataset,
    excluded_sentence_ids: set,
) -> int:
    kept = []
    removed = 0
    for wid, sid, sent_path in dataset.index:
        if (int(wid), str(sid)) in excluded_sentence_ids:
            removed += 1
        else:
            kept.append((wid, sid, sent_path))
    dataset.index = kept
    return removed


class HandwritingGenerationTrainer:
    def __init__(self, cfg, is_ddp=False, rank=0, world_size=1, local_rank=0):
        self.cfg = cfg
        self.is_ddp = is_ddp
        self.rank = rank
        self.world_size = world_size
        self.local_rank = local_rank


        self.device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")


        self._warmup_viz = self.cfg.TRAIN.get("FORCE_WARMUP_VIZ_ON_RESUME", True)


        split_mode, split_cfg = get_writer_split_policy_from_cfg(cfg.ENV)
        print(f"[Trainer] WRITER_SPLIT mode: {split_mode}", flush=True)
        if split_cfg:
            print(f"[Trainer] WRITER_SPLIT cfg: {split_cfg}", flush=True)

        self.train_writer_ids, self.test_writer_ids = get_writer_ids_from_dir(
            cfg.ENV.STYLE_DATASET_PATH,
            split_mode=split_mode,
            split_config=split_cfg,
        )


        style_ref_mode = getattr(cfg.ENV, 'STYLE_REF_MODE', 'representative')
        print(f"[Trainer] STYLE_REF_MODE: {style_ref_mode}", flush=True)


        self.stage1_only = getattr(cfg.TRAIN, 'STAGE1_ONLY', False)
        print(f"[Trainer] STAGE1_ONLY: {self.stage1_only}", flush=True)


        self.pen_state_class_weight = getattr(cfg.TRAIN, 'PEN_STATE_CLASS_WEIGHT', None)
        if self.pen_state_class_weight is not None:
            print(f"[Trainer] PEN_STATE_CLASS_WEIGHT: {self.pen_state_class_weight}", flush=True)
        else:
            print(f"[Trainer] PEN_STATE_CLASS_WEIGHT: None (균등 가중치)", flush=True)


        skip_font_ref = getattr(cfg.MODEL, 'USE_CONTEXT_AS_CONTENT', False)
        if skip_font_ref and self.rank == 0:
            print(f"[Trainer]  USE_CONTEXT_AS_CONTENT=True: Font reference loading SKIPPED", flush=True)


        self.n_gram_window = getattr(cfg.MODEL, 'N_GRAM_AWARE_SLIDING_WINDOW', 2)
        print(f"[Trainer] N_GRAM_AWARE_SLIDING_WINDOW: {self.n_gram_window}", flush=True)


        self.char_train_ds = CharGenDataset(
            style_pickles_root=cfg.ENV.STYLE_DATASET_PATH,
            rep_pickle_root=cfg.ENV.CHAR_DATASET_PATH,
            content_font_img_pkl=cfg.ENV.FONT_DATASET_IMG_PATH,
            character_dict_pkl=cfg.ENV.FONT_DATASET_DICT_PATH,
            writer_ids=self.train_writer_ids,
            num_style_refs=cfg.TRAIN.NUM_REFERENCE_IMGS,
            preload_data=getattr(cfg.ENV, 'PRELOAD_DATA', False),
            style_ref_mode=style_ref_mode,
            skip_font_reference=skip_font_ref,
            n_gram_window=self.n_gram_window,
        )


        tensor_cache_limit = int(getattr(cfg.ENV, "STYLE_TENSOR_CACHE_MAX_ITEMS", -1))
        self.char_train_ds._global_style_cache.set_tensor_cache_limit(tensor_cache_limit)
        if tensor_cache_limit > 0:
            print(f"[Trainer] STYLE_TENSOR_CACHE_MAX_ITEMS={tensor_cache_limit} (LRU)", flush=True)
        else:
            print(f"[Trainer] STYLE_TENSOR_CACHE_MAX_ITEMS=unlimited", flush=True)


        if not self.stage1_only:

            self.bigram_train_ds = BigramCharGenDataset(
                sent_pickles_root=cfg.ENV.SENT_DATASET_PATH,
                style_pickles_root=cfg.ENV.STYLE_DATASET_PATH,
                content_font_img_pkl=cfg.ENV.FONT_DATASET_IMG_PATH,
                character_dict_pkl=cfg.ENV.FONT_DATASET_DICT_PATH,
                writer_ids=self.train_writer_ids,
                num_style_refs=cfg.TRAIN.NUM_REFERENCE_IMGS,
                img_size=(cfg.ENV.IMG_H, cfg.ENV.IMG_W),
                preload_data=getattr(cfg.ENV, 'PRELOAD_DATA', False),
                style_ref_mode=style_ref_mode,
                skip_font_reference=skip_font_ref,
                n_gram_window=self.n_gram_window,
            )


            self.sent_train_ds = SentenceGenDataset(
                sent_pickles_root=cfg.ENV.SENT_DATASET_PATH,
                style_pickles_root = cfg.ENV.STYLE_DATASET_PATH,
                content_font_img_pkl = cfg.ENV.FONT_DATASET_IMG_PATH,
                character_dict_pkl   = cfg.ENV.FONT_DATASET_DICT_PATH,
                writer_ids = self.train_writer_ids,
                num_style_refs = cfg.TRAIN.NUM_REFERENCE_IMGS,
                use_rep_pickle_root = cfg.ENV.CHAR_DATASET_PATH,
                img_size=(cfg.ENV.IMG_H, cfg.ENV.IMG_W),
                preload_data=getattr(cfg.ENV, 'PRELOAD_DATA', False),
                style_ref_mode=style_ref_mode,
                skip_font_reference=skip_font_ref,
                n_gram_window=self.n_gram_window,
            )
            self.sent_val_ds = SentenceGenDataset(
                sent_pickles_root=cfg.ENV.SENT_DATASET_PATH,
                style_pickles_root = cfg.ENV.STYLE_DATASET_PATH,
                content_font_img_pkl = cfg.ENV.FONT_DATASET_IMG_PATH,
                character_dict_pkl   = cfg.ENV.FONT_DATASET_DICT_PATH,
                writer_ids = self.test_writer_ids,
                num_style_refs = cfg.VALID.NUM_REFERENCE_IMGS,
                use_rep_pickle_root = cfg.ENV.CHAR_DATASET_PATH,
                img_size=(cfg.ENV.IMG_H, cfg.ENV.IMG_W),
                preload_data=getattr(cfg.ENV, 'PRELOAD_DATA', False),
                style_ref_mode=style_ref_mode,
                skip_font_reference=skip_font_ref,
                n_gram_window=self.n_gram_window,
            )


            apply_to_train, sample_split_max_len = get_sample_split_train_filter_options_from_cfg(cfg.ENV)
            if apply_to_train:
                sample_split_test_list = get_sample_split_test_list_from_cfg(cfg.ENV)
                if sample_split_test_list is None:
                    print(
                        "[Trainer][WARN] SAMPLE_SPLIT.APPLY_TO_TRAIN=true but SAMPLE_SPLIT is disabled. "
                        "Skip sample-level train filtering.",
                        flush=True,
                    )
                else:
                    if not os.path.exists(sample_split_test_list):
                        raise FileNotFoundError(
                            f"SAMPLE_SPLIT.TEST_LIST not found: {sample_split_test_list}"
                        )

                    pair_quota = load_sample_split_pair_quota(
                        sample_split_test_list,
                        max_sentence_len=sample_split_max_len,
                    )
                    total_quota = sum(pair_quota.values())
                    print(
                        f"[Trainer] SAMPLE_SPLIT train filter enabled: {sample_split_test_list} "
                        f"(quota={total_quota}, keys={len(pair_quota)}, max_len={sample_split_max_len})",
                        flush=True,
                    )

                    excluded_sentence_ids, remaining_quota = _build_excluded_sentence_ids_from_quota(
                        self.sent_train_ds,
                        pair_quota,
                        sample_split_max_len,
                    )
                    removed_sent = _filter_sentence_dataset_index(self.sent_train_ds, excluded_sentence_ids)
                    removed_bigram = _filter_bigram_dataset_index(self.bigram_train_ds, excluded_sentence_ids)
                    missing_after_match = sum(v for v in remaining_quota.values() if v > 0)

                    print(
                        f"[Trainer] SAMPLE_SPLIT removed: sent={removed_sent}, bigram={removed_bigram}, "
                        f"missing_quota={missing_after_match}",
                        flush=True,
                    )
                    print(
                        f"[Trainer] After filter: train(sent)={len(self.sent_train_ds)}, "
                        f"train(bigram)={len(self.bigram_train_ds)}",
                        flush=True,
                    )

                    if len(self.sent_train_ds) == 0:
                        raise RuntimeError("SAMPLE_SPLIT filtering removed all sentence training samples.")
                    if len(self.bigram_train_ds) == 0:
                        raise RuntimeError("SAMPLE_SPLIT filtering removed all bigram training samples.")


        self.W = int(cfg.ENV.DATALOADER_WORKERS)

        prefetch_cfg = int(getattr(cfg.ENV, "DATALOADER_PREFETCH_FACTOR", 4))
        if prefetch_cfg < 1:
            prefetch_cfg = 1
        self.prefetch = (prefetch_cfg if self.W > 0 else None)

        persist = bool(getattr(cfg.ENV, "PERSISTENT_WORKERS", False)) if self.W > 0 else False
        pin_mem = bool(getattr(cfg.ENV, "PIN_MEMORY", False))

        if self.W > 0 and bool(getattr(cfg.ENV, "PRELOAD_DATA", False)):
            print(
                "[Trainer][WARN] PRELOAD_DATA=true with num_workers>0 may increase RAM usage per worker "
                "(cache copy/expansion).",
                flush=True,
            )


        def worker_init_fn(worker_id):
            import numpy as np
            import random

            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)


        from src.data.unique_writer_sampler import UniqueWriterBatchSampler

        if self.is_ddp:

            char_sampler = DistributedSampler(
                self.char_train_ds,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True,
                drop_last=True
            )
            self.char_loader = DataLoader(
                self.char_train_ds,
                batch_size=cfg.TRAIN.CHAR_BATCH_SIZE,
                sampler=char_sampler,
                num_workers=self.W, pin_memory=pin_mem, persistent_workers=persist,
                prefetch_factor=self.prefetch,
                collate_fn=self.char_train_ds.collate_fn,
                worker_init_fn=worker_init_fn,
            )
        else:

            char_batch_sampler = UniqueWriterBatchSampler(
                self.char_train_ds,
                batch_size=cfg.TRAIN.CHAR_BATCH_SIZE,
                drop_last=True,
                shuffle=True
            )
            self.char_loader = DataLoader(
                self.char_train_ds,
                batch_sampler=char_batch_sampler,
                num_workers=self.W, pin_memory=pin_mem, persistent_workers=persist,
                prefetch_factor=self.prefetch,
                collate_fn=self.char_train_ds.collate_fn,
                worker_init_fn=worker_init_fn,
            )


        if not self.stage1_only:
            if self.is_ddp:

                bigram_sampler = DistributedSampler(
                    self.bigram_train_ds,
                    num_replicas=self.world_size,
                    rank=self.rank,
                    shuffle=True,
                    drop_last=True
                )
                sent_sampler = DistributedSampler(
                    self.sent_train_ds,
                    num_replicas=self.world_size,
                    rank=self.rank,
                    shuffle=True,
                    drop_last=True
                )
                val_sampler = DistributedSampler(
                    self.sent_val_ds,
                    num_replicas=self.world_size,
                    rank=self.rank,
                    shuffle=False,
                    drop_last=False
                )

                self.bigram_loader = DataLoader(
                    self.bigram_train_ds,
                    batch_size=cfg.TRAIN.BIGRAM_BATCH_SIZE,
                    sampler=bigram_sampler,
                    num_workers=self.W, pin_memory=pin_mem, persistent_workers=persist,
                    prefetch_factor=self.prefetch,
                    collate_fn=self.bigram_train_ds.collate_fn,
                    worker_init_fn=worker_init_fn,
                )
                self.sent_loader = DataLoader(
                    self.sent_train_ds,
                    batch_size=cfg.TRAIN.SENT_BATCH_SIZE,
                    sampler=sent_sampler,
                    num_workers=self.W, pin_memory=pin_mem, persistent_workers=persist,
                    prefetch_factor=self.prefetch,
                    collate_fn=self.sent_train_ds.collate_fn,
                    worker_init_fn=worker_init_fn,
                )
                self.val_loader = DataLoader(
                    self.sent_val_ds,
                    batch_size=cfg.VALID.BATCH_SIZE,
                    sampler=val_sampler,
                    num_workers=self.W, pin_memory=pin_mem, persistent_workers=persist,
                    prefetch_factor=self.prefetch, drop_last=False,
                    collate_fn=self.sent_val_ds.collate_fn,
                    worker_init_fn=worker_init_fn,
                )
            else:

                bigram_batch_sampler = UniqueWriterBatchSampler(
                    self.bigram_train_ds,
                    batch_size=cfg.TRAIN.BIGRAM_BATCH_SIZE,
                    drop_last=True,
                    shuffle=True
                )
                sent_batch_sampler = UniqueWriterBatchSampler(
                    self.sent_train_ds,
                    batch_size=cfg.TRAIN.SENT_BATCH_SIZE,
                    drop_last=True,
                    shuffle=True
                )

                self.bigram_loader = DataLoader(
                    self.bigram_train_ds,
                    batch_sampler=bigram_batch_sampler,
                    num_workers=self.W, pin_memory=pin_mem, persistent_workers=persist,
                    prefetch_factor=self.prefetch,
                    collate_fn=self.bigram_train_ds.collate_fn,
                    worker_init_fn=worker_init_fn,
                )
                self.sent_loader = DataLoader(
                    self.sent_train_ds,
                    batch_sampler=sent_batch_sampler,
                    num_workers=self.W, pin_memory=pin_mem, persistent_workers=persist,
                    prefetch_factor=self.prefetch,
                    collate_fn=self.sent_train_ds.collate_fn,
                    worker_init_fn=worker_init_fn,
                )
                self.val_loader = DataLoader(
                    self.sent_val_ds, batch_size=cfg.VALID.BATCH_SIZE, shuffle=False,
                    num_workers=self.W, pin_memory=pin_mem, persistent_workers=persist,
                    prefetch_factor=self.prefetch, drop_last=False,
                    collate_fn=self.sent_val_ds.collate_fn,
                    worker_init_fn=worker_init_fn,
                )
            print_once(f"[Trainer] train(char)={len(self.char_train_ds)} "
                       f"train(bigram)={len(self.bigram_train_ds)} "
                       f"train(sent)={len(self.sent_train_ds)} valid(sent)={len(self.sent_val_ds)}")
        else:

            self.bigram_train_ds = None
            self.sent_train_ds = None
            self.sent_val_ds = None
            self.bigram_loader = None
            self.sent_loader = None
            self.val_loader = None
            print_once(f"[Trainer] STAGE1_ONLY mode: train(char)={len(self.char_train_ds)}")


        freeze_backbone_bn = getattr(cfg.MODEL, 'FREEZE_BACKBONE_BN', True)

        self.style_identifier = StyleIdentifier(
            style_dim=cfg.MODEL.STYLE_DIM,
            encoder_type=cfg.MODEL.ENCODER_TYPE,
            img_size=(cfg.ENV.IMG_H, cfg.ENV.IMG_W),
            base_layers=cfg.MODEL.BASE_LAYERS,
            base_nhead=cfg.MODEL.BASE_NHEAD,
            head_layers=cfg.MODEL.HEAD_LAYERS,
            head_nhead=cfg.MODEL.HEAD_NHEAD,
            patch_size=cfg.MODEL.PATCH_SIZE,
            freeze_backbone_bn=freeze_backbone_bn,
        )

        self.use_context_as_content = getattr(cfg.MODEL, 'USE_CONTEXT_AS_CONTENT', False)

        if self.use_context_as_content:

            self.font_encoder = None
            if self.rank == 0:
                print(f"[Trainer]  USE_CONTEXT_AS_CONTENT=True: Font Encoder 비활성화")
        else:


            encoder_vq_start = getattr(cfg.MODEL, 'ENCODER_VQ_START_ITER', -1)
            use_vq = (encoder_vq_start >= 0)

            self.font_encoder = FontEncoder(
                d_model=cfg.MODEL.FONT_DIM,
                base_nhead=cfg.MODEL.BASE_NHEAD,
                head_layers=cfg.MODEL.FONT_HEAD_LAYERS,

                use_vq=use_vq,
                vq_codebook_size=getattr(cfg.MODEL, 'ENCODER_VQ_CODEBOOK_SIZE', 512),
                vq_embed_dim=getattr(cfg.MODEL, 'ENCODER_VQ_EMBED_DIM', 128),
                vq_commitment_weight=getattr(cfg.MODEL, 'ENCODER_VQ_COMMITMENT_WEIGHT', 0.25),
                vq_codebook_decay=getattr(cfg.MODEL, 'ENCODER_VQ_CODEBOOK_DECAY', 0.99),
            ).to(self.device, non_blocking=True)

        self.context_encoder = ContextEncoder(
            model_name=cfg.MODEL.CONTEXT_BACKBONE,
            d_model=cfg.MODEL.CONTEXT_DIM,
            nhead=cfg.MODEL.CONTEXT_NHEAD,
            num_layers=cfg.MODEL.CONTEXT_LAYERS,
            dropout=cfg.MODEL.CONTEXT_DROPOUT,
            add_sin_posenc=True,
            freeze_backbone=getattr(cfg.MODEL, 'CONTEXT_FREEZE_BACKBONE', True),

            use_char_id_emb=getattr(cfg.MODEL, 'CONTEXT_USE_CHAR_ID_EMB', False),
            char_id_emb_dim=getattr(cfg.MODEL, 'CONTEXT_CHAR_ID_EMB_DIM', 64),
            max_char_id=getattr(cfg.MODEL, 'CONTEXT_MAX_CHAR_ID', 200000),

            context_token_dropout=getattr(cfg.MODEL, 'CONTEXT_TOKEN_DROPOUT_STAGE1', 0.0),
        ).to(self.device, non_blocking=True)


        self.hw_generator = HandwritingGenerator(
            d_model=cfg.MODEL.HWGEN_DIM,
            nhead=cfg.MODEL.BASE_NHEAD,
            writer_layers=cfg.MODEL.HWGEN_WRITER_LAYERS,
            glyph_layers=cfg.MODEL.HWGEN_GLYPH_LAYERS,
            use_rope=getattr(cfg.MODEL, 'USE_ROPE', True),
            rope_base=getattr(cfg.MODEL, 'ROPE_BASE', 100.0),

            use_context_gating=getattr(cfg.MODEL, 'USE_CONTEXT_GATING', False),
            context_gate_init=getattr(cfg.MODEL, 'CONTEXT_GATE_INIT', 0.5),
            context_gate_tokenwise=getattr(cfg.MODEL, 'CONTEXT_GATE_TOKENWISE', False),
            context_gate_cap=getattr(cfg.MODEL, 'CONTEXT_GATE_CAP', 1.0),
            use_context_pen_from_glyph_only=getattr(cfg.MODEL, 'CONTEXT_PEN_FROM_GLYPH_ONLY', False),

            use_residual_vq=getattr(cfg.MODEL, 'STAGE1_USE_DECODER_VQ', False),
            rvq_num_quantizers=getattr(cfg.MODEL, 'DECODER_VQ_NUM_QUANTIZERS', 2),
            rvq_codebook_size=getattr(cfg.MODEL, 'DECODER_VQ_CODEBOOK_SIZE', 512),
            rvq_embed_dim=getattr(cfg.MODEL, 'DECODER_VQ_EMBED_DIM', 64),
            rvq_commitment_weight=getattr(cfg.MODEL, 'DECODER_VQ_COMMITMENT_WEIGHT', 0.25),
            rvq_codebook_decay=getattr(cfg.MODEL, 'DECODER_VQ_CODEBOOK_DECAY', 0.99),
            rvq_clamp_scale=getattr(cfg.MODEL, 'DECODER_VQ_CLAMP_SCALE', 0.02),
            rvq_gate_init=getattr(cfg.MODEL, 'DECODER_VQ_GATE_INIT', 0.5),

            use_context_decoder=getattr(cfg.MODEL, 'USE_CONTEXT_DECODER', True),
        ).to(self.device, non_blocking=True)

        self.full_model = FullModel(
            style_identifier=self.style_identifier,
            font_encoder=self.font_encoder,
            handwriting_generator=self.hw_generator,
            context_encoder=self.context_encoder,
            use_context_as_content=self.use_context_as_content,

            n_gram_window=getattr(cfg.MODEL, 'N_GRAM_AWARE_SLIDING_WINDOW', 2),
        ).to(self.device, non_blocking=True)


        if self.is_ddp:
            self.full_model = DDP(
                self.full_model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=True,
            )
            if self.rank == 0:
                print(f"[Trainer] DDP enabled: {self.world_size} GPUs")


        model_params = self.full_model.module.parameters() if self.is_ddp else self.full_model.parameters()
        self.optimizer = optim.AdamW(model_params, lr=cfg.HYPERPARAMETER.BASE_LR, fused=True)


        from src.train.lr_scheduler import StageWarmupScheduler
        self.lr_scheduler = StageWarmupScheduler(
            self.optimizer,
            base_lr=cfg.HYPERPARAMETER.BASE_LR,
            initial_warmup_iters=getattr(cfg.HYPERPARAMETER, 'WARMUP_ITERS', 20000),
            stage_warmup_iters=getattr(cfg.HYPERPARAMETER, 'STAGE_WARMUP_ITERS', 1000)
        )


        self.style_loss_fn = WriterGlyphNCELoss(temperature=0.07)


        self.decoder_vq_loss_fn = RVQLoss(
            gate_regularization=0.01,
            num_mixtures=20,
        )

        self.decoder_vq_loss_weight_target = getattr(cfg.TRAIN, 'DECODER_VQ_LOSS_WEIGHT', 0.3)
        self.decoder_vq_warmup_iters = getattr(cfg.TRAIN, 'DECODER_VQ_WARMUP_ITERS', 5000)


        self.encoder_vq_loss_weight_target = getattr(cfg.TRAIN, 'ENCODER_VQ_LOSS_WEIGHT', 0.5)
        self.encoder_vq_warmup_iters = getattr(cfg.TRAIN, 'ENCODER_VQ_WARMUP_ITERS', 10000)


        self.stage_configs = {
            'stage1': {
                'name': 'Char',
                'temperature': 0.07,
                'grad_clip': getattr(self.cfg.TRAIN, 'STAGE1_GRAD_CLIP_NORM', 1.0),
                'loss_weights': {
                    'style': self.cfg.TRAIN.CHAR_STEP_STYLE_LOSS_WEIGHT,
                    'char': self.cfg.TRAIN.CHAR_STEP_CHAR_GEN_LOSS_WEIGHT,
                },
                'end_iter': self.cfg.TRAIN.CHAR_STAGE_ITERS,
            },
            'stage2': {
                'name': 'Char+Bigram',
                'temperature': getattr(self.cfg.TRAIN, 'STAGE2_CONTRASTIVE_TEMPERATURE', 0.07),
                'grad_clip': getattr(self.cfg.TRAIN, 'STAGE2_GRAD_CLIP_NORM', 1.0),
                'loss_weights': {
                    'style': self.cfg.TRAIN.BIGRAM_STEP_STYLE_LOSS_WEIGHT,
                    'char': self.cfg.TRAIN.BIGRAM_STEP_CHAR_GEN_LOSS_WEIGHT,
                    'bigram': self.cfg.TRAIN.BIGRAM_STEP_BIGRAM_GEN_LOSS_WEIGHT,
                    'vdl': getattr(self.cfg.TRAIN, 'BIGRAM_STEP_VDL_LOSS_WEIGHT', 0.0),
                },
                'end_iter': self.cfg.TRAIN.BIGRAM_STAGE_ITERS,
            },
            'stage3': {
                'name': 'Char+Sentence',
                'temperature': getattr(self.cfg.TRAIN, 'STAGE3_CONTRASTIVE_TEMPERATURE', 0.07),
                'grad_clip': getattr(self.cfg.TRAIN, 'STAGE3_GRAD_CLIP_NORM', 1.0),
                'loss_weights': {
                    'style': self.cfg.TRAIN.SENT_STEP_STYLE_LOSS_WEIGHT,
                    'char': self.cfg.TRAIN.SENT_STEP_CHAR_GEN_LOSS_WEIGHT,
                    'sent': self.cfg.TRAIN.SENT_STEP_SENT_GEN_LOSS_WEIGHT,
                    'vdl': getattr(self.cfg.TRAIN, 'SENT_STEP_VDL_LOSS_WEIGHT', 0.0),
                },
                'end_iter': self.cfg.HYPERPARAMETER.MAX_ITER,
            }
        }
        self.current_stage = 'stage1'


        self.encoder_vq_start_iter = getattr(self.cfg.MODEL, 'ENCODER_VQ_START_ITER', -1)
        self.decoder_vq_start_iter = getattr(self.cfg.MODEL, 'DECODER_VQ_START_ITER', -1)


        self.encoder_vq_enabled = False
        self.decoder_vq_enabled = False


        resume_path = _normalize_cfg_path(getattr(cfg.SAVE, "RESUME_PATH", None))
        if resume_path and os.path.exists(resume_path):

            self.exp_dir = os.path.dirname(os.path.dirname(resume_path))
        else:

            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.exp_dir = os.path.join(cfg.ENV.SAVE_MODEL_DIR, cfg.ENV.MODE, timestamp)

        os.makedirs(self.exp_dir, exist_ok=True)
        self.ckpt_dir = os.path.join(self.exp_dir, "checkpoints"); os.makedirs(self.ckpt_dir, exist_ok=True)

        if self.rank == 0:
            self.snap_dir = os.path.join(self.exp_dir, "snapshots");  os.makedirs(self.snap_dir, exist_ok=True)


            tb_dir = os.path.join(self.exp_dir, "tensorboard")
            os.makedirs(tb_dir, exist_ok=True)
            self.tb = SummaryWriter(log_dir=tb_dir)


            self.viz_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="viz")
        else:
            self.snap_dir = None
            self.tb = None
            self.viz_executor = None


        self.global_iter = 0
        self.best_val = float("inf")


        self.use_amp = getattr(cfg.HYPERPARAMETER, 'USE_AMP', False)


        self.grad_accum_steps_stage1 = getattr(cfg.HYPERPARAMETER, 'GRAD_ACCUM_STEPS_STAGE1', 1)
        self.grad_accum_steps_stage2 = getattr(cfg.HYPERPARAMETER, 'GRAD_ACCUM_STEPS_STAGE2', 1)
        self.grad_accum_steps_stage3_char = getattr(cfg.HYPERPARAMETER, 'GRAD_ACCUM_STEPS_STAGE3_CHAR', 1)
        self.grad_accum_steps_stage3_sent = getattr(cfg.HYPERPARAMETER, 'GRAD_ACCUM_STEPS_STAGE3_SENT', 2)


        self.grad_accum_steps = self.grad_accum_steps_stage1


        self.scaler = torch.cuda.amp.GradScaler(
            enabled=self.use_amp,
            init_scale=2.**16,
            growth_interval=2000
        )

        if self.use_amp:
            print(f"[Trainer]  AMP ENABLED (메모리 절감 모드)")
        else:
            print(f"[Trainer]  AMP DISABLED (SDT 방식, NaN 안정성 강화)")

        print(f"[Trainer]  Gradient Accumulation (Stage별):")
        print(f"   → Stage 1: Char {cfg.TRAIN.CHAR_BATCH_SIZE} (accum={self.grad_accum_steps_stage1})")
        print(f"   → Stage 2: Char {cfg.TRAIN.CHAR_BATCH_SIZE} + Bigram {cfg.TRAIN.BIGRAM_BATCH_SIZE} (accum={self.grad_accum_steps_stage2})")
        print(f"   → Stage 3: Char {cfg.TRAIN.CHAR_BATCH_SIZE} (accum={self.grad_accum_steps_stage3_char}) + Sent {cfg.TRAIN.SENT_BATCH_SIZE}×{self.grad_accum_steps_stage3_sent} (style detach)")


        if not self.is_ddp and torch.cuda.device_count() > 1:
            self.full_model = torch.nn.DataParallel(self.full_model)
            print(f"[Trainer] DataParallel enabled: {torch.cuda.device_count()} GPUs")


        self.font_img_dict = self.char_train_ds.content_font_img


        self._init_vq_branches()


        resume_path = _normalize_cfg_path(getattr(self.cfg.SAVE, "RESUME_PATH", None))
        resume_dir  = _normalize_cfg_path(getattr(self.cfg.SAVE, "RESUME_DIR",  None))
        resume_best = getattr(self.cfg.SAVE, "RESUME_BEST", False)
        if resume_path and os.path.isfile(resume_path):
            step, best = load_checkpoint(
                resume_path, self.full_model, self.device, optimizer=self.optimizer, strict=False
            )
            self.global_iter = step
            if best is not None: self.best_val = best
        elif resume_path:
            print(f"[Resume]  RESUME_PATH not found, start from scratch: {resume_path}")
        elif resume_dir and os.path.isdir(resume_dir):
            if resume_best and os.path.isfile(os.path.join(resume_dir, "ckpt_best.pt")):
                step, best = load_checkpoint(
                    os.path.join(resume_dir, "ckpt_best.pt"),
                    self.full_model, self.device, optimizer=self.optimizer, strict=False
                )
            else:
                out = load_latest_checkpoint(
                    resume_dir, self.full_model, self.device, optimizer=self.optimizer, strict=False
                )
                if out is not None:
                    step, best = out
                else:
                    step, best = 0, None
            self.global_iter = step
            if best is not None: self.best_val = best
        elif resume_dir:
            print(f"[Resume]  RESUME_DIR not found, start from scratch: {resume_dir}")


        self._load_start_checkpoint()


    def _load_start_checkpoint(self):
        start_path = _normalize_cfg_path(getattr(self.cfg.SAVE, "START_PATH", None))
        if not start_path or not os.path.isfile(start_path):
            return


        start_modules = getattr(self.cfg.SAVE, "START_MODULES", {})
        load_style_id = start_modules.get("STYLE_IDENTIFIER", False)
        load_font_enc = start_modules.get("FONT_ENCODER", False)
        load_hw_gen = start_modules.get("HANDWRITING_GENERATOR", False)
        load_ctx_enc = start_modules.get("CONTEXT_ENCODER", False)
        load_rvq_codebook = start_modules.get("RVQ_CODEBOOK", False)


        if not any([load_style_id, load_font_enc, load_hw_gen, load_ctx_enc, load_rvq_codebook]):
            return


        ckpt = torch.load(start_path, map_location=self.device)
        src_sd = ckpt.get("model", ckpt)


        has_module = any(k.startswith("module.") for k in src_sd.keys())
        if has_module:
            src_sd = {k.replace("module.", ""): v for k, v in src_sd.items()}


        fm = self.full_model.module if self.is_ddp else self.full_model
        tgt_sd = fm.state_dict()

        loaded_modules = []
        loaded_keys = 0


        if load_style_id:
            prefix = "style_identifier."
            matched = {k: v for k, v in src_sd.items() if k.startswith(prefix)}
            for k, v in matched.items():
                if k in tgt_sd and tgt_sd[k].shape == v.shape:
                    tgt_sd[k] = v
                    loaded_keys += 1
            if matched:
                loaded_modules.append("StyleIdentifier")


        if load_font_enc and not self.use_context_as_content:
            prefix = "font_encoder."
            matched = {k: v for k, v in src_sd.items() if k.startswith(prefix)}
            for k, v in matched.items():
                if k in tgt_sd and tgt_sd[k].shape == v.shape:
                    tgt_sd[k] = v
                    loaded_keys += 1
            if matched:
                loaded_modules.append("FontEncoder")
        elif load_font_enc and self.use_context_as_content:
            if self.rank == 0:
                print("[START_PATH]  Font Encoder 로드 스킵 (USE_CONTEXT_AS_CONTENT=True)")


        if load_hw_gen:
            prefix = "handwriting_generator."
            matched = {k: v for k, v in src_sd.items() if k.startswith(prefix)}
            for k, v in matched.items():
                if k in tgt_sd and tgt_sd[k].shape == v.shape:
                    tgt_sd[k] = v
                    loaded_keys += 1
            if matched:
                loaded_modules.append("HandwritingGenerator")


        if load_ctx_enc:
            prefix = "context_encoder."
            matched = {k: v for k, v in src_sd.items() if k.startswith(prefix)}
            for k, v in matched.items():
                if k in tgt_sd and tgt_sd[k].shape == v.shape:
                    tgt_sd[k] = v
                    loaded_keys += 1
            if matched:
                loaded_modules.append("ContextEncoder")


        if load_rvq_codebook:
            vq_patterns = [
                "handwriting_generator.rvq_branch.rvq.quantizers",
            ]

            if not self.use_context_as_content:
                vq_patterns.extend([
                    "font_encoder.vq_adapter.vq.embedding",
                    "font_encoder.vq_adapter.vq.ema_cluster_size",
                    "font_encoder.vq_adapter.vq.ema_weight",
                ])
            codebook_loaded = 0
            for k, v in src_sd.items():
                if any(pat in k for pat in vq_patterns):
                    if k in tgt_sd and tgt_sd[k].shape == v.shape:
                        tgt_sd[k] = v
                        loaded_keys += 1
                        codebook_loaded += 1
            if codebook_loaded > 0:
                loaded_modules.append(f"RVQ_Codebook({codebook_loaded})")


        fm.load_state_dict(tgt_sd, strict=False)


        filename = os.path.basename(start_path)
        print(f"\n[START_PATH] {filename} → {', '.join(loaded_modules)} ({loaded_keys} keys)")
        print(f"[START_PATH] Starting from iter 0 with initialized weights\n")


    def _diagnose_nan_source(self):
        print("\n" + "-"*60)
        print("[NaN DIAGNOSIS] Checking model state...")


        si = self.style_identifier.module if isinstance(self.style_identifier, torch.nn.DataParallel) else self.style_identifier
        bn_nan_count = 0
        for name, module in si.backbone.named_modules():
            if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
                rm_nan = not torch.isfinite(module.running_mean).all() if module.running_mean is not None else False
                rv_nan = not torch.isfinite(module.running_var).all() if module.running_var is not None else False
                if rm_nan or rv_nan:
                    bn_nan_count += 1
                    if bn_nan_count <= 3:
                        print(f"   BatchNorm '{name}': running_mean NaN={rm_nan}, running_var NaN={rv_nan}")

        if bn_nan_count > 0:
            print(f"   Total {bn_nan_count} BatchNorm layers have NaN running stats!")
            print(f"  → This is the ROOT CAUSE of NaN propagation")
        else:
            print(f"   BatchNorm running stats are clean")


        weight_nan_count = 0
        for name, param in si.named_parameters():
            if not torch.isfinite(param).all():
                weight_nan_count += 1
                if weight_nan_count <= 3:
                    nan_ratio = (~torch.isfinite(param)).sum().item() / param.numel()
                    print(f"   Weight '{name}': {nan_ratio*100:.1f}% NaN/Inf")

        if weight_nan_count > 0:
            print(f"   Total {weight_nan_count} parameters have NaN values!")
        else:
            print(f"   Model weights are clean")


        if self.use_amp and hasattr(self, 'scaler'):
            scale = self.scaler.get_scale()
            print(f"   GradScaler scale: {scale:.2e}")
            if scale < 1.0:
                print(f"   Scale is very low - may indicate gradient issues")

        print("-"*60 + "\n")

    def _find_latest_checkpoint(self, before_iter: int) -> Optional[str]:
        import glob as _glob
        search_dirs = []
        for d in [
            getattr(self, "ckpt_dir", None),
            getattr(self, "exp_dir", None),
            os.path.dirname(_normalize_cfg_path(getattr(self.cfg.SAVE, "RESUME_PATH", None)) or ""),
            _normalize_cfg_path(getattr(self.cfg.SAVE, "RESUME_DIR", None)),
        ]:
            if d and os.path.isdir(d) and d not in search_dirs:
                search_dirs.append(d)

        candidates = []
        for d in search_dirs:
            pattern = os.path.join(d, "ckpt_*.pt")
            for p in _glob.glob(pattern):
                base = os.path.basename(p)
                num_str = base.replace("ckpt_", "").replace(".pt", "")
                try:
                    it = int(num_str)
                except ValueError:
                    continue
                if it <= before_iter:
                    candidates.append((it, p))

        if not candidates:
            return None

        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]

    def _force_freeze_backbone_bn(self):
        si = self.style_identifier.module if isinstance(
            self.style_identifier, torch.nn.DataParallel) else self.style_identifier
        if hasattr(si, '_freeze_backbone_batchnorm'):
            si.freeze_backbone_bn = True
            si._freeze_backbone_batchnorm()
            print(" [Recovery] Backbone BN force-frozen to eval mode")

    def _attempt_nan_recovery(self, stage_name: str) -> bool:
        ckpt_path = self._find_latest_checkpoint(self.global_iter)
        if ckpt_path is None:
            print(f" [{stage_name}] No checkpoint found for recovery")
            return False

        print(f" [{stage_name}] Attempting auto-recovery from {ckpt_path}...")
        try:
            step = self.load_checkpoint(ckpt_path, strict=False)
            self._force_freeze_backbone_bn()
            self._nan_count = 0
            self._consec_grad_explosion = 0
            print(f" [{stage_name}] Recovered from checkpoint (iter {step})")
            return True
        except Exception as e:
            print(f" [{stage_name}] Recovery failed: {e}")
            return False


    def _init_vq_branches(self):
        from src.model.residual_vq import FontVQBranch, ResidualVQBranch

        encoder_vq_start = getattr(self.cfg.MODEL, 'ENCODER_VQ_START_ITER', -1)
        decoder_vq_start = getattr(self.cfg.MODEL, 'DECODER_VQ_START_ITER', -1)


        if self.use_context_as_content:

            print(f"[Init]  USE_CONTEXT_AS_CONTENT=True: Font Encoder VQ 해당 없음")
        elif encoder_vq_start >= 0:
            if self.font_encoder is not None and hasattr(self.font_encoder, 'vq_adapter') and self.font_encoder.vq_adapter is not None:
                K = self.font_encoder.vq_adapter.num_embeddings
                C = self.font_encoder.vq_adapter.in_proj.out_features
                print(f"[Init]  Encoder VQAdapter ready: K={K}, C={C} (will activate at iter {encoder_vq_start})")
            else:
                print(f"[Init]  Encoder VQAdapter not found (check FontEncoder initialization)")
        else:
            print(f"[Init] Encoder VQ disabled (START_ITER={encoder_vq_start})")


        if decoder_vq_start >= 0:
            hw_gen = self.hw_generator.module if isinstance(self.hw_generator, torch.nn.DataParallel) else self.hw_generator
            if hw_gen.rvq_branch is None:
                print(f"[Init]  Creating Decoder VQ branch (will activate at iter {decoder_vq_start})...")
                hw_gen.rvq_branch = ResidualVQBranch(
                    d_model=self.cfg.MODEL.HWGEN_DIM,
                    num_quantizers=getattr(self.cfg.MODEL, 'DECODER_VQ_NUM_QUANTIZERS', 2),
                    codebook_size=getattr(self.cfg.MODEL, 'DECODER_VQ_CODEBOOK_SIZE', 512),
                    vq_dim=getattr(self.cfg.MODEL, 'DECODER_VQ_EMBED_DIM', 64),
                    commitment_weight=getattr(self.cfg.MODEL, 'DECODER_VQ_COMMITMENT_WEIGHT', 0.25),
                    codebook_decay=getattr(self.cfg.MODEL, 'DECODER_VQ_CODEBOOK_DECAY', 0.99),
                    clamp_scale=getattr(self.cfg.MODEL, 'DECODER_VQ_CLAMP_SCALE', 0.02),
                    gate_init=getattr(self.cfg.MODEL, 'DECODER_VQ_GATE_INIT', 0.5),
                ).to(self.device)
                hw_gen._last_vq_loss = None
                hw_gen._last_vq_gate = None
                print(f"[Init]  Decoder VQ branch created: L={hw_gen.rvq_branch.rvq.num_quantizers}, K={hw_gen.rvq_branch.rvq.quantizers[0].num_embeddings}, C={hw_gen.rvq_branch.rvq.quantizers[0].embedding_dim}")
        else:
            print(f"[Init] Decoder VQ disabled (START_ITER={decoder_vq_start})")


    def build_inputs_unified(self, batch, mode='char'):
        device = self.device


        inputs = {
            'style_imgs': batch["style_reference"].to(device, non_blocking=True),
            'style_writer_ids': batch["style_labels"].to(device, non_blocking=True),
            'char_imgs': batch["font_reference"].to(device, non_blocking=True),
        }


        if mode == 'char':

            trajs = batch["target_traj"].to(device, non_blocking=True)
            inputs['curr_trajs'] = trajs.unsqueeze(1)
            inputs['loss_mask'] = batch["seq_loss_mask"].to(device, non_blocking=True)
            inputs['seq_chars'] = batch["seq_chars"].to(device, non_blocking=True).int()


            if "target_len" in batch:
                inputs['traj_lens'] = batch["target_len"].unsqueeze(1).to(device, non_blocking=True)
            else:


                valid_traj = (trajs[..., PEN_STATE_RANGE].sum(dim=-1) > 0)
                inputs['traj_lens'] = valid_traj.sum(dim=-1).unsqueeze(1)


            inputs['gt_trajs'] = inputs['curr_trajs']

        elif mode in ['bigram', 'sentence']:

            inputs['curr_trajs'] = batch["seq_trajs"].to(device, non_blocking=True)
            inputs['traj_lens'] = batch["seq_traj_lens"].to(device, non_blocking=True)
            inputs['loss_mask'] = batch["seq_loss_mask"].to(device, non_blocking=True)
            seq_chars_full = batch["seq_chars"].to(device, non_blocking=True).long()


            B, S, T = inputs['curr_trajs'].shape[:3]


            arange_t = torch.arange(T, device=device).view(1, 1, T)
            valid_t = (arange_t < inputs['traj_lens'].unsqueeze(-1)) & inputs['loss_mask'].unsqueeze(-1)
            inputs['gt_trajs'] = inputs['curr_trajs'] * valid_t.unsqueeze(-1).float()


            if mode == 'bigram' and "curr_indices" in batch:
                curr_idx = batch["curr_indices"].to(device, non_blocking=True).long()
                prev_idx = torch.clamp(curr_idx - 1, min=0)
                inputs['indices'] = torch.stack([prev_idx, curr_idx], dim=1)


                bix = torch.arange(B, device=device).unsqueeze(1).expand(B, 2)
                idx_2 = inputs['indices']
                inputs['seq_chars'] = seq_chars_full[bix, idx_2]


                inputs['context_seq_chars'] = seq_chars_full
            else:

                inputs['seq_chars'] = seq_chars_full

        return inputs

    def _run_step_unified(self, batch, mode='char', style_cache=None):

        inputs = self.build_inputs_unified(batch, mode=mode)


        alpha = self._get_2pass_alpha(self.global_iter)


        with torch.amp.autocast('cuda', enabled=self.use_amp):
            device = self.device
            mdl = self.hw_generator.module if isinstance(self.hw_generator, torch.nn.DataParallel) else self.hw_generator
            D = getattr(mdl, "d_model", self.cfg.MODEL.HWGEN_DIM)

            gt_trajs = inputs['curr_trajs']
            loss_mask = inputs['loss_mask']
            traj_lens = inputs.get('traj_lens')
            B, S, T = gt_trajs.shape[:3]


            if alpha > 0.0:

                with torch.no_grad():

                    was_training_fm = self.full_model.training
                    was_training_hw = self.hw_generator.training
                    self.full_model.eval()
                    self.hw_generator.eval()


                    gt_trajs_flat = gt_trajs.view(B * S, T, 6)
                    gt_traj_embs_flat = mdl.seq_to_emb(gt_trajs_flat)
                    gt_traj_embs = gt_traj_embs_flat.view(B, S, T, D)


                    if traj_lens is not None:
                        traj_lens_bc = traj_lens.expand(B, S) if traj_lens.size(1) == 1 else traj_lens
                        arange_t = torch.arange(T, device=device).view(1, 1, T)
                        valid_mask = (arange_t < traj_lens_bc.unsqueeze(-1))
                        gt_traj_embs = gt_traj_embs * valid_mask.unsqueeze(-1).float()


                    result_pass1 = self.full_model(
                        style_imgs=inputs['style_imgs'],
                        char_imgs=inputs['char_imgs'],
                        curr_traj_embs=gt_traj_embs,
                        writer_ids=inputs['style_writer_ids'],
                        loss_mask=loss_mask,
                        seq_chars=inputs.get('seq_chars'),
                        context_seq_chars=inputs.get('context_seq_chars'),
                        indices=inputs.get('indices'),
                        global_iter=self.global_iter,
                        style_cache=style_cache,
                        traj_lens=traj_lens,
                        gt_trajs=inputs.get('curr_trajs'),
                    )

                    pred_gmm1 = result_pass1["pred_gmm"]


                    if self.n_gram_window == 1 and mode in ('bigram', 'sentence'):
                        pred_gmm1 = convert_unigram_gmm_to_bigram(
                            pred_gmm1, inputs['curr_trajs'], num_mixtures=20
                        )

                    pred_seq = get_seq_from_gmm(pred_gmm1.view(B*S, T, -1), decode="argmax_onehot")
                    pred_seq = pred_seq.view(B, S, T, 6)
                    pred_delta = pred_seq[..., 0:2]


                    if was_training_fm:
                        self.full_model.train()
                    if was_training_hw:
                        self.hw_generator.train()


                mixed_trajs = gt_trajs.clone()
                mixed_trajs[..., 0:2] = (1.0 - alpha) * gt_trajs[..., 0:2] + alpha * pred_delta


                curr_trajs = mixed_trajs
            else:

                curr_trajs = gt_trajs


            curr_trajs_flat = curr_trajs.view(B * S, T, 6)
            curr_traj_embs_flat = mdl.seq_to_emb(curr_trajs_flat)
            curr_traj_embs = curr_traj_embs_flat.view(B, S, T, D)


            if traj_lens is not None:
                traj_lens_bc = traj_lens.expand(B, S) if traj_lens.size(1) == 1 else traj_lens
                arange_t = torch.arange(T, device=device).view(1, 1, T)
                valid_mask = (arange_t < traj_lens_bc.unsqueeze(-1))
                curr_traj_embs = curr_traj_embs * valid_mask.unsqueeze(-1).float()


            result = self.full_model(
                    style_imgs=inputs['style_imgs'],
                    char_imgs=inputs['char_imgs'],
                    curr_traj_embs=curr_traj_embs,
                    writer_ids=inputs['style_writer_ids'],
                    loss_mask=loss_mask,
                    seq_chars=inputs.get('seq_chars'),
                    context_seq_chars=inputs.get('context_seq_chars'),
                    indices=inputs.get('indices'),
                    global_iter=self.global_iter,
                    style_cache=style_cache,
                    traj_lens=traj_lens,
                    gt_trajs=inputs.get('curr_trajs'),
            )

            pred_gmm = result["pred_gmm"]
            style_cache_out = result["style_cache"]


            if self.n_gram_window == 1 and mode in ('bigram', 'sentence'):
                pred_gmm = convert_unigram_gmm_to_bigram(
                    pred_gmm, inputs['curr_trajs'], num_mixtures=20
                )

            gen_loss = self._compute_seq_loss(pred_gmm, inputs['gt_trajs'], loss_mask)


        vdl_loss = torch.tensor(0.0, device=self.device)
        vdl_stats = {
            "centroid": torch.tensor(0.0, device=self.device),
            "top": torch.tensor(0.0, device=self.device),
            "bottom": torch.tensor(0.0, device=self.device)
        }
        vdl_enabled = False
        if mode == 'bigram':
            vdl_enabled = getattr(self.cfg.TRAIN, 'BIGRAM_STEP_VDL_LOSS_WEIGHT', 0.0) > 0.0
        elif mode == 'sentence':
            vdl_enabled = getattr(self.cfg.TRAIN, 'SENT_STEP_VDL_LOSS_WEIGHT', 0.0) > 0.0
        if vdl_enabled:


            vdl_loss, vdl_stats = self._compute_vertical_drift_loss(
                pred_gmm=pred_gmm,
                gt_traj=inputs['curr_trajs'],
                loss_mask=loss_mask,
                traj_lens=traj_lens,
            )


        decoder_vq_loss = torch.tensor(0.0, device=self.device)
        decoder_vq_stats = {}
        if self.decoder_vq_enabled and self.hw_generator.training:
            mdl = self.hw_generator.module if isinstance(self.hw_generator, torch.nn.DataParallel) else self.hw_generator
            if hasattr(mdl, '_last_vq_loss') and mdl._last_vq_loss is not None:
                decoder_vq_loss = mdl._last_vq_loss

                if hasattr(mdl, '_last_vq_gate') and mdl._last_vq_gate is not None:
                    decoder_vq_stats['decoder_vq_gate_mean'] = mdl._last_vq_gate.item()


        encoder_vq_loss = torch.tensor(0.0, device=self.device)
        encoder_vq_stats = {}

        if result.get('font_vq_loss') is not None:
            encoder_vq_loss = result['font_vq_loss']

            font_vq_info = result.get('font_vq_info', {})
            if font_vq_info:
                if 'perplexity' in font_vq_info:
                    perp = font_vq_info['perplexity']
                    if torch.is_tensor(perp):
                        perp = perp.mean() if perp.numel() > 1 else perp
                        encoder_vq_stats['font_vq_perplexity'] = float(perp.item())
                    else:
                        encoder_vq_stats['font_vq_perplexity'] = float(perp)
                if 'code_usage' in font_vq_info:
                    usage = font_vq_info['code_usage']
                    if torch.is_tensor(usage):
                        usage = usage.mean() if usage.numel() > 1 else usage
                        encoder_vq_stats['font_vq_usage'] = int(usage.item())
                    else:
                        encoder_vq_stats['font_vq_usage'] = int(usage)


        style_loss = torch.tensor(0.0, device=self.device)
        if style_cache is None and style_cache_out is not None:
            writer_emb = style_cache_out['writer_emb']
            glyph_emb = style_cache_out['glyph_emb']
            style_loss = self.style_loss_fn(writer_emb, glyph_emb, inputs['style_writer_ids'])["total"]

        return {
            'gen_loss': gen_loss,
            'style_loss': style_loss,
            'decoder_vq_loss': decoder_vq_loss,
            'decoder_vq_stats': decoder_vq_stats,
            'encoder_vq_loss': encoder_vq_loss,
            'encoder_vq_stats': encoder_vq_stats,
            'vdl_loss': vdl_loss,
            'vdl_stats': vdl_stats,
            'pred_gmm': pred_gmm,
            'gt_trajs': inputs['gt_trajs'],
            'gt_trajs_unmasked': inputs['curr_trajs'],
            'loss_mask': inputs['loss_mask'],
            'seq_chars': inputs.get('seq_chars'),
            'style_cache': style_cache_out,
        }


    def build_inputs_from_sentence_batch(self, batch):
        device = self.device
        style_ref_imgs   = batch["style_reference"].to(device, non_blocking=True)
        style_writer_ids = batch["style_labels"].to(device, non_blocking=True)
        font_ref_imgs    = batch["font_reference"].to(device, non_blocking=True)
        trajs_all        = batch["seq_trajs"].to(device, non_blocking=True)
        traj_lens        = batch["seq_traj_lens"].to(device, non_blocking=True)
        loss_mask_bool   = batch["seq_loss_mask"].to(device, non_blocking=True)
        seq_chars        = batch["seq_chars"].to(device, non_blocking=True).int()

        B, S_max, T_max, _ = trajs_all.shape
        D = getattr(self.hw_generator, "d_model", self.cfg.MODEL.HWGEN_DIM)

        mdl = self.hw_generator.module if isinstance(self.hw_generator, torch.nn.DataParallel) else self.hw_generator


        flat = trajs_all.reshape(B * S_max, T_max, 6)
        flat_emb = mdl.seq_to_emb(flat)
        sentence_curr_traj_embs = flat_emb.reshape(B, S_max, T_max, -1)


        arange_t = torch.arange(T_max, device=device).view(1, 1, T_max)
        valid_t = (arange_t < traj_lens.unsqueeze(-1)) & loss_mask_bool.unsqueeze(-1)
        gt_traj_tensor = trajs_all * valid_t.unsqueeze(-1).float()

        sentence_loss_mask_f = loss_mask_bool.float()

        return (style_ref_imgs, style_writer_ids, font_ref_imgs,
                sentence_curr_traj_embs,
                sentence_loss_mask_f, gt_traj_tensor, seq_chars, S_max, B)


    def _writer_set(self, ds) -> set:

        if hasattr(ds, "writer_paths"):
            return set(int(k) for k in ds.writer_paths.keys())
        if hasattr(ds, "items"):
            s = set()
            try:
                for tup in ds.items:
                    wid = tup[0]
                    try: s.add(int(wid))
                    except: pass
            except Exception:
                pass
            return s
        if hasattr(ds, "writer_style_pkl"):
            return set(int(k) for k in ds.writer_style_pkl.keys())

        return set(self.train_writer_ids or [])

    def _get_char_batch_for_writers(self, writer_ids: List[int]) -> Dict[str, Any]:

        if not hasattr(self, '_char_sample_indices'):
            import time
            print("[Trainer]  Building char sample index cache...", flush=True)
            t0 = time.perf_counter()

            self._char_sample_indices = {}


            if not hasattr(self.char_train_ds, 'index'):
                raise RuntimeError("[CRITICAL] char_train_ds.index is missing")

            if len(self.char_train_ds.index) == 0:
                raise RuntimeError("[CRITICAL] char_train_ds.index is empty")


            invalid_count = 0
            for idx, item in enumerate(self.char_train_ds.index):

                if not isinstance(item, (tuple, list)) or len(item) < 1:
                    invalid_count += 1
                    if invalid_count <= 5:
                        print(f"[WARN] Invalid char index item at {idx}: {type(item)}, {item}", flush=True)
                    continue

                try:
                    wid = int(item[0])
                except (ValueError, TypeError) as e:
                    invalid_count += 1
                    if invalid_count <= 5:
                        print(f"[WARN] Invalid wid in char index at {idx}: {item[0]}, error: {e}", flush=True)
                    continue

                if wid not in self._char_sample_indices:
                    self._char_sample_indices[wid] = []
                self._char_sample_indices[wid].append(idx)

            if invalid_count > 0:
                print(f"[WARN] Skipped {invalid_count} invalid char index items", flush=True)

            if len(self._char_sample_indices) == 0:
                raise RuntimeError("[CRITICAL] No valid char samples found after filtering")


            self._char_fallback_wids = list(self._char_sample_indices.keys())

            elapsed = time.perf_counter() - t0
            total_samples = sum(len(indices) for indices in self._char_sample_indices.values())
            print(f"[Trainer]  Char sample index cache built: {len(self._char_sample_indices)} writers, "
                  f"{total_samples} samples, {elapsed:.3f}s", flush=True)


        indices = []
        for wid in writer_ids:
            wid_list = self._char_sample_indices.get(wid)
            if wid_list and len(wid_list) > 0:

                indices.append(wid_list[random.randrange(len(wid_list))])
            else:

                fallback_wid = self._char_fallback_wids[random.randrange(len(self._char_fallback_wids))]
                fallback_list = self._char_sample_indices[fallback_wid]
                indices.append(fallback_list[random.randrange(len(fallback_list))])


        samples = [self.char_train_ds[idx] for idx in indices]
        return self.char_train_ds.collate_fn(samples)

    def _get_bigram_batch_for_writers(self, writer_ids: List[int]) -> Tuple[Dict[str, Any], bool]:

        if not hasattr(self, '_bigram_sample_items'):
            import time
            print("[Trainer]  Building bigram sample item cache...", flush=True)
            t0 = time.perf_counter()

            self._bigram_sample_items = {}


            if not hasattr(self.bigram_train_ds, 'index'):
                raise RuntimeError("[CRITICAL] bigram_train_ds.index is missing")

            if len(self.bigram_train_ds.index) == 0:
                raise RuntimeError("[CRITICAL] bigram_train_ds.index is empty")


            invalid_count = 0
            for item in self.bigram_train_ds.index:

                if not isinstance(item, (tuple, list)) or len(item) < 1:
                    invalid_count += 1
                    if invalid_count <= 5:
                        print(f"[WARN] Invalid bigram index item: {type(item)}, {item}", flush=True)
                    continue

                try:
                    wid = int(item[0])
                except (ValueError, TypeError) as e:
                    invalid_count += 1
                    if invalid_count <= 5:
                        print(f"[WARN] Invalid wid in bigram index: {item[0]}, error: {e}", flush=True)
                    continue

                if wid not in self._bigram_sample_items:
                    self._bigram_sample_items[wid] = []
                self._bigram_sample_items[wid].append(item)

            if invalid_count > 0:
                print(f"[WARN] Skipped {invalid_count} invalid bigram index items", flush=True)

            if len(self._bigram_sample_items) == 0:
                raise RuntimeError("[CRITICAL] No valid bigram samples found after filtering")


            self._bigram_fallback_wids = list(self._bigram_sample_items.keys())

            elapsed = time.perf_counter() - t0
            total_samples = sum(len(items) for items in self._bigram_sample_items.values())
            print(f"[Trainer]  Bigram sample item cache built: {len(self._bigram_sample_items)} writers, "
                  f"{total_samples} samples, {elapsed:.3f}s", flush=True)


        items = []
        fallback_occurred = False
        for wid in writer_ids:
            wid_list = self._bigram_sample_items.get(wid)
            if wid_list and len(wid_list) > 0:

                items.append(wid_list[random.randrange(len(wid_list))])
            else:

                fallback_wid = self._bigram_fallback_wids[random.randrange(len(self._bigram_fallback_wids))]
                fallback_list = self._bigram_sample_items[fallback_wid]
                items.append(fallback_list[random.randrange(len(fallback_list))])
                fallback_occurred = True


        return self.bigram_train_ds.collate_fn(items), fallback_occurred

    def _get_sent_batch_for_writers(self, writer_ids: List[int]) -> Dict[str, Any]:

        if not hasattr(self, '_sent_sample_indices'):
            import time
            print("[Trainer]  Building sentence sample index cache...", flush=True)
            t0 = time.perf_counter()

            self._sent_sample_indices = {}


            if not hasattr(self.sent_train_ds, 'index'):
                raise RuntimeError("[CRITICAL] sent_train_ds.index is missing")

            if len(self.sent_train_ds.index) == 0:
                raise RuntimeError("[CRITICAL] sent_train_ds.index is empty")


            invalid_count = 0
            for idx, item in enumerate(self.sent_train_ds.index):

                if not isinstance(item, (tuple, list)) or len(item) < 1:
                    invalid_count += 1
                    if invalid_count <= 5:
                        print(f"[WARN] Invalid sent index item at {idx}: {type(item)}, {item}", flush=True)
                    continue

                try:
                    wid = int(item[0])
                except (ValueError, TypeError) as e:
                    invalid_count += 1
                    if invalid_count <= 5:
                        print(f"[WARN] Invalid wid in sent index at {idx}: {item[0]}, error: {e}", flush=True)
                    continue

                if wid not in self._sent_sample_indices:
                    self._sent_sample_indices[wid] = []
                self._sent_sample_indices[wid].append(idx)

            if invalid_count > 0:
                print(f"[WARN] Skipped {invalid_count} invalid sent index items", flush=True)

            if len(self._sent_sample_indices) == 0:
                raise RuntimeError("[CRITICAL] No valid sentence samples found after filtering")


            self._sent_fallback_wids = list(self._sent_sample_indices.keys())

            elapsed = time.perf_counter() - t0
            total_samples = sum(len(indices) for indices in self._sent_sample_indices.values())
            print(f"[Trainer]  Sentence sample index cache built: {len(self._sent_sample_indices)} writers, "
                  f"{total_samples} samples, {elapsed:.3f}s", flush=True)


        indices = []
        missing_wids = []

        for wid in writer_ids:
            wid_list = self._sent_sample_indices.get(wid)
            if wid_list and len(wid_list) > 0:

                indices.append(wid_list[random.randrange(len(wid_list))])
            else:

                missing_wids.append(wid)


        if missing_wids:
            print(f"\n{'='*80}")
            print(f"[CRITICAL] _get_sent_batch_for_writers: Missing writers in sent_train_ds!")
            print(f"  Requested: {len(writer_ids)} writers")
            print(f"  Missing ({len(missing_wids)}): {missing_wids[:10]}...")
            print(f"  Available writers: {len(self._sent_sample_indices)}")
            print(f"  This breaks Stage3 Writer ID synchronization!")
            print(f"{'='*80}\n", flush=True)
            raise ValueError(f"[CRITICAL] {len(missing_wids)} writers not found in sent_train_ds. "
                           f"Cannot synchronize with Char batch. Missing: {missing_wids[:5]}")


        samples = [self.sent_train_ds[idx] for idx in indices]
        return self.sent_train_ds.collate_fn(samples)


    def _pick_unique_wids(self, pool: List[int], B: int) -> List[int]:
        pool = list(pool)
        if len(pool) >= B:
            return random.sample(pool, B)
        if len(pool) == 0:
            return []
        picks = []
        used = set()
        while len(picks) < B:
            w = random.choice(pool)
            if w not in used:
                picks.append(w); used.add(w)
            else:
                if len(used) == len(pool):
                    picks.append(w)
        return picks

    def _init_sync_writer_pools(self):

        cW = self._writer_set(self.char_train_ds)
        bW = self._writer_set(self.bigram_train_ds)
        sW = self._writer_set(self.sent_train_ds)

        self._pool_stage1 = sorted(cW)
        self._pool_stage2 = sorted(cW & bW)
        self._pool_stage3 = sorted(cW & bW & sW)
        if not self._pool_stage1: print_once("[WARN] Stage1 writer pool empty; fallback to dataloader.")
        if not self._pool_stage2: print_once("[WARN] Stage2 writer pool empty; fallback to dataloader.")
        if not self._pool_stage3: print_once("[WARN] Stage3 writer pool empty; fallback to dataloader.")
        self._cursor = {1: 0, 2: 0, 3: 0}

    def _next_wid(self, stage: int) -> int:
        pool = {1: self._pool_stage1, 2: self._pool_stage2, 3: self._pool_stage3}.get(stage, [])
        if not pool:
            return random.choice(self.train_writer_ids if self.train_writer_ids else [0])
        i = self._cursor[stage] % len(pool)
        self._cursor[stage] += 1
        return pool[i]


    def _update_stage_config(self, stage: str):
        if stage not in self.stage_configs:
            print(f"[WARN] Unknown stage '{stage}', using default config")
            return

        self.current_stage = stage
        config = self.stage_configs[stage]


        self.lr_scheduler.set_stage(stage, self.global_iter)


        new_temp = config['temperature']
        old_temp = self.style_loss_fn.supcon_loss.temperature
        if abs(new_temp - old_temp) > 1e-6:
            self.style_loss_fn.supcon_loss.temperature = new_temp
            self.style_loss_fn.supcon_loss.base_temperature = new_temp
            print(f"[Stage Config] {stage.upper()}: temperature {old_temp:.3f} → {new_temp:.3f}", flush=True)


        print(f"[Stage Config] {config['name']}: grad_clip_norm = {config['grad_clip']:.1f}", flush=True)


        weights_str = ', '.join([f"{k}={v:.2f}" for k, v in config['loss_weights'].items()])
        print(f"[Stage Config] {config['name']}: loss_weights = {{{weights_str}}}", flush=True)


        if self.context_encoder is not None:
            stage_num = {'stage1': 1, 'stage2': 2, 'stage3': 3}.get(stage, 1)
            dropout_key = f'CONTEXT_TOKEN_DROPOUT_STAGE{stage_num}'
            new_dropout = getattr(self.cfg.MODEL, dropout_key, 0.0)
            old_dropout = self.context_encoder.context_token_dropout
            if abs(new_dropout - old_dropout) > 1e-6:
                self.context_encoder.context_token_dropout = new_dropout

                if new_dropout < 0:
                    print(f"[Stage Config] {stage.upper()}: Context 완전 비활성화 (dropout={new_dropout:.1f})", flush=True)
                elif new_dropout == 0:
                    print(f"[Stage Config] {stage.upper()}: Context 100% 사용 (dropout=0.0)", flush=True)
                else:
                    print(f"[Stage Config] {stage.upper()}: context_token_dropout {old_dropout:.2f} → {new_dropout:.2f} ({(1-new_dropout)*100:.0f}% 사용)", flush=True)

    def _check_and_enable_vq(self):


        if self.encoder_vq_start_iter >= 0 and self.global_iter >= self.encoder_vq_start_iter and not self.encoder_vq_enabled:
            if self.use_context_as_content or self.font_encoder is None:

                if not hasattr(self, "_encoder_vq_skip_logged"):
                    self._encoder_vq_skip_logged = True
                    print(f"[VQ]  Encoder VQ skipped (USE_CONTEXT_AS_CONTENT=True or FontEncoder=None)", flush=True)
            else:
                print(f"\n{'='*80}")
                print(f"[VQ]  Encoder VQ 활성화: iter {self.global_iter}")
                print(f"  → Font Encoder VQ 학습 시작")
                print(f"{'='*80}\n", flush=True)
                if hasattr(self.font_encoder, 'use_vq'):
                    self.font_encoder.use_vq = True
                self.encoder_vq_enabled = True


        if self.decoder_vq_start_iter >= 0 and self.global_iter >= self.decoder_vq_start_iter and not self.decoder_vq_enabled:
            print(f"\n{'='*80}")
            print(f"[VQ]  Decoder VQ 활성화: iter {self.global_iter}")
            print(f"  → Decoder VQ 학습 시작")
            print(f"  → Warmup: {self.decoder_vq_warmup_iters} iters")
            print(f"{'='*80}\n", flush=True)
            self.decoder_vq_enabled = True

            hw_gen = self.hw_generator.module if isinstance(self.hw_generator, torch.nn.DataParallel) else self.hw_generator
            if hasattr(hw_gen, 'use_residual_vq'):
                hw_gen.use_residual_vq = True


    def load_checkpoint(self, ckpt_path: str, strict: bool = False):

        model_to_load = self.full_model.module if self.is_ddp else self.full_model
        step, best_val = load_checkpoint(
            ckpt_path, model_to_load, self.device, self.optimizer, strict=strict
        )
        self.global_iter = step
        if best_val is not None:
            self.best_val = best_val
        return step

    def train(self):
        self.full_model.train()
        max_iter = self.cfg.HYPERPARAMETER.MAX_ITER

        self._init_sync_writer_pools()


        char_end    = self.cfg.TRAIN.CHAR_STAGE_ITERS
        bigram_end  = self.cfg.TRAIN.BIGRAM_STAGE_ITERS


        if self.stage1_only:
            char_end = max_iter
            print(f"[STAGE1_ONLY] char_end 확장: {self.cfg.TRAIN.CHAR_STAGE_ITERS} → {max_iter}")

        num_loaders = 1 if self.stage1_only else 3
        print(f"[Trainer] Initializing DataLoaders ({self.W} workers × {num_loaders} loaders × {self.prefetch} prefetch)...", flush=True)
        init_start = time.perf_counter()

        print(f"[Trainer] Starting char_loader ({self.W} workers)...", flush=True)
        it_char   = iter(self.char_loader)
        print(f"[Trainer] char_loader ready ({time.perf_counter()-init_start:.1f}s)", flush=True)


        it_bigram = None
        it_sent = None
        if not self.stage1_only:
            print(f"[Trainer] Starting bigram_loader ({self.W} workers)...", flush=True)
            it_bigram = iter(self.bigram_loader)
            print(f"[Trainer] bigram_loader ready ({time.perf_counter()-init_start:.1f}s)", flush=True)

            print(f"[Trainer] Starting sent_loader ({self.W} workers)...", flush=True)
            it_sent   = iter(self.sent_loader)
        print(f"[Trainer] All DataLoaders ready! Total init time: {time.perf_counter()-init_start:.1f}s", flush=True)


        print("[Trainer] Starting training loop (fp32 mode)...", flush=True)
        first_fetch_start = time.perf_counter()

        it_char, stage1_done = self._train_stage1(it_char, char_end, max_iter, first_fetch_start)
        if stage1_done:
            return

        it_bigram = self._train_stage2(it_bigram, bigram_end, max_iter)
        it_char = self._train_stage3(it_char, max_iter)

    def _train_stage1(self, it_char, char_end, max_iter, first_fetch_start):

        self._update_stage_config('stage1')
        self.grad_accum_steps = self.grad_accum_steps_stage1
        self._nonfinite_count = 0
        self._nan_count = 0
        self._consec_grad_explosion = 0
        while self.global_iter < min(char_end, max_iter):
            self.global_iter += 1
            iter_start = time.perf_counter()


            self._check_and_enable_vq()


            if self.global_iter == 1:
                first_fetch_time = (time.perf_counter() - first_fetch_start) * 1000.0
                print(f"[Trainer]  First batch fetch took {first_fetch_time:.1f}ms ({first_fetch_time/1000:.1f}s)", flush=True)


            self.optimizer.zero_grad()
            total_loss_accum = 0.0
            r_last = None
            c_batch_last = None
            encoder_vq_loss_str = ""

            for accum_idx in range(self.grad_accum_steps):
                c_batch, it_char = self._fetch_next(it_char, self.char_loader)


                r = self._run_char_step(c_batch)
                step_loss = (
                    r["style"] * self.cfg.TRAIN.CHAR_STEP_STYLE_LOSS_WEIGHT +
                    r["char"]  * self.cfg.TRAIN.CHAR_STEP_CHAR_GEN_LOSS_WEIGHT
                ) / self.grad_accum_steps


                if self.encoder_vq_enabled:
                    encoder_vq_loss_char = r.get("encoder_vq_loss", None)
                    if encoder_vq_loss_char is not None:
                        if self.encoder_vq_warmup_iters > 0 and self.encoder_vq_start_iter >= 0:
                            warmup_progress = min(1.0, (self.global_iter - self.encoder_vq_start_iter) / self.encoder_vq_warmup_iters)
                            encoder_vq_weight = self.encoder_vq_loss_weight_target * warmup_progress
                        else:
                            encoder_vq_weight = self.encoder_vq_loss_weight_target
                        step_loss = step_loss + encoder_vq_loss_char * encoder_vq_weight / self.grad_accum_steps
                        if accum_idx == self.grad_accum_steps - 1 and encoder_vq_loss_char.item() > 0:
                            encoder_vq_loss_str = f" enc_vq={encoder_vq_loss_char.item():.4f}(w={encoder_vq_weight:.2f})"

                total_loss_accum += step_loss.item() * self.grad_accum_steps
                r_last = r
                c_batch_last = c_batch


                if self.use_amp:
                    self.scaler.scale(step_loss).backward()
                else:
                    step_loss.backward()

            r = r_last
            c_batch = c_batch_last
            total_loss = torch.tensor(total_loss_accum, device=self.device)


            if not torch.isfinite(total_loss).all():
                self._nan_count = getattr(self, '_nan_count', 0) + 1
                print(f"\n{'='*80}")
                print(f"[!!! NaN/Inf Detected] at iteration {self.global_iter} (consecutive: {self._nan_count})")
                print(f"  total_loss: {total_loss.item()}")
                print(f"  r['style']: {r['style'].item()}")
                print(f"  r['char']: {r['char'].item()}")
                print(f"{'='*80}\n")

                self._diagnose_nan_source()

                if self._nan_count >= 3:
                    if self._attempt_nan_recovery("Stage1"):
                        self.full_model.train()
                        continue
                    raise ValueError(
                        f"[Stage1] NaN detected {self._nan_count} times consecutively. "
                        f"Model corrupted. Auto-recovery failed.")

                print(f" Skipping batch due to NaN/Inf...")
                self.optimizer.zero_grad(set_to_none=True)
                continue
            else:
                self._nan_count = 0


            clip_val = self.stage_configs[self.current_stage]['grad_clip']
            GRAD_EXPLOSION_THRESHOLD = getattr(self.cfg.TRAIN, 'GRAD_EXPLOSION_THRESHOLD', 100.0)
            CONSEC_EXPLOSION_LIMIT = 50
            skip_optim = False

            if self.use_amp:
                if clip_val > 0.0:
                    self.scaler.unscale_(self.optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.full_model.parameters(), clip_val)
                    if grad_norm > clip_val * GRAD_EXPLOSION_THRESHOLD:
                        print(f"\n [Stage1] GRADIENT EXPLOSION: {grad_norm:.2e} >> skip batch at iter {self.global_iter}\n", flush=True)
                        skip_optim = True
                    elif grad_norm > clip_val * 10:
                        print(f"\n  [Stage1] Large gradient: {grad_norm:.2f} (clip={clip_val:.1f}) at iter {self.global_iter}\n")

                if skip_optim:
                    self.optimizer.zero_grad()
                    self.scaler.update()
                else:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
            else:
                if clip_val > 0.0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.full_model.parameters(), clip_val)
                    if grad_norm > clip_val * GRAD_EXPLOSION_THRESHOLD:
                        print(f"\n [Stage1] GRADIENT EXPLOSION: {grad_norm:.2e} >> skip batch at iter {self.global_iter}\n", flush=True)
                        skip_optim = True
                    elif grad_norm > clip_val * 10:
                        print(f"\n  [Stage1] Large gradient: {grad_norm:.2f} (clip={clip_val:.1f}) at iter {self.global_iter}\n")

                if skip_optim:
                    self.optimizer.zero_grad()
                else:
                    self.optimizer.step()


            self._consec_grad_explosion = getattr(self, '_consec_grad_explosion', 0)
            if skip_optim:
                self._consec_grad_explosion += 1
                if self._consec_grad_explosion >= CONSEC_EXPLOSION_LIMIT:
                    print(f"\n [Stage1] {CONSEC_EXPLOSION_LIMIT} consecutive gradient explosions! "
                          f"Pre-emptive recovery to prevent BN corruption...")
                    if self._attempt_nan_recovery("Stage1-PreEmptive"):
                        self.full_model.train()
                        continue
                    print(f" [Stage1] Pre-emptive recovery failed, continuing with caution...")
            else:
                self._consec_grad_explosion = 0


            current_lr = self.lr_scheduler.step(self.global_iter)

            elapsed_ms = (time.perf_counter() - iter_start) * 1000.0


            total_loss_val = total_loss.item()
            style_loss_val = r['style'].item()
            char_loss_val = r['char'].item()


            writer_info = ""
            if 'style_labels' in c_batch:
                wids = c_batch['style_labels'].cpu().numpy()
                unique_wids = np.unique(wids)
                writer_info = f"W={len(unique_wids)}/{len(wids)} "


            timing_str = f"elapsed={elapsed_ms:.1f}ms"


            lr_str = f" lr={current_lr:.2e}" if self.global_iter < 2000 or (self.global_iter - self.lr_scheduler.stage_start_iter) < 500 else ""


            alpha = self._get_2pass_alpha(self.global_iter)
            alpha_str = f" 2pass_α={alpha:.3f}" if alpha > 0.0 else ""

            print(f"[train][Stage1-Char] it={self.global_iter} {writer_info}"
                  f"total={total_loss_val:.4f} style={style_loss_val:.4f} char_gen={char_loss_val:.4f}{encoder_vq_loss_str}{alpha_str} {timing_str}{lr_str}", flush=True)


            if self.rank == 0 and self.global_iter >= self.cfg.TRAIN.SNAPSHOT_BEGIN and self.global_iter % self.cfg.TRAIN.SNAPSHOT_ITERS == 0:
                self.tb.add_scalar("stage1/total_loss", total_loss.item(), self.global_iter)
                self.tb.add_scalar("stage1/char_gen_loss", r["char"].item(), self.global_iter)
                self.tb.add_scalar("stage1/style_nce_loss", r["style"].item(), self.global_iter)


                if self.encoder_vq_enabled and r.get("encoder_vq_loss") is not None:
                    self.tb.add_scalar("stage1/encoder_vq_loss", r["encoder_vq_loss"].item(), self.global_iter)

                    encoder_vq_stats = r.get("encoder_vq_stats", {})
                    if 'font_vq_perplexity' in encoder_vq_stats:
                        self.tb.add_scalar("stage1/font_vq_perplexity", encoder_vq_stats['font_vq_perplexity'], self.global_iter)
                    if 'font_vq_usage' in encoder_vq_stats:
                        self.tb.add_scalar("stage1/font_vq_usage", encoder_vq_stats['font_vq_usage'], self.global_iter)


            if self.rank == 0 and (self.global_iter == self.cfg.TRAIN.STYLE_REF_VISUALIZATION_BEGIN or (self.global_iter > self.cfg.TRAIN.STYLE_REF_VISUALIZATION_BEGIN and self.global_iter % self.cfg.TRAIN.STYLE_REF_VISUALIZATION_ITERS == 0)):

                style_imgs_vis = c_batch["style_reference"]
                writer_ids_vis = c_batch["style_labels"]
                visualize_style_images_tb(self.tb, style_imgs_vis, writer_ids_vis, self.global_iter, tag="stage1/style_imgs")

                del style_imgs_vis, writer_ids_vis
                gc.collect()
                torch.cuda.empty_cache()


            emb_vis_begin = getattr(self.cfg.TRAIN, 'EMBEDDING_VIS_BEGIN', 5000)
            emb_vis_iters = getattr(self.cfg.TRAIN, 'EMBEDDING_VIS_ITERS', 10000)
            emb_vis_max_points = getattr(self.cfg.TRAIN, 'EMBEDDING_VIS_MAX_POINTS', 200)

            if self.rank == 0 and self.context_encoder is not None and (
                self.global_iter == emb_vis_begin or
                (self.global_iter > emb_vis_begin and self.global_iter % emb_vis_iters == 0)
            ):
                try:
                    with torch.no_grad():
                        seq_chars_vis = c_batch["seq_chars"].to(self.device).long()
                        font_ref_vis = c_batch["font_reference"].to(self.device)

                        content_embs_vis, _, _ = self.full_model.encode_content(
                            font_ref_vis, seq_chars=seq_chars_vis
                        )
                        text_mask_vis = (seq_chars_vis > 0)

                        context_embs_vis = self.context_encoder(
                            seq_chars_vis,
                            text_mask_vis,
                            apply_dropout=False,
                            disable_hard_dropout=True,
                        )


                        sentences_vis = []
                        for b in range(seq_chars_vis.size(0)):
                            chars = seq_chars_vis[b].cpu().tolist()
                            sent = "".join([chr(c) if c > 0 else "" for c in chars])
                            sentences_vis.append(sent)

                        visualize_content_vs_context_v3(
                            content_embs_vis, context_embs_vis, seq_chars_vis,
                            self.tb, self.global_iter,
                            sentences=sentences_vis,
                            max_points=min(100, emb_vis_max_points),
                            tag="stage1/emb_analysis"
                        )
                        del content_embs_vis, context_embs_vis, seq_chars_vis, font_ref_vis, text_mask_vis
                except Exception as e:
                    print(f"[WARN] Stage1 Embedding visualization failed: {e}", flush=True)
                gc.collect()
                torch.cuda.empty_cache()

            if self.rank == 0 and (self.global_iter == self.cfg.TRAIN.VISUALIZATION_BEGIN or (self.global_iter > self.cfg.TRAIN.VISUALIZATION_BEGIN and self.global_iter % self.cfg.TRAIN.VISUALIZATION_ITERS == 0)):
                num_show = 5
                font_char_strs = r["font_char_strs"]
                sel = random.sample(range(len(font_char_strs)), min(num_show, len(font_char_strs)))


                style_ref_imgs   = c_batch["style_reference"].to(self.device, non_blocking=True)
                style_writer_ids = c_batch["style_labels"].to(self.device, non_blocking=True)
                char_imgs        = c_batch["font_reference"].to(self.device, non_blocking=True)
                seq_chars        = c_batch["seq_chars"].to(self.device, non_blocking=True).int()


                idx1 = torch.zeros(seq_chars.size(0), 1, dtype=torch.long, device=self.device)
                res_inf = self._infer_for_batch(style_ref_imgs, char_imgs, style_writer_ids,
                                                seq_chars, indices=idx1)
                flat_gmms = self._flatten_gmms_for_chars(res_inf)


                if len(flat_gmms) == 0 or (sel and len(flat_gmms) < max(sel) + 1):
                    print(f"[WARN] Stage1 Char visualization skipped: flat_gmms length ({len(flat_gmms)}) < max(sel) ({max(sel) if sel else 'N/A'})", flush=True)
                    continue

                pred_list_sel = [flat_gmms[i] for i in sel]

                print_once(
                    f"[train][Stage1-Char] it={self.global_iter} "
                    f"res_inf_batches={len(res_inf)} flat_gmms_n={len(flat_gmms)} sel_n={len(pred_list_sel)}"
                )

                gt_for_show = [r["gt_char"][i, 0].detach().cpu() for i in sel]


                self.viz_executor.submit(
                    visualize_snapshot_chars,
                    tb_writer=self.tb,
                    pred_gmms=[r["pred_gmm"][i, 0, :, :] for i in sel],
                    gt_coords_list=gt_for_show,
                    characters=[font_char_strs[i] for i in sel],
                    step=self.global_iter,
                    font_dataset=self._font_panel(),
                    IMG_SIZE=self.cfg.ENV.IMG_H,
                    mode="train-char",
                    coord_space="delta",
                    infer_gmms=[g.detach().cpu() for g in pred_list_sel]
                )


            if self.rank == 0 and self.global_iter >= self.cfg.TRAIN.CHECKPOINT_BEGIN and self.global_iter % self.cfg.TRAIN.CHECKPOINT_ITERS == 0:
                save_path = os.path.join(self.ckpt_dir, f"ckpt_{self.global_iter}.pt")

                model_to_save = self.full_model.module if self.is_ddp else self.full_model
                save_checkpoint(save_path, model_to_save, self.optimizer,
                                self.global_iter, self.best_val,
                                config=getattr(self.cfg, "as_dict", lambda: None)(),
                                model_config=extract_model_config(self.cfg))


        if self.stage1_only:
            print(f"\n[STAGE1_ONLY] Stage 1 완료 (iter={self.global_iter}). 학습 종료.")
            if self.rank == 0 and self.tb is not None:
                self.tb.close()
            return it_char, True

        return it_char, False

    def _train_stage2(self, it_bigram, bigram_end, max_iter):

        self._update_stage_config('stage2')
        self.grad_accum_steps = self.grad_accum_steps_stage2
        vdl_weight = getattr(self.cfg.TRAIN, 'BIGRAM_STEP_VDL_LOSS_WEIGHT', 0.0)


        print("\n" + "="*80)
        print("[VALIDATION] Stage2 Writer ID Sync Check")
        print("="*80)
        b_test, _ = self._fetch_next(iter(self.bigram_loader), self.bigram_loader)
        b_test_wids = b_test["style_labels"].cpu().numpy().tolist()
        c_test = self._get_char_batch_for_writers(b_test_wids)
        c_test_wids = c_test["style_labels"].cpu().numpy().tolist()

        if b_test_wids == c_test_wids:
            print(f" Writer ID Match: Bigram={b_test_wids[:5]}... == Char")
        else:
            print(f" Writer ID Mismatch!")
            print(f"   Bigram: {b_test_wids}")
            print(f"   Char:   {c_test_wids}")
            raise ValueError("Stage2 Writer ID mismatch detected!")


        char_cache_size = len(self._char_sample_indices) if hasattr(self, '_char_sample_indices') and self._char_sample_indices else 0
        print(f" Char sample cache: {char_cache_size} writers")
        print("="*80 + "\n", flush=True)

        while self.global_iter < min(bigram_end, max_iter):
            self.global_iter += 1
            start_time = time.perf_counter()


            self._check_and_enable_vq()


            self.optimizer.zero_grad()
            total_loss_accum = 0.0
            rc_last = None
            rb_last = None
            c_batch_last = None
            b_batch_last = None
            decoder_vq_loss_str = ""
            encoder_vq_loss_str = ""
            vdl_loss_str = ""
            style_cache_for_accum = None
            style_loss_accum = 0.0


            time_fetch = 0.0
            time_forward = 0.0

            for accum_idx in range(self.grad_accum_steps):

                t_fetch_start = time.perf_counter()
                b_batch, it_bigram = self._fetch_next(it_bigram, self.bigram_loader)
                bigram_writer_ids = b_batch["style_labels"].cpu().tolist()
                c_batch = self._get_char_batch_for_writers(bigram_writer_ids)
                time_fetch += (time.perf_counter() - t_fetch_start)


                t_forward_start = time.perf_counter()
                if accum_idx == 0:
                    rc = self._run_char_step(c_batch)
                    style_cache_for_accum = {k: v.detach() for k, v in rc["style_cache"].items()}
                    style_loss_accum = rc["style"].detach().item()
                else:
                    rc = self._run_char_step(c_batch, style_cache=style_cache_for_accum)
                    rc["style"] = torch.tensor(0.0, device=self.device)

                rb = self._run_bigram_step(b_batch, style_cache=style_cache_for_accum)
                time_forward += (time.perf_counter() - t_forward_start)


                step_loss = (
                    rc["char"]  * self.cfg.TRAIN.BIGRAM_STEP_CHAR_GEN_LOSS_WEIGHT +
                    rc["style"] * self.cfg.TRAIN.BIGRAM_STEP_STYLE_LOSS_WEIGHT +
                    rb["char"]  * self.cfg.TRAIN.BIGRAM_STEP_BIGRAM_GEN_LOSS_WEIGHT
                ) / self.grad_accum_steps


                if vdl_weight > 0.0:
                    vdl_loss = rb.get("vdl_loss", None)
                    if vdl_loss is not None:
                        step_loss = step_loss + vdl_loss * vdl_weight / self.grad_accum_steps
                        if accum_idx == self.grad_accum_steps - 1:
                            vdl_loss_str = f" vdl={vdl_loss.detach().item():.4f}(w={vdl_weight:.2f})"


                if self.decoder_vq_enabled:
                    decoder_vq_c = rc.get("decoder_vq_loss", None)
                    decoder_vq_b = rb.get("decoder_vq_loss", None)
                    if decoder_vq_c is not None or decoder_vq_b is not None:
                        decoder_vq_total = (decoder_vq_c if decoder_vq_c is not None else 0.0) + \
                                          (decoder_vq_b if decoder_vq_b is not None else 0.0)
                        warmup_iters = getattr(self, "decoder_vq_warmup_iters", 0) or 0
                        start_iter = getattr(self, "decoder_vq_start_iter", -1)
                        target_w = getattr(self, "decoder_vq_loss_weight_target", 0.0)
                        if warmup_iters > 0 and start_iter >= 0:
                            warmup_progress = min(1.0, max(0.0, (self.global_iter - start_iter) / float(warmup_iters)))
                            decoder_vq_weight = target_w * warmup_progress
                        else:
                            decoder_vq_weight = target_w
                        step_loss = step_loss + decoder_vq_total * decoder_vq_weight / self.grad_accum_steps
                        if accum_idx == self.grad_accum_steps - 1:
                            decoder_vq_loss_str = f" dec_vq={decoder_vq_total.detach().item():.4f}(w={decoder_vq_weight:.2f})"

                if self.encoder_vq_enabled:
                    encoder_vq_c = rc.get("encoder_vq_loss", None)
                    encoder_vq_b = rb.get("encoder_vq_loss", None)
                    if encoder_vq_c is not None or encoder_vq_b is not None:
                        encoder_vq_total = (encoder_vq_c if encoder_vq_c is not None else 0.0) + \
                                          (encoder_vq_b if encoder_vq_b is not None else 0.0)
                        warmup_iters = getattr(self, "encoder_vq_warmup_iters", 0) or 0
                        start_iter = getattr(self, "encoder_vq_start_iter", -1)
                        target_w = getattr(self, "encoder_vq_loss_weight_target", 0.0)
                        if warmup_iters > 0 and start_iter >= 0:
                            warmup_progress = min(1.0, max(0.0, (self.global_iter - start_iter) / float(warmup_iters)))
                            encoder_vq_weight = target_w * warmup_progress
                        else:
                            encoder_vq_weight = target_w
                        step_loss = step_loss + encoder_vq_total * encoder_vq_weight / self.grad_accum_steps
                        if accum_idx == self.grad_accum_steps - 1:
                            encoder_vq_loss_str = f" enc_vq={encoder_vq_total.detach().item():.4f}(w={encoder_vq_weight:.2f})"

                total_loss_accum += step_loss.item() * self.grad_accum_steps
                rc_last = rc
                rb_last = rb
                c_batch_last = c_batch
                b_batch_last = b_batch


                t_backward_start = time.perf_counter()
                if self.use_amp:
                    self.scaler.scale(step_loss).backward()
                else:
                    step_loss.backward()

            time_backward = (time.perf_counter() - t_backward_start)

            rc = rc_last
            rb = rb_last
            c_batch = c_batch_last
            b_batch = b_batch_last
            total_loss = torch.tensor(total_loss_accum, device=self.device)


            if not torch.isfinite(total_loss).all():
                self._nan_count = getattr(self, '_nan_count', 0) + 1
                print(f"\n{'='*80}")
                print(f"[!!! NaN/Inf Detected] at iteration {self.global_iter} (consecutive: {self._nan_count})")
                print(f"  total_loss: {total_loss.item()}")
                print(f"  rc['char']: {rc['char'].item()}, rc['style']: {rc['style'].item()}")
                print(f"  rb['char']: {rb['char'].item()}")
                print(f"{'='*80}\n")

                self._diagnose_nan_source()

                if self._nan_count >= 3:
                    if self._attempt_nan_recovery("Stage2"):
                        self.full_model.train()
                        continue
                    raise ValueError(
                        f"[Stage2] NaN detected {self._nan_count} times consecutively. "
                        f"Model corrupted. Auto-recovery failed.")

                print(f" Skipping batch due to NaN/Inf...")
                self.optimizer.zero_grad(set_to_none=True)
                continue
            else:
                self._nan_count = 0


            t_optim_start = time.perf_counter()
            clip_val = self.stage_configs[self.current_stage]['grad_clip']
            GRAD_EXPLOSION_THRESHOLD = getattr(self.cfg.TRAIN, 'GRAD_EXPLOSION_THRESHOLD', 100.0)
            CONSEC_EXPLOSION_LIMIT = 50
            skip_optim = False

            if self.use_amp:
                if clip_val > 0.0:
                    self.scaler.unscale_(self.optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.full_model.parameters(), clip_val)
                    if grad_norm > clip_val * GRAD_EXPLOSION_THRESHOLD:
                        print(f"\n [Stage2] GRADIENT EXPLOSION: {grad_norm:.2e} >> skip batch at iter {self.global_iter}\n", flush=True)
                        skip_optim = True
                    elif grad_norm > clip_val * 10:
                        print(f"\n  [Stage2] Large gradient: {grad_norm:.2f} (clip={clip_val:.1f}) at iter {self.global_iter}\n")

                if skip_optim:
                    self.optimizer.zero_grad()
                    self.scaler.update()
                else:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
            else:
                if clip_val > 0.0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.full_model.parameters(), clip_val)
                    if grad_norm > clip_val * GRAD_EXPLOSION_THRESHOLD:
                        print(f"\n [Stage2] GRADIENT EXPLOSION: {grad_norm:.2e} >> skip batch at iter {self.global_iter}\n", flush=True)
                        skip_optim = True
                    elif grad_norm > clip_val * 10:
                        print(f"\n  [Stage2] Large gradient: {grad_norm:.2f} (clip={clip_val:.1f}) at iter {self.global_iter}\n")

                if skip_optim:
                    self.optimizer.zero_grad()
                else:
                    self.optimizer.step()


            self._consec_grad_explosion = getattr(self, '_consec_grad_explosion', 0)
            if skip_optim:
                self._consec_grad_explosion += 1
                if self._consec_grad_explosion >= CONSEC_EXPLOSION_LIMIT:
                    print(f"\n [Stage2] {CONSEC_EXPLOSION_LIMIT} consecutive gradient explosions! "
                          f"Pre-emptive recovery...")
                    if self._attempt_nan_recovery("Stage2-PreEmptive"):
                        self.full_model.train()
                        continue
            else:
                self._consec_grad_explosion = 0

            time_optim = (time.perf_counter() - t_optim_start)


            current_lr = self.lr_scheduler.step(self.global_iter)

            elapsed_ms = (time.perf_counter() - start_time) * 1000.0


            time_fetch_ms = time_fetch * 1000.0
            time_forward_ms = time_forward * 1000.0
            time_backward_ms = time_backward * 1000.0
            time_optim_ms = time_optim * 1000.0

            if (self.global_iter - self.lr_scheduler.stage_start_iter) < 20:
                profile_str = f" [fetch={time_fetch_ms:.0f}ms fwd={time_forward_ms:.0f}ms bwd={time_backward_ms:.0f}ms opt={time_optim_ms:.0f}ms]"
            else:
                profile_str = ""


            c_wids = c_batch['style_labels'].cpu().numpy()
            b_wids = b_batch['style_labels'].cpu().numpy()


            if not np.array_equal(c_wids, b_wids):
                print(f"\n  [Stage2][WRITER_MISMATCH] it={self.global_iter}")
                print(f"    Char wids: {c_wids[:5].tolist()}")
                print(f"    Bigram wids: {b_wids[:5].tolist()}\n", flush=True)


            unique_wids = np.unique(b_wids)
            writer_info = f"W={len(unique_wids)}/{len(b_wids)} "


            lr_str = f" lr={current_lr:.2e}" if (self.global_iter - self.lr_scheduler.stage_start_iter) < 500 else ""


            alpha = self._get_2pass_alpha(self.global_iter)
            alpha_str = f" 2pass_α={alpha:.3f}" if alpha > 0.0 else ""

            bigram_style_str = "cache" if style_cache_for_accum is not None else f"{rb['style'].item():.4f}"
            print(f"[train][Stage2-Char+Bigram] it={self.global_iter} {writer_info}"
                  f"total={total_loss.item():.4f} "
                  f"char(style={style_loss_accum:.4f}, char={rc['char'].item():.4f}) "
                  f"bigram(style={bigram_style_str}, bigram={rb['char'].item():.4f})"
                  f"{decoder_vq_loss_str}{encoder_vq_loss_str}{vdl_loss_str}{alpha_str} "
                  f"elapsed={elapsed_ms:.2f}ms{profile_str}{lr_str}", flush=True)


            if self.global_iter == self.cfg.TRAIN.STYLE_REF_VISUALIZATION_BEGIN or (self.global_iter > self.cfg.TRAIN.STYLE_REF_VISUALIZATION_BEGIN and self.global_iter % self.cfg.TRAIN.STYLE_REF_VISUALIZATION_ITERS == 0):

                style_imgs_vis = c_batch["style_reference"]
                writer_ids_vis = c_batch["style_labels"]
                visualize_style_images_tb(self.tb, style_imgs_vis, writer_ids_vis, self.global_iter, tag="stage2/style_imgs")

                del style_imgs_vis, writer_ids_vis
                gc.collect()
                torch.cuda.empty_cache()


            emb_vis_begin = getattr(self.cfg.TRAIN, 'EMBEDDING_VIS_BEGIN', 5000)
            emb_vis_iters = getattr(self.cfg.TRAIN, 'EMBEDDING_VIS_ITERS', 10000)
            emb_vis_max_points = getattr(self.cfg.TRAIN, 'EMBEDDING_VIS_MAX_POINTS', 200)

            if self.rank == 0 and self.context_encoder is not None and (
                self.global_iter == emb_vis_begin or
                (self.global_iter > emb_vis_begin and self.global_iter % emb_vis_iters == 0)
            ):
                try:
                    with torch.no_grad():

                        seq_chars_vis = c_batch["seq_chars"].to(self.device).long()
                        font_ref_vis = c_batch["font_reference"].to(self.device)

                        content_embs_vis, _, _ = self.full_model.encode_content(
                            font_ref_vis, seq_chars=seq_chars_vis
                        )
                        text_mask_vis = (seq_chars_vis > 0)

                        context_embs_vis = self.context_encoder(
                            seq_chars_vis,
                            text_mask_vis,
                            apply_dropout=False,
                            disable_hard_dropout=True,
                        )


                        sentences_vis = []
                        for b in range(seq_chars_vis.size(0)):
                            chars = seq_chars_vis[b].cpu().tolist()
                            sent = "".join([chr(c) if c > 0 else "" for c in chars])
                            sentences_vis.append(sent)

                        visualize_content_vs_context_v3(
                            content_embs_vis, context_embs_vis, seq_chars_vis,
                            self.tb, self.global_iter,
                            sentences=sentences_vis,
                            max_points=min(100, emb_vis_max_points),
                            tag="stage2/emb_analysis"
                        )
                        del content_embs_vis, context_embs_vis, seq_chars_vis, font_ref_vis, text_mask_vis
                except Exception as e:
                    print(f"[WARN] Stage2 Embedding visualization failed: {e}", flush=True)
                gc.collect()
                torch.cuda.empty_cache()

            if self.global_iter >= self.cfg.TRAIN.SNAPSHOT_BEGIN and self.global_iter % self.cfg.TRAIN.SNAPSHOT_ITERS == 0:
                self.tb.add_scalar("stage2/total_loss", total_loss.item(), self.global_iter)
                self.tb.add_scalar("stage2/char_loss", rc["char"].item(), self.global_iter)
                self.tb.add_scalar("stage2/bigram_loss", rb["char"].item(), self.global_iter)
                self.tb.add_scalar("stage2/style_loss", style_loss_accum, self.global_iter)
                if vdl_weight > 0.0 and rb.get("vdl_loss") is not None:
                    self.tb.add_scalar("stage2/vdl_loss", rb["vdl_loss"].item(), self.global_iter)
                    vdl_stats = rb.get("vdl_stats", {})
                    if isinstance(vdl_stats, dict):

                        if "centroid" in vdl_stats:
                            self.tb.add_scalar("stage2/vdl_centroid", vdl_stats["centroid"].item(), self.global_iter)
                        if "top" in vdl_stats:
                            self.tb.add_scalar("stage2/vdl_top", vdl_stats["top"].item(), self.global_iter)
                        if "bottom" in vdl_stats:
                            self.tb.add_scalar("stage2/vdl_bottom", vdl_stats["bottom"].item(), self.global_iter)

                        for key in ["delta_centroid_pred_avg", "delta_centroid_gt_avg",
                                     "centroid_pred_avg", "centroid_gt_avg"]:
                            if key in vdl_stats:
                                val = vdl_stats[key]
                                self.tb.add_scalar(f"stage2/vdl_{key}",
                                                   val.item() if torch.is_tensor(val) else float(val),
                                                   self.global_iter)


                if self.decoder_vq_enabled:
                    decoder_vq_c = rc.get("decoder_vq_loss", None)
                    decoder_vq_b = rb.get("decoder_vq_loss", None)
                    if decoder_vq_c is not None or decoder_vq_b is not None:
                        decoder_vq_val = (
                            (decoder_vq_c.item() if decoder_vq_c is not None else 0.0) +
                            (decoder_vq_b.item() if decoder_vq_b is not None else 0.0)
                        )
                        self.tb.add_scalar("stage2/decoder_vq_loss", decoder_vq_val, self.global_iter)


                    decoder_stats_c = rc.get("decoder_vq_stats", {})
                    decoder_stats_b = rb.get("decoder_vq_stats", {})
                    if 'decoder_vq_gate_mean' in decoder_stats_c:
                        self.tb.add_scalar("stage2/decoder_vq_gate_char", decoder_stats_c['decoder_vq_gate_mean'], self.global_iter)
                    if 'decoder_vq_gate_mean' in decoder_stats_b:
                        self.tb.add_scalar("stage2/decoder_vq_gate_bigram", decoder_stats_b['decoder_vq_gate_mean'], self.global_iter)

                if self.encoder_vq_enabled and not self.use_context_as_content:
                    encoder_vq_c = rc.get("encoder_vq_loss", None)
                    encoder_vq_b = rb.get("encoder_vq_loss", None)
                    if encoder_vq_c is not None or encoder_vq_b is not None:
                        encoder_vq_val = (
                            (encoder_vq_c.item() if encoder_vq_c is not None else 0.0) +
                            (encoder_vq_b.item() if encoder_vq_b is not None else 0.0)
                        )
                        self.tb.add_scalar("stage2/encoder_vq_loss", encoder_vq_val, self.global_iter)


                    encoder_stats_c = rc.get("encoder_vq_stats", {})
                    encoder_stats_b = rb.get("encoder_vq_stats", {})
                    if 'encoder_vq_recon' in encoder_stats_c:
                        self.tb.add_scalar("stage2/encoder_vq_recon_char", encoder_stats_c['encoder_vq_recon'], self.global_iter)
                    if 'encoder_vq_recon' in encoder_stats_b:
                        self.tb.add_scalar("stage2/encoder_vq_recon_bigram", encoder_stats_b['encoder_vq_recon'], self.global_iter)


            if not skip_optim and self.rank == 0 and (self.global_iter == self.cfg.TRAIN.VISUALIZATION_BEGIN or (self.global_iter > self.cfg.TRAIN.VISUALIZATION_BEGIN and self.global_iter % self.cfg.TRAIN.VISUALIZATION_ITERS == 0)):
                print(f"\n{'='*80}\n [Stage2-VIS] Starting visualization at iter {self.global_iter} (VIZ_BEGIN={self.cfg.TRAIN.VISUALIZATION_BEGIN}, VIZ_ITERS={self.cfg.TRAIN.VISUALIZATION_ITERS})\n{'='*80}\n", flush=True)

                pred_gmm_b = rb["pred_gmm"]
                gt_b       = rb["gt_traj"]
                Bb         = pred_gmm_b.size(0)

                if Bb > 0:
                    sel_b = random.sample(range(Bb), min(5, Bb))


                    Sb = pred_gmm_b.size(1)
                    curr_idx = 1 if Sb > 1 else 0


                    pred_sel = [pred_gmm_b[i, 1].detach().cpu() for i in sel_b]
                    gt_sel   = [gt_b[i, 1].detach().cpu()       for i in sel_b]


                    style_ref_imgs   = b_batch["style_reference"].to(self.device, non_blocking=True)
                    style_writer_ids = b_batch["style_labels"].to(self.device, non_blocking=True)
                    char_imgs        = b_batch["font_reference"].to(self.device, non_blocking=True)
                    seq_chars_full   = b_batch["seq_chars"].to(self.device, non_blocking=True).long()
                    pair_idx_all     = rb["pair_indices"].to(self.device, non_blocking=True).long()


                    B_inf = char_imgs.size(0)
                    bix_inf = torch.arange(B_inf, device=self.device).unsqueeze(1).expand(B_inf, 2)
                    seq_chars_bigram = seq_chars_full[bix_inf, pair_idx_all]


                    print_once(f"[DEBUG-BIGRAM-INF] char_imgs.shape={char_imgs.shape}, "
                              f"seq_chars_bigram.shape={seq_chars_bigram.shape}, "
                              f"seq_chars_full.shape={seq_chars_full.shape}, pair_idx_all.shape={pair_idx_all.shape}")


                    gt_trajs_b = b_batch["seq_trajs"].to(self.device, non_blocking=True) if "seq_trajs" in b_batch else None
                    use_gt_prev_test = getattr(self.cfg.TRAIN, 'USE_GT_PREV_FOR_INFERENCE', False)


                    res_inf = self._infer_for_batch(
                        style_ref_imgs, char_imgs, style_writer_ids,
                        seq_chars_bigram,
                        indices=pair_idx_all,
                        context_seq_chars=seq_chars_full,
                        gt_trajs=gt_trajs_b, use_gt_prev=use_gt_prev_test
                    )


                    if sel_b:
                        first_idx = sel_b[0]
                        gmm_list = res_inf[first_idx]["gmm"]
                        print_once(f"[DEBUG-BIGRAM-INF] res_inf[{first_idx}]['gmm'] has {len(gmm_list)} elements, "
                                  f"shapes: {[g.shape for g in gmm_list]}")


                    pred_list_sel_b = [res_inf[i]["gmm"][1].detach().cpu() for i in sel_b]


                    prev_pos_sel = pair_idx_all[sel_b, 0]
                    curr_pos_sel = pair_idx_all[sel_b, 1]
                    codes_mat    = seq_chars_full[sel_b]


                    prev_codes = [ int(codes_mat[j, int(prev_pos_sel[j])].item()) for j in range(len(sel_b)) ]
                    curr_codes = [ int(codes_mat[j, int(curr_pos_sel[j])].item()) for j in range(len(sel_b)) ]

                    if hasattr(self.bigram_train_ds, "decode_code"):
                        prev_chars = [ self.bigram_train_ds.decode_code(c) for c in prev_codes ]
                        curr_chars = [ self.bigram_train_ds.decode_code(c) for c in curr_codes ]
                    else:
                        prev_chars = [ chr(c) if c > 0 else " " for c in prev_codes ]
                        curr_chars = [ chr(c) if c > 0 else " " for c in curr_codes ]


                    char_list = [ f"{p}{c}" for p, c in zip(prev_chars, curr_chars) ]


                    bigram_context = char_list
                    print_once(f"[VIS-BIGRAM] sel_b={sel_b}, prev_pos={prev_pos_sel.tolist()}, "
                              f"curr_pos={curr_pos_sel.tolist()}, context={bigram_context}")


                    if len(sel_b) > 0:
                        idx_vis = sel_b[0]


                        gt_lens_from_batch = b_batch["seq_traj_lens"][idx_vis].cpu()
                        print_once(f"[DEBUG-BIGRAM-VIS] gt_lens_from_batch: {gt_lens_from_batch.tolist()}, "
                                  f"gt_b.shape: {gt_b.shape}, idx_vis: {idx_vis}")


                        gt_bigram = []
                        for s in range(2):
                            traj_s = gt_b[idx_vis, s]
                            traj_len = b_batch["seq_traj_lens"][idx_vis, s].item()
                            if traj_len > 0:
                                gt_bigram.append(traj_s[:traj_len].detach().cpu())
                            else:

                                dummy = torch.zeros(1, 6)
                                dummy[0, 5] = 1.0
                                gt_bigram.append(dummy)
                                print(f"[WARN-BIGRAM-VIS] s={s} empty trajectory, using dummy EOC", flush=True)


                        pred_bigram_full = pred_gmm_b[idx_vis].detach().cpu()
                        pred_bigram = [pred_bigram_full[0], pred_bigram_full[1]]


                        print_once(f"[DEBUG-BIGRAM-VIS] pred_gmm shape: prev={pred_bigram[0].shape}, curr={pred_bigram[1].shape}")


                        infer_bigram = res_inf[idx_vis]["gmm"]
                        print_once(f"[DEBUG-BIGRAM-VIS] infer_gmm len: {len(infer_bigram)}, "
                                  f"shapes: {[g.shape for g in infer_bigram]}")


                        bigram_str = char_list[0]
                        sentence_chars_bigram = list(bigram_str)


                        print(f" [Stage2-Bigram-VIS] Submitting visualization at iter {self.global_iter}, bigram='{bigram_str}' → TB tag: train-bigram/sentence_panel", flush=True)
                        future = self.viz_executor.submit(
                            visualize_snapshot_sentence,
                            gt_coords_list=gt_bigram,
                            pred_gmm=pred_bigram,
                            sentence_chars=sentence_chars_bigram,
                            step=self.global_iter,
                            font_dataset=self._font_panel(),
                            tb_writer=self.tb,
                            IMG_SIZE=self.cfg.ENV.IMG_H,
                            mode="train-bigram",
                            coord_space="delta",
                            infer_gmm=infer_bigram,
                            n_gram_window=self.n_gram_window,
                        )

                        def _check_viz_error(fut):
                            try:
                                fut.result()
                                print(f" [Stage2-Bigram-VIS] Visualization completed successfully at iter {self.global_iter}", flush=True)
                            except Exception as e:
                                print(f"\n [ERROR-BIGRAM-VIZ] Visualization failed at iter {self.global_iter}: {e}\n", flush=True)
                                import traceback
                                traceback.print_exc()
                        future.add_done_callback(_check_viz_error)


                pred_gmm_c     = rc["pred_gmm"]
                gt_char_c      = rc["gt_char"]
                font_char_strs = rc["font_char_strs"]
                Bc = pred_gmm_c.size(0)

                if Bc > 0:
                    sel_c = random.sample(range(Bc), min(5, Bc))


                    try:
                        style_ref_imgs   = c_batch["style_reference"].to(self.device, non_blocking=True)
                        style_writer_ids = c_batch["style_labels"].to(self.device, non_blocking=True)
                        char_imgs        = c_batch["font_reference"].to(self.device, non_blocking=True)
                        seq_chars        = c_batch["seq_chars"].to(self.device, non_blocking=True).int()
                        idx1 = torch.zeros(seq_chars.size(0), 1, dtype=torch.long, device=self.device)
                        res_inf = self._infer_for_batch(style_ref_imgs, char_imgs, style_writer_ids,
                                                        seq_chars, indices=idx1)
                        flat_gmms = self._flatten_gmms_for_chars(res_inf)


                        if len(flat_gmms) < max(sel_c) + 1:
                            print(f"[WARN] Char visualization skipped: flat_gmms length ({len(flat_gmms)}) < max(sel_c) ({max(sel_c)})", flush=True)
                        else:
                            pred_list_sel_c = [flat_gmms[i] for i in sel_c]

                            gt_for_show_c = [gt_char_c[i, 0].detach().cpu() for i in sel_c]


                            print(f" [Stage2-Char-VIS] Submitting visualization at iter {self.global_iter}, chars={[font_char_strs[i] for i in sel_c]} → TB tag: train-char/char_panel", flush=True)
                            future_char = self.viz_executor.submit(
                                visualize_snapshot_chars,
                                tb_writer=self.tb,
                                pred_gmms=[pred_gmm_c[i, 0].detach().cpu() for i in sel_c],
                                gt_coords_list=gt_for_show_c,
                                characters=[font_char_strs[i] for i in sel_c],
                                step=self.global_iter,
                                font_dataset=self._font_panel(),
                                IMG_SIZE=self.cfg.ENV.IMG_H,
                                mode="train-char",
                                coord_space="delta",
                                infer_gmms=[g.detach().cpu() for g in pred_list_sel_c],
                            )

                            def _check_char_viz_error(fut):
                                try:
                                    fut.result()
                                    print(f" [Stage2-Char-VIS] Visualization completed successfully at iter {self.global_iter}", flush=True)
                                except Exception as e:
                                    print(f"\n [ERROR-CHAR-VIZ] Visualization failed at iter {self.global_iter}: {e}\n", flush=True)
                                    import traceback
                                    traceback.print_exc()
                            future_char.add_done_callback(_check_char_viz_error)
                    except Exception as e:
                        print(f"\n [ERROR] Char inference/visualization failed at iter {self.global_iter}: {e}\n", flush=True)
                        import traceback
                        traceback.print_exc()


            if self.rank == 0 and self.global_iter >= self.cfg.TRAIN.CHECKPOINT_BEGIN and self.global_iter % self.cfg.TRAIN.CHECKPOINT_ITERS == 0:
                save_path = os.path.join(self.ckpt_dir, f"ckpt_{self.global_iter}.pt")

                model_to_save = self.full_model.module if self.is_ddp else self.full_model
                save_checkpoint(save_path, model_to_save, self.optimizer,
                                self.global_iter, self.best_val,
                                config=getattr(self.cfg, "as_dict", lambda: None)(),
                                model_config=extract_model_config(self.cfg))

        return it_bigram

    def _train_stage3(self, it_char, max_iter):

        self._update_stage_config('stage3')


        use_bigram_in_stage3 = getattr(self.cfg.TRAIN, 'STAGE3_USE_BIGRAM', False)


        bigram_vdl_weight = 0.0


        if use_bigram_in_stage3:
            print(f"[Stage3] Bigram enabled (Char writer-synchronized sampling, VDL handled by Sentence VDL)")


        print("\n" + "="*80)
        print("[VALIDATION] Stage3 Writer ID Sync Check (Char → Sent)")
        print("="*80)


        c_test, _ = self._fetch_next(iter(self.char_loader), self.char_loader)
        c_test_wids = c_test["style_labels"].cpu().numpy().tolist()


        test_sent_batch_size = self.cfg.TRAIN.SENT_BATCH_SIZE


        wids_first_half = c_test_wids[:test_sent_batch_size]
        wids_second_half = c_test_wids[test_sent_batch_size:test_sent_batch_size*2] if len(c_test_wids) >= test_sent_batch_size*2 else c_test_wids[:test_sent_batch_size]

        s_test_1 = self._get_sent_batch_for_writers(wids_first_half)
        s_test_1_wids = s_test_1["style_labels"].cpu().numpy().tolist()

        s_test_2 = self._get_sent_batch_for_writers(wids_second_half)
        s_test_2_wids = s_test_2["style_labels"].cpu().numpy().tolist()

        match_1 = (wids_first_half == s_test_1_wids)
        match_2 = (wids_second_half == s_test_2_wids)

        if match_1 and match_2:
            print(f" Writer ID Sync OK (Char → Sent, batch_size={test_sent_batch_size})")
            print(f"   Char[0:{test_sent_batch_size}]:  {wids_first_half[:5]}... → Sent: {s_test_1_wids[:5]}...")
            print(f"   Char[{test_sent_batch_size}:{test_sent_batch_size*2}]: {wids_second_half[:5]}... → Sent: {s_test_2_wids[:5]}...")
        else:
            print(f" Writer ID Sync FAILED!")
            print(f"   Char[0:{test_sent_batch_size}]:  {wids_first_half}")
            print(f"   Sent (1st):  {s_test_1_wids}")
            print(f"   Char[{test_sent_batch_size}:{test_sent_batch_size*2}]: {wids_second_half}")
            print(f"   Sent (2nd):  {s_test_2_wids}")
            raise ValueError(f"[CRITICAL] Stage3 Writer ID sync failed! Check _get_sent_batch_for_writers. batch_size={test_sent_batch_size}")


        char_cache_size = len(self._char_sample_indices) if hasattr(self, '_char_sample_indices') and self._char_sample_indices else 0
        sent_cache_size = len(self._sent_sample_indices) if hasattr(self, '_sent_sample_indices') and self._sent_sample_indices else 0
        print(f" Sample cache status: Char={char_cache_size} writers, Sent={sent_cache_size} writers")
        print("="*80 + "\n", flush=True)


        sent_accum_steps = self.grad_accum_steps_stage3_sent
        sent_batch_size = self.cfg.TRAIN.SENT_BATCH_SIZE
        char_batch_size = self.cfg.TRAIN.CHAR_BATCH_SIZE


        if char_batch_size != sent_batch_size * sent_accum_steps:
            print(f"[WARN] Stage3: Char batch({char_batch_size}) != Sent batch({sent_batch_size}) × accum({sent_accum_steps})")
            print(f"       Writer ID 동기화가 정확하지 않을 수 있습니다.", flush=True)

        while self.global_iter < max_iter:
            self.global_iter += 1
            start_time = time.perf_counter()


            mem_before = torch.cuda.memory_allocated() / 1024**3
            mem_reserved = torch.cuda.memory_reserved() / 1024**3


            if self.global_iter % 100 == 0:
                torch.cuda.empty_cache()


            self._check_and_enable_vq()

            self.optimizer.zero_grad()
            total_loss_accum = 0.0


            c_batch, it_char = self._fetch_next(it_char, self.char_loader)
            rc = self._run_char_step(c_batch)

            char_loss = (
                rc["style"] * self.cfg.TRAIN.SENT_STEP_STYLE_LOSS_WEIGHT +
                rc["char"]  * self.cfg.TRAIN.SENT_STEP_CHAR_GEN_LOSS_WEIGHT
            )


            if self.encoder_vq_enabled:
                encoder_vq_char = rc.get("encoder_vq_loss", None)
                if encoder_vq_char is not None:
                    if self.encoder_vq_warmup_iters > 0 and self.encoder_vq_start_iter >= 0:
                        warmup_progress = min(1.0, (self.global_iter - self.encoder_vq_start_iter) / self.encoder_vq_warmup_iters)
                        encoder_vq_weight = self.encoder_vq_loss_weight_target * warmup_progress
                    else:
                        encoder_vq_weight = self.encoder_vq_loss_weight_target
                    char_loss = char_loss + encoder_vq_char * encoder_vq_weight


            if self.use_amp:
                self.scaler.scale(char_loss).backward()
            else:
                char_loss.backward()

            total_loss_accum += char_loss.item()


            char_writer_ids = c_batch["style_labels"].cpu().numpy().tolist()
            actual_char_batch = len(char_writer_ids)


            style_cache_char = rc.get("style_cache")


            is_viz_iter = self.rank == 0 and (
                (self.global_iter == self.cfg.TRAIN.VISUALIZATION_BEGIN) or
                (self.global_iter > self.cfg.TRAIN.VISUALIZATION_BEGIN and
                 self.global_iter % self.cfg.TRAIN.VISUALIZATION_ITERS == 0) or
                getattr(self, '_warmup_viz', False)
            )


            viz_char_data = None
            if is_viz_iter:
                viz_char_data = {
                    "pred_gmm": rc["pred_gmm"].detach().cpu() if "pred_gmm" in rc else None,
                    "gt_char": rc["gt_char"].detach().cpu() if "gt_char" in rc else None,
                    "font_char_strs": rc.get("font_char_strs"),

                    "style_reference": c_batch["style_reference"].clone(),
                    "style_labels": c_batch["style_labels"].clone(),
                    "font_reference": c_batch["font_reference"].clone(),
                    "seq_chars": c_batch["seq_chars"].clone(),
                }


            if "pred_gmm" in rc:
                del rc["pred_gmm"]
            if "gt_char" in rc:
                del rc["gt_char"]
            if "loss_mask" in rc:
                del rc["loss_mask"]


            for key in ["curr_trajs", "gt_trajs", "prev_trajs", "char_imgs", "style_imgs"]:
                if key in c_batch:
                    del c_batch[key]

            del char_loss


            torch.cuda.empty_cache()


            bigram_loss_str = ""
            if use_bigram_in_stage3:


                b_batch, fallback_occurred = self._get_bigram_batch_for_writers(char_writer_ids)

                if fallback_occurred:


                    if not hasattr(self, '_stage3_bigram_fallback_count'):
                        self._stage3_bigram_fallback_count = 0
                    self._stage3_bigram_fallback_count += 1

                    if self.rank == 0 and self.global_iter % 1000 == 0:
                        print(f"[WARN] Stage3 iter={self.global_iter}: Bigram fallback occurred "
                              f"(total: {self._stage3_bigram_fallback_count}), skipping bigram step", flush=True)


                    if self._stage3_bigram_fallback_count > 100 and self.global_iter % 5000 == 0:
                        print(f"\n{'='*80}")
                        print(f"[!!! WARNING] Stage3 Bigram fallback occurred {self._stage3_bigram_fallback_count} times!")
                        print(f"  This may indicate missing bigram data for many writers.")
                        print(f"  Consider: (1) Generating more bigram data, or (2) Disabling Stage3 bigram")
                        print(f"{'='*80}\n", flush=True)


                    rb = None
                    bigram_loss_str = " bigram=SKIPPED(fallback)"
                else:


                    style_cache_bigram = {
                        k: v.detach() if isinstance(v, torch.Tensor) else v
                        for k, v in style_cache_char.items()
                    } if style_cache_char is not None else None

                    rb = self._run_bigram_step(b_batch, style_cache=style_cache_bigram)

                    bigram_loss = (
                        rb["char"] * getattr(self.cfg.TRAIN, 'STAGE3_BIGRAM_STEP_BIGRAM_GEN_LOSS_WEIGHT', 0.5)
                    )


                    if bigram_vdl_weight > 0.0:
                        bigram_vdl_loss = rb.get("vdl_loss", None)
                        if bigram_vdl_loss is not None:
                            bigram_loss = bigram_loss + bigram_vdl_loss * bigram_vdl_weight
                            bigram_loss_str = f" bigram={rb['char'].item():.4f}+vdl({bigram_vdl_loss.item():.4f})"
                        else:
                            bigram_loss_str = f" bigram={rb['char'].item():.4f}"
                    else:
                        bigram_loss_str = f" bigram={rb['char'].item():.4f}"


                    if self.use_amp:
                        self.scaler.scale(bigram_loss).backward()
                    else:
                        bigram_loss.backward()

                    total_loss_accum += bigram_loss.item()


                    del bigram_loss
                    del style_cache_bigram
                    if "pred_gmm" in rb:
                        del rb["pred_gmm"]
                    for key in ["curr_trajs", "gt_trajs", "char_imgs"]:
                        if key in b_batch:
                            del b_batch[key]
                    torch.cuda.empty_cache()


            rs_last = None
            s_batch_last = None
            sent_loss_accum = 0.0
            vdl_loss_last = None
            vdl_stats_last = None
            if style_cache_char is None:
                raise RuntimeError(f"[Stage3] style_cache is None at iter {self.global_iter}. Check _run_char_step return value.")


            sent_batch_size = self.cfg.TRAIN.SENT_BATCH_SIZE


            expected_accum_steps = self.grad_accum_steps_stage3_sent
            max_possible_accum = actual_char_batch // sent_batch_size

            if expected_accum_steps > max_possible_accum:
                print(f"[WARN] Stage3 iter={self.global_iter}: sent_accum_steps={expected_accum_steps} > max possible={max_possible_accum} (Char {actual_char_batch} / Sent {sent_batch_size})", flush=True)
                expected_accum_steps = max(1, max_possible_accum)

            if expected_accum_steps == 0:
                print(f"[ERROR] Stage3 iter={self.global_iter}: expected_accum_steps=0! Char batch={actual_char_batch}, Sent batch={sent_batch_size}", flush=True)
                raise RuntimeError(f"Stage3 accumulation steps cannot be 0. Check batch sizes.")

            for sent_accum_idx in range(expected_accum_steps):

                start_idx = sent_accum_idx * sent_batch_size
                end_idx = min(start_idx + sent_batch_size, actual_char_batch)
                wids_slice = char_writer_ids[start_idx:end_idx]


                if len(wids_slice) < sent_batch_size:
                    needed = sent_batch_size - len(wids_slice)
                    wids_slice = wids_slice + char_writer_ids[:needed]
                    print(f"[WARN] Stage3 iter={self.global_iter} accum_idx={sent_accum_idx}: Padded {needed} writers from start (got {end_idx-start_idx}/{sent_batch_size})", flush=True)


                s_batch = self._get_sent_batch_for_writers(wids_slice)


                s_batch_wids = s_batch["style_labels"].cpu().numpy().tolist()

                if wids_slice != s_batch_wids:
                    print(f"\n{'='*80}")
                    print(f"[!!! CRITICAL] Stage3 Writer ID Mismatch at iter={self.global_iter}, accum_idx={sent_accum_idx}")
                    print(f"  Char wids (slice):  {wids_slice[:10]}...")
                    print(f"  Sent wids (batch):  {s_batch_wids[:10]}...")
                    print(f"  This breaks style_cache synchronization!")
                    print(f"{'='*80}\n", flush=True)
                    raise ValueError(f"[CRITICAL] Stage3 Writer ID mismatch! style_cache cannot be shared. "
                                   f"Char: {wids_slice[:5]}, Sent: {s_batch_wids[:5]}")


                actual_slice_len = min(end_idx - start_idx, sent_batch_size)
                if actual_slice_len < sent_batch_size:

                    style_cache_slice = {}
                    for k, v in style_cache_char.items():

                        part1 = v[start_idx:end_idx]
                        needed = sent_batch_size - part1.size(0)
                        if needed > 0:
                            part2 = v[:needed]
                            style_cache_slice[k] = torch.cat([part1, part2], dim=0).detach()
                        else:
                            style_cache_slice[k] = part1.detach()
                else:

                    style_cache_slice = {
                        k: v[start_idx:end_idx].detach()
                        for k, v in style_cache_char.items()
                    }


                B_sent = s_batch["seq_trajs"].size(0)
                S_sent = s_batch["seq_trajs"].size(1)
                T_sent = s_batch["seq_trajs"].size(2)
                seq_lens = s_batch["seq_lengths"]
                traj_lens = s_batch["seq_traj_lens"]


                rs = self._run_sentence_step(s_batch, style_cache=style_cache_slice)


                sent_loss = rs["sent"] * self.cfg.TRAIN.SENT_STEP_SENT_GEN_LOSS_WEIGHT / expected_accum_steps


                vdl_weight_dynamic = self._get_sent_vdl_weight(self.global_iter)
                if vdl_weight_dynamic > 0.0:
                    vdl_loss = rs.get("vdl_loss", None)
                    if vdl_loss is not None:
                        sent_loss = sent_loss + vdl_loss * vdl_weight_dynamic / expected_accum_steps
                        if sent_accum_idx == expected_accum_steps - 1:
                            vdl_loss_last = vdl_loss.detach()
                            vdl_stats_last = rs.get("vdl_stats", {})
                            vdl_stats_last['weight'] = vdl_weight_dynamic


                if self.decoder_vq_enabled:
                    decoder_vq_sent = rs.get("decoder_vq_loss", None)
                    if decoder_vq_sent is not None:
                        if self.decoder_vq_warmup_iters > 0 and self.decoder_vq_start_iter >= 0:
                            warmup_progress = min(1.0, (self.global_iter - self.decoder_vq_start_iter) / self.decoder_vq_warmup_iters)
                            decoder_vq_weight = self.decoder_vq_loss_weight_target * warmup_progress
                        else:
                            decoder_vq_weight = self.decoder_vq_loss_weight_target
                        sent_loss = sent_loss + decoder_vq_sent * decoder_vq_weight / expected_accum_steps


                if self.encoder_vq_enabled:
                    encoder_vq_sent = rs.get("encoder_vq_loss", None)
                    if encoder_vq_sent is not None:
                        if self.encoder_vq_warmup_iters > 0 and self.encoder_vq_start_iter >= 0:
                            warmup_progress = min(1.0, (self.global_iter - self.encoder_vq_start_iter) / self.encoder_vq_warmup_iters)
                            encoder_vq_weight = self.encoder_vq_loss_weight_target * warmup_progress
                        else:
                            encoder_vq_weight = self.encoder_vq_loss_weight_target
                        sent_loss = sent_loss + encoder_vq_sent * encoder_vq_weight / expected_accum_steps


                if self.use_amp:
                    self.scaler.scale(sent_loss).backward()
                else:
                    sent_loss.backward()

                sent_loss_accum += sent_loss.item() * expected_accum_steps


                if sent_accum_idx < expected_accum_steps - 1:

                    if "pred_gmm_sent" in rs:
                        del rs["pred_gmm_sent"]
                    if "gt_traj_list" in rs:
                        del rs["gt_traj_list"]
                    for key in ["seq_trajs", "font_reference", "style_reference"]:
                        if key in s_batch:
                            del s_batch[key]
                    del rs, s_batch, sent_loss, style_cache_slice
                else:

                    rs_last = rs
                    s_batch_last = s_batch


            if rs_last is None or s_batch_last is None:
                raise RuntimeError(f"[Stage3] iter={self.global_iter}: Sentence loop not executed! expected_accum_steps={expected_accum_steps}")

            rs = rs_last
            s_batch = s_batch_last
            total_loss_accum += sent_loss_accum
            total_loss = torch.tensor(total_loss_accum, device=self.device)


            if not torch.isfinite(total_loss).all():
                self._nan_count = getattr(self, '_nan_count', 0) + 1
                print(f"\n{'='*80}")
                print(f"[!!! NaN/Inf Detected] at iteration {self.global_iter} (consecutive: {self._nan_count})")
                print(f"  total_loss: {total_loss.item()}")
                print(f"  rc['style']: {rc['style'].item()}")
                print(f"  rc['char']: {rc['char'].item()}")
                print(f"  rs['sent']: {rs['sent'].item()}")
                print(f"{'='*80}\n")

                self._diagnose_nan_source()

                if self._nan_count >= 3:
                    if self._attempt_nan_recovery("Stage3"):
                        self.full_model.train()
                        continue
                    raise ValueError(
                        f"[Stage3] NaN detected {self._nan_count} times consecutively. "
                        f"Model corrupted. Auto-recovery failed.")

                print(f" Skipping batch due to NaN/Inf...")
                self.optimizer.zero_grad(set_to_none=True)
                del total_loss
                continue
            else:
                self._nan_count = 0


            clip_val = self.stage_configs[self.current_stage]['grad_clip']
            GRAD_EXPLOSION_THRESHOLD = getattr(self.cfg.TRAIN, 'GRAD_EXPLOSION_THRESHOLD', 100.0)
            CONSEC_EXPLOSION_LIMIT = 50
            skip_optim = False

            if self.use_amp:
                if clip_val > 0.0:
                    self.scaler.unscale_(self.optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.full_model.parameters(), clip_val)
                    if grad_norm > clip_val * GRAD_EXPLOSION_THRESHOLD:
                        print(f"\n [Stage3] GRADIENT EXPLOSION: {grad_norm:.2e} >> skip batch at iter {self.global_iter}\n", flush=True)
                        skip_optim = True
                    elif grad_norm > clip_val * 10:
                        print(f"\n  [Stage3] Large gradient: {grad_norm:.2f} (clip={clip_val:.1f}) at iter {self.global_iter}\n")

                if skip_optim:
                    self.optimizer.zero_grad()
                    self.scaler.update()
                else:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
            else:
                if clip_val > 0.0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.full_model.parameters(), clip_val)
                    if grad_norm > clip_val * GRAD_EXPLOSION_THRESHOLD:
                        print(f"\n [Stage3] GRADIENT EXPLOSION: {grad_norm:.2e} >> skip batch at iter {self.global_iter}\n", flush=True)
                        skip_optim = True
                    elif grad_norm > clip_val * 10:
                        mem_peak = torch.cuda.max_memory_allocated() / 1024**3
                        mem_current = torch.cuda.memory_allocated() / 1024**3
                        print(f"\n  [Stage3] Large gradient: {grad_norm:.2f} (clip={clip_val:.1f}) at iter {self.global_iter}")
                        print(f"    Memory: current={mem_current:.2f}GB peak={mem_peak:.2f}GB\n", flush=True)

                if skip_optim:
                    self.optimizer.zero_grad()
                else:
                    self.optimizer.step()


            self._consec_grad_explosion = getattr(self, '_consec_grad_explosion', 0)
            if skip_optim:
                self._consec_grad_explosion += 1
                if self._consec_grad_explosion >= CONSEC_EXPLOSION_LIMIT:
                    print(f"\n [Stage3] {CONSEC_EXPLOSION_LIMIT} consecutive gradient explosions! "
                          f"Pre-emptive recovery...")
                    if self._attempt_nan_recovery("Stage3-PreEmptive"):
                        self.full_model.train()
                        continue
            else:
                self._consec_grad_explosion = 0


            self.optimizer.zero_grad(set_to_none=True)


            if self.global_iter % 10 == 0:
                torch.cuda.empty_cache()


            current_lr = self.lr_scheduler.step(self.global_iter)

            elapsed_ms = (time.perf_counter() - start_time) * 1000.0


            c_wids_full = c_batch['style_labels'].cpu().numpy()
            unique_wids = np.unique(c_wids_full)
            writer_info = f"W={len(unique_wids)}/{len(c_wids_full)} "


            lr_str = f" lr={current_lr:.2e}" if (self.global_iter - self.lr_scheduler.stage_start_iter) < 500 else ""


            vq_str = ""
            if self.decoder_vq_enabled or (self.encoder_vq_enabled and not self.use_context_as_content):

                decoder_vq_c = rc.get("decoder_vq_loss", None)
                decoder_vq_s = rs.get("decoder_vq_loss", None)
                encoder_vq_c = rc.get("encoder_vq_loss", None)
                encoder_vq_s = rs.get("encoder_vq_loss", None)


                decoder_vq_val = 0.0
                if decoder_vq_c is not None:
                    decoder_vq_val += decoder_vq_c.item()
                if decoder_vq_s is not None:
                    decoder_vq_val += decoder_vq_s.item()

                encoder_vq_val = 0.0
                if encoder_vq_c is not None:
                    encoder_vq_val += encoder_vq_c.item()
                if encoder_vq_s is not None:
                    encoder_vq_val += encoder_vq_s.item()

                if self.decoder_vq_enabled and decoder_vq_val > 0:
                    vq_str += f" dec_vq={decoder_vq_val:.4f}"
                if (self.encoder_vq_enabled and not self.use_context_as_content) and encoder_vq_val > 0:
                    vq_str += f" enc_vq={encoder_vq_val:.4f}"


            alpha = self._get_2pass_alpha(self.global_iter)
            alpha_str = f" 2pass_α={alpha:.3f}" if alpha > 0.0 else ""


            vdl_str = ""
            if vdl_loss_last is not None:
                vdl_weight_current = vdl_stats_last.get('weight', 0.0) if vdl_stats_last else 0.0
                vdl_str = f" sent_vdl={vdl_loss_last.item():.4f}(w={vdl_weight_current:.3f})"


            stage_name = "Char+Bigram+Sentence" if use_bigram_in_stage3 else "Char+Sentence"

            print(
                f"[train][Stage3-{stage_name}] it={self.global_iter} {writer_info}"
                f"total={total_loss.item():.4f} "
                f"char={rc['char'].item():.4f}{bigram_loss_str} "
                f"sent={rs['sent'].item():.4f} "
                f"style={rc['style'].item():.4f}{vq_str}{vdl_str}{alpha_str} elapsed={elapsed_ms:.2f}ms{lr_str}",
                flush=True
            )


            if self.global_iter == self.cfg.TRAIN.STYLE_REF_VISUALIZATION_BEGIN or (self.global_iter > self.cfg.TRAIN.STYLE_REF_VISUALIZATION_BEGIN and self.global_iter % self.cfg.TRAIN.STYLE_REF_VISUALIZATION_ITERS == 0):
                print(f"[DEBUG] Stage3 style visualization triggered at iter={self.global_iter}", flush=True)

                style_imgs_vis = c_batch["style_reference"]
                writer_ids_vis = c_batch["style_labels"]
                visualize_style_images_tb(self.tb, style_imgs_vis, writer_ids_vis, self.global_iter, tag="stage3/style_imgs_char")


                style_imgs_sent_vis = s_batch["style_reference"]
                writer_ids_sent_vis = s_batch["style_labels"]
                visualize_style_images_tb(self.tb, style_imgs_sent_vis, writer_ids_sent_vis, self.global_iter, tag="stage3/style_imgs_sent")


                del style_imgs_vis, writer_ids_vis, style_imgs_sent_vis, writer_ids_sent_vis
                gc.collect()
                torch.cuda.empty_cache()


            emb_vis_begin = getattr(self.cfg.TRAIN, 'EMBEDDING_VIS_BEGIN', 5000)
            emb_vis_iters = getattr(self.cfg.TRAIN, 'EMBEDDING_VIS_ITERS', 10000)
            emb_vis_max_points = getattr(self.cfg.TRAIN, 'EMBEDDING_VIS_MAX_POINTS', 200)

            if self.rank == 0 and self.context_encoder is not None and (
                self.global_iter == emb_vis_begin or
                (self.global_iter > emb_vis_begin and self.global_iter % emb_vis_iters == 0)
            ):
                print(f"[DEBUG] Content vs Context Embedding visualization at iter={self.global_iter}", flush=True)
                try:
                    with torch.no_grad():

                        seq_chars_vis = s_batch["seq_chars"].to(self.device).long()
                        font_ref_vis = s_batch["font_reference"].to(self.device)


                        content_embs_vis, _, _ = self.full_model.encode_content(
                            font_ref_vis, seq_chars=seq_chars_vis
                        )


                        text_mask_vis = (seq_chars_vis > 0)
                        context_embs_vis = self.context_encoder(
                            seq_chars_vis,
                            text_mask_vis,
                            apply_dropout=False,
                            disable_hard_dropout=True,
                        )


                        sentences_vis = []
                        for b in range(seq_chars_vis.size(0)):
                            chars = seq_chars_vis[b].cpu().tolist()
                            sent = "".join([chr(c) if c > 0 else "" for c in chars])
                            sentences_vis.append(sent)

                        visualize_content_vs_context_v3(
                            content_embs_vis,
                            context_embs_vis,
                            seq_chars_vis,
                            self.tb,
                            self.global_iter,
                            sentences=sentences_vis,
                            max_points=min(100, emb_vis_max_points),
                            tag="stage3/emb_analysis"
                        )


                        del content_embs_vis, context_embs_vis, seq_chars_vis, font_ref_vis, text_mask_vis
                except Exception as e:
                    print(f"[WARN] Embedding visualization failed: {e}", flush=True)

                gc.collect()
                torch.cuda.empty_cache()

            if (self.global_iter >= self.cfg.TRAIN.SNAPSHOT_BEGIN and self.global_iter % self.cfg.TRAIN.SNAPSHOT_ITERS == 0) or self._warmup_viz:
                self.tb.add_scalar("stage3/total_loss", total_loss.item(), self.global_iter)
                self.tb.add_scalar("stage3/char_loss", rc["char"].item(), self.global_iter)
                self.tb.add_scalar("stage3/sent_loss", rs["sent"].item(), self.global_iter)
                self.tb.add_scalar("stage3/style_loss", rc["style"].item(), self.global_iter)


                if use_bigram_in_stage3 and 'rb' in locals() and rb is not None:
                    self.tb.add_scalar("stage3/bigram_loss", rb["char"].item(), self.global_iter)


                if vdl_loss_last is not None:
                    vdl_weight_current = self._get_sent_vdl_weight(self.global_iter)
                    self.tb.add_scalar("stage3/sent_vdl_loss", vdl_loss_last.item(), self.global_iter)
                    self.tb.add_scalar("stage3/sent_vdl_weight", vdl_weight_current, self.global_iter)

                    if isinstance(vdl_stats_last, dict):

                        if "centroid" in vdl_stats_last:
                            self.tb.add_scalar("stage3/sent_vdl_centroid", vdl_stats_last["centroid"].item(), self.global_iter)
                        if "top" in vdl_stats_last:
                            self.tb.add_scalar("stage3/sent_vdl_top", vdl_stats_last["top"].item(), self.global_iter)
                        if "bottom" in vdl_stats_last:
                            self.tb.add_scalar("stage3/sent_vdl_bottom", vdl_stats_last["bottom"].item(), self.global_iter)

                        for key in ["delta_centroid_pred_avg", "delta_centroid_gt_avg",
                                     "centroid_pred_avg", "centroid_gt_avg"]:
                            if key in vdl_stats_last:
                                val = vdl_stats_last[key]
                                self.tb.add_scalar(f"stage3/sent_vdl_{key}",
                                                   val.item() if torch.is_tensor(val) else float(val),
                                                   self.global_iter)


                if self.decoder_vq_enabled:
                    decoder_vq_c = rc.get("decoder_vq_loss", None)
                    decoder_vq_s = rs.get("decoder_vq_loss", None)
                    if decoder_vq_c is not None or decoder_vq_s is not None:
                        decoder_vq_val = (
                            (decoder_vq_c.item() if decoder_vq_c is not None else 0.0) +
                            (decoder_vq_s.item() if decoder_vq_s is not None else 0.0)
                        )
                        self.tb.add_scalar("stage3/decoder_vq_loss", decoder_vq_val, self.global_iter)


                    decoder_stats_c = rc.get("decoder_vq_stats", {})
                    decoder_stats_s = rs.get("decoder_vq_stats", {})
                    if 'decoder_vq_gate_mean' in decoder_stats_c:
                        self.tb.add_scalar("stage3/decoder_vq_gate_char", decoder_stats_c['decoder_vq_gate_mean'], self.global_iter)
                    if 'decoder_vq_gate_mean' in decoder_stats_s:
                        self.tb.add_scalar("stage3/decoder_vq_gate_sent", decoder_stats_s['decoder_vq_gate_mean'], self.global_iter)

                if self.encoder_vq_enabled and not self.use_context_as_content:
                    encoder_vq_c = rc.get("encoder_vq_loss", None)
                    encoder_vq_s = rs.get("encoder_vq_loss", None)
                    if encoder_vq_c is not None or encoder_vq_s is not None:
                        encoder_vq_val = (
                            (encoder_vq_c.item() if encoder_vq_c is not None else 0.0) +
                            (encoder_vq_s.item() if encoder_vq_s is not None else 0.0)
                        )
                        self.tb.add_scalar("stage3/encoder_vq_loss", encoder_vq_val, self.global_iter)


                    encoder_stats_c = rc.get("encoder_vq_stats", {})
                    encoder_stats_s = rs.get("encoder_vq_stats", {})
                    if 'encoder_vq_recon' in encoder_stats_c:
                        self.tb.add_scalar("stage3/encoder_vq_recon_char", encoder_stats_c['encoder_vq_recon'], self.global_iter)
                    if 'encoder_vq_recon' in encoder_stats_s:
                        self.tb.add_scalar("stage3/encoder_vq_recon_sent", encoder_stats_s['encoder_vq_recon'], self.global_iter)


            if is_viz_iter and viz_char_data is not None:

                pred_gmm_c     = viz_char_data["pred_gmm"]
                gt_char_c      = viz_char_data["gt_char"]
                font_char_strs = viz_char_data["font_char_strs"]
                Bc = pred_gmm_c.size(0) if pred_gmm_c is not None else 0

                if Bc > 0:
                    sel_c = random.sample(range(Bc), min(5, Bc))


                    style_ref_imgs   = viz_char_data["style_reference"].to(self.device, non_blocking=True)
                    style_writer_ids = viz_char_data["style_labels"].to(self.device, non_blocking=True)
                    char_imgs        = viz_char_data["font_reference"].to(self.device, non_blocking=True)
                    seq_chars        = viz_char_data["seq_chars"].to(self.device, non_blocking=True).int()
                    idx1 = torch.zeros(seq_chars.size(0), 1, dtype=torch.long, device=self.device)
                    res_inf = self._infer_for_batch(style_ref_imgs, char_imgs, style_writer_ids,
                                                    seq_chars, indices=idx1)
                    flat_gmms = self._flatten_gmms_for_chars(res_inf)
                    pred_list_sel_c = [flat_gmms[i] for i in sel_c]

                    gt_for_show_c = [gt_char_c[i, 0].detach().cpu() for i in sel_c]


                    self.viz_executor.submit(
                        visualize_snapshot_chars,
                        tb_writer=self.tb,
                        pred_gmms=[pred_gmm_c[i, 0].detach().cpu() for i in sel_c],
                        gt_coords_list=gt_for_show_c,
                        characters=[font_char_strs[i] for i in sel_c],
                        step=self.global_iter,
                        font_dataset=self._font_panel(),
                        IMG_SIZE=self.cfg.ENV.IMG_H,
                        mode="train-char",
                        coord_space="delta",
                        infer_gmms=[g.detach().cpu() for g in pred_list_sel_c],
                    )


                pred_gmm_sent      = rs["pred_gmm_sent"]
                sentence_char_ids  = rs["sentence_char_ids"]


                gt_traj_list = s_batch["seq_trajs"]

                b0 = 0
                ids_row = sentence_char_ids[b0].tolist()

                sent_chars_b0 = [chr(int(i)) if int(i) > 0 else "" for i in ids_row]


                gt_lens_b0 = s_batch["seq_traj_lens"][b0].cpu()
                gt_list_b0 = []
                for s in range(gt_traj_list.size(1)):
                    traj_len = int(gt_lens_b0[s].item())
                    if traj_len > 0:
                        gt_list_b0.append(gt_traj_list[b0, s, :traj_len].detach().cpu())
                    else:
                        gt_list_b0.append(torch.zeros(0, 6))


                style_ref_imgs   = s_batch["style_reference"][b0:b0+1].to(self.device, non_blocking=True)
                style_writer_ids = s_batch["style_labels"][b0:b0+1].to(self.device, non_blocking=True)
                char_imgs        = s_batch["font_reference"][b0:b0+1].to(self.device, non_blocking=True)
                seq_chars        = s_batch["seq_chars"][b0:b0+1].to(self.device, non_blocking=True).int()

                res_inf = self._infer_for_batch(style_ref_imgs, char_imgs, style_writer_ids, seq_chars, indices=None)
                pred_gmm_infer_raw = res_inf[0]['gmm']


                S_gt = len(sent_chars_b0)
                if len(pred_gmm_infer_raw) > S_gt:
                    pred_gmm_infer = pred_gmm_infer_raw[:S_gt]
                    print_once(f"[VIS]  Inference generated {len(pred_gmm_infer_raw)} chars, truncated to GT length {S_gt}")
                else:
                    pred_gmm_infer = pred_gmm_infer_raw


                self.viz_executor.submit(
                    visualize_snapshot_sentence,
                    gt_coords_list=gt_list_b0,
                    pred_gmm=pred_gmm_sent[b0].detach().cpu(),
                    sentence_chars=sent_chars_b0,
                    step=self.global_iter,
                    font_dataset=self._font_panel(),
                    tb_writer=self.tb,
                    IMG_SIZE=self.cfg.ENV.IMG_H,
                    mode="train-sent",
                    coord_space="delta",
                    infer_gmm=pred_gmm_infer,
                    n_gram_window=self.n_gram_window,
                )


                del gt_list_b0, pred_gmm_infer, pred_gmm_infer_raw, res_inf


                if viz_char_data is not None:
                    del viz_char_data
                    viz_char_data = None

                torch.cuda.empty_cache()

                self._warmup_viz = False

            if self.rank == 0 and self.global_iter >= self.cfg.TRAIN.CHECKPOINT_BEGIN and self.global_iter % self.cfg.TRAIN.CHECKPOINT_ITERS == 0:
                save_path = os.path.join(self.ckpt_dir, f"ckpt_{self.global_iter}.pt")

                model_to_save = self.full_model.module if self.is_ddp else self.full_model
                save_checkpoint(save_path, model_to_save, self.optimizer,
                                self.global_iter, self.best_val,
                                config=getattr(self.cfg, "as_dict", lambda: None)(),
                                model_config=extract_model_config(self.cfg))

            if self.global_iter >= self.cfg.TRAIN.VALIDATE_BEGIN and self.global_iter % self.cfg.TRAIN.VALIDATE_ITERS == 0:
                val = self.validate(self.global_iter)
                if val < self.best_val:
                    self.best_val = val
                    if self.rank == 0:
                        best_path = os.path.join(self.ckpt_dir, f"ckpt_valid_{self.global_iter}.pt")

                        model_to_save = self.full_model.module if self.is_ddp else self.full_model
                        save_checkpoint(best_path, model_to_save, self.optimizer,
                                        self.global_iter, self.best_val,
                                        config=getattr(self.cfg, "as_dict", lambda: None)())


        print_once("[Trainer] Waiting for all visualization tasks to complete...")
        self.viz_executor.shutdown(wait=True)
        print_once("[Trainer] All visualization tasks completed. Training finished.")

        return it_char


    def _run_char_step(self, batch: Dict[str, Any], style_cache: Dict[str, torch.Tensor] = None) -> Dict[str, Any]:
        result = self._run_step_unified(batch, mode='char', style_cache=style_cache)

        pred_gmm = result['pred_gmm']
        gt_char = result['gt_trajs']


        if self.global_iter >= self.cfg.TRAIN.CHECKPOINT_BEGIN and self.global_iter % self.cfg.TRAIN.CHECKPOINT_ITERS == 0:
            with torch.no_grad():

                gt_sample = gt_char[0, 0]
                gt_pen = gt_sample[:, PEN_STATE_RANGE]
                gt_has_any_pen = (gt_pen.sum(dim=-1) > 0).float()
                gt_valid_count = gt_has_any_pen.sum().item()
                gt_eoc_indices = (gt_pen[:, PEN_CLASS["EOC"]] > 0.5).nonzero(as_tuple=True)[0]

                print(f"\n[dbg][iter{self.global_iter}] === GT Analysis ===")
                print(f"  GT shape: {gt_char.shape}, Valid points: {int(gt_valid_count)}")
                print(f"  GT EOC indices: {gt_eoc_indices.tolist()[:5]} (showing first 5)")
                if len(gt_eoc_indices) > 0:
                    print(f"  GT PEN_STATE at EOC: {gt_pen[gt_eoc_indices[0]].tolist()}")


                sample_gmm = pred_gmm[0, 0]
                T_pred = sample_gmm.shape[0]
                z_pi, z_mu1, z_mu2, z_s1, z_s2, z_rho, z_pen = get_mixture_coef(sample_gmm.reshape(-1, 124))


                pen_prob = torch.softmax(z_pen, dim=-1)
                eoc_prob = pen_prob[:, PEN_CLASS["EOC"]]
                eoc_max = eoc_prob.max().item()
                eoc_max_idx = eoc_prob.argmax().item()
                eoc_ge_05 = (eoc_prob >= 0.5).sum().item()


                pm_idx = PEN_CLASS["PM"]
                pu_idx = PEN_CLASS["PU"]
                cursive_eoc_idx = PEN_CLASS["CURSIVE_EOC"]
                eoc_idx = PEN_CLASS["EOC"]

                print(f"\n[dbg][iter{self.global_iter}] === Pred Analysis ===")
                print(f"  Pred GMM shape: {pred_gmm.shape}")
                print(f"  PEN_STATE logits range: [{z_pen.min():.2f}, {z_pen.max():.2f}]")
                print(f"  PEN_STATE prob (mean): PM={pen_prob[:, pm_idx].mean():.4f}, PU={pen_prob[:, pu_idx].mean():.4f}, CURSIVE_EOC={pen_prob[:, cursive_eoc_idx].mean():.4f}, EOC={pen_prob[:, eoc_idx].mean():.4f}")
                print(f"  EOC prob: max={eoc_max:.4f} @idx={eoc_max_idx}, count(>=0.5)={eoc_ge_05}/{T_pred}")
                print(f"  mu (mean): x={z_mu1.mean():.4f}, y={z_mu2.mean():.4f}")
                print(f"  sigma (mean): x={z_s1.mean():.4f}, y={z_s2.mean():.4f}")


        seq_chars = result['seq_chars']
        font_char_strs = [chr(int(seq_chars[i, 0].item())) if int(seq_chars[i, 0]) > 0 else ""
                          for i in range(seq_chars.size(0))]

        return {
            "style": result['style_loss'],
            "char": result['gen_loss'],
            "decoder_vq_loss": result.get('decoder_vq_loss', torch.tensor(0.0, device=self.device)),
            "encoder_vq_loss": result.get('encoder_vq_loss', torch.tensor(0.0, device=self.device)),
            "pred_gmm": pred_gmm,
            "gt_char": gt_char,
            "font_char_strs": font_char_strs,
            "style_cache": result['style_cache']
        }

    def _run_bigram_step(self, bg_batch: Dict[str, Any], style_cache: Dict[str, torch.Tensor] = None) -> Dict[str, Any]:
        result = self._run_step_unified(bg_batch, mode='bigram', style_cache=style_cache)


        if "curr_indices" in bg_batch:
            curr_idx = bg_batch["curr_indices"].to(self.device, non_blocking=True).long()
            prev_idx = torch.clamp(curr_idx - 1, min=0)
            pair_indices = torch.stack([prev_idx, curr_idx], dim=1)
        else:

            B = result['pred_gmm'].size(0)
            pair_indices = torch.stack([
                torch.zeros(B, dtype=torch.long, device=self.device),
                torch.ones(B, dtype=torch.long, device=self.device)
            ], dim=1)

        return {
            "style": result['style_loss'],
            "char": result['gen_loss'],
            "decoder_vq_loss": result.get('decoder_vq_loss', torch.tensor(0.0, device=self.device)),
            "encoder_vq_loss": result.get('encoder_vq_loss', torch.tensor(0.0, device=self.device)),
            "vdl_loss": result.get('vdl_loss', torch.tensor(0.0, device=self.device)),
            "vdl_stats": result.get('vdl_stats', {}),
            "pred_gmm": result['pred_gmm'],
            "gt_traj": result['gt_trajs_unmasked'],
            "style_cache": result['style_cache'],
            "pair_indices": pair_indices,
        }


    def _run_sentence_step(self, batch: Dict[str, Any], style_cache: Dict[str, torch.Tensor] = None) -> Dict[str, Any]:

        result_sent = self._run_step_unified(batch, mode='sentence', style_cache=style_cache)

        return {
            "style": result_sent['style_loss'],
            "sent": result_sent['gen_loss'],
            "decoder_vq_loss": result_sent.get('decoder_vq_loss', torch.tensor(0.0, device=self.device)),
            "encoder_vq_loss": result_sent.get('encoder_vq_loss', torch.tensor(0.0, device=self.device)),
            "vdl_loss": result_sent.get('vdl_loss', torch.tensor(0.0, device=self.device)),
            "vdl_stats": result_sent.get('vdl_stats', {}),
            "pred_gmm_sent": result_sent['pred_gmm'],
            "sentence_char_ids": result_sent['seq_chars'],
            "gt_traj_list": result_sent['gt_trajs'],
            "style_cache": result_sent['style_cache']
        }


    @torch.no_grad()
    def validate(self, step: int) -> float:
        import random
        self.full_model.eval()


        val_batch_size = self.cfg.VALID.BATCH_SIZE
        dataset_size = len(self.sent_val_ds)


        all_batches = dataset_size // val_batch_size
        max_val_batches = min(20, all_batches)

        print(f"[VALID] dataset_size={dataset_size}, batch_size={val_batch_size}, max_batches={max_val_batches}")

        total_loss = 0.0
        total_samples = 0
        total_batches = 0


        vis_gt_list = None
        vis_sent_chars = None
        vis_pred_gmm = None
        vis_selected = False


        for batch_idx, batch in enumerate(self.val_loader):

            if batch_idx >= max_val_batches:
                break
            B = batch["seq_trajs"].size(0)


            if batch_idx % 10 == 0:
                print(f"[VALID] Processing batch {batch_idx}/{max_val_batches}...", flush=True)


            style_ref_imgs   = batch["style_reference"].to(self.device, non_blocking=True)
            style_writer_ids = batch["style_labels"].to(self.device, non_blocking=True)
            char_imgs        = batch["font_reference"].to(self.device, non_blocking=True)
            seq_chars        = batch["seq_chars"].to(self.device, non_blocking=True).int()
            gt_trajs_batch   = batch["seq_trajs"]


            res_inf_list = self._infer_for_batch(style_ref_imgs, char_imgs, style_writer_ids, seq_chars, indices=None)


            for b in range(B):
                pred_gmm_infer = res_inf_list[b]['gmm']
                S_gt = char_imgs.size(1)


                if len(pred_gmm_infer) > S_gt:
                    pred_gmm_infer = pred_gmm_infer[:S_gt]


                S_inf = len(pred_gmm_infer)
                if S_inf == 0:
                    continue

                T_max = max(g.size(0) for g in pred_gmm_infer)
                C = pred_gmm_infer[0].size(1)

                pred_gmm_tensor = torch.zeros(1, S_inf, T_max, C)
                for s_idx, gmm in enumerate(pred_gmm_infer):
                    T_s = gmm.size(0)
                    pred_gmm_tensor[0, s_idx, :T_s] = gmm


                gt_orig = gt_trajs_batch[b:b+1, :S_inf].cpu()
                T_orig = gt_orig.size(2)

                if T_orig < T_max:
                    gt_tensor = torch.zeros(1, S_inf, T_max, 6)
                    gt_tensor[:, :, :T_orig] = gt_orig
                elif T_orig > T_max:
                    gt_tensor = gt_orig[:, :, :T_max]
                else:
                    gt_tensor = gt_orig


                loss_mask_orig = batch["seq_loss_mask"][b:b+1, :S_inf].cpu()
                if loss_mask_orig.dim() == 3:
                    T_mask = loss_mask_orig.size(2)
                    if T_mask < T_max:
                        loss_mask = torch.zeros(1, S_inf, T_max, dtype=loss_mask_orig.dtype)
                        loss_mask[:, :, :T_mask] = loss_mask_orig
                    elif T_mask > T_max:
                        loss_mask = loss_mask_orig[:, :, :T_max]
                    else:
                        loss_mask = loss_mask_orig
                else:
                    loss_mask = loss_mask_orig

                pred_gmm_dev = pred_gmm_tensor.to(self.device)
                gt_dev = gt_tensor.to(self.device)


                sample_loss = self._compute_seq_loss(
                    pred_gmm_dev,
                    gt_dev,
                    loss_mask.to(self.device),
                    suppress_debug=True
                )

                total_loss += sample_loss.item()
                total_samples += 1


                if not vis_selected and batch_idx == 0 and B > 0:
                    vis_sample_idx = random.randint(0, B - 1)
                    if b == vis_sample_idx:

                        gt_lens_vis = batch["seq_traj_lens"][b].cpu()
                        vis_gt_list = []
                        for s in range(gt_trajs_batch.size(1)):
                            traj_len = int(gt_lens_vis[s].item())
                            if traj_len > 0:
                                vis_gt_list.append(gt_trajs_batch[b, s, :traj_len].detach().cpu())
                            else:
                                vis_gt_list.append(torch.zeros(0, 6))

                        seq_chars_text = batch.get("seq_chars_text", None)
                        if seq_chars_text is not None and len(seq_chars_text) > b:
                            vis_sent_chars = list(seq_chars_text[b])
                        else:
                            ids_row = batch["seq_chars"][b].tolist()
                            vis_sent_chars = [chr(int(i)) if int(i) > 0 else "" for i in ids_row]
                        vis_pred_gmm = pred_gmm_infer
                        vis_selected = True

            total_batches += 1


            if (batch_idx + 1) % 20 == 0:
                interim_avg = total_loss / total_samples if total_samples > 0 else 0.0
                print(f"[VALID] Batch {batch_idx+1}/{max_val_batches}: avg_loss={interim_avg:.4f} (samples={total_samples})", flush=True)


        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        print(f"[VALID] step={step} loss={avg_loss:.4f} (samples={total_samples}, batches={total_batches})")
        self.tb.add_scalar("val/inference_loss", avg_loss, step)


        if vis_gt_list is not None and vis_pred_gmm is not None and vis_sent_chars is not None:
            visualize_snapshot_sentence(
                gt_coords_list=vis_gt_list,
                pred_gmm=None,
                sentence_chars=vis_sent_chars,
                step=step,
                font_dataset=self._font_panel(),
                tb_writer=self.tb,
                IMG_SIZE=self.cfg.ENV.IMG_H,
                mode="valid-sent",
                coord_space="delta",
                infer_gmm=vis_pred_gmm,
                n_gram_window=self.n_gram_window,
            )

        self.full_model.train()
        return avg_loss


    def _compute_seq_loss(self, pred_gmm, gt_traj, loss_mask, suppress_debug=False):
        device = pred_gmm.device

        if not isinstance(loss_mask, torch.Tensor):
            raise TypeError("loss_mask must be a Tensor")
        loss_mask = loss_mask.to(device, non_blocking=True)


        debug_validation = getattr(self.cfg.TRAIN, 'DEBUG_LOSS_VALIDATION', False)

        if debug_validation:

            B, S, T, C = gt_traj.shape
            assert C >= 6, f"[LOSS_VALIDATION] gt_traj last dim must be >=6 (expanded format), got {C}"

            pen_states = gt_traj[..., PEN_STATE_RANGE]
            pen_state_sums = pen_states.sum(dim=-1)


            invalid_onehot = ((pen_state_sums != 0.0) & (torch.abs(pen_state_sums - 1.0) > 1e-5))
            if invalid_onehot.any():
                invalid_indices = torch.nonzero(invalid_onehot, as_tuple=False)
                sample_idx = invalid_indices[0].tolist()
                b, s, t = sample_idx
                raise ValueError(
                    f"[LOSS_VALIDATION] GT pen_state is not one-hot!\n"
                    f"  Location: batch={b}, seq={s}, time={t}\n"
                    f"  Pen state: {pen_states[b, s, t].tolist()}\n"
                    f"  Sum: {pen_state_sums[b, s, t].item():.6f} (expected 1.0 or 0.0)\n"
                    f"  Total violations: {invalid_onehot.sum().item()}/{invalid_onehot.numel()}"
                )


            if loss_mask.dim() == 2:
                assert loss_mask.shape == (B, S), \
                    f"[LOSS_VALIDATION] loss_mask shape {loss_mask.shape} doesn't match gt_traj [B,S]=({B},{S})"
            elif loss_mask.dim() == 3:
                assert loss_mask.shape == (B, S, T), \
                    f"[LOSS_VALIDATION] loss_mask shape {loss_mask.shape} doesn't match gt_traj [B,S,T]=({B},{S},{T})"
            else:
                raise ValueError(f"[LOSS_VALIDATION] loss_mask must be 2D or 3D, got {loss_mask.dim()}D")


        loss_mask_2d = loss_mask.any(dim=-1) if loss_mask.dim() == 3 else loss_mask
        loss_mask_2d = (loss_mask_2d > 0).to(device, non_blocking=True)

        gts = gt_traj.to(device, non_blocking=True)
        B, S, T, C = gts.shape

        assert C >= 6, f"gt_traj last dim must be >=6 (expanded format), got {C}"


        time_mask = (gts[..., PEN_STATE_RANGE].sum(dim=-1) > 0)


        char_mask3d = loss_mask_2d.unsqueeze(-1).expand_as(time_mask)
        valid_mask = (time_mask & char_mask3d).reshape(-1)
        total_len = valid_mask.sum().float()
        if total_len.item() == 0:
            return torch.tensor(0., device=device)


        preds = pred_gmm.contiguous().view(B*S*T, -1).to(torch.float32)
        gts_f = gts.contiguous().view(B*S*T, -1).to(torch.float32)
        valid_mask = valid_mask.view(-1, 1).to(torch.float32)


        z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr, z_pen_logits = get_mixture_coef(preds)


        if not suppress_debug and self.global_iter % 10000 == 0 and getattr(self, '_debug_loss_printed', -1) != self.global_iter:
            print(f"\n[DEBUG][LOSS] Iteration {self.global_iter}")
            print(f"  GT pen_data shape: {gts_f[:, PEN_STATE_RANGE].shape}")
            print(f"  GT pen_data (first 3):\n{gts_f[:3, PEN_STATE_RANGE]}")
            print(f"  GT pen_data sum per row (should be 1.0 for one-hot): {gts_f[:5, PEN_STATE_RANGE].sum(dim=1)}")
            print(f"  Pred pen_logits (first 3):\n{z_pen_logits[:3]}")
            print(f"  Pred sigma1 mean/std: {z_sigma1.mean():.4f}/{z_sigma1.std():.4f}")
            print(f"  Pred sigma2 mean/std: {z_sigma2.mean():.4f}/{z_sigma2.std():.4f}")
            print(f"  GT x1_data (first 3): {gts_f[:3, 0]}")
            print(f"  GT x2_data (first 3): {gts_f[:3, 1]}")
            print(f"  Valid mask sum: {valid_mask.sum()}/{valid_mask.numel()}")
            print(f"  Valid mask (first 10): {valid_mask[:10].squeeze().tolist()}")
            print(f"  Valid mask min/max/mean: {valid_mask.min():.4f}/{valid_mask.max():.4f}/{valid_mask.mean():.4f}")
            self._debug_loss_printed = self.global_iter

        moving_all, state_loss = get_pen_loss(
            z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr, z_pen_logits,
            gts_f[:, 0:1], gts_f[:, 1:2], gts_f[:, PEN_STATE_RANGE],
            step=self.global_iter if not suppress_debug else None,
            time_mask=valid_mask,
            class_weight=self.pen_state_class_weight,
        )


        if (moving_all is None) or torch.isnan(moving_all).any() or torch.isnan(state_loss).any():
            return torch.tensor(0., device=device, dtype=torch.float32)


        denom = valid_mask.sum().clamp_min(1.0)
        nll = moving_all.view(-1).sum() / denom


        if not suppress_debug and self.global_iter % 10000 == 0 and getattr(self, '_loss_debug_printed', -1) != self.global_iter:
            total_weighted = nll + 1.5 * state_loss
            print(f"[LOSS_DEBUG][iter={self.global_iter}] MDN_NLL={nll.item():.4f}, State_Loss={state_loss.item():.4f} (weight=1.5, SDT=2.0), Total={total_weighted.item():.4f}")
            self._loss_debug_printed = self.global_iter


        return nll + 1.5 * state_loss

    def _compute_vertical_drift_loss(
        self,
        pred_gmm: torch.Tensor,
        gt_traj: torch.Tensor,
        loss_mask: torch.Tensor,
        traj_lens: Optional[torch.Tensor],
        eps: float = 1e-6,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        device = pred_gmm.device
        if pred_gmm.dim() != 4:
            raise ValueError(f"pred_gmm must be 4D [B,S,T,C], got {pred_gmm.shape}")
        B, S, T, C = pred_gmm.shape
        if S <= 1:
            zero = torch.tensor(0.0, device=device)
            return zero, {"centroid": zero, "top": zero, "bottom": zero}


        if traj_lens is not None:
            arange_t = torch.arange(T, device=device).view(1, 1, T)
            time_valid = (arange_t < traj_lens.unsqueeze(-1))
        else:
            time_valid = (gt_traj[..., PEN_STATE_RANGE].sum(dim=-1) > 0)

        time_valid_f = time_valid.float()


        flat = pred_gmm.reshape(B * S * T, C).to(torch.float32)
        pi, _, mu2, _, _, _, _ = get_mixture_coef(flat)
        dy_pred = (pi * mu2).sum(dim=1).view(B, S, T)


        dy_gt = gt_traj[..., TRAJ_INDEX_EXPANDED["Y"]].to(torch.float32)


        dy_pred_m = dy_pred * time_valid_f
        dy_gt_m = dy_gt * time_valid_f

        dy_pred_flat = dy_pred_m.reshape(B, S * T)
        dy_gt_flat = dy_gt_m.reshape(B, S * T)

        y_abs_pred = dy_pred_flat.cumsum(dim=1).reshape(B, S, T)
        y_abs_gt = dy_gt_flat.cumsum(dim=1).reshape(B, S, T)


        pu_idx = TRAJ_INDEX_EXPANDED["PU"]
        pen_down = (gt_traj[..., pu_idx] < 0.5) & time_valid
        pen_down_f = pen_down.float()


        w = pen_down_f.sum(dim=2)
        valid_char = w > 0
        w_safe = w.clamp_min(1.0)


        centroid_pred = (y_abs_pred * pen_down_f).sum(dim=2) / w_safe
        centroid_gt = (y_abs_gt * pen_down_f).sum(dim=2) / w_safe


        top_pred = self._compute_percentile_per_char(y_abs_pred, pen_down, percentile=10.0)
        top_gt = self._compute_percentile_per_char(y_abs_gt, pen_down, percentile=10.0)


        bottom_pred = self._compute_percentile_per_char(y_abs_pred, pen_down, percentile=90.0)
        bottom_gt = self._compute_percentile_per_char(y_abs_gt, pen_down, percentile=90.0)


        d_centroid_pred = centroid_pred[:, 1:] - centroid_pred[:, :-1]
        d_centroid_gt = centroid_gt[:, 1:] - centroid_gt[:, :-1]

        d_top_pred = top_pred[:, 1:] - top_pred[:, :-1]
        d_top_gt = top_gt[:, 1:] - top_gt[:, :-1]

        d_bottom_pred = bottom_pred[:, 1:] - bottom_pred[:, :-1]
        d_bottom_gt = bottom_gt[:, 1:] - bottom_gt[:, :-1]


        valid_pair = valid_char[:, :-1] & valid_char[:, 1:]


        if loss_mask.dim() == 2:
            char_mask = (loss_mask > 0)
        elif loss_mask.dim() == 3:
            char_mask = (loss_mask.sum(dim=-1) > 0)
        else:
            raise ValueError(f"loss_mask must be 2D or 3D, got {loss_mask.dim()}D")


        pair_loss_mask = char_mask[:, 1:]
        valid_pair = valid_pair & pair_loss_mask

        valid_pair_f = valid_pair.float()
        n_valid_pairs = valid_pair_f.sum().clamp_min(1.0)

        if not valid_pair.any():
            zero = torch.tensor(0.0, device=device)
            return zero, {"centroid": zero, "top": zero, "bottom": zero}


        loss_centroid = ((d_centroid_pred - d_centroid_gt) ** 2 * valid_pair_f).sum() / n_valid_pairs
        loss_top = ((d_top_pred - d_top_gt) ** 2 * valid_pair_f).sum() / n_valid_pairs
        loss_bottom = ((d_bottom_pred - d_bottom_gt) ** 2 * valid_pair_f).sum() / n_valid_pairs


        w_centroid = getattr(self.cfg.TRAIN, 'VDL_CENTROID_WEIGHT', 2.0)
        w_top = getattr(self.cfg.TRAIN, 'VDL_TOP_WEIGHT', 1.0)
        w_bottom = getattr(self.cfg.TRAIN, 'VDL_BOTTOM_WEIGHT', 1.0)

        total = w_centroid * loss_centroid + w_top * loss_top + w_bottom * loss_bottom


        stats = {
            "centroid": loss_centroid.detach(),
            "top": loss_top.detach(),
            "bottom": loss_bottom.detach(),

            "delta_centroid_pred_avg": d_centroid_pred[valid_pair].mean().detach() if valid_pair.any() else torch.tensor(0.0, device=device),
            "delta_centroid_gt_avg": d_centroid_gt[valid_pair].mean().detach() if valid_pair.any() else torch.tensor(0.0, device=device),
            "delta_top_pred_avg": d_top_pred[valid_pair].mean().detach() if valid_pair.any() else torch.tensor(0.0, device=device),
            "delta_top_gt_avg": d_top_gt[valid_pair].mean().detach() if valid_pair.any() else torch.tensor(0.0, device=device),
            "delta_bottom_pred_avg": d_bottom_pred[valid_pair].mean().detach() if valid_pair.any() else torch.tensor(0.0, device=device),
            "delta_bottom_gt_avg": d_bottom_gt[valid_pair].mean().detach() if valid_pair.any() else torch.tensor(0.0, device=device),

            "centroid_pred_avg": centroid_pred[valid_char].mean().detach() if valid_char.any() else torch.tensor(0.0, device=device),
            "centroid_gt_avg": centroid_gt[valid_char].mean().detach() if valid_char.any() else torch.tensor(0.0, device=device),
            "top_pred_avg": top_pred[valid_char].mean().detach() if valid_char.any() else torch.tensor(0.0, device=device),
            "top_gt_avg": top_gt[valid_char].mean().detach() if valid_char.any() else torch.tensor(0.0, device=device),
            "bottom_pred_avg": bottom_pred[valid_char].mean().detach() if valid_char.any() else torch.tensor(0.0, device=device),
            "bottom_gt_avg": bottom_gt[valid_char].mean().detach() if valid_char.any() else torch.tensor(0.0, device=device),
        }

        return total, stats

    def _compute_percentile_per_char(
        self,
        y_abs: torch.Tensor,
        pen_down: torch.Tensor,
        percentile: float = 90.0,
    ) -> torch.Tensor:
        B, S, T = y_abs.shape
        device = y_abs.device
        q = percentile / 100.0

        pen_down_f = pen_down.float()
        n_valid = pen_down_f.sum(dim=2)


        centroid = (y_abs * pen_down_f).sum(dim=2) / n_valid.clamp_min(1.0)


        inf = torch.tensor(float('inf'), device=device, dtype=y_abs.dtype)
        y_masked = torch.where(pen_down, y_abs, inf)


        y_sorted, _ = torch.sort(y_masked, dim=2)


        k = torch.clamp((n_valid - 1) * q, min=0).long()
        k = torch.clamp(k, max=T - 1)


        result = torch.gather(y_sorted, 2, k.unsqueeze(-1)).squeeze(-1)


        short_mask = n_valid < 3
        result = torch.where(short_mask, centroid, result)

        return result


    def _get_sent_vdl_weight(self, global_iter):
        start_iter = getattr(self.cfg.TRAIN, 'SENT_VDL_WARMUP_START_ITER', 70000)
        end_iter = getattr(self.cfg.TRAIN, 'SENT_VDL_WARMUP_END_ITER', 100000)
        start_weight = getattr(self.cfg.TRAIN, 'SENT_VDL_WARMUP_START_WEIGHT', 0.0)
        end_weight = getattr(self.cfg.TRAIN, 'SENT_VDL_WARMUP_END_WEIGHT', 0.02)


        if not hasattr(self, '_vdl_warmup_validated'):
            self._vdl_warmup_validated = True

            if start_iter >= end_iter:
                print(f"[WARN] VDL warmup: start_iter ({start_iter}) >= end_iter ({end_iter})")
                print(f"       Using constant weight: {end_weight:.4f}")

            bigram_end = getattr(self.cfg.TRAIN, 'BIGRAM_STAGE_ITERS', 50000)
            if start_iter < bigram_end:
                print(f"[WARN] VDL warmup starts at {start_iter} < Stage3 start ({bigram_end})")
                print(f"       Warmup will start from iteration {bigram_end}")


        if start_iter >= end_iter:
            return end_weight

        if global_iter < start_iter:
            return start_weight
        elif global_iter >= end_iter:
            return end_weight
        else:

            progress = (global_iter - start_iter) / (end_iter - start_iter)
            return start_weight + (end_weight - start_weight) * progress

    def _get_2pass_alpha(self, global_iter):
        start_iter = getattr(self.cfg.TRAIN, 'TWOPASS_START_ITER', -1)
        if start_iter < 0 or global_iter < start_iter:
            return 0.0

        alpha_max = getattr(self.cfg.TRAIN, 'TWOPASS_ALPHA_MAX', 0.4)
        ramp_iters = getattr(self.cfg.TRAIN, 'TWOPASS_RAMP_ITERS', 20000)

        progress = (global_iter - start_iter) / max(1, ramp_iters)
        progress = max(0.0, min(1.0, progress))

        return progress * alpha_max


    def _fetch_next(self, it, loader):
        try:
            batch = next(it)
            return batch, it
        except StopIteration:

            new_it = iter(loader)
            batch = next(new_it)
            return batch, new_it

    def _font_panel(self):
        class _FP:
            def __init__(self, d): self.d = d
            def get_char_img(self, ch):
                if isinstance(ch, torch.Tensor):
                    t = ch.detach().cpu().float()
                    if t.ndim == 2: t = t.unsqueeze(0)
                    return t
                if isinstance(ch, np.ndarray):
                    t = torch.from_numpy(ch).float()
                    if t.ndim == 2: t = t.unsqueeze(0)
                    return t

                img = self.d.get(ch, np.zeros((64,64), dtype=np.float32))
                t = torch.from_numpy(img).float()
                if t.ndim == 2: t = t.unsqueeze(0)
                return t
        return _FP(self.font_img_dict)


    @torch.no_grad()
    def _infer_for_batch(self, style_ref_imgs, sentence_char_imgs, style_writer_ids,
                        sentence_char_ids, indices=None, context_seq_chars=None,
                        max_len=None, gt_trajs=None, use_gt_prev=False):
        max_len = max_len or getattr(self.cfg.VALID, "MAX_LEN", 120)

        if isinstance(self.full_model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
            fm = self.full_model.module
        else:
            fm = self.full_model
        was_training = fm.training
        fm.eval()
        try:
            res = fm.inference(
                style_imgs=style_ref_imgs,
                char_imgs=sentence_char_imgs,
                writer_ids=style_writer_ids,
                global_iter=self.global_iter,
                max_len=max_len,
                seq_chars=sentence_char_ids,
                context_seq_chars=context_seq_chars,
                indices=indices,
                gt_trajs=gt_trajs,
                use_gt_prev=use_gt_prev,
            )
        finally:
            if was_training: fm.train()
        return res

    def _flatten_gmms_for_chars(self, res):
        flat = []
        for b in range(len(res)):
            for g in res[b]["gmm"]:
                flat.append(g)
        return flat

    def _select_indices(self, flat, sel):
        return [flat[i] for i in sel]
