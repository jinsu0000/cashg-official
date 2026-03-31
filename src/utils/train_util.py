from __future__ import annotations
import os
import glob
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math

from src.utils.logger import print_once
from src.utils.tb_util import (
    visualize_sentence_level,
    visualize_character_level,

)
from src.data.data_utils import (
    denormalize_xy_abs_symmetric,
    denormalize_height_based,
    delta_to_abs_norm,
)
from src.config.constants import TRAJ_INDEX, TRAJ_INDEX_EXPANDED, TRAJ_DIM, TRAJ_DIM_EXPANDED, PEN_STATE_RANGE


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim: int, max_len: int = 4096, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_len = max_len
        self.base = base


        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)


        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None

    def _update_cos_sin_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        if seq_len > self._seq_len_cached or self._cos_cached is None or self._cos_cached.device != device:
            self._seq_len_cached = max(seq_len, self._seq_len_cached)


            t = torch.arange(self._seq_len_cached, device=device, dtype=torch.float32)


            freqs = torch.einsum('i,j->ij', t, self.inv_freq.to(device))


            emb = torch.cat([freqs, freqs], dim=-1)

            self._cos_cached = emb.cos().to(dtype)
            self._sin_cached = emb.sin().to(dtype)

    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x[..., :self.dim // 2], x[..., self.dim // 2:]
        return torch.cat([-x2, x1], dim=-1)

    def apply_rotary_pos_emb(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = q.shape[0]
        seq_len_q = q.shape[1]
        seq_len_k = k.shape[1]


        max_len = max(seq_len_q, seq_len_k)
        if position_ids is not None:
            max_pos = position_ids.max().item() + 1
            max_len = max(max_len, max_pos)

        self._update_cos_sin_cache(max_len, q.device, q.dtype)


        if position_ids is None:

            cos_q = self._cos_cached[:seq_len_q].unsqueeze(0)
            sin_q = self._sin_cached[:seq_len_q].unsqueeze(0)
            cos_k = self._cos_cached[:seq_len_k].unsqueeze(0)
            sin_k = self._sin_cached[:seq_len_k].unsqueeze(0)
        else:


            if position_ids.dim() == 1:
                position_ids = position_ids.unsqueeze(0)


            pos_q = position_ids[:, :seq_len_q].clamp(0, self._seq_len_cached - 1)
            cos_q = torch.gather(
                self._cos_cached.unsqueeze(0).expand(batch_size, -1, -1),
                dim=1,
                index=pos_q.unsqueeze(-1).expand(-1, -1, self.dim)
            )
            sin_q = torch.gather(
                self._sin_cached.unsqueeze(0).expand(batch_size, -1, -1),
                dim=1,
                index=pos_q.unsqueeze(-1).expand(-1, -1, self.dim)
            )


            pos_k = position_ids[:, :seq_len_k].clamp(0, self._seq_len_cached - 1)
            cos_k = torch.gather(
                self._cos_cached.unsqueeze(0).expand(batch_size, -1, -1),
                dim=1,
                index=pos_k.unsqueeze(-1).expand(-1, -1, self.dim)
            )
            sin_k = torch.gather(
                self._sin_cached.unsqueeze(0).expand(batch_size, -1, -1),
                dim=1,
                index=pos_k.unsqueeze(-1).expand(-1, -1, self.dim)
            )


        if q.dim() == 4:
            cos_q = cos_q.unsqueeze(2)
            sin_q = sin_q.unsqueeze(2)
            cos_k = cos_k.unsqueeze(2)
            sin_k = sin_k.unsqueeze(2)


        q_embed = (q * cos_q) + (self.rotate_half(q) * sin_q)
        k_embed = (k * cos_k) + (self.rotate_half(k) * sin_k)

        return q_embed, k_embed

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.apply_rotary_pos_emb(q, k, position_ids)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096, dropout: float = 0.0):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32)
                        * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)
        self.dropout = nn.Dropout(p=dropout)
        self.dim = d_model

    def forward(self, x: torch.Tensor, step: int = None) -> torch.Tensor:
        if step is None:
            S = x.size(1)
            x = x * math.sqrt(self.dim)
            x = x + self.pe[:S].unsqueeze(0)
            return self.dropout(x)

        i = min(int(step), self.pe.size(0)-1)
        pe_i = self.pe[i].view(1,1,-1)
        x = x.clone()
        x[:, -1, :] = x[:, -1, :] * math.sqrt(self.dim) + pe_i.squeeze(1)
        return self.dropout(x)


def random_double_sampling(x: torch.Tensor, ratio: float = 0.25) -> Tuple[torch.Tensor, torch.Tensor]:
    B, L, N, D = x.shape
    x = rearrange(x, "B L N D -> B N L D")
    noise = torch.rand(B, N, L, device=x.device)
    ids_shuffle = torch.argsort(noise, dim=2)
    anchor_tokens = int(L * ratio)
    pos_tokens = int(L * 2 * ratio)
    ids_anchor = ids_shuffle[:, :, :anchor_tokens]
    ids_pos = ids_shuffle[:, :, anchor_tokens:pos_tokens]
    x_anchor = torch.gather(x, dim=2, index=ids_anchor.unsqueeze(-1).repeat(1, 1, 1, D))
    x_pos = torch.gather(x, dim=2, index=ids_pos.unsqueeze(-1).repeat(1, 1, 1, D))
    return x_anchor, x_pos


def make_square_subsequent_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    m = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
    return m.masked_fill(m == 1, float("-inf")).masked_fill(m == 0, 0.0)


def generate_contextual_square_mask(context_len: int, seq_len: int, device: torch.device) -> torch.Tensor:
    N = context_len + seq_len
    mask = torch.zeros(N, N, device=device)
    causal = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1) * float("-inf")
    mask[context_len:, context_len:] = causal
    return mask


def get_mixture_coef(
    output: torch.Tensor,
    num_mixtures: int = 20,
    sigma_eps: float = 1e-3,
):
    C = 4 + 6*num_mixtures
    assert output.dim() == 2 and output.size(1) == C, f"get_mixture_coef expects [N,{C}], got {output.shape}"

    z = output
    pen_logits = z[:, :4]
    rest = z[:, 4:]
    pi_logits, mu1, mu2, s1_raw, s2_raw, rho_raw = torch.split(rest, num_mixtures, dim=1)

    pi = F.softmax(pi_logits, dim=-1)
    sigma1 = F.softplus(s1_raw) + sigma_eps
    sigma2 = F.softplus(s2_raw) + sigma_eps
    corr = torch.tanh(rho_raw).clamp(-0.99999, 0.99999)

    return pi, mu1, mu2, sigma1, sigma2, corr, pen_logits


def get_seq_from_gmm(
    gmm_pred: torch.Tensor,
    num_mixtures: int = 20,
    decode: str = "argmax_onehot",
):
    assert gmm_pred.dim() == 3, f"get_seq_from_gmm expects [B,T,C], got {gmm_pred.shape}"
    B, T, C = gmm_pred.shape
    flat = gmm_pred.reshape(B*T, C)

    pi, mu1, mu2, s1, s2, rho, pen_logits = get_mixture_coef(flat, num_mixtures)


    if decode in ["argmax", "argmax_onehot"]:
        k = torch.argmax(pi, dim=1, keepdim=True)
        dx = torch.gather(mu1, 1, k)
        dy = torch.gather(mu2, 1, k)
    elif decode == "expectation":
        dx = (pi * mu1).sum(dim=1, keepdim=True)
        dy = (pi * mu2).sum(dim=1, keepdim=True)
    else:
        raise ValueError("decode must be 'expectation' or 'argmax_onehot'")


    pen_idx = torch.argmax(pen_logits, dim=1)


    pen_onehot = torch.nn.functional.one_hot(pen_idx, num_classes=4).float()
    pm = pen_onehot[:, 0:1]
    pu = pen_onehot[:, 1:2]
    cursive_eoc = pen_onehot[:, 2:3]
    eoc = pen_onehot[:, 3:4]
    seq = torch.cat([dx, dy, pm, pu, cursive_eoc, eoc], dim=1).reshape(B, T, 6)

    return seq


def compute_bigram_to_unigram_offset(
    bigram_traj: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    B, S, T, D = bigram_traj.shape
    device = bigram_traj.device


    pen_state_sum = bigram_traj[..., 2:6].sum(dim=-1)
    time_valid = (pen_state_sum > 0).float()

    bi_dx = bigram_traj[..., 0] * time_valid
    bi_dy = bigram_traj[..., 1] * time_valid


    bi_dx_flat = bi_dx.reshape(B, S * T)
    bi_dy_flat = bi_dy.reshape(B, S * T)
    sent_x_abs = bi_dx_flat.cumsum(dim=1).reshape(B, S, T)
    sent_y_abs = bi_dy_flat.cumsum(dim=1).reshape(B, S, T)


    char_has_valid = (time_valid.sum(dim=2) > 0)

    x_for_max = sent_x_abs.clone()
    x_for_max[time_valid == 0] = -1e9
    char_bbox_right, _ = x_for_max.max(dim=2)


    for s in range(S):
        invalid_mask = ~char_has_valid[:, s]
        if invalid_mask.any():
            if s == 0:
                char_bbox_right[invalid_mask, s] = 0.0
            else:
                char_bbox_right[invalid_mask, s] = char_bbox_right[invalid_mask, s - 1]


    origin_x = torch.zeros(B, S, device=device)
    if S > 1:
        origin_x[:, 1:] = char_bbox_right[:, :-1]


    origin_y = torch.zeros(B, S, device=device)


    uni_x_abs = sent_x_abs - origin_x.unsqueeze(-1)
    uni_y_abs = sent_y_abs - origin_y.unsqueeze(-1)


    uni_dx = uni_x_abs.clone()
    uni_dy = uni_y_abs.clone()
    uni_dx[:, :, 1:] = uni_x_abs[:, :, 1:] - uni_x_abs[:, :, :-1]
    uni_dy[:, :, 1:] = uni_y_abs[:, :, 1:] - uni_y_abs[:, :, :-1]


    offset_dx = (bi_dx - uni_dx) * time_valid
    offset_dy = (bi_dy - uni_dy) * time_valid

    return offset_dx, offset_dy


def convert_unigram_gmm_to_bigram(
    pred_gmm: torch.Tensor,
    gt_traj: torch.Tensor,
    num_mixtures: int = 20,
) -> torch.Tensor:
    B, S, T, C = pred_gmm.shape
    M = num_mixtures


    offset_dx, offset_dy = compute_bigram_to_unigram_offset(gt_traj)


    pred_gmm_out = pred_gmm.clone()

    mu1_start = 4 + M
    mu1_end   = 4 + 2 * M
    mu2_start = 4 + 2 * M
    mu2_end   = 4 + 3 * M


    offset_dx_exp = offset_dx.unsqueeze(-1).expand(B, S, T, M)
    offset_dy_exp = offset_dy.unsqueeze(-1).expand(B, S, T, M)

    pred_gmm_out[..., mu1_start:mu1_end] = pred_gmm[..., mu1_start:mu1_end] + offset_dx_exp
    pred_gmm_out[..., mu2_start:mu2_end] = pred_gmm[..., mu2_start:mu2_end] + offset_dy_exp

    return pred_gmm_out


def convert_unigram_delta_list_to_bigram(
    delta_list: List[np.ndarray],
) -> List[np.ndarray]:
    S = len(delta_list)
    if S == 0:
        return delta_list

    result = []

    anchor_x = 0.0
    anchor_y = 0.0

    origin_x = 0.0
    origin_y = 0.0

    for s in range(S):
        delta = delta_list[s]
        if delta is None or len(delta) == 0:
            result.append(delta)
            continue

        delta = delta.copy()
        dx = delta[:, 0]
        dy = delta[:, 1]


        uni_abs_x = np.cumsum(dx)
        uni_abs_y = np.cumsum(dy)


        sent_abs_x = uni_abs_x + origin_x
        sent_abs_y = uni_abs_y + origin_y


        bi_abs_x = sent_abs_x - anchor_x
        bi_abs_y = sent_abs_y - anchor_y
        bi_dx = np.zeros_like(bi_abs_x)
        bi_dy = np.zeros_like(bi_abs_y)
        bi_dx[0] = bi_abs_x[0]
        bi_dy[0] = bi_abs_y[0]
        bi_dx[1:] = bi_abs_x[1:] - bi_abs_x[:-1]
        bi_dy[1:] = bi_abs_y[1:] - bi_abs_y[:-1]

        out = delta.copy()
        out[:, 0] = bi_dx
        out[:, 1] = bi_dy
        result.append(out)


        anchor_x = sent_abs_x[-1]
        anchor_y = sent_abs_y[-1]

        pen_valid = delta[:, 2:6].sum(axis=-1) > 0
        if pen_valid.any():
            valid_x = sent_abs_x[pen_valid]
            origin_x = float(valid_x.max())
        else:
            origin_x = anchor_x
        origin_y = 0.0

    return result


def convert_unigram_gmm_list_to_bigram(
    gmm_list: List[torch.Tensor],
    gt_coords_list: List[torch.Tensor | np.ndarray],
    num_mixtures: int = 20,
) -> List[torch.Tensor]:
    S_gmm = len(gmm_list)
    S_gt  = len(gt_coords_list)
    S = min(S_gmm, S_gt)

    if S == 0:
        return gmm_list


    valid_indices = []
    gmm_tensors = []
    gt_tensors = []

    for i in range(S):
        g = gmm_list[i]
        gt = gt_coords_list[i]
        if g is None or gt is None:
            continue
        if isinstance(gt, np.ndarray):
            gt = torch.from_numpy(gt).float()
        if gt.numel() == 0 or g.numel() == 0:
            continue
        valid_indices.append(i)
        gmm_tensors.append(g)
        gt_tensors.append(gt)

    if len(valid_indices) == 0:
        return gmm_list


    S_valid = len(valid_indices)
    T_max_gmm = max(g.size(0) for g in gmm_tensors)
    T_max_gt  = max(g.size(0) for g in gt_tensors)
    T_max = max(T_max_gmm, T_max_gt)
    C = gmm_tensors[0].size(-1)


    ref_device = gmm_tensors[0].device
    ref_dtype  = gmm_tensors[0].dtype

    gmm_packed = torch.zeros(1, S_valid, T_max, C, device=ref_device, dtype=ref_dtype)
    gt_packed  = torch.zeros(1, S_valid, T_max, 6, device=ref_device, dtype=ref_dtype)

    for j, (g, gt) in enumerate(zip(gmm_tensors, gt_tensors)):
        gmm_packed[0, j, :g.size(0)] = g.to(device=ref_device, dtype=ref_dtype)
        gt_packed[0, j, :gt.size(0)] = gt.to(device=ref_device, dtype=ref_dtype)


    gmm_converted = convert_unigram_gmm_to_bigram(gmm_packed, gt_packed, num_mixtures)


    result = list(gmm_list)
    for j, idx in enumerate(valid_indices):
        orig_len = gmm_tensors[j].size(0)
        result[idx] = gmm_converted[0, j, :orig_len]

    return result


class Seq2Emb(nn.Module):
    def __init__(self, hid_dim: int, in_dim: int = 6, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 256)
        self.fc2 = nn.Linear(256, hid_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        from src.config.constants import TRAJ_DIM_EXPANDED

        assert seq.dim() == 3 and seq.size(-1) == TRAJ_DIM_EXPANDED, \
            f"Seq2Emb expects [B,T,6] (Expanded format only), got {seq.shape}"

        x = F.relu(self.fc1(seq))
        x = self.drop(x)
        return self.fc2(x)


class Emb2Seq(nn.Module):
    def __init__(self, hid_dim: int, output_dim: int = 124, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(hid_dim, 256)
        self.fc2 = nn.Linear(256, output_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 3, f"Emb2Seq expects [B,T,H], got {x.shape}"
        B, T, H = x.shape
        h = F.relu(self.fc1(x.reshape(B*T, H)))
        h = self.drop(h)
        y = self.fc2(h).reshape(B, T, -1)
        return y


def check_tensor(name: str, tensor: Any):
    if torch.is_tensor(tensor):
        arr = tensor.detach().cpu().numpy()
        print(
            f"[CHECK] {name} | shape={tuple(arr.shape)} "
            f"min={arr.min() if arr.size>0 else 'NA'} "
            f"max={arr.max() if arr.size>0 else 'NA'} "
            f"mean={arr.mean() if arr.size>0 else 'NA'} "
            f"std={arr.std() if arr.size>0 else 'NA'} "
            f"nan={torch.isnan(tensor).any().item()} "
            f"inf={torch.isinf(tensor).any().item()}"
        )
    else:
        print(f"[CHECK] {name} is not a tensor")


def _has_module_prefix(state_dict: Dict[str, torch.Tensor]) -> bool:
    return any(k.startswith("module.") for k in state_dict.keys())


def strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not _has_module_prefix(state_dict):
        return state_dict
    return {k.replace("module.", "", 1): v for k, v in state_dict.items()}


def add_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if _has_module_prefix(state_dict):
        return state_dict
    return {f"module.{k}": v for k, v in state_dict.items()}


def extract_model_config(cfg) -> Dict[str, Any]:
    model_cfg = getattr(cfg, 'MODEL', cfg)
    train_cfg = getattr(cfg, 'TRAIN', None)

    return {

        "USE_CONTEXT_AS_CONTENT": getattr(model_cfg, 'USE_CONTEXT_AS_CONTENT', False),


        "CONTEXT_BACKBONE": getattr(model_cfg, 'CONTEXT_BACKBONE', 'google/canine-c'),
        "CONTEXT_DIM": getattr(model_cfg, 'CONTEXT_DIM', 256),
        "CONTEXT_LAYERS": getattr(model_cfg, 'CONTEXT_LAYERS', 2),
        "CONTEXT_NHEAD": getattr(model_cfg, 'CONTEXT_NHEAD', 8),
        "CONTEXT_FREEZE_BACKBONE": getattr(model_cfg, 'CONTEXT_FREEZE_BACKBONE', True),
        "CONTEXT_USE_CHAR_ID_EMB": getattr(model_cfg, 'CONTEXT_USE_CHAR_ID_EMB', False),
        "CONTEXT_CHAR_ID_EMB_DIM": getattr(model_cfg, 'CONTEXT_CHAR_ID_EMB_DIM', 64),
        "CONTEXT_MAX_CHAR_ID": getattr(model_cfg, 'CONTEXT_MAX_CHAR_ID', 200000),

        "CONTEXT_TOKEN_DROPOUT_STAGE1": getattr(model_cfg, 'CONTEXT_TOKEN_DROPOUT_STAGE1', 0.0),
        "CONTEXT_TOKEN_DROPOUT_STAGE2": getattr(model_cfg, 'CONTEXT_TOKEN_DROPOUT_STAGE2', 0.0),
        "CONTEXT_TOKEN_DROPOUT_STAGE3": getattr(model_cfg, 'CONTEXT_TOKEN_DROPOUT_STAGE3', 0.0),


        "FONT_DIM": getattr(model_cfg, 'FONT_DIM', 256),


        "HWGEN_DIM": getattr(model_cfg, 'HWGEN_DIM', 256),
        "HWGEN_WRITER_LAYERS": getattr(model_cfg, 'HWGEN_WRITER_LAYERS', 2),
        "HWGEN_GLYPH_LAYERS": getattr(model_cfg, 'HWGEN_GLYPH_LAYERS', 2),
        "USE_CONTEXT_GATING": getattr(model_cfg, 'USE_CONTEXT_GATING', False),
        "CONTEXT_GATE_INIT": getattr(model_cfg, 'CONTEXT_GATE_INIT', 0.5),


        "STYLE_DIM": getattr(model_cfg, 'STYLE_DIM', 256),


        "USE_ROPE": getattr(model_cfg, 'USE_ROPE', True),
        "ROPE_BASE": getattr(model_cfg, 'ROPE_BASE', 100.0),


        "N_GRAM_AWARE_SLIDING_WINDOW": getattr(model_cfg, 'N_GRAM_AWARE_SLIDING_WINDOW', 2),
        "USE_CONTEXT_DECODER": getattr(model_cfg, 'USE_CONTEXT_DECODER', True),
    }


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    step: int,
    best_val: Optional[float] = None,
    config: Optional[Dict[str, Any]] = None,
    model_config: Optional[Dict[str, Any]] = None,
):
    ckpt = {
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "best_val": best_val,
        "config": config,
        "model_config": model_config,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(ckpt, path)
    print_once(f"[CKPT] saved: {path}")


def load_checkpoint(
    ckpt_path: str,
    model: torch.nn.Module,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
    *,
    strict: bool = False,
) -> Tuple[int, Optional[float]]:
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    print_once(f"[CKPT] loading: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    sd = ckpt["model"]


    model_is_dp = isinstance(model, torch.nn.DataParallel)
    has_module = _has_module_prefix(sd)
    if model_is_dp and not has_module:
        sd = add_module_prefix(sd)
    if (not model_is_dp) and has_module:
        sd = strip_module_prefix(sd)

    missing, unexpected = model.load_state_dict(sd, strict=strict)
    if strict is False and (missing or unexpected):
        print_once(f"[CKPT] non-strict: missing={len(missing)} unexpected={len(unexpected)}")

    if optimizer is not None and ckpt.get("optimizer") is not None:
        try:
            optimizer.load_state_dict(ckpt["optimizer"])

            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(device)
            print_once(f"[CKPT] optimizer state moved to {device}")
        except Exception as e:
            print_once(f"[CKPT] optimizer load failed, continue fresh: {e}")

    step = int(ckpt.get("step", 0))
    best_val = ckpt.get("best_val", None)
    print_once(f"[CKPT] loaded. resume step={step}, best_val={best_val}")
    return step, best_val


def load_checkpoint_config(ckpt_path: str, device: str = "cpu") -> Dict[str, Any]:
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    model_config = ckpt.get("model_config", {})
    full_config = ckpt.get("config", {})
    model_section = full_config.get("MODEL", {}) if isinstance(full_config, dict) else {}

    if not model_config:

        if model_section:
            model_config = {
                "USE_CONTEXT_AS_CONTENT": model_section.get("USE_CONTEXT_AS_CONTENT", False),
            }
            print_once(f"[CKPT]  model_config not found, extracted from config")
        else:
            print_once(f"[CKPT]  No config found in checkpoint, using defaults")
    else:
        print_once(f"[CKPT]  model_config loaded: {list(model_config.keys())}")


    if model_section:
        extra_keys = [

            "USE_CONTEXT_AS_CONTENT",

            "STYLE_DIM",
            "FONT_DIM",
            "FONT_HEAD_LAYERS",
            "HWGEN_DIM",
            "HWGEN_WRITER_LAYERS",
            "HWGEN_GLYPH_LAYERS",

            "ENCODER_TYPE",
            "BASE_LAYERS",
            "BASE_NHEAD",
            "HEAD_LAYERS",
            "HEAD_NHEAD",
            "PATCH_SIZE",

            "CONTEXT_BACKBONE",
            "CONTEXT_DIM",
            "CONTEXT_LAYERS",
            "CONTEXT_NHEAD",
            "CONTEXT_DROPOUT",
            "CONTEXT_FREEZE_BACKBONE",
            "CONTEXT_USE_CHAR_ID_EMB",
            "CONTEXT_CHAR_ID_EMB_DIM",
            "CONTEXT_MAX_CHAR_ID",

            "USE_ROPE",
            "ROPE_BASE",
            "USE_CONTEXT_GATING",
            "CONTEXT_GATE_INIT",

            "ENCODER_VQ_START_ITER",
            "ENCODER_VQ_CODEBOOK_SIZE",
            "ENCODER_VQ_EMBED_DIM",
            "ENCODER_VQ_COMMITMENT_WEIGHT",
            "ENCODER_VQ_CODEBOOK_DECAY",

            "DECODER_VQ_START_ITER",
            "DECODER_VQ_NUM_QUANTIZERS",
            "DECODER_VQ_CODEBOOK_SIZE",
            "DECODER_VQ_EMBED_DIM",
            "DECODER_VQ_COMMITMENT_WEIGHT",
            "DECODER_VQ_CODEBOOK_DECAY",
            "DECODER_VQ_CLAMP_SCALE",
            "DECODER_VQ_GATE_INIT",
        ]
        for key in extra_keys:
            if key not in model_config and key in model_section:
                model_config[key] = model_section[key]

    return model_config


def load_latest_checkpoint(
    ckpt_dir: str,
    model: torch.nn.Module,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
    *,
    pattern: str = "ckpt_*.pt",
    strict: bool = False,
) -> Optional[Tuple[int, Optional[float]]]:
    if not os.path.isdir(ckpt_dir):
        print_once(f"[CKPT] dir not found: {ckpt_dir}")
        return None
    paths = glob.glob(os.path.join(ckpt_dir, pattern))
    if not paths:
        print_once(f"[CKPT] no files in {ckpt_dir}")
        return None

    def _parse_step(p: str) -> int:
        m = re.search(r"ckpt_(\d+)\.pt$", os.path.basename(p))
        return int(m.group(1)) if m else -1

    paths.sort(key=_parse_step)
    latest = paths[-1]
    return load_checkpoint(latest, model, device, optimizer, strict=strict)


def _first_eoc_index(norm_traj_np: np.ndarray) -> Optional[int]:
    if norm_traj_np.shape[1] <= TRAJ_INDEX["EOC"]:
        return None
    hits = np.where(norm_traj_np[:, TRAJ_INDEX["EOC"]] >= 1)[0]
    return int(hits[0]) if len(hits) > 0 else None


def visualize_snapshot_chars(
    tb_writer,
    pred_gmms: List[torch.Tensor],
    gt_coords_list: List[torch.Tensor],
    characters: List[str],
    step: int,
    font_dataset,
    IMG_SIZE: int = 64,
    mode: str = "train",
    coord_space: str = "delta",
    gt_start_xy_list: Optional[List[np.ndarray]] = None,
    char_gap: int = 10,
    infer_gmms: Optional[List[torch.Tensor]] = None,
):
    def _ensure_btC(x: torch.Tensor) -> torch.Tensor:

        if x.dim() == 2:
            return x.unsqueeze(0)
        if x.dim() == 3 and x.size(0) == 1:
            return x
        raise AssertionError(f"expected [T,C] or [1,T,C], got {tuple(x.shape)}")

    char_imgs = []
    gt_abs_list = []
    pred_abs_list = []
    infer_abs_list = []

    for idx, (pred_gmm, gt_coords, ch) in enumerate(zip(pred_gmms, gt_coords_list, characters)):

        if isinstance(pred_gmm, list):
            pred_gmm = torch.stack(pred_gmm, dim=0)
        pred_btC = _ensure_btC(pred_gmm)


        if isinstance(gt_coords, np.ndarray):
            gt_coords = torch.tensor(gt_coords, dtype=torch.float32)


        with torch.no_grad():
            traj_pred_bt = get_seq_from_gmm(pred_btC)
            traj_pred = traj_pred_bt[0]
            if coord_space == "delta":
                traj_pred = delta_to_abs_norm(traj_pred, start_xy=None)
            traj_pred_np = traj_pred.detach().cpu().numpy()


            from src.config.constants import TRAJ_INDEX_EXPANDED
            cursive_eoc = traj_pred_np[:, TRAJ_INDEX_EXPANDED["CURSIVE_EOC"]] > 0.5
            eoc = traj_pred_np[:, TRAJ_INDEX_EXPANDED["EOC"]] > 0.5
            any_eoc = cursive_eoc | eoc
            eoc_indices = np.where(any_eoc)[0]
            if len(eoc_indices) > 0:
                first_eoc = eoc_indices[0]
                traj_pred_np = traj_pred_np[:first_eoc+1]


        infer_abs_np = None
        if infer_gmms is not None and idx < len(infer_gmms) and infer_gmms[idx] is not None:
            inf_btC = _ensure_btC(infer_gmms[idx])

            print_once(f"[VIS-DEBUG] Inference GMM shape: {inf_btC.shape}, char={ch}")

            with torch.no_grad():
                traj_inf_bt = get_seq_from_gmm(inf_btC)
                traj_inf = traj_inf_bt[0]

                print_once(f"[VIS-DEBUG] Generated traj shape: {traj_inf.shape}")

                if coord_space == "delta":
                    traj_inf = delta_to_abs_norm(traj_inf, start_xy=None)
                infer_abs_np = traj_inf.detach().cpu().numpy()


                from src.config.constants import TRAJ_INDEX_EXPANDED
                cursive_eoc_inf = infer_abs_np[:, TRAJ_INDEX_EXPANDED["CURSIVE_EOC"]] > 0.5
                eoc_inf = infer_abs_np[:, TRAJ_INDEX_EXPANDED["EOC"]] > 0.5
                any_eoc_inf = cursive_eoc_inf | eoc_inf
                print_once(f"[VIS-DEBUG] EOC values: {infer_abs_np[:5, TRAJ_INDEX_EXPANDED['EOC']]} (first 5)")
                print_once(f"[VIS-DEBUG] EOC indices: {np.where(any_eoc_inf)[0]}")


                eoc_indices_inf = np.where(any_eoc_inf)[0]
                if len(eoc_indices_inf) > 0:
                    first_eoc_inf = eoc_indices_inf[0]
                    infer_abs_np = infer_abs_np[:first_eoc_inf+1]
                    print_once(f"[VIS-DEBUG] After EOC cut: shape={infer_abs_np.shape}")
                else:
                    print_once(f"[VIS-DEBUG]   No EOC found! shape={infer_abs_np.shape}")


        traj_gt_np = gt_coords.detach().cpu().numpy()
        if coord_space == "delta":
            start_xy = None if gt_start_xy_list is None else gt_start_xy_list[idx]
            traj_gt_np = delta_to_abs_norm(traj_gt_np, start_xy=start_xy)


        gt_abs   = denormalize_height_based(traj_gt_np,   IMG_SIZE)
        pred_abs = denormalize_height_based(traj_pred_np, IMG_SIZE)
        infer_abs = denormalize_height_based(infer_abs_np, IMG_SIZE) if infer_abs_np is not None else None


        font_char = ch[-1] if ch and len(ch) > 0 else ch
        char_img = font_dataset.get_char_img(font_char)

        char_imgs.append(char_img)
        gt_abs_list.append(gt_abs)
        pred_abs_list.append(pred_abs)
        infer_abs_list.append(infer_abs)


    visualize_character_level(
        char_imgs=char_imgs,
        gt_coords_list=gt_abs_list,
        pred_coords_list=pred_abs_list,
        infer_coords_list=infer_abs_list,
        chars=characters,
        step=step,
        tb_writer=tb_writer,
        tag_prefix=mode,
        image_size=IMG_SIZE,
        char_gap=char_gap
    )


def visualize_snapshot_sentence(
    gt_coords_list: List[torch.Tensor | np.ndarray],
    pred_gmm: torch.Tensor,
    sentence_chars: List[str],
    step: int,
    font_dataset,
    tb_writer,
    IMG_SIZE: int = 64,
    mode: str = "train",
    coord_space: str = "delta",
    gt_start_xy_list: Optional[List[np.ndarray]] = None,
    infer_gmm: Optional[torch.Tensor | List[torch.Tensor]] = None,
    n_gram_window: int = 2,
):
    def _ensure_btC(x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            return x.unsqueeze(0)
        if x.dim() == 3 and x.size(0) == 1:
            return x
        raise AssertionError(f"expected [T,C] or [1,T,C], got {tuple(x.shape)}")


    if pred_gmm is not None:
        if isinstance(pred_gmm, list):

            pass
        else:

            if pred_gmm.dim() == 4 and pred_gmm.size(1) == 1:
                pred_gmm = pred_gmm.squeeze(1)


    S = len(sentence_chars)


    if pred_gmm is not None:
        if isinstance(pred_gmm, list):
            if len(pred_gmm) > S:
                pred_gmm = pred_gmm[:S]
                print_once(f"[VIS]  pred_gmm (list) truncated from {len(pred_gmm)} to {S} chars")
        else:
            if pred_gmm.size(0) > S:
                pred_gmm = pred_gmm[:S]
                print_once(f"[VIS]  pred_gmm (tensor) truncated from {pred_gmm.size(0)} to {S} chars")

    if infer_gmm is not None:
        if isinstance(infer_gmm, list):
            if len(infer_gmm) > S:
                infer_gmm = infer_gmm[:S]
                print_once(f"[VIS]  infer_gmm (list) truncated from {len(infer_gmm)} to {S} chars")
        else:
            if infer_gmm.size(0) > S:
                infer_gmm = infer_gmm[:S]
                print_once(f"[VIS]  infer_gmm (tensor) truncated from {infer_gmm.size(0)} to {S} chars")

    char_imgs = []
    characters = []


    pred_delta_list = []
    infer_delta_list = []
    gt_delta_list = []

    for i in range(S):
        ch = sentence_chars[i]
        characters.append(ch)
        font_img = font_dataset.get_char_img(ch)
        if font_img is not None:
            char_imgs.append(font_img.cpu())
        else:

            char_imgs.append(None)


        gt_coords = gt_coords_list[i] if i < len(gt_coords_list) else None
        if gt_coords is None or (isinstance(gt_coords, torch.Tensor) and gt_coords.numel() == 0):
            gt_delta_list.append(None)
            pred_delta_list.append(None)
            infer_delta_list.append(None)
            continue


        if pred_gmm is not None:

            if isinstance(pred_gmm, list):
                pred_seq = pred_gmm[i] if i < len(pred_gmm) else None
            else:
                pred_seq = pred_gmm[i] if i < pred_gmm.size(0) else None

            if pred_seq is not None:
                pred_btC = _ensure_btC(pred_seq)
                with torch.no_grad():
                    traj_pred_bt = get_seq_from_gmm(pred_btC)
                    traj_pred_delta = traj_pred_bt[0].detach().cpu().numpy()


                    cursive_eoc = traj_pred_delta[:, TRAJ_INDEX_EXPANDED["CURSIVE_EOC"]] > 0.5
                    eoc = traj_pred_delta[:, TRAJ_INDEX_EXPANDED["EOC"]] > 0.5
                    any_eoc = cursive_eoc | eoc
                    eoc_indices = np.where(any_eoc)[0]
                    if len(eoc_indices) > 0:
                        traj_pred_delta = traj_pred_delta[:eoc_indices[0]+1]


                    pred_delta_list.append(traj_pred_delta)
            else:
                pred_delta_list.append(None)
        else:

            pred_delta_list.append(None)


        infer_delta_np = None
        if infer_gmm is not None:
            if isinstance(infer_gmm, list):
                inf_g = infer_gmm[i] if i < len(infer_gmm) else None
            else:
                inf_g = infer_gmm[i] if i < infer_gmm.size(0) else None

            if inf_g is not None:
                inf_btC = _ensure_btC(inf_g)
                with torch.no_grad():
                    traj_inf_bt = get_seq_from_gmm(inf_btC)
                    infer_delta_np = traj_inf_bt[0].detach().cpu().numpy()


                    cursive_eoc = infer_delta_np[:, TRAJ_INDEX_EXPANDED["CURSIVE_EOC"]] > 0.5
                    eoc = infer_delta_np[:, TRAJ_INDEX_EXPANDED["EOC"]] > 0.5
                    any_eoc = cursive_eoc | eoc
                    eoc_indices = np.where(any_eoc)[0]
                    if len(eoc_indices) > 0:
                        infer_delta_np = infer_delta_np[:eoc_indices[0]+1]


        infer_delta_list.append(infer_delta_np)


        traj_gt_np = gt_coords.detach().cpu().numpy() if isinstance(gt_coords, torch.Tensor) else gt_coords
        gt_delta_list.append(traj_gt_np)


    gt_abs_list = []
    pred_abs_list = []
    infer_abs_list = []


    current_xy_gt = None
    current_xy_pred = None

    for i in range(S):
        if gt_delta_list[i] is None:
            gt_abs_list.append(None)
            pred_abs_list.append(None)
            continue


        if coord_space == "delta":
            traj_gt_abs = delta_to_abs_norm(gt_delta_list[i], start_xy=current_xy_gt)
            if traj_gt_abs is not None and len(traj_gt_abs) > 0:
                current_xy_gt = traj_gt_abs[-1, :2].copy()
        else:
            traj_gt_abs = gt_delta_list[i]


        traj_pred_abs = None
        if pred_delta_list[i] is not None:
            traj_pred_abs = delta_to_abs_norm(pred_delta_list[i], start_xy=current_xy_pred)
            if traj_pred_abs is not None and len(traj_pred_abs) > 0:
                current_xy_pred = traj_pred_abs[-1, :2].copy()

        gt_abs = denormalize_height_based(traj_gt_abs, IMG_SIZE)
        pred_abs = denormalize_height_based(traj_pred_abs, IMG_SIZE) if traj_pred_abs is not None else None
        gt_abs_list.append(gt_abs)
        pred_abs_list.append(pred_abs)


    current_xy_infer = None
    for i in range(S):
        if gt_delta_list[i] is None:
            infer_abs_list.append(None)
            continue

        traj_infer_abs = None
        if infer_delta_list[i] is not None:
            traj_infer_abs = delta_to_abs_norm(infer_delta_list[i], start_xy=current_xy_infer)
            if traj_infer_abs is not None and len(traj_infer_abs) > 0:
                current_xy_infer = traj_infer_abs[-1, :2].copy()

        infer_abs = denormalize_height_based(traj_infer_abs, IMG_SIZE) if traj_infer_abs is not None else None
        infer_abs_list.append(infer_abs)

    visualize_sentence_level(
        tb_writer=tb_writer,
        char_imgs=char_imgs,
        gt_coords_list=gt_abs_list,
        pred_coords_list=pred_abs_list,
        infer_coords_list=infer_abs_list,
        sentence_chars=characters,
        tag_prefix=mode,
        image_size=IMG_SIZE,
        step=step,
    )
