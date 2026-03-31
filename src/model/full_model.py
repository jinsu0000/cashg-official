from typing import Optional, Dict, Any, List, Tuple, Union
import torch
import torch.nn as nn
from einops import rearrange

from src.model.font_encoder import FontEncoder
from src.model.context_encoder import ContextEncoder
from src.model.handwriting_generator import HandwritingGenerator
from src.model.style_identifier import StyleIdentifier
from src.utils.logger import print_once, print_trace
from src.utils.train_util import (
    convert_unigram_delta_list_to_bigram,
    convert_unigram_gmm_list_to_bigram,
)


class FullModel(nn.Module):
    def __init__(self, style_identifier: StyleIdentifier,
                 font_encoder: Optional[FontEncoder],
                 handwriting_generator: HandwritingGenerator,
                 context_encoder: Optional[ContextEncoder]=None,
                 use_context_as_content: bool = False,
                 n_gram_window: int = 2):
        super().__init__()
        self.style_identifier = style_identifier
        self.font_encoder = font_encoder
        self.handwriting_generator = handwriting_generator
        self.context_encoder = context_encoder
        self.use_context_as_content = use_context_as_content
        self.n_gram_window = n_gram_window

        if n_gram_window == 1:
            print_once(f"[FullModel]  ABLATION: Unigram mode (n_gram_window=1, 이전 글자 정보 사용 안 함)")
        elif n_gram_window >= 3:
            print_once(f"[FullModel]  WARNING: n_gram_window={n_gram_window} 요청됨, 현재 구조는 최대 2 지원. 2로 제한됨.")
            self.n_gram_window = 2


        if use_context_as_content:
            if context_encoder is None:
                raise ValueError("[FullModel] use_context_as_content=True requires context_encoder!")
            print_once("[FullModel]  USE_CONTEXT_AS_CONTENT=True: Font Encoder 비활성화, Context Encoder를 Content로 사용")
        else:
            if font_encoder is None:
                raise ValueError("[FullModel] use_context_as_content=False requires font_encoder!")
            print_once("[FullModel]  USE_CONTEXT_AS_CONTENT=False: Font Encoder 사용 (기본 모드)")

    def _encode_content_cached(
        self,
        seq_chars: torch.Tensor,
        device: torch.device
    ) -> torch.Tensor:
        B, S = seq_chars.shape
        seq_chars_long = seq_chars.to(device=device, dtype=torch.long)


        flat_chars = seq_chars_long.reshape(-1)
        valid_mask = flat_chars > 0
        valid_chars = flat_chars[valid_mask]

        if valid_chars.numel() == 0:

            D = self.context_encoder.d_model
            return torch.zeros(B, S, 1, D, device=device, dtype=torch.float32)

        unique_chars = torch.unique(valid_chars)
        N_unique = unique_chars.size(0)
        N_total = B * S
        N_valid = valid_chars.numel()


        if not hasattr(self, '_content_cache_logged'):
            self._content_cache_logged = True
            reduction = (1.0 - N_unique / N_valid) * 100 if N_valid > 0 else 0
            print(f"[DEBUG] ======================================", flush=True)
            print(f"[DEBUG] _encode_content_cached() 호출됨!", flush=True)
            print(f"[DEBUG] Loop 처리로 Content Emb 생성", flush=True)
            print_once(f"[FullModel] Content Emb Caching: {N_unique}/{N_valid} unique chars ({reduction:.1f}% reduction)")
            print_once(f"[FullModel] Batch size: B={B}, S={S}, Total={N_total}, Valid={N_valid}, Unique={N_unique}")
            print(f"[DEBUG] ======================================", flush=True)


        char_input = unique_chars.unsqueeze(1)
        char_mask = torch.ones(N_unique, 1, dtype=torch.bool, device=device)


        unique_embs = self.context_encoder(
            char_input,
            char_mask,
            apply_dropout=False,
            disable_hard_dropout=True,
        )
        unique_embs = unique_embs.squeeze(2).squeeze(1)
        D = unique_embs.size(-1)


        matches = flat_chars.unsqueeze(1) == unique_chars.unsqueeze(0)


        content_embs_flat = torch.matmul(matches.float(), unique_embs)


        content_embs = content_embs_flat.reshape(B, S, 1, D)
        return content_embs

    def encode_content(
        self,
        char_imgs: torch.Tensor,
        seq_chars: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[dict]]:
        B, S, _, H, W = char_imgs.shape
        dev = char_imgs.device

        font_vq_loss = None
        font_vq_info = None

        if self.use_context_as_content:


            if seq_chars is None:
                raise ValueError("[FullModel] use_context_as_content=True requires seq_chars!")


            content_embs = self._encode_content_cached(seq_chars, dev)
        else:

            x = char_imgs.reshape(B * S, 1, H, W)
            e, font_vq_loss, font_vq_info = self.font_encoder(x)
            content_embs = e.reshape(B, S, 1, -1)


        if seq_chars is not None:
            if seq_chars.size(1) == S:
                is_space = (seq_chars == ord(' '))
                if is_space.any():
                    space_emb = self.handwriting_generator.SPACE_EMB.view(1, 1, 1, -1)
                    space_mask = is_space.unsqueeze(-1).unsqueeze(-1)
                    content_embs = torch.where(
                        space_mask.to(dev),
                        space_emb.expand(B, S, 1, -1).to(dev, dtype=content_embs.dtype),
                        content_embs
                    )

        return content_embs, font_vq_loss, font_vq_info

    def forward(self,
        *,
        style_imgs: torch.Tensor,
        char_imgs: torch.Tensor,
        curr_traj_embs: torch.Tensor,
        writer_ids: torch.Tensor = None,
        loss_mask: torch.Tensor = None,
        seq_chars: Optional[torch.Tensor]=None,
        context_seq_chars: Optional[torch.Tensor]=None,
        indices: Optional[torch.Tensor]=None,
        global_iter: int = 0,
        style_cache: Optional[Dict[str,torch.Tensor]] = None,
        traj_lens: Optional[torch.Tensor] = None,
        gt_trajs: Optional[torch.Tensor] = None,
        **kwargs
    ):
        dev: torch.device = char_imgs.device
        nb: bool = (dev.type == "cuda")
        B, S = char_imgs.size(0), char_imgs.size(1)


        if loss_mask is None:
            loss_mask = torch.ones(B, S, device=dev, dtype=torch.bool)
        else:
            loss_mask = loss_mask.to(device=dev, dtype=torch.bool, non_blocking=nb)


        if style_cache is not None:
            writer_emb  = style_cache["writer_emb"]
            glyph_emb   = style_cache["glyph_emb"]
            writer_mem  = style_cache["writer_style"]
            glyph_feat  = style_cache["glyph_style"]
        else:
            writer_emb, glyph_emb, writer_mem, glyph_feat = self.style_identifier(style_imgs)

        writer_memory_base = writer_mem.to(dev, non_blocking=nb)
        glyph_memory_base = rearrange(glyph_feat, 'b t n c -> b (t n) c').to(dev, non_blocking=nb)


        content_embs, font_vq_loss, font_vq_info = self.encode_content(char_imgs, seq_chars)
        content_embs = content_embs.to(dev, non_blocking=nb)


        ctx_chars = context_seq_chars if context_seq_chars is not None else seq_chars
        context_memory: Optional[torch.Tensor] = None

        if (self.context_encoder is not None) and (ctx_chars is not None):
            text_mask = (ctx_chars > 0).to(device=dev, dtype=torch.bool, non_blocking=nb)
            ctx_chars_long = ctx_chars.to(device=dev, dtype=torch.long, non_blocking=nb)

            context_memory = self.context_encoder(ctx_chars_long, text_mask, apply_dropout=True)
        else:
            context_memory = None


        if (context_memory is not None) and (context_memory.numel() > 0):
            S_char = char_imgs.size(1)
            S_ctx  = context_memory.size(1)


            if indices is not None:

                max_idx = indices.max().item()
                min_idx = indices.min().item()
                if max_idx >= S_ctx or min_idx < 0:
                    raise ValueError(
                        f"[CONTEXT_VALIDATION] indices out of range!\n"
                        f"  indices range: [{min_idx}, {max_idx}]\n"
                        f"  context_memory shape: [B={B}, S_ctx={S_ctx}, 1, D={context_memory.size(-1)}]\n"
                        f"  Valid range: [0, {S_ctx-1}]\n"
                        f"  indices shape: {indices.shape}\n"
                        f"  indices[:3]: {indices[:3].tolist() if len(indices) > 0 else 'empty'}"
                    )


            if indices is not None:

                indices = indices.to(device=dev, dtype=torch.long, non_blocking=nb)


                if indices.size(1) == 1:
                    idx = indices.expand(B, S_char)
                elif indices.size(1) == 2:
                    idx = indices
                else:
                    idx = indices

                bidx = torch.arange(B, device=dev).unsqueeze(1).expand(B, idx.size(1))
                idx_clamp = idx.clamp(0, S_ctx - 1)
                context_memory = context_memory[bidx, idx_clamp, :, :]

                if context_memory.size(1) != S_char:
                    if context_memory.size(1) > S_char:
                        context_memory = context_memory[:, -S_char:, :, :]
                    else:

                        pad = torch.zeros(
                            B, S_char - context_memory.size(1), 1, context_memory.size(-1),
                            device=dev, dtype=context_memory.dtype
                        )
                        context_memory = torch.cat([context_memory, pad], dim=1)
            else:
                if S_ctx != S_char:
                    if S_ctx > S_char:
                        context_memory = context_memory[:, :S_char, :, :]
                    else:
                        pad = torch.zeros(
                            B, S_char - S_ctx, 1, context_memory.size(-1),
                            device=dev, dtype=context_memory.dtype
                        )
                        context_memory = torch.cat([context_memory, pad], dim=1)


        T = curr_traj_embs.size(2)
        prev_traj_embs = torch.zeros_like(curr_traj_embs)
        prev_content_embs = torch.zeros_like(content_embs)

        if S > 1 and self.n_gram_window >= 2:
            prev_traj_embs[:, 1:, :, :] = curr_traj_embs[:, :-1, :, :]
            prev_content_embs[:, 1:, :, :] = content_embs[:, :-1, :, :]


        writer_memory_exp = writer_memory_base.unsqueeze(1).expand(B, S, *writer_memory_base.shape[1:]).contiguous()
        glyph_memory_exp  = glyph_memory_base.unsqueeze(1).expand(B, S, *glyph_memory_base.shape[1:]).contiguous()


        BS = B * S

        content_flat    = content_embs.reshape(BS, 1, -1)
        prev_cont_flat  = prev_content_embs.reshape(BS, 1, -1)

        curr_traj_flat  = curr_traj_embs.reshape(BS, T, -1)
        prev_traj_flat  = prev_traj_embs.reshape(BS, T, -1)

        writer_mem_flat = writer_memory_exp.reshape(BS, writer_memory_exp.size(2), -1)
        glyph_mem_flat  = glyph_memory_exp.reshape(BS, glyph_memory_exp.size(2), -1)

        context_flat: Optional[torch.Tensor] = None
        if context_memory is not None:
            context_flat = context_memory.reshape(BS, 1, -1)


        char_mask_flat = loss_mask.reshape(BS)


        pred_gmm_flat = self.handwriting_generator(
            content_embs      = content_flat,
            curr_traj_embs    = curr_traj_flat,
            writer_memory     = writer_mem_flat,
            glyph_memory      = glyph_mem_flat,
            context_memory    = context_flat,
            prev_traj_embs    = prev_traj_flat,
            prev_content_embs = prev_cont_flat,
            char_mask         = char_mask_flat,
        )

        C = pred_gmm_flat.size(-1)
        pred_gmm = pred_gmm_flat.reshape(B, S, T, C)


        seq_mask = loss_mask.to(device=dev, dtype=pred_gmm.dtype, non_blocking=nb).unsqueeze(-1).unsqueeze(-1)
        pred_gmm = pred_gmm * seq_mask


        return {
            "pred_gmm": pred_gmm,
            "font_vq_loss": font_vq_loss,
            "font_vq_info": font_vq_info,
            "style_cache": {
                "writer_emb": writer_emb.to(dev, non_blocking=nb),
                "glyph_emb":  glyph_emb.to(dev, non_blocking=nb),
                "writer_style": writer_memory_base,
                "glyph_style":  glyph_feat.to(dev, non_blocking=nb),
            }
        }

    @torch.no_grad()
    def inference(
        self,
        *,
        style_imgs: torch.Tensor,
        char_imgs: torch.Tensor,
        writer_ids: torch.Tensor,
        global_iter: int,
        max_len: int = 256,
        seq_chars: Optional[torch.Tensor] = None,
        context_seq_chars: Optional[torch.Tensor] = None,
        indices: Optional[torch.Tensor] = None,
        gt_trajs: Optional[torch.Tensor] = None,
        use_gt_prev: bool = False,
    ):
        dev: torch.device = style_imgs.device
        nb: bool = (dev.type == "cuda")
        B, S = char_imgs.size(0), char_imgs.size(1)


        writer_emb, glyph_emb, writer_mem, glyph_feat = self.style_identifier(style_imgs)
        writer_memory = writer_mem.to(dev, non_blocking=nb)
        glyph_memory  = rearrange(glyph_feat, 'b t n c -> b (t n) c').to(dev, non_blocking=nb)


        content_embs, _, _ = self.encode_content(char_imgs, seq_chars=seq_chars)
        content_embs = content_embs.to(dev, non_blocking=nb)


        ctx_chars = context_seq_chars if context_seq_chars is not None else seq_chars

        context_memory_full: Optional[torch.Tensor] = None
        seq_chars_full: Optional[torch.Tensor] = None

        if (self.context_encoder is not None) and (ctx_chars is not None):

            seq_chars_full = ctx_chars.to(device=dev, dtype=torch.long, non_blocking=nb)
            tm_full = (seq_chars_full > 0).to(dtype=torch.bool, non_blocking=nb)

            context_memory_full = self.context_encoder(seq_chars_full, tm_full, apply_dropout=True)
            if context_memory_full.dim() == 3:
                context_memory_full = context_memory_full.unsqueeze(2)

        if indices is None:
            indices = torch.arange(S, device=dev).view(1, S).expand(B, S)
        else:
            indices = indices.to(device=dev, dtype=torch.long, non_blocking=nb)
            if indices.size(1) != S:
                if indices.size(1) > S:
                    indices = indices[:, :S]
                else:
                    pad = torch.zeros(B, S - indices.size(1), dtype=indices.dtype, device=dev)
                    indices = torch.cat([indices, pad], dim=1)
            if seq_chars_full is not None:
                S_full = seq_chars_full.size(1)
                indices = indices.clamp_(0, max(S_full - 1, 0))

        bix = torch.arange(B, device=dev).unsqueeze(1).expand(B, S)

        if context_memory_full is not None:
            context_memory = context_memory_full[bix, indices, 0, :]
            context_memory = context_memory.unsqueeze(2)
        else:
            context_memory = None


        out = []
        prev_traj_emb_cache = [None] * B
        prev_valid_idx_cache = [-1] * B


        if seq_chars_full is not None:

            valid_lengths = (seq_chars_full > 0).sum(dim=1)
        else:
            valid_lengths = torch.full((B,), S, device=dev, dtype=torch.long)

        for b in range(B):
            gmm_row = []
            traj_row = []

            S_valid = min(int(valid_lengths[b].item()), S)

            for s in range(S_valid):
                curr_content = content_embs[b:b+1, s:s+1, 0, :]
                curr_context = None if context_memory is None else context_memory[b:b+1, s:s+1, 0, :]
                w_mem_b = writer_memory[b:b+1]
                g_mem_b = glyph_memory[b:b+1]


                use_prefix = (s > 0) and (self.n_gram_window >= 2)


                if use_prefix and (prev_traj_emb_cache[b] is not None):
                    prev_content = content_embs[b:b+1, s-1:s, 0, :]

                    prefix_embs = torch.cat([prev_content, prev_traj_emb_cache[b]], dim=1)
                else:
                    prefix_embs = None

                if b == 0 and s < 2:
                    prefix_info = "None" if prefix_embs is None else f"shape={prefix_embs.shape}"
                    print(f"[DEBUG-INF] b={b}, s={s}, prefix_embs={prefix_info}, use_prefix={use_prefix}")


                is_space = (seq_chars_full is not None and
                           s < seq_chars_full.size(1) and
                           seq_chars_full[b, s] == ord(' '))
                max_len_char = 10 if is_space else max_len

                gmm_curr, traj_curr = self.handwriting_generator.inference(
                    content_emb=curr_content,
                    writer_memory=w_mem_b,
                    glyph_memory=g_mem_b,
                    max_len=max_len_char,
                    context_memory=curr_context,
                    prefix_embs=prefix_embs,
                    return_traj=True,

                )


                next_is_space = False
                if seq_chars_full is not None and (s + 1) < seq_chars_full.size(1):
                    next_is_space = (seq_chars_full[b, s + 1] == ord(' '))
                if next_is_space:
                    if gmm_curr is not None and gmm_curr.numel() > 0:
                        gmm_curr[-1, :4] = torch.tensor(
                            [-10.0, -10.0, -10.0, 10.0],
                            device=gmm_curr.device,
                            dtype=gmm_curr.dtype,
                        )
                    if traj_curr is not None and traj_curr.numel() > 0:
                        traj_curr[-1, 2:6] = torch.tensor(
                            [0.0, 0.0, 0.0, 1.0],
                            device=traj_curr.device,
                            dtype=traj_curr.dtype,
                        )

                gmm_row.append(gmm_curr)
                traj_row.append(traj_curr)

                if use_gt_prev and gt_trajs is not None:
                    gt_traj_curr = gt_trajs[b, s, :, :]
                    valid_mask = (gt_traj_curr.abs().sum(dim=-1) > 0)
                    if valid_mask.any():
                        traj_curr_gt = gt_traj_curr[valid_mask]
                        traj_emb = self.handwriting_generator.seq_to_emb(
                            traj_curr_gt.unsqueeze(0).to(device=dev, non_blocking=nb)
                        )
                        prev_traj_emb_cache[b] = traj_emb
                        prev_valid_idx_cache[b] = s
                        if b == 0 and s == 0:
                            print(f"[DEBUG] use_gt_prev=True: b={b}, s={s}, GT traj shape={traj_curr_gt.shape}, "
                                  f"AR traj shape={traj_curr.shape if traj_curr is not None else None}")
                    else:
                        prev_traj_emb_cache[b] = None

                elif traj_curr is None or traj_curr.numel() == 0:
                    prev_traj_emb_cache[b] = None

                else:

                    if self.n_gram_window >= 2:
                        traj_emb = self.handwriting_generator.seq_to_emb(
                            traj_curr.unsqueeze(0).to(device=dev, non_blocking=nb)
                        )

                        prev_traj_emb_cache[b] = traj_emb
                        prev_valid_idx_cache[b] = s


            if self.n_gram_window == 1 and len(gmm_row) > 0:
                traj_uni_np_list = []
                for traj in traj_row:
                    if traj is None or traj.numel() == 0:
                        traj_uni_np_list.append(None)
                    else:
                        traj_uni_np_list.append(traj.detach().cpu().numpy())

                traj_bigram_np_list = convert_unigram_delta_list_to_bigram(traj_uni_np_list)


                num_mixtures = 20
                for g in gmm_row:
                    if g is None or g.numel() == 0:
                        continue
                    C = int(g.size(-1))
                    if C > 4 and (C - 4) % 6 == 0:
                        num_mixtures = (C - 4) // 6
                    break

                gmm_row = convert_unigram_gmm_list_to_bigram(
                    gmm_row,
                    traj_bigram_np_list,
                    num_mixtures=num_mixtures,
                )

            out.append({"gmm": gmm_row})
        return out
