from typing import Tuple, Union, Optional, List, Dict
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import math

from src.utils.logger import print_once, print_trace
from src.model.transformer_decoder import TransformerDecoder, TransformerDecoderLayer
from src.model.residual_vq import ResidualVQBranch

from src.utils.train_util import (
    Seq2Emb, Emb2Seq, get_seq_from_gmm,
    make_square_subsequent_mask, get_mixture_coef
)
from src.config.constants import TRAJ_INDEX, PEN_STATE_DIM


class HandwritingGenerator(nn.Module):
    def __init__(
        self,
        d_model=256,
        nhead=8,
        dim_feedforward=2048,
        writer_layers=2,
        glyph_layers=2,
        context_layers=1,
        dropout=0.1,
        activation="relu",
        normalize_before=True,
        return_intermediate_dec=False,
        pen_state_dim=PEN_STATE_DIM,
        use_rope=True,
        rope_base=100.0,

        use_residual_vq=False,
        rvq_num_quantizers=2,
        rvq_codebook_size=512,
        rvq_embed_dim=64,
        rvq_commitment_weight=0.25,
        rvq_codebook_decay=0.99,
        rvq_clamp_scale=0.05,
        rvq_gate_init=0.0,

        use_context_gating=False,
        context_gate_init=0.5,
        context_gate_tokenwise=False,
        context_gate_cap=1.0,

        use_context_decoder=True,


        use_context_pen_from_glyph_only=False,
    ):
        super().__init__()
        print_once(f"[HandwritingGenerator] Initializing with d_model={d_model}, nhead={nhead}, "
                   f"writer_layers={writer_layers}, glyph_layers={glyph_layers}, use_rope={use_rope}")

        self.use_rope = use_rope
        self.d_model = d_model
        self.use_residual_vq = use_residual_vq
        self.use_context_gating = use_context_gating
        self.context_gate_tokenwise = bool(context_gate_tokenwise)
        self.context_gate_cap = float(max(0.0, min(1.0, context_gate_cap)))
        self.use_context_decoder = use_context_decoder
        self.use_context_pen_from_glyph_only = bool(use_context_pen_from_glyph_only)

        if not use_context_decoder:
            print_once(f"[HandwritingGenerator]  ABLATION: Context Decoder DISABLED (use_context_decoder=False)")
        if self.use_context_pen_from_glyph_only:
            print_once("[HandwritingGenerator]  Pen-state head source: glyph-only (context bypass for pen logits)")


        if use_context_gating:
            if self.context_gate_tokenwise:


                self.context_gate = nn.Parameter(torch.tensor(context_gate_init))
                self.context_gate_linear = nn.Linear(d_model, 1, bias=False)

                nn.init.zeros_(self.context_gate_linear.weight)
                print_once(
                    f"[HandwritingGenerator]  Context Gating(token-wise) enabled: "
                    f"init={context_gate_init:.2f}, cap={self.context_gate_cap:.2f}"
                )
            else:
                self.context_gate = nn.Parameter(torch.tensor(context_gate_init))
                self.context_gate_linear = None
                print_once(
                    f"[HandwritingGenerator]  Context Gating(scalar) enabled: "
                    f"init={context_gate_init:.2f}, cap={self.context_gate_cap:.2f}"
                )
        else:
            self.context_gate = None
            self.context_gate_linear = None


        writer_dec_layer = TransformerDecoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation, normalize_before,
            use_rope=use_rope, rope_base=rope_base
        )
        self.writer_decoder = TransformerDecoder(
            writer_dec_layer, writer_layers, nn.LayerNorm(d_model)
        )

        glyph_dec_layer = TransformerDecoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation, normalize_before,
            use_rope=use_rope, rope_base=rope_base
        )
        self.glyph_decoder = TransformerDecoder(
            glyph_dec_layer, glyph_layers, nn.LayerNorm(d_model)
        )

        context_dec_layer = TransformerDecoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation, normalize_before,
            use_rope=use_rope, rope_base=rope_base
        )
        self.context_decoder = TransformerDecoder(
            context_dec_layer, context_layers, nn.LayerNorm(d_model)
        )

        self.pen_state_dim = pen_state_dim
        N_MIX = 20
        out_feature_cnt = 6


        self.emb_to_seq = Emb2Seq(d_model, self.pen_state_dim + N_MIX * out_feature_cnt)
        self.seq_to_emb = Seq2Emb(d_model)

        self.BOS_EMB = nn.Parameter(torch.randn(1, d_model))
        self.SPACE_EMB = nn.Parameter(torch.randn(1, d_model))


        self.z_alpha = 0.5


        if self.use_residual_vq:
            self.rvq_branch = ResidualVQBranch(
                d_model=d_model,
                num_quantizers=rvq_num_quantizers,
                codebook_size=rvq_codebook_size,
                vq_dim=rvq_embed_dim,
                commitment_weight=rvq_commitment_weight,
                codebook_decay=rvq_codebook_decay,
                clamp_scale=rvq_clamp_scale,
                gate_init=rvq_gate_init,
            )
            print_once(f"[HandwritingGenerator]  Residual VQ enabled: L={rvq_num_quantizers}, "
                       f"K={rvq_codebook_size}, C={rvq_embed_dim}, clamp={rvq_clamp_scale}")


            self._last_vq_loss = None
            self._last_vq_gate = None
        else:
            self.rvq_branch = None

            print_once(f"[HandwritingGenerator]  Residual VQ disabled (baseline mode)")

    def _compute_context_gate(self, gate_input: torch.Tensor) -> torch.Tensor:
        if self.context_gate_linear is not None:
            logits = self.context_gate_linear(gate_input) + self.context_gate.view(1, 1, 1)
            gate = torch.sigmoid(logits)
        else:
            gate = torch.sigmoid(self.context_gate)

        if self.context_gate_cap < 1.0:
            gate = gate * self.context_gate_cap
        return gate


    def _g_update_z(self, prev_z, char_emb):
        if prev_z is None:
            return char_emb
        if prev_z.dim() == 2:
            prev_z = prev_z.squeeze(0)
        a = self.z_alpha
        return a * prev_z + (1.0 - a) * char_emb


    def _apply_rvq_correction(
        self,
        gmm_params: torch.Tensor,
        delta: torch.Tensor,
        gate: torch.Tensor,
    ) -> torch.Tensor:
        K, L, C = gmm_params.shape
        num_mixtures = 20


        pen_logits = gmm_params[..., :self.pen_state_dim]
        rest = gmm_params[..., self.pen_state_dim:]


        pi_logits = rest[..., :num_mixtures]
        mu1 = rest[..., num_mixtures:2*num_mixtures]
        mu2 = rest[..., 2*num_mixtures:3*num_mixtures]
        s1_raw = rest[..., 3*num_mixtures:4*num_mixtures]
        s2_raw = rest[..., 4*num_mixtures:5*num_mixtures]
        rho_raw = rest[..., 5*num_mixtures:]


        delta_x = delta[..., 0:1]
        delta_y = delta[..., 1:2]

        mu1_corrected = mu1 + gate * delta_x
        mu2_corrected = mu2 + gate * delta_y


        rest_corrected = torch.cat([
            pi_logits,
            mu1_corrected,
            mu2_corrected,
            s1_raw,
            s2_raw,
            rho_raw,
        ], dim=-1)

        gmm_params_corrected = torch.cat([pen_logits, rest_corrected], dim=-1)

        return gmm_params_corrected

    def forward(
        self,
        *,

        content_embs: torch.Tensor,
        curr_traj_embs: torch.Tensor,
        writer_memory: torch.Tensor,
        glyph_memory: torch.Tensor,
        context_memory: Optional[torch.Tensor],
        char_mask: Optional[torch.Tensor],

        prev_traj_embs: Optional[torch.Tensor] = None,
        prev_content_embs: Optional[torch.Tensor] = None,
        context_memory_prev: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        dev: torch.device = content_embs.device
        nb: bool = (dev.type == "cuda")

        BS, _, D = content_embs.shape
        Tc = curr_traj_embs.size(1)

        dtype = content_embs.dtype


        if char_mask is None:
            char_mask_b = torch.ones(BS, dtype=torch.bool, device=dev)
        else:
            char_mask_b = (char_mask.view(BS, -1)[:, 0] > 0).to(dtype=torch.bool, device=dev, non_blocking=nb)


        curr_valid = (curr_traj_embs.abs().sum(dim=2) > 0)
        Tc_len = curr_valid.sum(dim=1)
        has_prev_inputs = (prev_traj_embs is not None) and (prev_content_embs is not None)

        if has_prev_inputs:
            prev_valid = (prev_traj_embs.abs().sum(dim=2) > 0)
            Tp_len = prev_valid.sum(dim=1)
            prev_on = (Tp_len > 0)
        else:
            Tp_len = torch.zeros(BS, dtype=torch.long, device=dev)
            prev_on = torch.zeros(BS, dtype=torch.bool, device=dev)
            has_prev_inputs = False


        Tp_max_eff = int(Tp_len.max().item()) if has_prev_inputs else 0
        Tc_max_eff = int(Tc_len.max().item())
        has_prev_any = bool(prev_on.any().item()) if has_prev_inputs else False

        Lmax = (1 + Tp_max_eff if has_prev_any else 0) + 1 + Tc_max_eff

        input_emb = torch.zeros(BS, Lmax, D, device=dev, dtype=dtype)


        if has_prev_any:
            idx_rows = prev_on.nonzero(as_tuple=False).squeeze(1)
            if idx_rows.numel() > 0:
                input_emb[idx_rows, 0, :] = prev_content_embs[idx_rows, 0, :].to(dtype=dtype)


        if has_prev_any and Tp_max_eff > 0:
            prev_traj_cut = prev_traj_embs[:, :Tp_max_eff, :].to(dtype=dtype)
            on_mask = prev_on.view(BS, 1, 1).to(dtype=dtype)
            input_emb[:, 1:1+Tp_max_eff, :] = prev_traj_cut * on_mask


        content_idx = torch.where(prev_on, 1 + Tp_len, torch.zeros_like(Tp_len))
        rows = torch.arange(BS, device=dev)
        input_emb[rows, content_idx, :] = content_embs[:, 0, :]


        if Tc_max_eff > 0:
            arT  = torch.arange(Tc_max_eff, device=dev).unsqueeze(0).expand(BS, Tc_max_eff)
            dest = (content_idx.unsqueeze(1) + 1 + arT).clamp_(0, Lmax - 1)
            src  = curr_traj_embs[:, :Tc_max_eff, :].to(dtype=dtype)
            input_emb.scatter_(1, dest.unsqueeze(-1).expand(-1, -1, D), src)


        position_ids = None
        if self.use_rope:

            position_ids = torch.arange(Lmax, device=dev, dtype=torch.long).unsqueeze(0).expand(BS, -1)


        ar = torch.arange(Lmax, device=dev).unsqueeze(0)
        content_idx = torch.where(prev_on, 1 + Tp_len, torch.zeros_like(Tp_len))


        prev_span_len = (1 + Tp_len).clamp(min=0)
        prev_valid_mask = (ar < prev_span_len.unsqueeze(1)) & prev_on.unsqueeze(1)


        curr_left  = content_idx.unsqueeze(1)
        curr_right = content_idx.unsqueeze(1) + 1 + Tc_len.unsqueeze(1)
        curr_valid_mask = (ar >= curr_left) & (ar < curr_right)

        valid = prev_valid_mask | curr_valid_mask
        tgt_kpm = ~valid


        causal_mask = make_square_subsequent_mask(Lmax, device=dev)


        w_out = self.writer_decoder(input_emb, writer_memory,
                                    tgt_mask=causal_mask, tgt_key_padding_mask=tgt_kpm,
                                    position_ids=position_ids)
        g_out = self.glyph_decoder(w_out, glyph_memory,
                                   tgt_mask=causal_mask, tgt_key_padding_mask=tgt_kpm,
                                   position_ids=position_ids)


        if self.use_context_decoder and context_memory is not None:
            c_out = self.context_decoder(g_out, context_memory,
                                         tgt_mask=causal_mask, tgt_key_padding_mask=tgt_kpm,
                                         position_ids=position_ids)
            if self.use_context_gating:


                gate = self._compute_context_gate(g_out)
                decode_out = g_out + gate * (c_out - g_out)
            else:

                decode_out = c_out
        else:
            decode_out = g_out


        if Tc > 0:
            start = content_idx
            arT = torch.arange(Tc, device=dev).unsqueeze(0).expand(BS, Tc)
            idx = (start.unsqueeze(1) + arT).clamp_(0, Lmax - 1)
            bix = torch.arange(BS, device=dev).unsqueeze(1).expand(BS, Tc)
            decode_curr = decode_out[bix, idx, :]
        else:
            decode_curr = decode_out[:, 0:0, :]

        pred = self.emb_to_seq(decode_curr)


        if self.use_context_pen_from_glyph_only and Tc > 0:
            glyph_curr = g_out[bix, idx, :]
            pred_glyph = self.emb_to_seq(glyph_curr)
            pred[..., :self.pen_state_dim] = pred_glyph[..., :self.pen_state_dim]


        if (~char_mask_b).any():
            bad = (~char_mask_b).nonzero(as_tuple=False).squeeze(1)
            if bad.numel() > 0:
                pred[bad] = 0.0

        return pred

    def inference(
        self,
        *,
        content_emb: torch.Tensor,
        writer_memory: torch.Tensor,
        glyph_memory: torch.Tensor,
        max_len: int,
        context_memory: Optional[torch.Tensor] = None,
        prefix_embs: Optional[torch.Tensor] = None,
        return_traj: bool = False,

    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        dev: torch.device = content_emb.device
        nb: bool = (dev.type == "cuda")
        D = content_emb.shape[-1]
        dtype = content_emb.dtype


        L_p = 0 if (prefix_embs is None) else int(prefix_embs.size(1))
        L_total = L_p + 1 + max_len

        raw = torch.zeros(1, L_total, D, device=dev, dtype=dtype)
        if L_p > 0:
            raw[:, :L_p, :] = prefix_embs.to(device=dev, dtype=dtype, non_blocking=nb)

        raw[:, L_p, :] = content_emb[0, 0]


        causal = make_square_subsequent_mask(L_total, device=dev)
        pred_gmm: Optional[torch.Tensor] = None
        traj_pts: List[torch.Tensor] = []

        for t in range(max_len):
            cur = L_p + 1 + t
            x = raw[:, :cur, :]


            position_ids = None
            if self.use_rope:
                position_ids = torch.arange(cur, device=dev, dtype=torch.long).unsqueeze(0)

            w_out = self.writer_decoder(x, writer_memory, tgt_mask=causal[:cur, :cur],
                                        position_ids=position_ids)
            g_out = self.glyph_decoder(w_out, glyph_memory, tgt_mask=causal[:cur, :cur],
                                       position_ids=position_ids)


            if self.use_context_decoder and context_memory is not None:
                c_out = self.context_decoder(g_out, context_memory, tgt_mask=causal[:cur, :cur],
                                            position_ids=position_ids)
                if self.use_context_gating:

                    gate = self._compute_context_gate(g_out)
                    dec = g_out + gate * (c_out - g_out)
                else:

                    dec = c_out
            else:
                dec = g_out

            h_t = dec[:, L_p + t, :]
            z_t = self.emb_to_seq(h_t.unsqueeze(1))


            if self.use_context_pen_from_glyph_only:
                h_t_glyph = g_out[:, L_p + t, :]
                z_t_glyph = self.emb_to_seq(h_t_glyph.unsqueeze(1))
                z_t[..., :self.pen_state_dim] = z_t_glyph[..., :self.pen_state_dim]


            apply_vq_inf = self.use_residual_vq and self.rvq_branch is not None
            if apply_vq_inf and hasattr(self, '_last_trained_gate'):
                apply_vq_inf = (self._last_trained_gate > 0.1)
            elif apply_vq_inf:
                apply_vq_inf = False

            if apply_vq_inf:

                with torch.no_grad():
                    delta_global, gate, _, _ = self.rvq_branch(h_t.unsqueeze(1), theta_line=None)
                    z_t = self._apply_rvq_correction(z_t, delta_global, gate)

            if pred_gmm is None:
                C = z_t.size(-1)
                pred_gmm = torch.zeros(max_len, C, device=dev, dtype=z_t.dtype)
            pred_gmm[t] = z_t[0, 0]


            coord_t = get_seq_from_gmm(
                z_t, decode="argmax_onehot"
            )


            cursive_eoc = coord_t[0, 0, 4].item()
            eoc = coord_t[0, 0, 5].item()
            is_eoc = (cursive_eoc > 0.5) or (eoc > 0.5)


            z_pen_logits = z_t[0, 0, :4]
            pen_probs = torch.softmax(z_pen_logits, dim=0)
            if return_traj:
                traj_pts.append(coord_t[0, 0].detach())
            x_next = self.seq_to_emb(coord_t)
            if x_next.dtype != dtype:
                x_next = x_next.to(dtype=dtype)
            if cur < L_total:
                raw[:, cur, :] = x_next[0, 0]


            if is_eoc:
                pred_gmm = pred_gmm[:t+1]
                break

        if pred_gmm is None:
            pred_gmm = torch.zeros(0, 124, device=dev, dtype=dtype)

        if return_traj:

            traj = torch.stack(traj_pts, dim=0) if len(traj_pts) > 0 else torch.zeros(0, 6, device=dev, dtype=dtype)
            return pred_gmm, traj
        else:
            return pred_gmm
