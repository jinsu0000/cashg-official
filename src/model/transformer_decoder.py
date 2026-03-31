import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional
from src.utils.logger import print_once, print_trace
from src.model.attention_rope import MultiheadAttentionWithRoPE, MultiheadCrossAttentionWithRoPE
import copy

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, use_rope=True, rope_base=10000.0):
        super().__init__()
        print_once(f"[TransformerDecoderLayer]: d_model={d_model}, nhead={nhead}, dim_feedforward={dim_feedforward}, dropout={dropout}, activation={activation}, normalize_before={normalize_before}, use_rope={use_rope}")

        self.use_rope = use_rope


        if use_rope:

            self.self_attn = MultiheadAttentionWithRoPE(
                d_model, nhead, dropout=dropout, batch_first=True,
                use_rope=True, rope_base=rope_base
            )
            self.multihead_attn = MultiheadCrossAttentionWithRoPE(
                d_model, nhead, dropout=dropout, batch_first=True,
                use_rope=True, rope_base=rope_base
            )
        else:

            self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
            self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = getattr(torch.nn.functional, activation) if isinstance(activation, str) else activation
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask=None, memory_mask=None,
                     tgt_key_padding_mask=None, memory_key_padding_mask=None,
                     pos=None, query_pos=None, position_ids=None):


        print_trace(
            f"[TransformerDecoder] forward: tgt.shape={tgt.shape}, "
            f"memory.shape={memory.shape}, "
            f"tgt_mask = {tgt_mask.shape if tgt_mask is not None else None}"
        )


        if self.use_rope:

            tgt2, _ = self.self_attn(
                tgt, tgt, tgt,
                attn_mask=tgt_mask,
                key_padding_mask=tgt_key_padding_mask,
                position_ids=position_ids
            )
        else:

            q = k = self.with_pos_embed(tgt, query_pos)
            tgt2, _ = self.self_attn(
                q, k, tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask
            )
        print_trace(f"[DecoderLayer] after self_attn: tgt2.shape={tgt2.shape}")

        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)


        if self.use_rope:

            tgt2, _ = self.multihead_attn(
                tgt, memory, memory,
                attn_mask=memory_mask,
                key_padding_mask=memory_key_padding_mask,
                query_position_ids=position_ids
            )
        else:

            tgt2, _ = self.multihead_attn(
                self.with_pos_embed(tgt, query_pos),
                self.with_pos_embed(memory, pos),
                memory,
                attn_mask=memory_mask,
                key_padding_mask=memory_key_padding_mask
            )
        print_trace(f"[DecoderLayer] after cross_attn: tgt2.shape={tgt2.shape}")
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)


        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        print_trace(f"[DecoderLayer] after FFN: tgt.shape={tgt.shape}")
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask=None, memory_mask=None,
                    tgt_key_padding_mask=None, memory_key_padding_mask=None,
                    pos=None, query_pos=None, position_ids=None):
        print_trace(
            f"[TransformerDecoder] forward: tgt.shape={tgt.shape}, "
            f"memory.shape={memory.shape}, "
            f"tgt_mask = {tgt_mask.shape if tgt_mask is not None else None}"
        )
        tgt2 = self.norm1(tgt)


        if self.use_rope:
            tgt2, _ = self.self_attn(
                tgt2, tgt2, tgt2,
                attn_mask=tgt_mask,
                key_padding_mask=tgt_key_padding_mask,
                position_ids=position_ids
            )
        else:
            q = k = self.with_pos_embed(tgt2, query_pos)
            tgt2, _ = self.self_attn(q, k, tgt2, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)
        print_trace(f"[DecoderLayer] after self_attn: tgt2.shape={tgt2.shape}")

        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm2(tgt)


        if self.use_rope:
            tgt2, _ = self.multihead_attn(
                tgt, memory, memory,
                attn_mask=memory_mask,
                key_padding_mask=memory_key_padding_mask,
                query_position_ids=position_ids
            )
        else:
            tgt2, _ = self.multihead_attn(
                self.with_pos_embed(tgt, query_pos),
                self.with_pos_embed(memory, pos),
                memory,
                attn_mask=memory_mask,
                key_padding_mask=memory_key_padding_mask
            )
        print_trace(f"[DecoderLayer] after cross_attn: tgt2.shape={tgt2.shape}")

        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        print_trace(f"[DecoderLayer] after FFN: tgt.shape={tgt.shape}")
        return tgt

    def forward(self, tgt, memory,
                tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None,
                pos=None, query_pos=None, position_ids=None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos, position_ids)
        else:
            return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos, position_ids)

class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        print_once(f"[TransformerDecoder] init num_layers={num_layers}, norm={norm}, return_intermediate={return_intermediate}")
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None,
                pos=None, query_pos=None, position_ids=None):
        print_trace(
            f"[TransformerDecoder] forward: tgt.shape={tgt.shape}, "
            f"memory.shape={memory.shape}, "
            f"tgt_mask = {tgt_mask.shape if tgt_mask is not None else None}"
        )
        output = tgt
        intermediate = []
        for idx, layer in enumerate(self.layers):
            output = layer(
                output, memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                pos=pos, query_pos=query_pos,
                position_ids=position_ids
            )
            print_trace(f"[TransformerDecoder] after layer{idx}: output.shape={output.shape}")
            if self.return_intermediate:
                intermediate.append(self.norm(output) if self.norm is not None else output)

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate[-1] = output

        print_trace(f"[TransformerDecoder] final output.shape={output.shape}")
        if self.return_intermediate:
            return torch.stack(intermediate)
        return output
