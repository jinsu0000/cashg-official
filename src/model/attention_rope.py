import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple
import math

from src.utils.train_util import RotaryPositionalEmbedding


class MultiheadAttentionWithRoPE(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        batch_first: bool = True,
        use_rope: bool = True,
        rope_base: float = 10000.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.use_rope = use_rope
        self.batch_first = batch_first

        assert self.head_dim * num_heads == embed_dim, \
            f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"


        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)
        self.scaling = 1.0 / math.sqrt(self.head_dim)


        if use_rope:
            self.rope = RotaryPositionalEmbedding(
                dim=self.head_dim,
                max_len=4096,
                base=rope_base
            )

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

        if self.q_proj.bias is not None:
            nn.init.constant_(self.q_proj.bias, 0.)
            nn.init.constant_(self.k_proj.bias, 0.)
            nn.init.constant_(self.v_proj.bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attn_mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        need_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:

        if not self.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        B, T, D = query.shape
        S = key.size(1)


        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)


        q = q.view(B, T, self.num_heads, self.head_dim)
        k = k.view(B, S, self.num_heads, self.head_dim)
        v = v.view(B, S, self.num_heads, self.head_dim)


        if self.use_rope:

            q, k = self.rope(q, k, position_ids)


        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)


        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scaling


        if attn_mask is not None:

            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
            scores = scores + attn_mask

        if key_padding_mask is not None:

            scores = scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf')
            )


        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)


        output = torch.matmul(attn_weights, v)


        output = output.transpose(1, 2).contiguous()
        output = output.view(B, T, D)


        output = self.out_proj(output)


        if not self.batch_first:
            output = output.transpose(0, 1)


        if need_weights:

            attn_weights = attn_weights.mean(dim=1)
            return output, attn_weights
        else:
            return output, None


class MultiheadCrossAttentionWithRoPE(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        batch_first: bool = True,
        use_rope: bool = True,
        rope_base: float = 10000.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.use_rope = use_rope
        self.batch_first = batch_first

        assert self.head_dim * num_heads == embed_dim, \
            f"embed_dim must be divisible by num_heads"


        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)
        self.scaling = 1.0 / math.sqrt(self.head_dim)


        if use_rope:
            self.rope_q = RotaryPositionalEmbedding(
                dim=self.head_dim,
                max_len=4096,
                base=rope_base
            )


        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

        if self.q_proj.bias is not None:
            nn.init.constant_(self.q_proj.bias, 0.)
            nn.init.constant_(self.k_proj.bias, 0.)
            nn.init.constant_(self.v_proj.bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attn_mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        query_position_ids: Optional[Tensor] = None,
        need_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        if not self.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        B, T, D = query.shape
        M = key.size(1)


        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)


        q = q.view(B, T, self.num_heads, self.head_dim)
        k = k.view(B, M, self.num_heads, self.head_dim)
        v = v.view(B, M, self.num_heads, self.head_dim)


        if self.use_rope and query_position_ids is not None:

            q_rot, _ = self.rope_q(q, q, query_position_ids)
            q = q_rot


        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)


        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scaling


        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
            scores = scores + attn_mask

        if key_padding_mask is not None:
            scores = scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf')
            )

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        output = torch.matmul(attn_weights, v)
        output = output.transpose(1, 2).contiguous()
        output = output.view(B, T, D)
        output = self.out_proj(output)

        if not self.batch_first:
            output = output.transpose(0, 1)

        if need_weights:
            attn_weights = attn_weights.mean(dim=1)
            return output, attn_weights
        else:
            return output, None
