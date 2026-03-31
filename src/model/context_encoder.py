import torch
import torch.nn as nn
from typing import List, Tuple, Optional
from transformers import AutoTokenizer, AutoModel, T5EncoderModel

class SinPosEnc(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32)
                        * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        L = x.size(1)
        return x + self.pe[:L].unsqueeze(0)


class ContextEncoder(nn.Module):
    def __init__(
        self,
        model_name: str = "google/canine-c",
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 2,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        add_sin_posenc: bool = True,
        freeze_backbone: bool = True,
        min_chars_for_canine: int = None,

        use_char_id_emb: bool = False,
        char_id_emb_dim: int = 64,
        max_char_id: int = 200000,
        context_token_dropout: float = 0.0,
    ):
        super().__init__()
        self.model_name = model_name.lower()
        self.is_byt5 = "byt5" in self.model_name
        self.is_canine = "canine" in self.model_name
        self.freeze_backbone = freeze_backbone
        self.use_char_id_emb = use_char_id_emb
        self.d_model = d_model


        self.tokenizer = AutoTokenizer.from_pretrained(model_name)


        if self.is_byt5:

            self.backbone = T5EncoderModel.from_pretrained(model_name)
            hidden = self.backbone.config.d_model
        else:

            self.backbone = AutoModel.from_pretrained(model_name)
            hidden = self.backbone.config.hidden_size

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            print(f"[ContextEncoder] Backbone frozen (freeze_backbone=True)")
        else:
            print(f"[ContextEncoder] Backbone trainable (freeze_backbone=False) - {sum(p.numel() for p in self.backbone.parameters())/1e6:.1f}M params")


        if use_char_id_emb:
            self.char_id_emb = nn.Embedding(max_char_id, char_id_emb_dim, padding_idx=0)
            self.char_id_proj = nn.Linear(char_id_emb_dim, d_model)


            self.char_id_scale = nn.Parameter(torch.tensor(0.1))

            nn.init.normal_(self.char_id_emb.weight, mean=0.0, std=0.02)
            nn.init.xavier_uniform_(self.char_id_proj.weight, gain=0.1)
            nn.init.zeros_(self.char_id_proj.bias)
            print(f"[ContextEncoder] Char ID Embedding enabled: {max_char_id} x {char_id_emb_dim}d -> {d_model}d (init_scale=0.1)")
        else:
            self.char_id_emb = None
            self.char_id_proj = None
            self.char_id_scale = None

        self.proj = nn.Linear(hidden, d_model)


        nn.init.xavier_uniform_(self.proj.weight, gain=0.1)
        nn.init.zeros_(self.proj.bias)

        self.add_sin = add_sin_posenc
        if self.add_sin:
            self.posenc = SinPosEnc(d_model)


        use_extra_encoder = freeze_backbone and (num_layers > 0)

        if use_extra_encoder:
            enc_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                dropout=dropout, batch_first=True, activation="gelu"
            )
            self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
            print(f"[ContextEncoder] Extra Transformer Encoder: {num_layers} layers (backbone frozen)")
        else:
            self.encoder = None
            if freeze_backbone:
                print(f"[ContextEncoder] Extra Transformer Encoder: DISABLED (num_layers=0)")
            else:
                print(f"[ContextEncoder] Extra Transformer Encoder: DISABLED (backbone fine-tuning, not needed)")


        if self.is_canine:
            default_min = int(getattr(self.backbone.config, "downsampling_rate", 4))
            self.min_chars_for_canine = int(min_chars_for_canine or max(4, default_min))
        else:
            self.min_chars_for_canine = 1


        self.max_char_id = max_char_id


        self.context_token_dropout = context_token_dropout
        if context_token_dropout < 0.0:
            print(f"[ContextEncoder] Context Token Dropout: {context_token_dropout:.1f} (Context 완전 비활성화)")
        elif context_token_dropout > 0.0:
            print(f"[ContextEncoder] Context Token Dropout: {context_token_dropout:.2f} (training only)")


    def _safe_chr(self, v: int) -> str:
        try:
            v = int(v)
            if 0 < v <= 0x10FFFF:
                return chr(v)
        except Exception:
            pass
        return "?"


    def _compute_char_to_byte_mapping(self, text: str) -> List[Tuple[int, int]]:
        mappings = []
        byte_pos = 0
        for char in text:
            char_bytes = char.encode('utf-8')
            num_bytes = len(char_bytes)
            mappings.append((byte_pos, byte_pos + num_bytes))
            byte_pos += num_bytes
        return mappings


    def _encode_texts_byt5(self, texts: List[str], device, apply_dropout: bool = True) -> Tuple[torch.Tensor, torch.Tensor, List[List[Tuple[int, int]]]]:


        batch = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=200
        )
        batch = {k: v.to(device) for k, v in batch.items()}


        backbone_training = self.backbone.training

        if not apply_dropout:
            self.backbone.eval()


        with torch.cuda.amp.autocast(enabled=False):
            H = self.backbone(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"]
            ).last_hidden_state

        attn_mask = batch["attention_mask"].bool()
        input_ids = batch["input_ids"]


        special = (input_ids <= 2)
        kpm = (~attn_mask) | special

        M = self.proj(H)
        if self.add_sin:
            M = self.posenc(M)


        if apply_dropout and self.encoder is not None:

            M_ctx = self.encoder(M, src_key_padding_mask=kpm)
        else:

            M_ctx = M


        if not apply_dropout and backbone_training:
            self.backbone.train()


        char_mappings = [self._compute_char_to_byte_mapping(t) for t in texts]

        return M_ctx, kpm, char_mappings


    def _encode_texts_canine(self, texts: List[str], device, apply_dropout: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        batch = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        batch = {k: v.to(device) for k, v in batch.items()}


        backbone_training = self.backbone.training

        if not apply_dropout:
            self.backbone.eval()

        H = self.backbone(**batch).last_hidden_state

        input_ids = batch["input_ids"]
        attn_mask = batch["attention_mask"].bool()
        special = torch.zeros_like(input_ids, dtype=torch.bool)
        for sid in self.tokenizer.all_special_ids:
            special |= (input_ids == sid)
        kpm = (~attn_mask) | special

        M = self.proj(H)
        if self.add_sin:
            M = self.posenc(M)


        if apply_dropout and self.encoder is not None:

            M_ctx = self.encoder(M, src_key_padding_mask=kpm)
        else:

            M_ctx = M


        if not apply_dropout and backbone_training:
            self.backbone.train()

        return M_ctx, kpm

    def forward(
        self,
        seq_chars: torch.Tensor,
        text_mask: torch.Tensor,
        apply_dropout: bool = True,
        disable_hard_dropout: bool = False,
    ):
        B, S = seq_chars.shape
        device = seq_chars.device


        if (not disable_hard_dropout) and self.training and self.context_token_dropout < 0.0:
            return torch.zeros(B, S, 1, self.d_model, device=device, dtype=torch.float32)


        if apply_dropout and self.training and self.context_token_dropout > 0.0:
            dropout_mask = torch.rand(B, S, device=device) > self.context_token_dropout


            for b in range(B):
                for s in range(1, S):

                    if text_mask[b, s-1] and text_mask[b, s]:
                        if not dropout_mask[b, s-1] and not dropout_mask[b, s]:

                            if torch.rand(1, device=device).item() < 0.5:
                                dropout_mask[b, s-1] = True
                            else:
                                dropout_mask[b, s] = True


            text_mask = text_mask & dropout_mask


        rows = seq_chars.detach().cpu()
        mrows = text_mask.detach().cpu().bool()


        texts: List[str] = []
        all_indices: List[torch.Tensor] = []

        for b in range(B):


            idx_all = torch.nonzero((rows[b] > 0) & mrows[b], as_tuple=False).squeeze(-1)
            if idx_all.dim() == 0:
                idx_all = idx_all.unsqueeze(0)
            all_indices.append(idx_all)

            vals = rows[b, idx_all].to(torch.int64).tolist()
            s = "".join(self._safe_chr(v) for v in vals)


            if len(s) < self.min_chars_for_canine:
                s = s + (" " * (self.min_chars_for_canine - len(s)))
            texts.append(s)

        if self.is_byt5:

            M_ctx, kpm, char_mappings = self._encode_texts_byt5(texts, device, apply_dropout)
            D = M_ctx.size(-1)
            char_ctx = torch.zeros(B, S, D, device=M_ctx.device, dtype=M_ctx.dtype)

            for b in range(B):
                idx_all = all_indices[b]
                if idx_all.numel() == 0:
                    continue

                mapping = char_mappings[b]
                num_chars = min(len(mapping), idx_all.numel())

                for i in range(num_chars):
                    start_byte, end_byte = mapping[i]
                    dst_idx = idx_all[i].item()

                    if dst_idx >= S:
                        continue


                    if end_byte <= M_ctx.size(1):

                        byte_embs = M_ctx[b, start_byte:end_byte, :]
                        if byte_embs.size(0) > 0:

                            valid_mask = ~kpm[b, start_byte:end_byte]
                            if valid_mask.any():
                                char_ctx[b, dst_idx] = byte_embs[valid_mask].mean(dim=0)
                            else:
                                char_ctx[b, dst_idx] = byte_embs.mean(dim=0)

            char_ctx = self._add_char_id_embedding(char_ctx, seq_chars)
            return char_ctx.unsqueeze(2)

        else:

            M_ctx, kpm = self._encode_texts_canine(texts, device, apply_dropout)
            valid_tok = (~kpm)

            D = M_ctx.size(-1)
            char_ctx = torch.zeros(B, S, D, device=M_ctx.device, dtype=M_ctx.dtype)

            for b in range(B):
                pos_tok = torch.nonzero(valid_tok[b], as_tuple=False).squeeze(-1)
                if pos_tok.dim() == 0:
                    pos_tok = pos_tok.unsqueeze(0)
                idx_all = all_indices[b]

                L = min(pos_tok.numel(), idx_all.numel())
                if L == 0:
                    continue


                keep_local = mrows[b][idx_all][:L]
                if keep_local.any():
                    sel = torch.nonzero(keep_local, as_tuple=False).squeeze(-1)
                    if sel.dim() == 0:
                        sel = sel.unsqueeze(0)


                    sel_gpu = sel.to(device)
                    idx_all_gpu = idx_all.to(device)

                    src = pos_tok[:L][sel_gpu]
                    dst = idx_all_gpu[:L][sel_gpu]


                    valid_src = src < M_ctx.size(1)
                    valid_dst = dst < S
                    valid = valid_src & valid_dst
                    if valid.any():
                        char_ctx[b, dst[valid]] = M_ctx[b, src[valid], :]

            char_ctx = self._add_char_id_embedding(char_ctx, seq_chars)
            return char_ctx.unsqueeze(2)

    def _add_char_id_embedding(self, char_ctx: torch.Tensor, seq_chars: torch.Tensor) -> torch.Tensor:
        if not self.use_char_id_emb or self.char_id_emb is None:
            return char_ctx


        char_ids = seq_chars.clamp(0, self.max_char_id - 1).long()
        char_id_emb = self.char_id_emb(char_ids)
        char_id_proj = self.char_id_proj(char_id_emb)


        return char_ctx + self.char_id_scale * char_id_proj
