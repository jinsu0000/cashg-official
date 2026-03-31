import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from src.utils.train_util import PositionalEncoding
from einops import rearrange
from src.utils.logger import print_once, print_trace
from src.model.residual_vq import FontVQBranch, VQAdapter

class FontEncoder(nn.Module):
    def __init__(
        self,
        d_model=256,
        base_nhead=8,
        head_layers=3,

        use_vq=False,
        vq_codebook_size=256,
        vq_embed_dim=128,
        vq_commitment_weight=0.25,
        vq_codebook_decay=0.99,
    ):
        super().__init__()
        print_once(f"[FontEncoder] Initializing with d_model={d_model}, base_nhead={base_nhead}, head_layers={head_layers}")
        self.d_model = d_model
        self.use_vq = use_vq

        self.backbone = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            *list(models.resnet18(pretrained=True).children())[1:-2]
        )
        self.proj = nn.Linear(512, d_model)
        self.pos_encoding = PositionalEncoding(self.d_model)

        self.dim_feedforward = 2048
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=base_nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=0.1,
            activation='relu',
            batch_first=True
        )
        self.font_head = nn.TransformerEncoder(
            encoder_layer, num_layers=head_layers, norm=nn.LayerNorm(self.d_model)
        )


        if self.use_vq:
            self.vq_adapter = VQAdapter(
                d_model=d_model,
                vq_dim=vq_embed_dim,
                n_embed=vq_codebook_size,
                n_levels=1,
                decay=vq_codebook_decay,
                commitment_cost=vq_commitment_weight,
            )

            self.vq_branch = None
            print_once(f"[FontEncoder]  VQAdapter enabled: vq_dim={vq_embed_dim}, codebook={vq_codebook_size}, levels=1")


            self._last_font_vq_loss = None
            self._last_font_vq_info = None
            self._forward_count = 0
        else:
            self.vq_adapter = None
            self.vq_branch = None
            print_once(f"[FontEncoder]  Font VQ disabled (baseline mode)")

    def forward(self, x):

        print_once(f"[FontEncoder] *** Input shape: {x.shape}")
        B = x.shape[0]


        blank_mask = (x.abs().sum(dim=(1,2,3)) == 0)

        not_blank_idx = (~blank_mask).nonzero(as_tuple=False).squeeze(1)


        out = torch.zeros(B, self.d_model, device=x.device, dtype=x.dtype)

        if not_blank_idx.numel() > 0:
            feat = self.backbone(x[not_blank_idx])
            print_trace(f"[FontEncoder] Backbone output shape: {feat.shape}")
            feat = rearrange(feat, 'b c h w -> b (h w) c')
            print_trace(f"[FontEncoder] Rearranged feature shape: {feat.shape}")
            feat = self.proj(feat)
            print_trace(f"[FontEncoder] Projected feature shape: {feat.shape}")
            feat = self.pos_encoding(feat)
            feat = self.font_head(feat)
            print_trace(f"[FontEncoder] Font head output shape: {feat.shape}")


            if self.use_vq and self.vq_adapter is not None:

                feat_vq, vq_loss, vq_info = self.vq_adapter(feat)


                content_emb = feat_vq.mean(dim=1)


                if self.training:
                    self._last_font_vq_loss = vq_loss
                    self._last_font_vq_info = vq_info


                    self._forward_count += 1
                    if self._forward_count <= 100 and self._forward_count % 10 == 0:
                        try:
                            px = float(vq_info.get('perplexity', 0))
                            usage = int(vq_info.get('code_usage', 0))
                            K = self.vq_adapter.num_embeddings
                            print_once(f"[FontEncoder VQ] perplexity={px:.1f}, usage={usage}/{K} ({usage/K*100:.1f}%)")
                        except Exception:
                            pass
                else:
                    self._last_font_vq_loss = None
                    self._last_font_vq_info = None


                out[not_blank_idx] = content_emb.to(dtype=out.dtype)
                print_trace(f"[FontEncoder]  VQAdapter applied: perplexity={vq_info.get('perplexity', 0):.1f}")
            else:

                feat_pooled = feat.mean(dim=1)

                out[not_blank_idx] = feat_pooled.to(dtype=out.dtype)
                self._last_font_vq_loss = None
                self._last_font_vq_info = None

            print_trace(f"[FontEncoder] *** FontEncoder out shape: {out.shape}")
        else:
            print_trace(f"[FontEncoder] *** All blanks: output shape = {out.shape}")
            self._last_font_vq_loss = None
            self._last_font_vq_info = None


        vq_loss = self._last_font_vq_loss if hasattr(self, '_last_font_vq_loss') else None
        vq_info = self._last_font_vq_info if hasattr(self, '_last_font_vq_info') else None

        return out, vq_loss, vq_info
