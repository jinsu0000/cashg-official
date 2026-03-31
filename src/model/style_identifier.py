import torch.nn.functional as F
import torch
import torch.nn as nn

from einops import rearrange
from src.utils.logger import print_once, print_trace
import torchvision.models as models
from src.utils.train_util import PositionalEncoding, random_double_sampling


class StyleIdentifier(nn.Module):
    def __init__(self, style_dim=256, emb_dim = 256, encoder_type='RESNET18', img_size=(64, 64), base_layers=2, base_nhead=8, head_layers=1, head_nhead=8, patch_size=16,
                 freeze_backbone_bn=True):
        super().__init__()
        self.style_dim = style_dim
        self.patch_size = patch_size
        self.encoder_type = encoder_type
        self.freeze_backbone_bn = freeze_backbone_bn

        self.dim_feedforward = 2048


        if self.encoder_type == "RESNeXt50":
            self.backbone = nn.Sequential(
                nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
                *list(models.resnext50_32x4d(pretrained=True).children())[1:-2]
            )
            self.backbone_channels = 2048
        elif self.encoder_type == "RESNET18":

            self.backbone = nn.Sequential(*([nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)] +list(models.resnet18(pretrained=True).children())[1:-2]))
            self.backbone_channels = 512


        if self.backbone_channels != style_dim:
            print_once(f"[StyleIdentifier] Channel projection: {self.backbone_channels} → {style_dim}")
            self.channel_proj = nn.Conv2d(
                self.backbone_channels, style_dim,
                kernel_size=1,
                bias=False
            )
            self.d_model = style_dim
        else:

            print_once(f"[StyleIdentifier] No projection needed: backbone_channels == style_dim == {style_dim}")
            self.channel_proj = None
            self.d_model = self.backbone_channels


        if self.freeze_backbone_bn:
            self._freeze_backbone_batchnorm()


        self._init_first_conv()


        self.pos_encoding = PositionalEncoding(self.d_model)


        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=base_nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=0.1,
            activation='relu',
            batch_first=True
        )
        self.base_transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=base_layers, norm=None
        )


        self.writer_head = nn.TransformerEncoder(
            encoder_layer, num_layers=head_layers, norm=nn.LayerNorm(self.d_model)
        )
        self.glyph_head = nn.TransformerEncoder(
            encoder_layer, num_layers=head_layers, norm=nn.LayerNorm(self.d_model)
        )


        self.pro_mlp_writer = nn.Sequential(
            nn.Linear(self.d_model, 4096), nn.GELU(),
            nn.Linear(4096, emb_dim),
        )
        self.pro_mlp_glyph = nn.Sequential(
            nn.Linear(self.d_model, 4096), nn.GELU(),
            nn.Linear(4096, emb_dim),
        )


        self.apply(self._init_weights)

    def train(self, mode: bool = True):
        super().train(mode)
        if mode and self.freeze_backbone_bn:
            for module in self.backbone.modules():
                if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                    module.eval()
        return self

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        if isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def _init_first_conv(self):
        first_conv = self.backbone[0]
        if isinstance(first_conv, nn.Conv2d):

            nn.init.kaiming_normal_(first_conv.weight, mode='fan_out', nonlinearity='relu')
            print(f" [StyleIdentifier] First Conv2d initialized with Kaiming Normal")
            print(f"   → in_channels={first_conv.in_channels}, out_channels={first_conv.out_channels}")

    def _freeze_backbone_batchnorm(self):
        bn_count = 0
        for name, module in self.backbone.named_modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                module.eval()
                module.track_running_stats = True

                for param in module.parameters():
                    param.requires_grad = False
                bn_count += 1

        print(f" [StyleIdentifier] Froze {bn_count} BatchNorm layers in backbone")
        print(f"   → running_mean/running_var will NOT be updated during training")
        print(f"   → This prevents NaN propagation from gradient explosion")

    def _check_and_restore_batchnorm(self):
        nan_found = False
        for name, module in self.backbone.named_modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                if not torch.isfinite(module.running_mean).all():
                    print(f" [StyleIdentifier] NaN in running_mean of {name}")
                    nan_found = True
                if not torch.isfinite(module.running_var).all():
                    print(f" [StyleIdentifier] NaN in running_var of {name}")
                    nan_found = True
        return nan_found

    def train(self, mode=True):
        super().train(mode)
        if mode and self.freeze_backbone_bn:

            for module in self.backbone.modules():
                if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                    module.eval()
        return self

    def forward(self, style_imgs):

        if self.freeze_backbone_bn and self.training:
            for module in self.backbone.modules():
                if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                    module.eval()

        print_once(f"[StyleIdentifier] *** encoder_type: {self.encoder_type}, style_dim: {self.style_dim}, style_imgs: {style_imgs.shape[3:]}")
        B, N2, C, H, W = style_imgs.shape
        N = N2 // 2
        print_once(f"[StyleIdentifier] style_imgs input B: {B}, N2: {N2}, C: {C}, H: {H}, W: {W}, N: {N}")


        img_min, img_max = style_imgs.min().item(), style_imgs.max().item()
        img_mean, img_std = style_imgs.mean().item(), style_imgs.std().item()
        none_count = torch.isnan(style_imgs).sum().item()
        zero_imgs = (style_imgs.view(B*N2, -1).sum(dim=1) == 0).sum().item()


        if none_count > 0 or img_min < -0.1 or img_max > 1.1 or zero_imgs > B*N2*0.1:
            print(f"  [StyleIdentifier][BATCH_ANOMALY] B={B}, N2={N2} | img_range=[{img_min:.3f}, {img_max:.3f}], mean={img_mean:.3f}, std={img_std:.3f}")
            print(f"  [StyleIdentifier][BATCH_ANOMALY] NaN_count={none_count}, zero_imgs={zero_imgs}")

        x = style_imgs.view(B * N2, C, H, W)
        print_once(f"[StyleIdentifier] x after reshape : {x.shape}")

        feat = self.backbone(x)
        print_once(f"[StyleIdentifier] feat after backbone() : {feat.shape}")


        if self.channel_proj is not None:
            feat = self.channel_proj(feat)
            print_once(f"[StyleIdentifier] feat after channel_proj() : {feat.shape}")
        else:
            print_once(f"[StyleIdentifier] No channel projection (backbone_channels == style_dim)")


        if not torch.isfinite(feat).all():
            nan_count = (~torch.isfinite(feat)).sum().item()
            feat_mean = feat[torch.isfinite(feat)].mean().item() if torch.isfinite(feat).any() else 0.0
            print(f"  [StyleIdentifier][DEBUG] backbone output has {nan_count} NaN/Inf! mean={feat_mean:.3f}")

        h, w = feat.shape[2], feat.shape[3]
        P = h * w
        print_once(f"[StyleIdentifier] h: {h}, w: {w}, P: {P} after feat.shape[2], feat.shape[3]")

        feat = feat.view(B * N2, self.d_model, -1).permute(0, 2, 1)
        print_once(f"[StyleIdentifier] feat after view and permute: {feat.shape}")

        feat = self.pos_encoding(feat)


        if not torch.isfinite(feat).all():
            nan_count = (~torch.isfinite(feat)).sum().item()
            print(f"  [StyleIdentifier][DEBUG] pos_encoding output has {nan_count} NaN/Inf!")

        feat = self.base_transformer(feat)
        print_once(f"[StyleIdentifier] feat after pos_encoding() & base_transformer() : {feat.shape}")


        if not torch.isfinite(feat).all():
            nan_count = (~torch.isfinite(feat)).sum().item()
            print(f"  [StyleIdentifier][DEBUG] base_transformer output has {nan_count} NaN/Inf!")


        writer_feat = self.writer_head(feat)


        if not torch.isfinite(writer_feat).all():
            nan_count = (~torch.isfinite(writer_feat)).sum().item()
            print(f"  [StyleIdentifier][DEBUG] writer_head output has {nan_count} NaN/Inf!")

        glyph_feat = self.glyph_head(feat)


        if not torch.isfinite(glyph_feat).all():
            nan_count = (~torch.isfinite(glyph_feat)).sum().item()
            print(f"  [StyleIdentifier][DEBUG] glyph_head output has {nan_count} NaN/Inf!")

        print_once("[StyleIdentifier] WRITER feat:", writer_feat.shape, ", GLYPH feat:", glyph_feat.shape)

        writer_feat = rearrange(writer_feat, '(b p n) t c -> (p b) t n c',
                           b=B, p=2, n=N)
        glyph_feat = rearrange(glyph_feat, '(b p n) t c -> (p b) t n c',
                           b=B, p=2, n=N)
        print_once(f"[StyleIdentifier] writer_feat after rearrange: {writer_feat.shape}, glyph_feat after rearrange: {glyph_feat.shape}")

        writer_feat = rearrange(writer_feat, 'b t n c ->b (t n) c')


        writer_feat_mean = writer_feat.mean(dim=1)
        print_once(f"[StyleIdentifier] writer_feat_mean after mean(1): {writer_feat_mean.shape}")
        writer_feat_proj = self.pro_mlp_writer(writer_feat_mean)
        print_once(f"[StyleIdentifier] writer_feat_proj after pro_mlp_writer(): {writer_feat_proj.shape}")


        if not torch.isfinite(writer_feat_proj).all():
            nan_count = (~torch.isfinite(writer_feat_proj)).sum().item()
            print(f"  [StyleIdentifier][DEBUG] pro_mlp_writer output has {nan_count} NaN/Inf!")
            print(f"  [StyleIdentifier][DEBUG] writer_feat_mean: min={writer_feat_mean.min().item():.3f}, max={writer_feat_mean.max().item():.3f}, mean={writer_feat_mean.mean().item():.3f}")

        writer_query, writer_positive = torch.split(writer_feat_proj, B, dim=0)
        print_once(f"[StyleIdentifier] writer_query: {writer_query.shape}, writer_positive: {writer_positive.shape} after split")
        writer_embed = torch.stack([writer_query, writer_positive], dim=1)
        if self.encoder_type == "RESNET18":
            writer_embed = nn.functional.normalize(writer_embed, p=2, dim=2)
        print_once(f"[StyleIdentifier] writer_embed after stack: {writer_embed.shape}")


        glyph_feat = glyph_feat[:B, :]
        print_once(f"[StyleIdentifier] glyph_feat after view: {glyph_feat.shape}")

        glyph_anc, glyph_pos = random_double_sampling(glyph_feat)
        d_model = glyph_anc.shape[-1]
        print_once(f"[StyleIdentifier] random_double_sampling glyph_anc: {glyph_anc.shape}, glyph_pos: {glyph_pos.shape}, d_model: {d_model}")
        glyph_anc = glyph_anc.reshape(B, -1, d_model)
        glyph_pos = glyph_pos.reshape(B, -1, d_model)
        print_once(f"[StyleIdentifier] glyph_anc after resize: {glyph_anc.shape}, glyph_pos after resize: {glyph_pos.shape}")

        glyph_anc = torch.mean(glyph_anc, 1, keepdim=True)
        glyph_pos = torch.mean(glyph_pos, 1, keepdim=True)

        print_once(f"[StyleIdentifier] glyph_anc after mean(1): {glyph_anc.shape}, glyph_pos after mean(1): {glyph_pos.shape}")

        glyph_anc = self.pro_mlp_glyph(glyph_anc)
        print_once(f"[StyleIdentifier] glyph_anc after pro_mlp_glyph: {glyph_anc.shape}")
        glyph_pos = self.pro_mlp_glyph(glyph_pos)
        print_once(f"[StyleIdentifier] glyph_pos after pro_mlp_glyph: {glyph_pos.shape}")
        glyph_embed = torch.cat([glyph_anc, glyph_pos], dim=1)
        print_once(f"[StyleIdentifier] glyph_embed after cat: {glyph_embed.shape}")

        if self.encoder_type == "RESNET18":
            glyph_embed = nn.functional.normalize(glyph_embed, p=2, dim=2)


        writer_style = writer_feat[:B, :, :]
        glyph_style = glyph_feat

        print_once(f"[StyleIdentifier] *** Output writer_embed: {writer_embed.shape}, glyph_embed: {glyph_embed.shape}, writer_style: {writer_style.shape}, glyph_style: {glyph_style.shape}")


        if not torch.isfinite(writer_embed).all():
            nan_count = (~torch.isfinite(writer_embed)).sum().item()
            print(f" [StyleIdentifier][CRITICAL] writer_embed contains {nan_count} NaN/Inf!")
            print(f" This indicates gradient explosion. Check optimizer and grad clipping.")

        if not torch.isfinite(glyph_embed).all():
            nan_count = (~torch.isfinite(glyph_embed)).sum().item()
            print(f" [StyleIdentifier][CRITICAL] glyph_embed contains {nan_count} NaN/Inf!")

        if not torch.isfinite(writer_style).all():
            nan_count = (~torch.isfinite(writer_style)).sum().item()
            print(f" [StyleIdentifier][CRITICAL] writer_style contains {nan_count} NaN/Inf!")

        if not torch.isfinite(glyph_style).all():
            nan_count = (~torch.isfinite(glyph_style)).sum().item()
            print(f" [StyleIdentifier][CRITICAL] glyph_style contains {nan_count} NaN/Inf!")

        return writer_embed, glyph_embed, writer_style, glyph_style
