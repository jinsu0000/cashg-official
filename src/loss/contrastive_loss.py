import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils.logger import print_once, print_trace

class WriterGlyphNCELoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.supcon_loss = SupConLoss(temperature=temperature)

    def forward(self, writer_embed, glyph_embed, writer_labels):

        device = writer_embed.device


        if not torch.isfinite(writer_embed).all():
            nan_count = (~torch.isfinite(writer_embed)).sum().item()
            print(f" [WriterGlyphNCELoss][CRITICAL] writer_embed contains {nan_count} NaN/Inf!")
            print(f" This indicates gradient explosion. Batch will be skipped by trainer.")


        if not torch.isfinite(glyph_embed).all():
            nan_count = (~torch.isfinite(glyph_embed)).sum().item()
            print(f" [WriterGlyphNCELoss][CRITICAL] glyph_embed contains {nan_count} NaN/Inf!")
            print(f" This indicates gradient explosion. Batch will be skipped by trainer.")


        loss_writer = self.supcon_loss(
            writer_embed,
            labels=writer_labels
        )


        print_trace("writer_labels:", writer_labels[:10].cpu().numpy())
        print_trace("writer_embed mean/std:", writer_embed.mean().item(), writer_embed.std().item())
        print_trace("glyph_embed mean/std:", glyph_embed.mean().item(), glyph_embed.std().item())
        print_once("writer_embed shape:", writer_embed.shape)
        print_trace("unique label count:", len(torch.unique(writer_labels)))


        loss_glyph = self.supcon_loss(
            glyph_embed,
            labels=None
        )

        total_loss = loss_writer + loss_glyph
        return {
            'total': total_loss,
            'writer': loss_writer,
            'glyph': loss_glyph
        }

class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07, contrast_mode='all', base_temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        device = features.device

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...], at least 3 dims required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        print_once(f"[SupConLoss] features : {features.shape}")
        batch_size = features.shape[0]
        n_views = features.shape[1]


        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = n_views
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))


        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T), self.temperature
        )

        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()


        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is not None:

            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        elif mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        else:
            mask = mask.float().to(device)


        mask = mask.repeat(anchor_count, n_views)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask


        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))


        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)


        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()


        if not torch.isfinite(loss):
            print(f" [SupConLoss][CRITICAL] NaN/Inf loss detected!")
            print(f"   exp_logits max: {exp_logits.max().item():.2e}")
            print(f"   mask sum: {mask.sum().item()}")
            print(f"   This batch should be skipped by trainer!")

        return loss
