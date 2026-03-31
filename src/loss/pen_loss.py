import torch
import torch.nn as nn
import numpy as np
import sys
from src.config.constants import TRAJ_INDEX, TRAJ_DIM
from src.utils.logger import print_once


def _sanitize_mdn(z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr, z_pen_logits):

    z_pi = torch.nan_to_num(z_pi, nan=0.0, posinf=0.0, neginf=0.0)
    z_pi = z_pi + 1e-8
    z_pi = z_pi / z_pi.sum(dim=1, keepdim=True).clamp_min(1e-6)


    z_mu1 = torch.nan_to_num(z_mu1)
    z_mu2 = torch.nan_to_num(z_mu2)


    z_sigma1 = torch.nan_to_num(z_sigma1, nan=1.0, posinf=500.0, neginf=1.0).clamp(1e-6, 500.0)
    z_sigma2 = torch.nan_to_num(z_sigma2, nan=1.0, posinf=500.0, neginf=1.0).clamp(1e-6, 500.0)


    z_corr = torch.tanh(torch.nan_to_num(z_corr)).clamp(-0.999, 0.999)


    z_pen_logits = torch.nan_to_num(z_pen_logits)

    return z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr, z_pen_logits

def get_pen_loss(
    z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr, z_pen_logits,
    x1_data, x2_data, pen_data,
    step=None,
    time_mask=None,
    class_weight=None,
):
    device = z_pi.device
    x1_data = x1_data.to(device)
    x2_data = x2_data.to(device)
    pen_data = pen_data.to(device)


    if time_mask is None:
        with torch.no_grad():
            time_mask = (
                (pen_data.sum(dim=1) > 0) |
                (x1_data.abs().squeeze(1) > 0) |
                (x2_data.abs().squeeze(1) > 0)
            )

    z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr, z_pen_logits = \
        _sanitize_mdn(z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr, z_pen_logits)


    pdf = tf_2d_normal(x1_data, x2_data, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr)
    eps = 1e-8
    log_pdf = torch.log(pdf + eps)
    log_pi  = torch.log(z_pi + eps)
    log_sum = torch.logsumexp(log_pi + log_pdf, dim=1, keepdim=True)
    nll = -log_sum


    pen_targets = torch.argmax(pen_data, dim=1)


    if class_weight is not None:
        cw = torch.as_tensor(class_weight, dtype=torch.float32, device=device)
    else:
        cw = None
    ce_loss = nn.CrossEntropyLoss(reduction='none', weight=cw)
    pen_per = ce_loss(z_pen_logits, pen_targets)


    if time_mask is not None:
        m = time_mask.to(device=device, dtype=torch.float32).view(-1, 1)
        nll = nll * m
        m_pen = time_mask.to(device=device, dtype=torch.float32).view(-1)
        denom = m_pen.sum().clamp_min(1.0)
        state_loss = (pen_per * m_pen).sum() / denom
    else:
        state_loss = pen_per.mean()


    fail = False
    if torch.isnan(nll).any() or torch.isinf(nll).any() \
       or torch.isnan(state_loss).any() or torch.isinf(state_loss).any():
        print(f"\n[CRITICAL][get_pen_loss] NaN/Inf detected! step:{step}")
        fail = True

    if fail:
        print("==== DEBUG DUMP (get_pen_loss) ====")
        print("step:", step)
        print("z_pi (mixture weight, sum 1):", z_pi)
        print("z_mu1:", z_mu1)
        print("z_mu2:", z_mu2)
        print("z_sigma1:", z_sigma1)
        print("z_sigma2:", z_sigma2)
        print("z_corr:", z_corr)
        print("z_pen_logits:", z_pen_logits)
        print("x1_data:", x1_data)
        print("x2_data:", x2_data)
        print("pen_data:", pen_data)
        print("nll (pdf):", nll)
        print("result1 (loss):", nll)
        print("result2 (pen state loss):", state_loss)
        print("===============================")
        return None, None

    return nll, state_loss

def tf_2d_normal(x1, x2, mu1, mu2, s1, s2, rho):
    s1 = torch.clamp(s1, min=1e-6, max=500.0)
    s2 = torch.clamp(s2, min=1e-6, max=500.0)
    norm1 = x1 - mu1
    norm2 = x2 - mu2
    s1s2 = s1 * s2
    z = (norm1 / s1) ** 2 + (norm2 / s2) ** 2 - 2 * rho * norm1 * norm2 / s1s2
    neg_rho = torch.clamp(1 - rho ** 2, 1e-6, 1.0)
    result = torch.exp(-z / (2 * neg_rho))
    denom = 2 * np.pi * s1s2 * torch.sqrt(neg_rho)
    result = result / (denom + 1e-8)
    result = torch.nan_to_num(result, nan=0.0)
    return result
