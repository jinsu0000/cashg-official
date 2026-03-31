import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from src.utils.train_util import get_mixture_coef


def compute_vq_target_residual(
    gmm_pred: torch.Tensor,
    gt_traj: torch.Tensor,
    num_mixtures: int = 20,
) -> torch.Tensor:
    B, T, C = gmm_pred.shape


    flat = gmm_pred.reshape(B * T, C)
    pi, mu1, mu2, s1, s2, rho, pen_logits = get_mixture_coef(flat, num_mixtures)


    mu_x = (pi * mu1).sum(dim=1)
    mu_y = (pi * mu2).sum(dim=1)

    mu_x = mu_x.reshape(B, T)
    mu_y = mu_y.reshape(B, T)


    gt_x = gt_traj[..., 0]
    gt_y = gt_traj[..., 1]


    residual_x = gt_x - mu_x
    residual_y = gt_y - mu_y

    residual = torch.stack([residual_x, residual_y], dim=-1)

    return residual


def compute_rvq_loss(
    delta_pred: torch.Tensor,
    residual_target: torch.Tensor,
    gate: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    gate_regularization: float = 0.01,
) -> Tuple[torch.Tensor, dict]:
    B, T = delta_pred.shape[:2]


    recon_loss = F.l1_loss(delta_pred, residual_target, reduction='none')


    gate_loss = F.mse_loss(gate, torch.zeros_like(gate), reduction='none')


    if mask is not None:
        mask_expanded = mask.unsqueeze(-1).float()
        recon_loss = recon_loss * mask_expanded
        gate_loss = gate_loss * mask_expanded

        valid_count = mask.sum().clamp(min=1.0)
        recon_loss = recon_loss.sum() / valid_count / 2.0
        gate_loss = gate_loss.sum() / valid_count
    else:
        recon_loss = recon_loss.mean()
        gate_loss = gate_loss.mean()


    total_loss = recon_loss + gate_regularization * gate_loss


    with torch.no_grad():
        stats = {
            'rvq_recon': recon_loss.item(),
            'rvq_gate_reg': gate_loss.item(),
            'rvq_gate_mean': gate.mean().item(),
            'rvq_gate_std': gate.std().item(),
            'rvq_delta_norm': torch.norm(delta_pred, dim=-1).mean().item(),
            'rvq_residual_norm': torch.norm(residual_target, dim=-1).mean().item(),
        }

    return total_loss, stats


class RVQLoss(nn.Module):
    def __init__(
        self,
        gate_regularization: float = 0.01,
        num_mixtures: int = 20,
    ):
        super().__init__()
        self.gate_regularization = gate_regularization
        self.num_mixtures = num_mixtures

    def forward(
        self,
        gmm_pred: torch.Tensor,
        gt_traj: torch.Tensor,
        delta_pred: torch.Tensor,
        gate: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:

        residual_target = compute_vq_target_residual(
            gmm_pred, gt_traj, self.num_mixtures
        )


        loss, stats = compute_rvq_loss(
            delta_pred, residual_target, gate, mask, self.gate_regularization
        )

        return loss, stats


def test_rvq_loss():
    print("=== RVQ Loss Test ===")

    B, T, C = 4, 10, 124
    num_mixtures = 20


    gmm_pred = torch.randn(B, T, C)
    gt_traj = torch.randn(B, T, 5)
    delta_pred = torch.randn(B, T, 2) * 0.01
    gate = torch.sigmoid(torch.randn(B, T, 1))
    mask = torch.rand(B, T) > 0.2


    loss_fn = RVQLoss(gate_regularization=0.01, num_mixtures=num_mixtures)
    loss, stats = loss_fn(gmm_pred, gt_traj, delta_pred, gate, mask)

    print(f" Loss: {loss.item():.6f}")
    print(f" Stats:")
    for k, v in stats.items():
        print(f"   {k}: {v:.6f}")


    loss.backward()
    print(f" Backward pass OK")

    print("=== Test Passed ===\n")


if __name__ == "__main__":
    test_rvq_loss()
