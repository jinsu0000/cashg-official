import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from src.utils.logger import print_once


class VectorQuantizer(nn.Module):
    def __init__(
        self,
        num_embeddings: int = 512,
        embedding_dim: int = 64,
        commitment_cost: float = 0.25,
        decay: float = 0.99,
        epsilon: float = 1e-5,
    ):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon


        embedding = torch.randn(num_embeddings, embedding_dim)
        self.register_buffer("embedding", embedding)
        self.register_buffer("ema_cluster_size", torch.zeros(num_embeddings))
        self.register_buffer("ema_weight", embedding.clone())

    def forward(self, z_e: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        B, T, C = z_e.shape
        z_e_flat = z_e.reshape(-1, C)


        z_e_norm_sq = (z_e_flat ** 2).sum(dim=1, keepdim=True)
        emb_norm_sq = (self.embedding ** 2).sum(dim=1, keepdim=True).t()


        distances = z_e_norm_sq + emb_norm_sq
        distances = torch.addmm(distances, z_e_flat, self.embedding.t(), beta=1.0, alpha=-2.0)


        indices = torch.argmin(distances, dim=1)


        z_q_flat = F.embedding(indices, self.embedding)
        z_q = z_q_flat.reshape(B, T, C)


        if self.training:
            with torch.no_grad():

                ema_update = torch.zeros(self.num_embeddings, device=z_e.device)

                ema_update.scatter_add_(0, indices, torch.ones(indices.shape[0], dtype=torch.float32, device=z_e.device))

                ema_cluster_size = self.ema_cluster_size * self.decay + ema_update * (1 - self.decay)
                n = torch.sum(ema_cluster_size)
                ema_cluster_size = (ema_cluster_size + self.epsilon) / (n + self.num_embeddings * self.epsilon) * n
                self.ema_cluster_size.copy_(ema_cluster_size)


                dw = torch.zeros(self.num_embeddings, C, device=z_e.device, dtype=z_e.dtype)
                dw.index_add_(0, indices, z_e_flat.detach())

                ema_weight = self.ema_weight * self.decay + dw * (1 - self.decay)
                self.ema_weight.copy_(ema_weight)
                self.embedding.copy_(ema_weight / self.ema_cluster_size.unsqueeze(1))


        commitment_loss = self.commitment_cost * F.mse_loss(z_e, z_q.detach())


        z_q = z_e + (z_q - z_e).detach()

        indices = indices.reshape(B, T)

        return z_q, commitment_loss, indices


class ResidualVectorQuantizer(nn.Module):
    def __init__(
        self,
        num_quantizers: int = 2,
        num_embeddings: int = 512,
        embedding_dim: int = 64,
        commitment_cost: float = 0.25,
        decay: float = 0.99,
    ):
        super().__init__()

        self.num_quantizers = num_quantizers

        self.quantizers = nn.ModuleList([
            VectorQuantizer(
                num_embeddings=num_embeddings,
                embedding_dim=embedding_dim,
                commitment_cost=commitment_cost,
                decay=decay,
            )
            for _ in range(num_quantizers)
        ])

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, C = z.shape
        z_q_total = torch.zeros(B, T, C, dtype=z.dtype, device=z.device)
        residual = z.clone()
        total_loss = 0.0
        indices_list = []

        for quantizer in self.quantizers:
            z_q, loss, indices = quantizer(residual)
            z_q_total.add_(z_q)
            residual = residual - z_q
            total_loss = total_loss + loss
            indices_list.append(indices)

        indices_stacked = torch.stack(indices_list, dim=-1)
        return z_q_total, total_loss, indices_stacked


class VQAdapter(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        vq_dim: int = 128,
        n_embed: int = 512,
        n_levels: int = 1,
        decay: float = 0.99,
        commitment_cost: float = 0.25,
    ):
        super().__init__()


        self.in_proj = nn.Linear(d_model, vq_dim)


        if n_levels == 1:
            self.vq = VectorQuantizer(
                num_embeddings=n_embed,
                embedding_dim=vq_dim,
                commitment_cost=commitment_cost,
                decay=decay,
            )
        else:
            self.vq = ResidualVectorQuantizer(
                num_quantizers=n_levels,
                num_embeddings=n_embed,
                embedding_dim=vq_dim,
                commitment_cost=commitment_cost,
                decay=decay,
            )


        self.out_proj = nn.Linear(vq_dim, d_model)

        self.num_embeddings = n_embed

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        original_shape = x.shape
        needs_unsqueeze = (x.ndim == 2)

        if needs_unsqueeze:
            x = x.unsqueeze(1)


        z_e = self.in_proj(x)


        if isinstance(self.vq, VectorQuantizer):
            z_q, vq_loss, indices = self.vq(z_e)

            with torch.no_grad():
                K = self.vq.num_embeddings
                indices_flat = indices.flatten()
                if indices_flat.numel() > 0:
                    avg_probs = torch.bincount(indices_flat, minlength=K).float() / indices_flat.numel()
                    perplexity = torch.exp(-(avg_probs * (avg_probs + 1e-10).log()).sum())
                    usage = (avg_probs > 0).float().sum()
                else:
                    perplexity = torch.tensor(0.0, device=x.device)
                    usage = torch.tensor(0.0, device=x.device)
            info = {"perplexity": perplexity, "code_usage": usage, "indices": indices}
        else:
            z_q, vq_loss, indices = self.vq(z_e)

            info = {"indices": indices}


        x_q = self.out_proj(z_q)

        if needs_unsqueeze:
            x_q = x_q.squeeze(1)

        return x_q, vq_loss, info


class ResidualVQBranch(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        num_quantizers: int = 2,
        codebook_size: int = 512,
        vq_dim: int = 64,
        commitment_weight: float = 0.25,
        codebook_decay: float = 0.99,
        clamp_scale: float = 0.05,
        gate_init: float = 0.0,
    ):
        super().__init__()

        self.clamp_scale = clamp_scale


        self.encoder = nn.Linear(d_model, vq_dim)


        self.rvq = ResidualVectorQuantizer(
            num_quantizers=num_quantizers,
            num_embeddings=codebook_size,
            embedding_dim=vq_dim,
            commitment_cost=commitment_weight,
            decay=codebook_decay,
        )


        self.decoder = nn.Linear(vq_dim, 2)


        self.gate_net = nn.Linear(d_model, 1)
        nn.init.constant_(self.gate_net.bias, gate_init)

        print_once(f"[ResidualVQBranch] clamp_scale={clamp_scale}, gate_init={gate_init}")

    def _forward_impl(
        self,
        hidden: torch.Tensor,
        theta_line: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        BS, T, D = hidden.shape


        z_e = self.encoder(hidden)
        z_q, vq_loss, indices = self.rvq(z_e)


        delta = self.decoder(z_q)


        delta = torch.tanh(delta) * self.clamp_scale


        gate = torch.sigmoid(self.gate_net(hidden))


        return delta, gate, vq_loss, indices

    def forward(
        self,
        hidden: torch.Tensor,
        theta_line: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._forward_impl(hidden, theta_line)


class FontVQBranch(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        codebook_size: int = 256,
        vq_dim: int = 128,
        commitment_weight: float = 0.25,
        codebook_decay: float = 0.99,
    ):
        super().__init__()


        self.encoder = nn.Sequential(
            nn.Linear(d_model, vq_dim * 2),
            nn.ReLU(),
            nn.Linear(vq_dim * 2, vq_dim),
        )


        self.vq = VectorQuantizer(
            num_embeddings=codebook_size,
            embedding_dim=vq_dim,
            commitment_cost=commitment_weight,
            decay=codebook_decay,
        )


        self.decoder = nn.Sequential(
            nn.Linear(vq_dim, vq_dim * 2),
            nn.ReLU(),
            nn.Linear(vq_dim * 2, d_model),
        )

    def forward(
        self,
        font_feat: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, D = font_feat.shape


        z_e = self.encoder(font_feat)


        z_q, vq_loss, indices = self.vq(z_e.unsqueeze(1))
        z_q = z_q.squeeze(1)
        indices = indices.squeeze(1)


        content_emb = self.decoder(z_q)


        recon_loss = F.mse_loss(content_emb, font_feat.detach())

        return content_emb, vq_loss, recon_loss, indices


def test_font_vq_branch():
    print("=== FontVQBranch Test ===")

    B, D = 8, 256


    font_vq = FontVQBranch(
        d_model=D,
        codebook_size=256,
        vq_dim=128,
    )


    font_feat = torch.randn(B, D)


    content_emb, vq_loss, recon_loss, indices = font_vq(font_feat)

    print(f" Input font_feat: {font_feat.shape}")
    print(f" Output content_emb: {content_emb.shape}")
    print(f" VQ loss: {vq_loss.item():.6f}")
    print(f" Recon loss: {recon_loss.item():.6f}")
    print(f" Indices: {indices.shape}, unique codes: {len(indices.unique())}/{256}")


    total_loss = vq_loss + recon_loss
    total_loss.backward()
    print(f" Backward pass OK")

    print("=== Test Passed ===\n")


def test_residual_vq_branch():
    print("=== ResidualVQBranch Test ===")

    B, T, D = 4, 10, 256


    rvq_branch = ResidualVQBranch(
        d_model=D,
        num_quantizers=2,
        codebook_size=512,
        vq_dim=64,
        clamp_scale=0.02,
        gate_init=0.5,
    )


    hidden = torch.randn(B, T, D)
    theta_line = torch.randn(B, T) * 0.1


    delta_global, gate, vq_loss, indices = rvq_branch(hidden, theta_line)

    print(f" Input hidden: {hidden.shape}")
    print(f" Output delta_global: {delta_global.shape}")
    print(f" Output gate: {gate.shape}, range: [{gate.min():.3f}, {gate.max():.3f}]")
    print(f" VQ loss: {vq_loss.item():.6f}")
    print(f" Indices: {indices.shape}")
    print(f" Delta range: [{delta_global.min():.6f}, {delta_global.max():.6f}]")


    loss = vq_loss + delta_global.mean()
    loss.backward()
    print(f" Backward pass OK")

    print("=== Test Passed ===\n")


if __name__ == "__main__":
    test_font_vq_branch()
    test_residual_vq_branch()
