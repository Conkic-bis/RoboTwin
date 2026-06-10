"""Latent alignment objectives for the unified action bridge.

A2A assumes history latents and future-action latents live in a shared
geometry; when E_hist and E_act are learned independently the straight-line
transport path can be arbitrary. These objectives make the bridge geometry
explicit:

    L_jepa   JEPA-style predictive alignment: P(c, z_source) -> stopgrad(z1)
    L_nce    InfoNCE between condition c and target latent z1
    L_vicreg variance/invariance/covariance regularization between c and z1

Plus collapse diagnostics (variance / effective rank) for the latent metrics
in the experimental plan.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class JEPAPredictor(nn.Module):
    """Predict the future action latent from condition + source latent.

    z1_pred = P(c, z_source); trained against stopgrad(z1) so the action
    encoder is not dragged toward the predictor (standard JEPA asymmetry).
    """

    def __init__(self, cond_dim, latent_dim, hidden_dim=512, num_layers=2):
        super().__init__()
        layers = []
        in_dim = cond_dim + latent_dim
        for _ in range(num_layers):
            layers += [nn.Linear(in_dim, hidden_dim), nn.SiLU()]
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, latent_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, c, z_source):
        return self.net(torch.cat([c, z_source], dim=-1))


def jepa_loss(predictor, c, z_source, z1):
    z1_pred = predictor(c, z_source)
    return F.smooth_l1_loss(z1_pred, z1.detach())


def info_nce_loss(a, b, temperature=0.07):
    """Symmetric InfoNCE between two batches of paired features."""
    a = F.normalize(a, dim=1)
    b = F.normalize(b, dim=1)
    logits = a @ b.T / temperature
    labels = torch.arange(a.shape[0], device=a.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2


def vicreg_loss(a, b, sim_weight=25.0, var_weight=25.0, cov_weight=1.0):
    """VICReg between paired features (Bardes et al. 2022)."""
    sim = F.mse_loss(a, b)

    def variance(x):
        std = torch.sqrt(x.var(dim=0) + 1e-4)
        return F.relu(1.0 - std).mean()

    def covariance(x):
        n, d = x.shape
        x = x - x.mean(dim=0)
        cov = (x.T @ x) / max(n - 1, 1)
        off_diag = cov - torch.diag(torch.diag(cov))
        return (off_diag ** 2).sum() / d

    var = variance(a) + variance(b)
    cov = covariance(a) + covariance(b)
    return sim_weight * sim + var_weight * var + cov_weight * cov


@torch.no_grad()
def latent_collapse_metrics(z, prefix=""):
    """Collapse indicators: feature variance and effective rank.

    Effective rank = exp(entropy of normalized singular values) of the
    centered feature matrix; collapses toward 1 if latents degenerate.
    """
    z = z.detach().float()
    metrics = {f"{prefix}var": z.var(dim=0).mean().item()}
    try:
        zc = z - z.mean(dim=0)
        s = torch.linalg.svdvals(zc)
        p = s / (s.sum() + 1e-8)
        entropy = -(p * torch.log(p + 1e-8)).sum()
        metrics[f"{prefix}effective_rank"] = torch.exp(entropy).item()
    except Exception:  # noqa: BLE001 - SVD can fail on degenerate batches
        pass
    return metrics
