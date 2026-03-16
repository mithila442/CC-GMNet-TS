import torch.nn as nn
import torch


class GMLayer(nn.Module):
    """
    Gaussian Mixture Layer with DIAGONAL covariance (per §3.3 of the paper).

    Using diagonal covariance avoids:
      - torch.inverse on [D,D] matrices (numerically unstable for large D)
      - Astronomical normalization constants from (2π)^D * det(full_cov)
      - The need for geotorch.positive_definite

    Parameters stored:
      centers:  [C, K, D]  (class-conditioned) or [K, D]
      log_var:  [C, K, D]  (class-conditioned) or [K, D]
        — log of diagonal variance; exponentiated at runtime to ensure σ² > 0
    """

    def __init__(self, n_features, num_gaussians, device,
                 requires_grad=True, num_classes=1, class_conditioned=False):
        super(GMLayer, self).__init__()
        self.n_features = n_features
        self.num_gaussians = num_gaussians
        self.device = device
        self.class_conditioned = class_conditioned
        self.num_classes = num_classes

        if class_conditioned:
            self.centers = nn.Parameter(
                torch.randn(num_classes, num_gaussians, n_features) * 0.5,
                requires_grad=requires_grad,
            )
            # log-variance initialised to 0  →  σ² = 1
            self.log_var = nn.Parameter(
                torch.zeros(num_classes, num_gaussians, n_features),
                requires_grad=requires_grad,
            )
        else:
            self.centers = nn.Parameter(
                torch.randn(num_gaussians, n_features) * 0.5,
                requires_grad=requires_grad,
            )
            self.log_var = nn.Parameter(
                torch.zeros(num_gaussians, n_features),
                requires_grad=requires_grad,
            )

    # ------------------------------------------------------------------ #
    def compute_likelihoods(self, x):
        """
        x : [B, M, D]   (batch, bag_size, latent_dim)
        returns : [B, M, C*K]  (normalised soft-assignment probabilities)
        """
        B, M, D = x.shape

        if self.class_conditioned:
            # var = exp(log_var), clamped for stability
            var = torch.exp(self.log_var).clamp(min=1e-6)        # [C, K, D]

            # Expand for broadcasting
            x_exp      = x.unsqueeze(2).unsqueeze(2)             # [B, M, 1, 1, D]
            centers    = self.centers.unsqueeze(0).unsqueeze(0)   # [1, 1, C, K, D]
            var_exp    = var.unsqueeze(0).unsqueeze(0)            # [1, 1, C, K, D]

            diff = x_exp - centers                                # [B, M, C, K, D]

            # Mahalanobis (diagonal):  sum_d (x_d - μ_d)² / σ²_d
            mahal = (diff ** 2 / var_exp).sum(dim=-1)             # [B, M, C, K]

            # log-normalisation:  0.5 * sum_d log(σ²_d)  +  D/2 * log(2π)
            log_norm = 0.5 * self.log_var.sum(dim=-1)             # [C, K]
            log_norm = log_norm + 0.5 * D * torch.log(
                torch.tensor(2.0 * torch.pi, device=x.device)
            )
            log_norm = log_norm.unsqueeze(0).unsqueeze(0)         # [1, 1, C, K]

            log_probs = -0.5 * mahal - log_norm                  # [B, M, C, K]

            # Numerically stable softmax-style normalisation across C*K
            log_probs_flat = log_probs.reshape(B, M, -1)          # [B, M, C*K]
            probs = torch.softmax(log_probs_flat, dim=-1)         # [B, M, C*K]
            return probs

        else:
            var = torch.exp(self.log_var).clamp(min=1e-6)         # [K, D]

            x_exp   = x.unsqueeze(2)                              # [B, M, 1, D]
            centers = self.centers.unsqueeze(0).unsqueeze(0)       # [1, 1, K, D]
            var_exp = var.unsqueeze(0).unsqueeze(0)                # [1, 1, K, D]

            diff  = x_exp - centers                                # [B, M, K, D]
            mahal = (diff ** 2 / var_exp).sum(dim=-1)              # [B, M, K]

            log_norm = 0.5 * self.log_var.sum(dim=-1)              # [K]
            log_norm = log_norm + 0.5 * D * torch.log(
                torch.tensor(2.0 * torch.pi, device=x.device)
            )
            log_norm = log_norm.unsqueeze(0).unsqueeze(0)          # [1, 1, K]

            log_probs = -0.5 * mahal - log_norm                   # [B, M, K]
            probs = torch.softmax(log_probs, dim=-1)               # [B, M, K]
            return probs

    def forward(self, x):
        return self.compute_likelihoods(x)