"""Two-layer LLaVA-1.5 style projector used by LMOL."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class LMOLProjector(nn.Module):
    """Project CLIP ViT-L/14-336px visual tokens into Vicuna hidden space."""

    def __init__(self, vision_dim: int = 1024, text_hidden_dim: int = 4096) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(vision_dim, text_hidden_dim)
        self.linear_2 = nn.Linear(text_hidden_dim, text_hidden_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Apply Linear-GELU-Linear projection to every visual token."""

        hidden = F.gelu(self.linear_1(features), approximate="tanh")
        return self.linear_2(hidden)
