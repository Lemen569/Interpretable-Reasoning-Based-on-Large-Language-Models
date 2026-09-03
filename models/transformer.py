from __future__ import annotations

import torch
import torch.nn as nn


class FusionTransformer(nn.Module):
    """Adaptive fusion of scalar GNN and LLM scores.

    The paper describes score projection followed by a Transformer encoder.
    For two scalar streams, a compact one-token encoder is sufficient and
    keeps the interface explicit for both the full model and linear-fusion
    ablation.
    """

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 3,
        dropout: float = 0.1,
        device: str = "cuda",
    ):
        super().__init__()
        self.device = device
        self.gnn_projection = nn.Linear(1, d_model)
        self.llm_projection = nn.Linear(1, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output = nn.Sequential(nn.Linear(d_model, 1), nn.Sigmoid())

    def forward(self, gnn_scores: torch.Tensor, llm_scores: torch.Tensor) -> torch.Tensor:
        gnn_scores = gnn_scores.float().reshape(-1, 1)
        llm_scores = llm_scores.float().reshape(-1, 1)
        tokens = torch.stack(
            [self.gnn_projection(gnn_scores), self.llm_projection(llm_scores)],
            dim=1,
        )
        encoded = self.encoder(tokens).mean(dim=1)
        return self.output(encoded).squeeze(-1)

    def fuse_features(self, gnn_score: torch.Tensor, llm_score: torch.Tensor) -> float:
        with torch.no_grad():
            return float(self.forward(gnn_score, llm_score).reshape(-1)[0].item())

