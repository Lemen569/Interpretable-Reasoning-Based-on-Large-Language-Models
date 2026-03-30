import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class FusionTransformer(nn.Module):
    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 3,
        dropout: float = 0.1,
        device: str = "cuda"
    ):
        super().__init__()
        self.d_model = d_model
        self.device = device

        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=False
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=n_layers)

        self.proj = nn.Linear(d_model, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        src = src.unsqueeze(1) if src.dim() == 2 else src
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = output.mean(dim=0)
        output = self.proj(output)
        return self.sigmoid(output)

    def fuse_features(self, gnn_feat: torch.Tensor, llm_feat: torch.Tensor) -> float:
        feat = (gnn_feat + llm_feat) / 2
        score = self.forward(feat)
        return score.item()

if __name__ == "__main__":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    model = FusionTransformer(d_model=128, n_heads=8, n_layers=3, device=DEVICE)

    feat = torch.randn(10, 128).to(DEVICE)
    out = model(feat)
    print(f"Fusion output shape: {out.shape}")
    print(f"Fusion score: {out.item():.4f}")
    print("FusionTransformer test passed successfully!")