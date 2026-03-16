import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal PE. Supports sequences up to max_len."""
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)                         # [L, D]
        position = torch.arange(0, max_len).unsqueeze(1).float()   # [L, 1]
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))                # [1, L, D]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TransformerFEModule(nn.Module):
    """
    Transformer-based Feature Extraction Module (CC-GMNet-TS, §3.2).

    Pipeline:
      [B, T, in_dim]
        → linear projection           → [B, T, d_model]
        → prepend class token         → [B, T+1, d_model]
        → sinusoidal positional enc.
        → L Transformer encoder layers
        → attention pooling           → [B, d_model]
        → linear + Sigmoid            → [B, output_size]  ∈ [0,1]^d
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        output_size: int = 256,
        num_layers: int = 4,
        nhead: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_len: int = 512,
    ):
        super().__init__()
        self.output_size = output_size          # read by GMNet_Module

        # 1. Token projection
        self.input_proj = nn.Linear(input_dim, d_model)

        # 2. Learnable class token
        self.class_token = nn.Parameter(torch.randn(1, 1, d_model))

        # 3. Positional encoding (applied after prepending class token)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len=max_len + 1, dropout=dropout)

        # 4. Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,           # pre-norm: more stable with small batches
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 5. Attention pooling
        self.attn_proj = nn.Linear(d_model, 1)

        # 6. Output dropout + projection + sigmoid bound
        self.out_dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(d_model, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : [B, T, input_dim]   (B = batch × bag_size after DLQuantification reshape)
        returns : [B, output_size]
        """
        B, T, _ = x.shape

        # Project each timestep to d_model
        tokens = self.input_proj(x)                                # [B, T, d_model]

        # Prepend class token
        cls = self.class_token.expand(B, -1, -1)                  # [B, 1, d_model]
        tokens = torch.cat([cls, tokens], dim=1)                  # [B, T+1, d_model]

        # Positional encoding
        tokens = self.pos_enc(tokens)                             # [B, T+1, d_model]

        # Transformer encoding (all-to-all self-attention)
        hidden = self.transformer(tokens)                         # [B, T+1, d_model]

        # Attention pooling: scalar weight per position → weighted sum
        attn_w = F.softmax(self.attn_proj(hidden), dim=1)        # [B, T+1, 1]
        pooled = (hidden * attn_w).sum(dim=1)                    # [B, d_model]

        # Project and bound to [0, 1]^d  (§3.2, Eq. 4)
        return torch.sigmoid(self.out_proj(self.out_dropout(pooled)))  # [B, output_size]