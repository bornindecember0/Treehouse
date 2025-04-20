import torch
import torch.nn as nn

class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim=192, num_heads=3, mlp_ratio=4.0, dropout=0.1):
        super().__init__()

        # layer 1: layer norm + multi-head attention
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

        # layer 2: layer norm + multi-layer perceptron 
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # pass through layer 1
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0] 
        # in ViT, self-attention uses the same source for Q, K, V

        # pass through layer 2
        x = x + self.mlp(self.norm2(x))
        return x
