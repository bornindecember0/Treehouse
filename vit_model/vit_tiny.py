import torch
import torch.nn as nn
from .patch_embed import PatchEmbedding
from .transformer import TransformerEncoder

class ViTTiny(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3,
                 num_classes=200, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4.0, dropout=0.1):
        super().__init__()

        # patch embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)

        # transformer encoder
        self.transformer_encoder = nn.Sequential(*[
            TransformerEncoder(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

        # final normalization
        self.norm = nn.LayerNorm(embed_dim)

        # classification head
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x, return_embedding=False): #!!! change return_embedding=True if you want to use LLM integration
        x = self.patch_embed(x)   
        x = self.transformer_encoder(x)    
        x = self.norm(x)          
        cls_token = x[:, 0]       
        classification_output = self.head(cls_token)

        if return_embedding:
            return cls_token          # for LLM integration
        else:
            return classification_output   # for classification


