import torch
import torch.nn as nn
from .patch_embed import PatchEmbedding
from .transformer import TransformerEncoder
import torch.nn.functional as F

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

        # classification head v1
        self.head = nn.Linear(embed_dim, num_classes)
        
        # # v2 use all patch embedding not only the <cls>
        # self.attention_pool = nn.Linear(embed_dim, 1) 
        # self.head = nn.Linear(embed_dim * 2, num_classes)

        self.apply(self._init_weights)

        

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x, return_embedding=False): #!!! change return_embedding=True if you want to use LLM integration
        x = self.patch_embed(x)   
        x = self.transformer_encoder(x)    
        x = self.norm(x)   
        # v1       
        cls_token = x[:, 0]       
        classification_output = self.head(cls_token) #delete?

        if return_embedding:
            return cls_token          # for LLM integration
        else:
            return classification_output   # for classification



        # # v2 -> 
        # # Weighted attention pooling over patches
        # cls_token = x[:, 0]
        # patch_features = x[:, 1:]

        # # Calculate attention weights
        # # Output shape: [B, N, 1]
        # attn_scores = self.attention_pool(patch_features)
        # # Convert to [B, N] and apply softmax
        # attn_weights = F.softmax(attn_scores.squeeze(-1), dim=1)

        # # Weighted sum of patch features
        # weighted_features = torch.sum(patch_features * attn_weights.unsqueeze(-1), dim=1)
        
       
        # # Combine with CLS token
        # global_features = torch.cat([cls_token, weighted_features], dim=1)

        
        # classification_output = self.head(global_features)

        # if return_embedding:
        #     return global_features
        # else:
        #     return classification_output


