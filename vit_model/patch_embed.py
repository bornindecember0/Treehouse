import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, image_size=224, patch_size=16, in_chanels=3, embed_dim=192):
        super().__init__()
        self.image_size = image_size # e.g. 224 × 224
        self.patch_size = patch_size # e.g. 16 × 16
        self.num_patches = (image_size // patch_size) ** 2
        self.num_tokens = self.num_patches+1 # num_patches + 1 [CLS] token

        # convolution layer to extract and project patches
        self.projection = nn.Conv2d(
            in_chanels, 
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

        # CLS token and positional embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.dropout = nn.Dropout(0.1) # subjet to change

    def forward(self, x): 
        x = self.projection(x)               

        x = x.flatten(2)
        x = x.transpose(1, 2) # [batch_size, num_patches, embed_dim] for x

        batch_size = x.shape[0] 
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # [batch_size, 1, embed_dim] for tokens
        x = torch.cat((cls_tokens, x), dim=1) # [batch_size, num_patches+1, embed_dim] for x

        x = x + self.pos_embed                         
        x = self.dropout(x)                        
        return x
