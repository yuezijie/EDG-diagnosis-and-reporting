import torch
import torch.nn as nn

class CrossAttentionBlock(nn.Module):
    def __init__(self, embed_dim, nhead, context_dim=None):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.self_attn = nn.MultiheadAttention(embed_dim, nhead, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        context_dim = context_dim or embed_dim
        self.cross_attn = nn.MultiheadAttention(embed_dim, nhead, kdim=context_dim, vdim=context_dim, batch_first=True)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )

    def forward(self, x, context):
        x = x + self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.cross_attn(self.norm2(x), context, context)[0]
        x = x + self.ffn(self.norm3(x))
        return x