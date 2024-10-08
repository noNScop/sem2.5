import torch
from torch import nn

class PatchEmbedding(nn.Module):
    def __init__(self, input_dim, embedding_dim, patch_size):
        super().__init__()
        self.linear_projection = nn.Conv2d(input_dim, embedding_dim, patch_size, stride=patch_size)

    def forward(self, x):
        return torch.flatten(self.linear_projection(x), start_dim=2).permute(0, 2, 1)

class EmbeddingBlock(nn.Module):
    def __init__(self, img_size, input_dim, embedding_dim, patch_size):
        assert img_size % patch_size == 0, f"Image size ({img_size}) must be divisible by patch size ({patch_size})"
        super().__init__()

        self.patch_emb = PatchEmbedding(input_dim, embedding_dim, patch_size)
        self.class_emb = nn.Parameter(torch.randn(1, 1, embedding_dim))
        self.pos_emb = nn.Parameter(torch.randn(img_size**2 // patch_size**2 + 1, embedding_dim))

    def forward(self, x):
        x = self.patch_emb(x)

        batch_size = x.shape[0]
        x = torch.cat((self.class_emb.expand(batch_size, -1, -1), x), dim=1)
        return x + self.pos_emb

class ViTScript(nn.Module):
    def __init__(self,
                 img_size=224,
                 input_dim=3,
                 embedding_dim=768,
                 patch_size=16,
                 num_transformer_layers=12,
                 mlp_size=3072,
                 num_attn_heads=12,
                 mlp_dropout=0.1,
                 embedding_dropout=0.1,
                 num_classes=3):

        super().__init__()
        self.embedding_block = EmbeddingBlock(img_size, input_dim, embedding_dim, patch_size)

        transformer_layer = nn.TransformerEncoderLayer(d_model=embedding_dim,
                                                       nhead=num_attn_heads,
                                                       dim_feedforward=mlp_size,
                                                       dropout=mlp_dropout,
                                                       activation="gelu",
                                                       batch_first=True,
                                                       norm_first=True)

        self.encoder = nn.TransformerEncoder(transformer_layer, num_layers=num_transformer_layers)
        self.classifier = nn.Sequential(nn.LayerNorm(embedding_dim),
                                        nn.Linear(embedding_dim, num_classes))

    def forward(self, x):
        x = self.encoder(self.embedding_block(x))
        return self.classifier(x[:, 0])