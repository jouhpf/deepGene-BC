import torch
import torch.nn as nn

class Mlp(nn.Module):
    def __init__(self, input_dim: int, n_classes: int):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16, n_classes)
        )
    def forward(self, x):
        return self.classifier(x)


class DeepFMLite(nn.Module):
    def __init__(self, n_features, n_classes, embed_dim=4):
        super().__init__()
        self.n_features = n_features
        self.embed_dim = embed_dim
        self.n_classes = n_classes

        # ---------- Linear (Wide) ----------
        self.linear = nn.Linear(n_features, n_classes)

        # ---------- Embedding ----------
        self.embedding = nn.Embedding(n_features, embed_dim)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.01)

        # ---------- Deep ----------
        self.deep = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(32, n_classes)
        )

    def forward(self, x):
        """
        x: [B, N], binary (0/1)
        """
        # ---------- Linear ----------
        linear_part = self.linear(x)   # [B, C]

        # ---------- Embedding lookup ----------
        # v: [B, N, K]
        v = self.embedding.weight.unsqueeze(0) * x.unsqueeze(2)

        # ---------- FM ----------
        sum_v = v.sum(dim=1)                       # [B, K]
        sum_v_square = sum_v * sum_v               # (sum v)^2
        square_sum_v = (v * v).sum(dim=1)          # sum(v^2)

        fm_part = 0.5 * (sum_v_square - square_sum_v)
        fm_part = fm_part.sum(dim=1, keepdim=True) # [B, 1]
        fm_part = fm_part.repeat(1, self.n_classes)

        # ---------- Deep ----------
        deep_input = sum_v                          # [B, K]
        deep_part = self.deep(deep_input)           # [B, C]

        # ---------- Output ----------
        logits = linear_part + fm_part + deep_part
        return logits

