import torch
import torch.nn as nn
import torch.nn.functional as F


class OmicsEncoder(nn.Module):
    """
    A simple MLP encoder for individual omics modality.
    Includes a projection head for contrastive learning.
    """

    def __init__(self, input_dim, hidden_dim=256, projection_dim=128):
        super(OmicsEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )

        self.projection_head = nn.Sequential(
            nn.Linear(hidden_dim, projection_dim),
            nn.BatchNorm1d(projection_dim),
            nn.ReLU()
        )

    def forward(self, x):
        features = self.encoder(x)
        projection = self.projection_head(features)
        return F.normalize(projection, dim=1)  # L2-normalized embeddings
