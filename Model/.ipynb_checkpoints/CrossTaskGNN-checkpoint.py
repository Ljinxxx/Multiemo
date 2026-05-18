import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleGraphConv(nn.Module):
    """
    Simple GCN layer.

    x:   [1, N, D]
    adj: [1, N, N]
    """

    def __init__(self, in_dim, out_dim, dropout=0.1):
        super().__init__()

        self.linear = nn.Linear(in_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x, adj):
        h = torch.bmm(adj, x)
        h = self.linear(h)
        h = F.relu(h)
        h = self.dropout(h)
        h = self.norm(h)

        return h


class CrossTaskGNN(nn.Module):
    """
    Minimal GNN module.

    This version only does GNN-based emotion classification.
    It does not include identity disentanglement yet.

    Input:
        features: [1, N, D]
        adj:      [1, N, N]

    Output:
        gnn_features: [N, D]
        emotion_logits: [N, num_classes]
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_classes,
        dropout=0.1
    ):
        super().__init__()

        self.gcn1 = SimpleGraphConv(
            input_dim,
            hidden_dim,
            dropout
        )

        self.gcn2 = SimpleGraphConv(
            hidden_dim,
            hidden_dim,
            dropout
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, features, adj):
        h = self.gcn1(features, adj)
        h = self.gcn2(h, adj)

        h = h.squeeze(0)

        emotion_logits = self.classifier(h)

        return h, emotion_logits