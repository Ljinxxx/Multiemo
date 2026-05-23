import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleGraphConv(nn.Module):
    """
    Simple residual graph convolution layer.

    Input:
        x:   [1, N, D]
        adj: [1, N, N]

    Output:
        h:   [1, N, D]
    """

    def __init__(self, dim, dropout=0.1):
        super().__init__()

        self.linear = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, adj):
        message = torch.bmm(adj, x)
        h = self.linear(message)
        h = F.relu(h)
        h = self.dropout(h)

        # Residual inside GCN layer
        h = self.norm(x + h)

        return h


class CrossTaskGNN(nn.Module):
    """
    Residual GNN Adapter.

    This module does NOT directly output emotion logits.

    It only produces a small residual feature delta:
        gnn_delta = GNN(fc_outputs)

    Then TrainMultiEMO.py will use:
        refined_fc_outputs = fc_outputs + gnn_alpha * gnn_delta
        emotion_logits = self.model.mlp(refined_fc_outputs)

    This is much safer than directly replacing the original classifier.
    """

    def __init__(
        self,
        input_dim,
        hidden_dim=None,
        num_classes=None,
        dropout=0.1
    ):
        super().__init__()

        self.input_dim = input_dim

        self.gcn1 = SimpleGraphConv(
            dim=input_dim,
            dropout=dropout
        )

        self.gcn2 = SimpleGraphConv(
            dim=input_dim,
            dropout=dropout
        )

        self.adapter = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, input_dim)
        )

    def forward(self, features, adj):
        """
        features: [1, N, D]
        adj:      [1, N, N]

        return:
            gnn_delta: [N, D]
        """

        h = self.gcn1(features, adj)
        h = self.gcn2(h, adj)

        h = h.squeeze(0)

        # Use tanh to avoid overly large perturbation.
        gnn_delta = torch.tanh(self.adapter(h))

        return gnn_delta