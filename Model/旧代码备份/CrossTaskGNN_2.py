import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleGraphConv(nn.Module):
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
        h = self.norm(x + h)
        return h


class CrossTaskGNN(nn.Module):
    """
    Residual GNN Adapter.

    It does not replace the original classifier.
    It only generates a residual delta:

        refined_fc_outputs = fc_outputs + gnn_alpha * gnn_delta
    """

    def __init__(
        self,
        input_dim,
        hidden_dim=None,
        num_classes=None,
        dropout=0.1
    ):
        super().__init__()

        self.gcn1 = SimpleGraphConv(input_dim, dropout)
        self.gcn2 = SimpleGraphConv(input_dim, dropout)

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
        gnn_delta = torch.tanh(self.adapter(h))

        return gnn_delta