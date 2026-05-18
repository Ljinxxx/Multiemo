import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleGraphConv(nn.Module):
    """
    Memory-lite residual graph convolution layer.

    Input:
        x:   [B, L, D]
        adj: [B, L, L]

    Output:
        h:   [B, L, D]
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

        h = self.norm(x + h)

        return h


class GraphMultiTaskGNN(nn.Module):
    """
    Memory-lite Graph Multi-task GNN.

    Compared with the previous version:

    1. Use only one GCN layer.
    2. Extract valid utterances first.
    3. Emotion branch and identity branch are computed only on valid nodes.
    4. Keep the same return format, so TrainMultiEMO.py does not need major changes.

    Input:
        padded_fc_outputs: [B, L, D]
        adj:               [B, L, L]
        valid_mask:        [B, L]

    Output:
        gnn_delta:            [N, D]
        graph_emotion_logits: [N, num_emotions]
        identity_logits:      [N, num_speakers]
        emotion_feature:      [N, D]
        identity_feature:     [N, D]
    """

    def __init__(
        self,
        input_dim,
        num_emotions,
        num_speakers,
        dropout=0.1
    ):
        super().__init__()

        self.gcn = SimpleGraphConv(
            dim=input_dim,
            dropout=dropout
        )

        self.shared_norm = nn.LayerNorm(input_dim)

        self.emotion_proj = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(input_dim)
        )

        self.identity_proj = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(input_dim)
        )

        self.delta_adapter = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Tanh()
        )

        self.graph_emotion_classifier = nn.Linear(
            input_dim,
            num_emotions
        )

        self.identity_classifier = nn.Linear(
            input_dim,
            num_speakers
        )

    def forward(self, padded_fc_outputs, adj, valid_mask):
        """
        padded_fc_outputs: [B, L, D]
        adj:               [B, L, L]
        valid_mask:        [B, L]
        """

        h = self.gcn(
            padded_fc_outputs,
            adj
        )

        h = self.shared_norm(h)

        # Very important:
        # take valid nodes first, then run task-specific branches.
        # This avoids doing multiple MLPs on padded [B, L, D].
        h_valid = h[valid_mask]

        emotion_feature = self.emotion_proj(
            h_valid
        )

        identity_feature = self.identity_proj(
            h_valid
        )

        gnn_delta = self.delta_adapter(
            emotion_feature
        )

        graph_emotion_logits = self.graph_emotion_classifier(
            emotion_feature
        )

        identity_logits = self.identity_classifier(
            identity_feature
        )

        return (
            gnn_delta,
            graph_emotion_logits,
            identity_logits,
            emotion_feature,
            identity_feature
        )