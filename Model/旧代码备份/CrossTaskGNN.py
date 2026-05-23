import torch
import torch.nn as nn
import torch.nn.functional as F


class GradientReverseFunction(torch.autograd.Function):
    """
    Gradient Reversal Layer.

    Forward:
        y = x

    Backward:
        dL/dx = -lambda * dL/dy

    This is used to remove speaker identity information
    from emotion-related representations.
    """

    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


class GradientReverseLayer(nn.Module):
    def __init__(self, lambd=1.0):
        super().__init__()
        self.lambd = lambd

    def forward(self, x):
        return GradientReverseFunction.apply(x, self.lambd)


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
    Graph-based multi-task module with identity disentanglement.

    Tasks:
        1. Emotion recognition
        2. Speaker identity classification
        3. Adversarial speaker removal from emotion feature

    Input:
        padded_fc_outputs: [B, L, D]
        adj:               [B, L, L]
        valid_mask:        [B, L]

    Output:
        gnn_delta:            [N, D]
        graph_emotion_logits: [N, num_emotions]
        identity_logits:      [N, num_speakers]
        adv_identity_logits:  [N, num_speakers]
        emotion_feature:      [N, D]
        identity_feature:     [N, D]
    """

    def __init__(
        self,
        input_dim,
        num_emotions,
        num_speakers,
        dropout=0.1,
        grl_lambda=1.0
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

        self.grl = GradientReverseLayer(
            lambd=grl_lambda
        )

        self.adv_identity_classifier = nn.Linear(
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

        # Only keep valid utterances before task-specific branches.
        h_valid = h[valid_mask]

        emotion_feature = self.emotion_proj(
            h_valid
        )

        identity_feature = self.identity_proj(
            h_valid
        )

        # Residual delta used to refine emotion classification feature.
        gnn_delta = self.delta_adapter(
            emotion_feature
        )

        graph_emotion_logits = self.graph_emotion_classifier(
            emotion_feature
        )

        identity_logits = self.identity_classifier(
            identity_feature
        )

        # Adversarial identity prediction from emotion feature.
        # Because of GRL, this makes emotion_feature less speaker-informative.
        reversed_emotion_feature = self.grl(
            emotion_feature
        )

        adv_identity_logits = self.adv_identity_classifier(
            reversed_emotion_feature
        )

        return (
            gnn_delta,
            graph_emotion_logits,
            identity_logits,
            adv_identity_logits,
            emotion_feature,
            identity_feature
        )