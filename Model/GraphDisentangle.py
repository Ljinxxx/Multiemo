import torch
import torch.nn as nn
import torch.nn.functional as F



"""
Graph-based identity disentanglement module.

This file implements the graph-based multi-task module used after the
MultiEMO backbone. Given utterance-level multimodal representations, the
module builds graph-enhanced representations and separates them into:

1. emotion-oriented features for emotion recognition;
2. identity-oriented features for speaker identity modeling;
3. adversarial identity prediction from emotion features through a
   Gradient Reversal Layer (GRL).

The goal is to reduce speaker identity leakage in emotion-oriented features
while preserving an explicit identity branch for speaker-related information.
"""




# 梯度反转层：
# 前向传播时保持输入不变；
# 反向传播时将梯度乘以负系数。
# 这样可以让 emotion_feature 在训练过程中变得更难预测说话人身份，
# 从而减少情感表征中的身份信息泄露。
class GradientReverseFunction(torch.autograd.Function):
    """
    Gradient Reversal Layer.

    Forward:
        y = x

    Backward:
        dL/dx = -lambda * dL/dy

    This is used to reduce speaker identity information
    in the emotion-related feature.
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
    Memory-lite residual graph convolution.

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


# 图多任务身份解耦模块：
# 输入为 MultiEMO 得到的话语级融合表征；
# 首先通过图卷积建模话语之间的关系；
# 然后分出 emotion_feature 和 identity_feature 两个表征空间；
# identity_feature 用于正常身份分类；
# emotion_feature 经过梯度反转层后用于对抗身份分类。

class GraphMultiTaskGNN(nn.Module):
    """
    Graph-based emotion-identity multi-task module with identity disentanglement.

    It provides:
        1. graph emotion auxiliary prediction;
        2. identity prediction from identity feature;
        3. adversarial identity prediction from emotion feature through GRL;
        4. residual delta for emotion classification.

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
        h = self.gcn(
            padded_fc_outputs,
            adj
        )

        h = self.shared_norm(h)

        # Compute task-specific branches only for valid utterances.
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

        reversed_emotion_feature = self.grl(
            emotion_feature
        )
        # 对抗身份预测分支：
        # adv_identity_logits 由 emotion_feature 经过梯度反转层后得到。
        # 优化该分支时，身份分类器会学习预测说话人身份，
        # 但 emotion_feature 接收到的是反向梯度，
        # 因此会逐渐去除其中可预测的说话人身份信息。
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
