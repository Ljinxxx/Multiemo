import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalContrastiveLoss(nn.Module):
    """
    Ordinary three-pair cross-modal contrastive loss.

    Positive pairs:
        text_i  <-> audio_i
        text_i  <-> visual_i
        audio_i <-> visual_i
    """

    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def pair_contrastive_loss(self, x, y):
        if x.size(0) <= 1:
            return x.new_tensor(0.0)

        x = F.normalize(x, dim=-1)
        y = F.normalize(y, dim=-1)

        logits = torch.matmul(x, y.t()) / self.temperature
        targets = torch.arange(x.size(0), device=x.device)

        loss_xy = F.cross_entropy(logits, targets)
        loss_yx = F.cross_entropy(logits.t(), targets)

        return (loss_xy + loss_yx) / 2.0

    def forward(self, text_features, audio_features, visual_features):
        loss_ta = self.pair_contrastive_loss(
            text_features,
            audio_features
        )

        loss_tv = self.pair_contrastive_loss(
            text_features,
            visual_features
        )

        loss_av = self.pair_contrastive_loss(
            audio_features,
            visual_features
        )

        return (loss_ta + loss_tv + loss_av) / 3.0


def orthogonal_loss(emotion_feature, identity_feature):
    emotion_feature = F.normalize(
        emotion_feature,
        dim=-1
    )

    identity_feature = F.normalize(
        identity_feature,
        dim=-1
    )

    loss = torch.mean(
        torch.sum(
            emotion_feature * identity_feature,
            dim=-1
        ) ** 2
    )

    return loss


def compute_macro_f1(preds, labels, num_classes):
    f1_list = []

    for c in range(num_classes):
        tp = ((preds == c) & (labels == c)).sum().item()
        fp = ((preds == c) & (labels != c)).sum().item()
        fn = ((preds != c) & (labels == c)).sum().item()

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)

        f1 = (
            2 * precision * recall
            / (precision + recall + 1e-8)
        )

        f1_list.append(f1)

    return float(np.mean(f1_list) * 100.0)


def majority_baseline(labels, num_classes):
    counts = torch.bincount(
        labels.long(),
        minlength=num_classes
    )

    return (
        counts.max().float()
        / labels.numel()
        * 100.0
    ).item()
