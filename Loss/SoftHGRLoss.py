import torch
import torch.nn as nn


class SoftHGRLoss(nn.Module):
    """
    [MODIFIED]
    稳定版 Soft-HGR Loss。

    目的：
        最大化融合后的文本、音频、视觉特征之间的相关性。

    输入：
        f_t: [num_valid_utterances, model_dim]
        f_a: [num_valid_utterances, model_dim]
        f_v: [num_valid_utterances, model_dim]
    """

    def __init__(self):
        super().__init__()
        self.eps = 1e-8

    def center(self, feature):
        """
        [MODIFIED]
        对特征做中心化。
        """

        return feature - feature.mean(dim=0, keepdim=True)

    def covariance(self, feature):
        """
        [MODIFIED]
        计算特征维度上的协方差矩阵。

        feature: [N, D]
        返回:    [D, D]
        """

        num_samples = feature.shape[0]

        if num_samples <= 1:
            feature_dim = feature.shape[-1]
            return torch.zeros(
                feature_dim,
                feature_dim,
                device=feature.device,
                dtype=feature.dtype
            )

        feature = self.center(feature)

        cov = torch.matmul(feature.t(), feature) / (num_samples - 1)

        return cov

    def pair_soft_hgr(self, feature_X, feature_Y):
        """
        [MODIFIED]
        计算两个模态之间的 Soft-HGR 项。

        Soft-HGR 目标：
            maximize E[X^T Y] - 1/2 tr(Cov(X) Cov(Y))

        训练时我们最小化 loss，所以最后取负号。
        """

        feature_X = self.center(feature_X)
        feature_Y = self.center(feature_Y)

        # E[X^T Y]
        feature_mapping = torch.mean(
            torch.sum(feature_X * feature_Y, dim=-1)
        )

        cov_X = self.covariance(feature_X)
        cov_Y = self.covariance(feature_Y)

        covariance_penalty = torch.trace(
            torch.matmul(cov_X, cov_Y)
        )

        soft_hgr_score = feature_mapping - 0.5 * covariance_penalty

        return -soft_hgr_score

    def forward(self, f_t, f_a, f_v):
        """
        f_t, f_a, f_v:
            [num_valid_utterances, model_dim]
        """

        if f_t.shape[0] <= 1:
            return f_t.new_tensor(0.0)

        # [MODIFIED]
        # 三组模态相关性：
        # text-audio, text-visual, audio-visual
        loss_ta = self.pair_soft_hgr(f_t, f_a)
        loss_tv = self.pair_soft_hgr(f_t, f_v)
        loss_av = self.pair_soft_hgr(f_a, f_v)

        # [MODIFIED]
        # 取平均，让 loss 尺度稳定
        loss = (loss_ta + loss_tv + loss_av) / 3.0

        return loss