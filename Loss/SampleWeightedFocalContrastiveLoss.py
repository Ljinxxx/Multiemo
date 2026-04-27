import torch
import torch.nn as nn
import torch.nn.functional as F


class SampleWeightedFocalContrastiveLoss(nn.Module):
    """
    [MODIFIED]
    向量化 Sample-Weighted Focal Contrastive Loss。

    原始 SWFC 思想：
        1. 同类样本拉近；
        2. 不同类样本推远；
        3. 少数类样本权重大；
        4. 难样本权重大。

    这里保留原始思想，但修改实现方式：
        - 原代码逐样本循环；
        - 新代码使用相似度矩阵一次性计算；
        - 速度更快，逻辑更清晰，数值也更稳定。
    """

    def __init__(
        self,
        temp_param,
        focus_param,
        sample_weight_param,
        dataset,
        class_counts,
        device
    ):
        super().__init__()

        self.temp_param = temp_param
        self.focus_param = focus_param
        self.sample_weight_param = sample_weight_param
        self.dataset = dataset
        self.device = device
        self.eps = 1e-8

        if self.dataset == 'MELD':
            self.num_classes = 7
        elif self.dataset == 'IEMOCAP':
            self.num_classes = 6
        else:
            raise ValueError('Please choose either MELD or IEMOCAP')

        # [MODIFIED]
        # class_counts 来自 TrainMultiEMO.py 的 get_class_counts()
        self.class_counts = class_counts.float().to(self.device)

        # [MODIFIED]
        # 计算类别权重
        self.class_weights = self.get_sample_weights()

    def get_sample_weights(self):
        """
        [MODIFIED]
        类别权重：
            类别样本越少，权重越大。

        原论文形式类似：
            weight_c = (N / N_c) ^ alpha

        这里使用 mean-normalization，避免 loss 数值过大或过小。
        """

        total_counts = torch.sum(self.class_counts)

        class_weights = (
            total_counts / (self.class_counts + self.eps)
        ) ** self.sample_weight_param

        # [MODIFIED]
        # 用均值归一化，而不是 sum 归一化。
        # 这样整体 loss 尺度更接近 CE loss。
        class_weights = class_weights / class_weights.mean().clamp_min(self.eps)

        return class_weights

    def forward(self, features, labels):
        """
        features: [num_valid_utterances, feature_dim]
        labels:   [num_valid_utterances]

        注意：
            TrainMultiEMO.py 里传进来的 labels 已经去掉了 padding label -1。
        """

        labels = labels.long().to(self.device)
        features = features.to(self.device)

        num_samples = labels.shape[0]

        if num_samples <= 1:
            return features.new_tensor(0.0)

        # [MODIFIED]
        # 归一化后用点积等价于 cosine similarity
        features = F.normalize(features, dim=-1)

        # ================================
        # [MODIFIED] 计算两两相似度矩阵
        # similarity: [N, N]
        # ================================
        similarity = torch.matmul(features, features.t()) / self.temp_param

        # 去除自己和自己对比
        self_mask = torch.eye(
            num_samples,
            device=self.device,
            dtype=torch.bool
        )

        # 正样本 mask：标签相同且不是自己
        positive_mask = labels.unsqueeze(0).eq(labels.unsqueeze(1))
        positive_mask = positive_mask & (~self_mask)

        # 分母 mask：除了自己之外，其他样本都参与
        logits_mask = ~self_mask

        # ================================
        # [MODIFIED] 数值稳定处理
        # ================================
        similarity = similarity - similarity.max(dim=1, keepdim=True)[0].detach()

        exp_similarity = torch.exp(similarity) * logits_mask.float()

        denominator = exp_similarity.sum(dim=1, keepdim=True) + self.eps

        probability = exp_similarity / denominator

        # ================================
        # [MODIFIED] 只取正样本概率
        # ================================
        positive_probability_sum = (
            probability * positive_mask.float()
        ).sum(dim=1)

        positive_count = positive_mask.float().sum(dim=1)

        # [MODIFIED]
        # 有些 anchor 在 batch 里可能没有同类正样本。
        # 这些 anchor 不能计算对比损失，直接跳过。
        valid_anchor_mask = positive_count > 0

        if valid_anchor_mask.sum() == 0:
            return features.new_tensor(0.0)

        positive_probability = (
            positive_probability_sum[valid_anchor_mask]
            / positive_count[valid_anchor_mask].clamp_min(1.0)
        )

        valid_labels = labels[valid_anchor_mask]

        # ================================
        # [MODIFIED] 类别权重
        # ================================
        sample_weights = self.class_weights[valid_labels]

        # ================================
        # [MODIFIED] focal 项
        # 难样本 positive_probability 低，则权重更高
        # ================================
        focal_weights = (
            1.0 - positive_probability
        ).clamp_min(0.0) ** self.focus_param

        loss = (
            -sample_weights
            * focal_weights
            * torch.log(positive_probability + self.eps)
        )

        return loss.mean()