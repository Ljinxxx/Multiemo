import torch
import torch.nn as nn


class GatedCrossModalBlock(nn.Module):
    """
    [MODIFIED]
    新的跨模态融合块。

    原始 MultiAttn 的逻辑是：
        main modality 作为 Query，
        其他模态作为 Key / Value 做 cross-attention。

    这里的修改是：
        1. 仍然保留 cross-attention；
        2. 额外加入 gate 门控；
        3. 让模型自动学习“应该从辅助模态吸收多少信息”。
    """

    def __init__(self, model_dim, num_heads, hidden_dim, dropout_rate):
        super().__init__()

        # [MODIFIED-1]
        # 使用 PyTorch 原生多头注意力，输入格式为 [batch, seq_len, dim]
        self.cross_attn_1 = nn.MultiheadAttention(
            embed_dim=model_dim,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True
        )

        self.cross_attn_2 = nn.MultiheadAttention(
            embed_dim=model_dim,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True
        )

        # [MODIFIED-2]
        # 第一次跨模态融合的门控：
        # 输入 main 和 attn_out 的拼接，输出每个维度的 gate
        self.gate_1 = nn.Sequential(
            nn.Linear(model_dim * 2, model_dim),
            nn.Sigmoid()
        )

        # [MODIFIED-3]
        # 第二次跨模态融合的门控
        self.gate_2 = nn.Sequential(
            nn.Linear(model_dim * 2, model_dim),
            nn.Sigmoid()
        )

        self.norm_1 = nn.LayerNorm(model_dim)
        self.norm_2 = nn.LayerNorm(model_dim)
        self.norm_3 = nn.LayerNorm(model_dim)

        self.dropout = nn.Dropout(dropout_rate)

        # [MODIFIED-4]
        # FFN 保留 Transformer 风格结构
        self.ffn = nn.Sequential(
            nn.Linear(model_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, model_dim)
        )

    def forward(self, main_modality, modality_A, modality_B):
        """
        main_modality: [batch, seq_len, model_dim]
        modality_A:    [batch, seq_len, model_dim]
        modality_B:    [batch, seq_len, model_dim]

        返回：
            fused_main: [batch, seq_len, model_dim]
        """

        # ================================
        # [MODIFIED] 第一阶段：main attend to modality_A
        # ================================
        attn_out_1, _ = self.cross_attn_1(
            query=main_modality,
            key=modality_A,
            value=modality_A,
            need_weights=False
        )

        # [MODIFIED] gate 控制 modality_A 注入强度
        gate_1 = self.gate_1(
            torch.cat([main_modality, attn_out_1], dim=-1)
        )

        # [MODIFIED] 门控残差
        x = self.norm_1(
            main_modality + self.dropout(gate_1 * attn_out_1)
        )

        # ================================
        # [MODIFIED] 第二阶段：上一步结果 attend to modality_B
        # ================================
        attn_out_2, _ = self.cross_attn_2(
            query=x,
            key=modality_B,
            value=modality_B,
            need_weights=False
        )

        gate_2 = self.gate_2(
            torch.cat([x, attn_out_2], dim=-1)
        )

        x = self.norm_2(
            x + self.dropout(gate_2 * attn_out_2)
        )

        # ================================
        # FFN
        # ================================
        ffn_out = self.ffn(x)

        x = self.norm_3(
            x + self.dropout(ffn_out)
        )

        return x


class MultiAttn(nn.Module):
    """
    [MODIFIED]
    堆叠多个 GatedCrossModalBlock。

    保留原始 MultiAttn 的调用方式：
        query_modality, modality_A, modality_B -> fused query_modality
    """

    def __init__(self, num_layers, model_dim, num_heads, hidden_dim, dropout_rate):
        super().__init__()

        self.layers = nn.ModuleList([
            GatedCrossModalBlock(
                model_dim=model_dim,
                num_heads=num_heads,
                hidden_dim=hidden_dim,
                dropout_rate=dropout_rate
            )
            for _ in range(num_layers)
        ])

    def forward(self, query_modality, modality_A, modality_B):
        for layer in self.layers:
            query_modality = layer(query_modality, modality_A, modality_B)

        return query_modality


class MultiAttnModel(nn.Module):
    """
    [IMPORTANT]
    这个类名不要改。

    因为 Model/MultiEMO_Model.py 里面导入的是：
        from MultiAttn import MultiAttnModel

    所以我们保留 MultiAttnModel 的类名和接口。
    """

    def __init__(self, num_layers, model_dim, num_heads, hidden_dim, dropout_rate):
        super().__init__()

        # [MODIFIED]
        # 文本分支：text 作为主模态，融合 audio 和 visual
        self.multiattn_text = MultiAttn(
            num_layers=num_layers,
            model_dim=model_dim,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            dropout_rate=dropout_rate
        )

        # [MODIFIED]
        # 音频分支：audio 作为主模态，融合 text 和 visual
        self.multiattn_audio = MultiAttn(
            num_layers=num_layers,
            model_dim=model_dim,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            dropout_rate=dropout_rate
        )

        # [MODIFIED]
        # 视觉分支：visual 作为主模态，融合 text 和 audio
        self.multiattn_visual = MultiAttn(
            num_layers=num_layers,
            model_dim=model_dim,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            dropout_rate=dropout_rate
        )

    def forward(self, text_features, audio_features, visual_features):
        """
        text_features:   [batch, seq_len, model_dim]
        audio_features:  [batch, seq_len, model_dim]
        visual_features: [batch, seq_len, model_dim]
        """

        # [MODIFIED]
        # 三个模态分别作为主模态进行融合
        f_t = self.multiattn_text(
            text_features,
            audio_features,
            visual_features
        )

        f_a = self.multiattn_audio(
            audio_features,
            text_features,
            visual_features
        )

        f_v = self.multiattn_visual(
            visual_features,
            text_features,
            audio_features
        )

        return f_t, f_a, f_v