from DialogueRNN import BiModel
from MultiAttn import MultiAttnModel
from MLP import MLP

import torch
import torch.nn as nn


'''
MultiEMO consists of three key components:
unimodal context modeling, multimodal fusion, and emotion classification.

[MODIFIED]
Residual MultiAttn Skip version:

Original classifier input:
    fused_text + fused_audio + fused_visual

Modified classifier input:
    raw_text + raw_audio + raw_visual
    +
    fused_text + fused_audio + fused_visual

Purpose:
    Preserve original unimodal contextual features after MultiAttn fusion.
'''


class MultiEMO(nn.Module):
    def __init__(
        self,
        dataset,
        multi_attn_flag,
        roberta_dim,
        hidden_dim,
        dropout,
        num_layers,
        model_dim,
        num_heads,
        D_m_audio,
        D_m_visual,
        D_g,
        D_p,
        D_e,
        D_h,
        n_classes,
        n_speakers,
        listener_state,
        context_attention,
        D_a,
        dropout_rec,
        device
    ):
        super().__init__()

        self.dataset = dataset
        self.multi_attn_flag = multi_attn_flag

        # ==============================
        # Unimodal feature projection
        # ==============================
        self.text_fc = nn.Linear(roberta_dim, model_dim)

        self.text_dialoguernn = BiModel(
            model_dim,
            D_g,
            D_p,
            D_e,
            D_h,
            dataset,
            n_classes,
            n_speakers,
            listener_state,
            context_attention,
            D_a,
            dropout_rec,
            dropout,
            device
        )

        self.audio_fc = nn.Linear(D_m_audio, model_dim)

        self.audio_dialoguernn = BiModel(
            model_dim,
            D_g,
            D_p,
            D_e,
            D_h,
            dataset,
            n_classes,
            n_speakers,
            listener_state,
            context_attention,
            D_a,
            dropout_rec,
            dropout,
            device
        )

        self.visual_fc = nn.Linear(D_m_visual, model_dim)

        self.visual_dialoguernn = BiModel(
            model_dim,
            D_g,
            D_p,
            D_e,
            D_h,
            dataset,
            n_classes,
            n_speakers,
            listener_state,
            context_attention,
            D_a,
            dropout_rec,
            dropout,
            device
        )

        # ==============================
        # MultiAttn fusion module
        # ==============================
        self.multiattn = MultiAttnModel(
            num_layers,
            model_dim,
            num_heads,
            hidden_dim,
            dropout
        )

        # ============================================================
        # [MODIFIED-1]
        # 原始版本：
        #     self.fc = nn.Linear(model_dim * 3, model_dim)
        #
        # 因为原来只拼接：
        #     fused_text + fused_audio + fused_visual
        #
        # 现在拼接：
        #     raw_text + raw_audio + raw_visual
        #     +
        #     fused_text + fused_audio + fused_visual
        #
        # 所以输入维度从 3 * model_dim 变成 6 * model_dim。
        # ============================================================
        self.fc = nn.Linear(model_dim * 6, model_dim)

        if self.dataset == 'MELD':
            self.mlp = MLP(
                model_dim,
                model_dim * 2,
                n_classes,
                dropout
            )

        elif self.dataset == 'IEMOCAP':
            self.mlp = MLP(
                model_dim,
                model_dim,
                n_classes,
                dropout
            )

    def forward(
        self,
        texts,
        audios,
        visuals,
        speaker_masks,
        utterance_masks,
        padded_labels
    ):
        # ==============================
        # Text modality
        # ==============================
        text_features = self.text_fc(texts)

        # IEMOCAP uses additional textual context modeling.
        if self.dataset == 'IEMOCAP':
            text_features = self.text_dialoguernn(
                text_features,
                speaker_masks,
                utterance_masks
            )

        # ==============================
        # Audio modality
        # ==============================
        audio_features = self.audio_fc(audios)
        audio_features = self.audio_dialoguernn(
            audio_features,
            speaker_masks,
            utterance_masks
        )

        # ==============================
        # Visual modality
        # ==============================
        visual_features = self.visual_fc(visuals)
        visual_features = self.visual_dialoguernn(
            visual_features,
            speaker_masks,
            utterance_masks
        )

        # DialogueRNN output shape:
        #     [seq_len, batch_size, model_dim]
        #
        # MultiAttn input shape:
        #     [batch_size, seq_len, model_dim]
        text_features = text_features.transpose(0, 1)
        audio_features = audio_features.transpose(0, 1)
        visual_features = visual_features.transpose(0, 1)

        # ============================================================
        # [MODIFIED-2]
        # 在进入 MultiAttn 之前，保存原始单模态上下文特征。
        #
        # 这些 raw features 不是最初的 raw input，
        # 而是经过 fc 投影和 DialogueRNN 上下文建模后的单模态特征。
        # ============================================================
        raw_text_features = text_features
        raw_audio_features = audio_features
        raw_visual_features = visual_features

        # ==============================
        # MultiAttn fusion
        # ==============================
        if self.multi_attn_flag == True:
            fused_text_features, fused_audio_features, fused_visual_features = self.multiattn(
                text_features,
                audio_features,
                visual_features
            )
        else:
            fused_text_features = text_features
            fused_audio_features = audio_features
            fused_visual_features = visual_features

        # ============================================================
        # Flatten and remove padding utterances.
        #
        # 这里沿用你原代码的逻辑：
        #     padded_labels != -1 表示有效话语
        # ============================================================

        # ------------------------------
        # Raw unimodal features
        # ------------------------------
        raw_text_features = raw_text_features.reshape(
            -1,
            raw_text_features.shape[-1]
        )
        raw_text_features = raw_text_features[padded_labels != -1]

        raw_audio_features = raw_audio_features.reshape(
            -1,
            raw_audio_features.shape[-1]
        )
        raw_audio_features = raw_audio_features[padded_labels != -1]

        raw_visual_features = raw_visual_features.reshape(
            -1,
            raw_visual_features.shape[-1]
        )
        raw_visual_features = raw_visual_features[padded_labels != -1]

        # ------------------------------
        # MultiAttn fused features
        # ------------------------------
        fused_text_features = fused_text_features.reshape(
            -1,
            fused_text_features.shape[-1]
        )
        fused_text_features = fused_text_features[padded_labels != -1]

        fused_audio_features = fused_audio_features.reshape(
            -1,
            fused_audio_features.shape[-1]
        )
        fused_audio_features = fused_audio_features[padded_labels != -1]

        fused_visual_features = fused_visual_features.reshape(
            -1,
            fused_visual_features.shape[-1]
        )
        fused_visual_features = fused_visual_features[padded_labels != -1]

        # ============================================================
        # [MODIFIED-3]
        # 原始版本：
        #     fused_features = torch.cat(
        #         (fused_text_features,
        #          fused_audio_features,
        #          fused_visual_features),
        #         dim=-1
        #     )
        #
        # 修改后：
        #     把 MultiAttn 前后的特征都送入分类器。
        #
        # 这样分类器可以同时利用：
        #     1. 原始单模态上下文信息
        #     2. 融合后的跨模态信息
        # ============================================================
        fused_features = torch.cat(
            (
                raw_text_features,
                raw_audio_features,
                raw_visual_features,
                fused_text_features,
                fused_audio_features,
                fused_visual_features
            ),
            dim=-1
        )

        # ==============================
        # Classification
        # ==============================
        fc_outputs = self.fc(fused_features)
        mlp_outputs = self.mlp(fc_outputs)

        # ============================================================
        # 返回值保持和原始代码完全一致。
        #
        # 所以 Train/TrainMultiEMO.py 不用改：
        #     HGR loss 仍然使用 fused_text/audio/visual
        #     SWFC loss 仍然使用 fc_outputs
        #     CE loss 仍然使用 mlp_outputs
        # ============================================================
        return (
            fused_text_features,
            fused_audio_features,
            fused_visual_features,
            fc_outputs,
            mlp_outputs
        )