from DialogueRNN import BiModel
from MultiAttn import MultiAttnModel
from MLP import MLP

import torch
import torch.nn as nn


"""
MultiEMO Model

[MODIFIED VERSION: Auxiliary Unimodal Loss]

This version includes:

1. Dataset-specific classifier input strategy:
   - IEMOCAP uses Residual Skip:
       raw_text + raw_audio + raw_visual
       +
       fused_text + fused_audio + fused_visual

   - MELD keeps original fused-only input:
       fused_text + fused_audio + fused_visual

2. Masked MultiAttn:
   - Pass utterance_masks into MultiAttn to avoid attending to padding utterances.

3. Auxiliary unimodal classifiers:
   - raw_text_features   -> text_aux_classifier
   - raw_audio_features  -> audio_aux_classifier
   - raw_visual_features -> visual_aux_classifier

The auxiliary logits are returned to TrainMultiEMO.py,
where auxiliary CE losses are computed.
"""


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

        # ============================================================
        # Text modality
        # ============================================================
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

        # ============================================================
        # Audio modality
        # ============================================================
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

        # ============================================================
        # Visual modality
        # ============================================================
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

        # ============================================================
        # MultiAttn fusion
        # ============================================================
        self.multiattn = MultiAttnModel(
            num_layers,
            model_dim,
            num_heads,
            hidden_dim,
            dropout
        )

        # ============================================================
        # [MODIFIED-1]
        # Main classifier input dimension.
        #
        # IEMOCAP:
        #   raw_text + raw_audio + raw_visual
        #   +
        #   fused_text + fused_audio + fused_visual
        #   input dim = 6 * model_dim
        #
        # MELD:
        #   fused_text + fused_audio + fused_visual
        #   input dim = 3 * model_dim
        # ============================================================
        if self.dataset == 'IEMOCAP':
            self.fc = nn.Linear(model_dim * 6, model_dim)
        elif self.dataset == 'MELD':
            self.fc = nn.Linear(model_dim * 3, model_dim)
        else:
            raise ValueError("dataset must be either 'MELD' or 'IEMOCAP'")

        # ============================================================
        # Main MLP classifier
        # ============================================================
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

        # ============================================================
        # [MODIFIED-2]
        # Auxiliary unimodal classifiers.
        #
        # These heads directly supervise unimodal contextual features.
        # The purpose is to prevent multimodal fusion from weakening
        # useful unimodal information.
        # ============================================================
        self.text_aux_classifier = MLP(
            model_dim,
            model_dim,
            n_classes,
            dropout
        )

        self.audio_aux_classifier = MLP(
            model_dim,
            model_dim,
            n_classes,
            dropout
        )

        self.visual_aux_classifier = MLP(
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
        # ============================================================
        # Text feature projection
        # ============================================================
        text_features = self.text_fc(texts)

        # Original implementation:
        # IEMOCAP applies additional textual context modeling.
        if self.dataset == 'IEMOCAP':
            text_features = self.text_dialoguernn(
                text_features,
                speaker_masks,
                utterance_masks
            )

        # ============================================================
        # Audio feature projection + context modeling
        # ============================================================
        audio_features = self.audio_fc(audios)

        audio_features = self.audio_dialoguernn(
            audio_features,
            speaker_masks,
            utterance_masks
        )

        # ============================================================
        # Visual feature projection + context modeling
        # ============================================================
        visual_features = self.visual_fc(visuals)

        visual_features = self.visual_dialoguernn(
            visual_features,
            speaker_masks,
            utterance_masks
        )

        # ============================================================
        # Shape transform
        #
        # DialogueRNN output:
        #   [seq_len, batch_size, model_dim]
        #
        # MultiAttn input:
        #   [batch_size, seq_len, model_dim]
        # ============================================================
        text_features = text_features.transpose(0, 1)
        audio_features = audio_features.transpose(0, 1)
        visual_features = visual_features.transpose(0, 1)

        # ============================================================
        # Save raw unimodal contextual features before MultiAttn.
        #
        # These are not original raw input features.
        # They are already projected and context-modeled.
        # ============================================================
        raw_text_features = text_features
        raw_audio_features = audio_features
        raw_visual_features = visual_features

        # ============================================================
        # MultiAttn fusion with attention mask.
        #
        # utterance_masks:
        #   [batch_size, seq_len]
        #   1 = valid utterance
        #   0 = padding utterance
        # ============================================================
        if self.multi_attn_flag == True:
            attention_mask = utterance_masks.bool()

            fused_text_features, fused_audio_features, fused_visual_features = self.multiattn(
                text_features,
                audio_features,
                visual_features,
                attention_mask=attention_mask
            )

        else:
            fused_text_features = text_features
            fused_audio_features = audio_features
            fused_visual_features = visual_features

        # ============================================================
        # Flatten and remove padding utterances.
        #
        # padded_labels != -1 indicates valid utterances.
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
        # Auxiliary unimodal logits.
        #
        # These logits are used only for auxiliary CE losses.
        # Main prediction still uses mlp_outputs.
        # ============================================================
        text_aux_outputs = self.text_aux_classifier(raw_text_features)
        audio_aux_outputs = self.audio_aux_classifier(raw_audio_features)
        visual_aux_outputs = self.visual_aux_classifier(raw_visual_features)

        # ============================================================
        # Dataset-specific main classifier input.
        # ============================================================
        if self.dataset == 'IEMOCAP':
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

        elif self.dataset == 'MELD':
            fused_features = torch.cat(
                (
                    fused_text_features,
                    fused_audio_features,
                    fused_visual_features
                ),
                dim=-1
            )

        else:
            raise ValueError("dataset must be either 'MELD' or 'IEMOCAP'")

        # ============================================================
        # Main classification
        # ============================================================
        fc_outputs = self.fc(fused_features)
        mlp_outputs = self.mlp(fc_outputs)

        # ============================================================
        # [MODIFIED-4]
        # Return auxiliary logits in addition to original outputs.
        #
        # Original return:
        #   fused_text_features,
        #   fused_audio_features,
        #   fused_visual_features,
        #   fc_outputs,
        #   mlp_outputs
        #
        # New return:
        #   + text_aux_outputs
        #   + audio_aux_outputs
        #   + visual_aux_outputs
        # ============================================================
        return (
            fused_text_features,
            fused_audio_features,
            fused_visual_features,
            fc_outputs,
            mlp_outputs,
            text_aux_outputs,
            audio_aux_outputs,
            visual_aux_outputs
        )