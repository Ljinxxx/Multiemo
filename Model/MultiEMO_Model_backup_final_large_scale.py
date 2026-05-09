from DialogueRNN import BiModel
from MultiAttn import MultiAttnModel
from MLP import MLP

import torch
import torch.nn as nn


"""
MultiEMO Model

[MODIFIED VERSION: IEMOCAP final + MELD weak residual]

Dataset-specific strategy:

1. IEMOCAP:
   - Use Residual Skip:
       raw_text + raw_audio + raw_visual
       +
       fused_text + fused_audio + fused_visual

   - Use Masked MultiAttn:
       pass utterance_masks into MultiAttn

   - Use auxiliary unimodal classifiers:
       text/audio/visual auxiliary logits are returned
       and used in TrainMultiEMO.py

2. MELD:
   - Keep fused-only classifier input:
       fused_text + fused_audio + fused_visual

   - Do NOT use full Residual Skip.

   - Do NOT pass attention mask into MultiAttn,
     making MELD closer to original MultiEMO behavior.

   - Add MELD-only Weak Residual Fusion:
       fused_text   = fused_text   + alpha * raw_text
       fused_audio  = fused_audio  + alpha * raw_audio
       fused_visual = fused_visual + alpha * raw_visual

     where alpha is a learnable scalar initialized near 0.119.

   - Auxiliary logits are still returned for code consistency,
     but TrainMultiEMO.py disables AUX loss for MELD.
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
        # MultiAttn fusion module
        # ============================================================
        self.multiattn = MultiAttnModel(
            num_layers,
            model_dim,
            num_heads,
            hidden_dim,
            dropout
        )

        # ============================================================
        # Dataset-specific main classifier input dimension
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
        # [MODIFIED]
        # MELD-only weak residual coefficient.
        #
        # alpha = sigmoid(-2.0) ≈ 0.119
        #
        # This gives MELD a small amount of raw unimodal contextual
        # information without changing classifier input dimension.
        # ============================================================
        if self.dataset == 'MELD':
            self.meld_res_alpha = nn.Parameter(torch.tensor(-2.0))

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
        # Auxiliary unimodal classifiers
        #
        # IEMOCAP:
        #   used by auxiliary unimodal loss
        #
        # MELD:
        #   logits are returned, but AUX loss is disabled in training
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
        # Text modality
        # ============================================================
        text_features = self.text_fc(texts)

        # Original MultiEMO behavior:
        # IEMOCAP uses additional text DialogueRNN.
        if self.dataset == 'IEMOCAP':
            text_features = self.text_dialoguernn(
                text_features,
                speaker_masks,
                utterance_masks
            )

        # ============================================================
        # Audio modality
        # ============================================================
        audio_features = self.audio_fc(audios)

        audio_features = self.audio_dialoguernn(
            audio_features,
            speaker_masks,
            utterance_masks
        )

        # ============================================================
        # Visual modality
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
        # Save unimodal contextual features before MultiAttn.
        #
        # These are not original raw input features.
        # They are projected/contextual unimodal features.
        # ============================================================
        raw_text_features = text_features
        raw_audio_features = audio_features
        raw_visual_features = visual_features

        # ============================================================
        # Dataset-specific MultiAttn masking
        #
        # IEMOCAP:
        #   use attention mask
        #
        # MELD:
        #   do not pass attention mask
        #   keep closer to original MultiEMO behavior
        # ============================================================
        if self.multi_attn_flag == True:
            if self.dataset == 'IEMOCAP':
                attention_mask = utterance_masks.bool()

            elif self.dataset == 'MELD':
                attention_mask = None

            else:
                raise ValueError("dataset must be either 'MELD' or 'IEMOCAP'")

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
        # Flatten and remove padded utterances.
        #
        # padded_labels != -1 means valid utterance.
        # ============================================================

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
        # Auxiliary unimodal logits
        #
        # IEMOCAP:
        #   used by AUX loss
        #
        # MELD:
        #   returned for compatibility, but AUX loss is disabled
        # ============================================================
        text_aux_outputs = self.text_aux_classifier(raw_text_features)
        audio_aux_outputs = self.audio_aux_classifier(raw_audio_features)
        visual_aux_outputs = self.visual_aux_classifier(raw_visual_features)

        # ============================================================
        # Dataset-specific main classifier input
        #
        # IEMOCAP:
        #   Full residual skip:
        #       raw + fused
        #
        # MELD:
        #   Weak residual fusion:
        #       fused = fused + alpha * raw
        #   Then keep original fused-only concat:
        #       fused_text + fused_audio + fused_visual
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
            # ========================================================
            # [MODIFIED]
            # MELD-only Weak Residual Fusion.
            #
            # This is different from full residual skip.
            #
            # Full residual skip:
            #   cat(raw_text, raw_audio, raw_visual,
            #       fused_text, fused_audio, fused_visual)
            #
            # Weak residual fusion:
            #   fused_text   = fused_text   + alpha * raw_text
            #   fused_audio  = fused_audio  + alpha * raw_audio
            #   fused_visual = fused_visual + alpha * raw_visual
            #
            # The classifier input dimension remains 3 * model_dim.
            # ========================================================
            alpha = torch.sigmoid(self.meld_res_alpha)

            fused_text_features = (
                fused_text_features
                + alpha * raw_text_features
            )

            fused_audio_features = (
                fused_audio_features
                + alpha * raw_audio_features
            )

            fused_visual_features = (
                fused_visual_features
                + alpha * raw_visual_features
            )

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