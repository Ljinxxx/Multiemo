from DialogueRNN import BiModel
from MultiAttn import MultiAttnModel
from MLP import MLP

import torch
import torch.nn as nn


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

        # Text modality
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

        # Audio modality
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

        # Visual modality
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

        # MultiAttn fusion
        self.multiattn = MultiAttnModel(
            num_layers,
            model_dim,
            num_heads,
            hidden_dim,
            dropout
        )

        # Dataset-specific classifier input dimension
        if self.dataset == 'IEMOCAP':
            # IEMOCAP uses residual skip: raw + fused
            self.fc = nn.Linear(model_dim * 6, model_dim)

        elif self.dataset == 'MELD':
            # MELD uses fused-only
            self.fc = nn.Linear(model_dim * 3, model_dim)

        else:
            raise ValueError("dataset must be either 'MELD' or 'IEMOCAP'")

        # Main classifier
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

        # Auxiliary unimodal classifiers
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
        # Text modality
        text_features = self.text_fc(texts)

        # Keep original MultiEMO behavior:
        # IEMOCAP uses text DialogueRNN.
        if self.dataset == 'IEMOCAP':
            text_features = self.text_dialoguernn(
                text_features,
                speaker_masks,
                utterance_masks
            )

        # Audio modality
        audio_features = self.audio_fc(audios)

        audio_features = self.audio_dialoguernn(
            audio_features,
            speaker_masks,
            utterance_masks
        )

        # Visual modality
        visual_features = self.visual_fc(visuals)

        visual_features = self.visual_dialoguernn(
            visual_features,
            speaker_masks,
            utterance_masks
        )

        # [seq_len, batch_size, dim] -> [batch_size, seq_len, dim]
        text_features = text_features.transpose(0, 1)
        audio_features = audio_features.transpose(0, 1)
        visual_features = visual_features.transpose(0, 1)

        # Save unimodal contextual features before MultiAttn
        raw_text_features = text_features
        raw_audio_features = audio_features
        raw_visual_features = visual_features

        # Dataset-specific attention mask
        if self.multi_attn_flag == True:
            if self.dataset == 'IEMOCAP':
                attention_mask = utterance_masks.bool()

            elif self.dataset == 'MELD':
                # MELD does not pass attention mask.
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

        # Flatten and remove padded utterances
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

        # Auxiliary logits
        text_aux_outputs = self.text_aux_classifier(raw_text_features)
        audio_aux_outputs = self.audio_aux_classifier(raw_audio_features)
        visual_aux_outputs = self.visual_aux_classifier(raw_visual_features)

        # Dataset-specific fusion
        if self.dataset == 'IEMOCAP':
            # IEMOCAP uses residual skip: raw + fused
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
            # MELD uses fused-only
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