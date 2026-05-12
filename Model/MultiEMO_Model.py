from DialogueRNN import BiModel
from MultiAttn import MultiAttnModel
from MLP import MLP
from LineGraphAdapter import ParallelLineGraphLogitHead
from SpeakerRoleRelationGraph import SpeakerRoleRelationGraphDiscriminator

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
        device,
        use_line_graph=False,
        line_graph_dropout=0.1,
        line_graph_gate_init=-5.0,
        line_graph_use_vector_gate=False,
        line_graph_use_confidence_gate=True,
        line_graph_uncertainty_gamma=1.0,
        use_speaker_role_adv=False,
        speaker_adv_dropout=0.1,
        speaker_adv_hidden_dim=None,
        speaker_adv_max_pairs_per_dialogue=32
    ):
        super().__init__()

        self.dataset = dataset
        self.multi_attn_flag = multi_attn_flag
        self.use_line_graph = use_line_graph
        self.line_graph_use_confidence_gate = line_graph_use_confidence_gate
        self.line_graph_uncertainty_gamma = line_graph_uncertainty_gamma
        self.use_speaker_role_adv = use_speaker_role_adv

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

        # Parallel Line Graph Logit Head (Graph-v1b)
        if self.use_line_graph:
            self.line_graph_head = ParallelLineGraphLogitHead(
                model_dim=model_dim,
                num_classes=n_classes,
                hidden_dim=model_dim,
                dropout=line_graph_dropout,
                gate_init=line_graph_gate_init
            )
        else:
            self.line_graph_head = None

        # Speaker-Role Relation Graph Adversarial Discriminator
        if self.use_speaker_role_adv:
            self.speaker_relation_discriminator = SpeakerRoleRelationGraphDiscriminator(
                model_dim=model_dim,
                hidden_dim=speaker_adv_hidden_dim if speaker_adv_hidden_dim is not None else model_dim,
                dropout=speaker_adv_dropout,
                max_pairs_per_dialogue=speaker_adv_max_pairs_per_dialogue
            )
        else:
            self.speaker_relation_discriminator = None

    def forward(
        self,
        texts,
        audios,
        visuals,
        speaker_masks,
        utterance_masks,
        padded_labels,
        compute_speaker_relation=False,
        speaker_grl_lambda=0.0,
        return_fc_outputs_seq=False
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

        # Keep sequence-level references before flatten
        raw_text_seq = raw_text_features
        raw_audio_seq = raw_audio_features
        raw_visual_seq = raw_visual_features
        fused_text_seq = fused_text_features
        fused_audio_seq = fused_audio_features
        fused_visual_seq = fused_visual_features

        # Construct valid_mask for line graph adapter
        valid_mask = utterance_masks.bool()

        # Construct sequence-level classifier input
        if self.dataset == 'IEMOCAP':
            # IEMOCAP uses residual skip: raw + fused
            fused_features_seq = torch.cat(
                (
                    raw_text_seq,
                    raw_audio_seq,
                    raw_visual_seq,
                    fused_text_seq,
                    fused_audio_seq,
                    fused_visual_seq
                ),
                dim=-1
            )

        elif self.dataset == 'MELD':
            # MELD uses fused-only
            fused_features_seq = torch.cat(
                (
                    fused_text_seq,
                    fused_audio_seq,
                    fused_visual_seq
                ),
                dim=-1
            )

        else:
            raise ValueError("dataset must be either 'MELD' or 'IEMOCAP'")

        # Sequence-level fc
        fc_outputs_seq = self.fc(fused_features_seq)  # [B, T, D]

        # Flatten and remove padded utterances
        flat_label_mask = (padded_labels != -1)

        fc_outputs = fc_outputs_seq.reshape(-1, fc_outputs_seq.shape[-1])
        fc_outputs = fc_outputs[flat_label_mask]

        # Base logits from MLP classifier
        base_logits = self.mlp(fc_outputs)

        # Parallel Line Graph Logit Head
        if self.use_line_graph:
            graph_logits_delta_seq, graph_gate = self.line_graph_head(
                fc_outputs_seq, valid_mask
            )
            graph_logits_delta = graph_logits_delta_seq.reshape(
                -1, graph_logits_delta_seq.shape[-1]
            )
            graph_logits_delta = graph_logits_delta[flat_label_mask]

            if self.line_graph_use_confidence_gate:
                base_probs = torch.softmax(base_logits.detach(), dim=-1)
                confidence = torch.max(base_probs, dim=-1, keepdim=True)[0]
                uncertainty_gate = torch.pow(
                    1.0 - confidence, self.line_graph_uncertainty_gamma
                )
                mlp_outputs = base_logits + graph_gate * uncertainty_gate * graph_logits_delta
            else:
                mlp_outputs = base_logits + graph_gate * graph_logits_delta
        else:
            mlp_outputs = base_logits

        # Flatten raw/fused features for HGR loss and aux outputs
        raw_text_features = raw_text_seq.reshape(-1, raw_text_seq.shape[-1])
        raw_text_features = raw_text_features[flat_label_mask]

        raw_audio_features = raw_audio_seq.reshape(-1, raw_audio_seq.shape[-1])
        raw_audio_features = raw_audio_features[flat_label_mask]

        raw_visual_features = raw_visual_seq.reshape(-1, raw_visual_seq.shape[-1])
        raw_visual_features = raw_visual_features[flat_label_mask]

        fused_text_features = fused_text_seq.reshape(-1, fused_text_seq.shape[-1])
        fused_text_features = fused_text_features[flat_label_mask]

        fused_audio_features = fused_audio_seq.reshape(-1, fused_audio_seq.shape[-1])
        fused_audio_features = fused_audio_features[flat_label_mask]

        fused_visual_features = fused_visual_seq.reshape(-1, fused_visual_seq.shape[-1])
        fused_visual_features = fused_visual_features[flat_label_mask]

        # Auxiliary logits
        text_aux_outputs = self.text_aux_classifier(raw_text_features)
        audio_aux_outputs = self.audio_aux_classifier(raw_audio_features)
        visual_aux_outputs = self.visual_aux_classifier(raw_visual_features)

        # Speaker-Role Relation Graph Adversarial Branch
        speaker_relation_logits = None
        speaker_relation_labels = None

        if self.use_speaker_role_adv and compute_speaker_relation:
            # speaker_masks is [T, B, S] from dataloader, transpose to [B, T, S]
            speaker_masks_bt = speaker_masks.transpose(0, 1)
            speaker_relation_logits, speaker_relation_labels = \
                self.speaker_relation_discriminator(
                    fc_outputs_seq,
                    valid_mask,
                    speaker_masks_bt,
                    grl_lambda=speaker_grl_lambda
                )

        if compute_speaker_relation:
            if return_fc_outputs_seq:
                return (
                    fused_text_features,
                    fused_audio_features,
                    fused_visual_features,
                    fc_outputs,
                    mlp_outputs,
                    text_aux_outputs,
                    audio_aux_outputs,
                    visual_aux_outputs,
                    speaker_relation_logits,
                    speaker_relation_labels,
                    fc_outputs_seq
                )
            else:
                return (
                    fused_text_features,
                    fused_audio_features,
                    fused_visual_features,
                    fc_outputs,
                    mlp_outputs,
                    text_aux_outputs,
                    audio_aux_outputs,
                    visual_aux_outputs,
                    speaker_relation_logits,
                    speaker_relation_labels
                )
        else:
            if return_fc_outputs_seq:
                return (
                    fused_text_features,
                    fused_audio_features,
                    fused_visual_features,
                    fc_outputs,
                    mlp_outputs,
                    text_aux_outputs,
                    audio_aux_outputs,
                    visual_aux_outputs,
                    fc_outputs_seq
                )
            else:
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