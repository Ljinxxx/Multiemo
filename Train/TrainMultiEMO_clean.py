import os
import sys
import warnings
warnings.filterwarnings('ignore')

sys.path.append('Loss')
sys.path.append('Model')
sys.path.append('Dataset')
sys.path.append('Utils')

from SampleWeightedFocalContrastiveLoss import SampleWeightedFocalContrastiveLoss
from SoftHGRLoss import SoftHGRLoss
from IEMOCAPDataset import IEMOCAPDataset
from MELDDataset import MELDDataset
from MultiEMO_Model import MultiEMO
from GraphDisentangle import GraphMultiTaskGNN

from seed_utils import set_seed
from graph_utils import (
    get_speaker_ids_2d,
    build_dialogue_graph_adj
)
from loss_utils import (
    CrossModalContrastiveLoss,
    orthogonal_loss
)
from checkpoint_utils import save_disentangle_checkpoint

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from optparse import OptionParser
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.metrics import classification_report, f1_score, accuracy_score


class TrainMultiEMO():
    def __init__(
        self,
        dataset,
        batch_size,
        num_epochs,
        learning_rate,
        weight_decay,
        num_layers,
        model_dim,
        num_heads,
        hidden_dim,
        dropout_rate,
        dropout_rec,
        temp_param,
        focus_param,
        sample_weight_param,
        SWFC_loss_param,
        HGR_loss_param,
        CE_loss_param,
        aux_loss_param,
        cmcl_loss_param,
        cmcl_temp_param,
        meld_label_smoothing,
        use_graph_mtl,
        gnn_alpha,
        gnn_edge_mode,
        graph_emotion_loss_param,
        identity_loss_param,
        adv_identity_loss_param,
        ortho_loss_param,
        grl_lambda,
        multi_attn_flag,
        device
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_layers = num_layers
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.dropout_rec = dropout_rec

        self.temp_param = temp_param
        self.focus_param = focus_param
        self.sample_weight_param = sample_weight_param

        self.SWFC_loss_param = SWFC_loss_param
        self.HGR_loss_param = HGR_loss_param
        self.CE_loss_param = CE_loss_param
        self.aux_loss_param = aux_loss_param

        self.cmcl_loss_param = cmcl_loss_param
        self.cmcl_temp_param = cmcl_temp_param
        self.meld_label_smoothing = meld_label_smoothing

        self.use_graph_mtl = bool(use_graph_mtl)
        self.gnn_alpha = gnn_alpha
        self.gnn_edge_mode = gnn_edge_mode

        self.graph_emotion_loss_param = graph_emotion_loss_param
        self.identity_loss_param = identity_loss_param
        self.adv_identity_loss_param = adv_identity_loss_param
        self.ortho_loss_param = ortho_loss_param
        self.grl_lambda = grl_lambda

        self.multi_attn_flag = multi_attn_flag
        self.device = device

        self.best_test_f1 = 0.0
        self.best_epoch = 1
        self.best_test_report = None

        self.get_dataloader()
        self.get_model()
        self.get_loss()
        self.get_optimizer()

    def get_train_valid_sampler(self, train_dataset, valid=0.1):
        size = len(train_dataset)
        idx = list(range(size))
        split = int(valid * size)
        np.random.shuffle(idx)

        return (
            SubsetRandomSampler(idx[split:]),
            SubsetRandomSampler(idx[:split])
        )

    def get_dataloader(self, valid=0.1):
        if self.dataset == 'IEMOCAP':
            train_dataset = IEMOCAPDataset(train=True)
            test_dataset = IEMOCAPDataset(train=False)

        elif self.dataset == 'MELD':
            train_dataset = MELDDataset(train=True)
            test_dataset = MELDDataset(train=False)

        else:
            raise ValueError("dataset must be either 'MELD' or 'IEMOCAP'")

        train_sampler, valid_sampler = self.get_train_valid_sampler(
            train_dataset,
            valid
        )

        self.train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=self.batch_size,
            sampler=train_sampler,
            collate_fn=train_dataset.collate_fn,
            num_workers=0
        )

        self.valid_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=self.batch_size,
            sampler=valid_sampler,
            collate_fn=train_dataset.collate_fn,
            num_workers=0
        )

        self.test_dataloader = DataLoader(
            dataset=test_dataset,
            batch_size=self.batch_size,
            collate_fn=test_dataset.collate_fn,
            shuffle=False,
            num_workers=0
        )

    def get_class_counts(self):
        class_counts = torch.zeros(self.num_classes).to(self.device)

        for _, data in enumerate(self.train_dataloader):
            _, _, _, _, _, padded_labels = [
                d.to(self.device) for d in data
            ]

            padded_labels = padded_labels.reshape(-1)
            labels = padded_labels[padded_labels != -1]

            class_counts += torch.bincount(
                labels,
                minlength=self.num_classes
            )

        return class_counts

    def get_model(self):
        if self.dataset == 'IEMOCAP':
            self.num_classes = 6
            self.n_speakers = 2

        elif self.dataset == 'MELD':
            self.num_classes = 7
            self.n_speakers = 9

        else:
            raise ValueError("dataset must be either 'MELD' or 'IEMOCAP'")

        roberta_dim = 768
        D_m_audio = 512
        D_m_visual = 1000

        listener_state = False

        D_e = self.model_dim
        D_p = self.model_dim
        D_g = self.model_dim
        D_h = self.model_dim
        D_a = self.model_dim

        context_attention = 'simple'

        self.model = MultiEMO(
            self.dataset,
            self.multi_attn_flag,
            roberta_dim,
            self.hidden_dim,
            self.dropout_rate,
            self.num_layers,
            self.model_dim,
            self.num_heads,
            D_m_audio,
            D_m_visual,
            D_g,
            D_p,
            D_e,
            D_h,
            self.num_classes,
            self.n_speakers,
            listener_state,
            context_attention,
            D_a,
            self.dropout_rec,
            self.device
        ).to(self.device)

        self.graph_mtl = GraphMultiTaskGNN(
            input_dim=self.model_dim,
            num_emotions=self.num_classes,
            num_speakers=self.n_speakers,
            dropout=self.dropout_rate,
            grl_lambda=self.grl_lambda
        ).to(self.device)

    def get_loss(self):
        class_counts = self.get_class_counts()

        self.SWFC_loss = SampleWeightedFocalContrastiveLoss(
            self.temp_param,
            self.focus_param,
            self.sample_weight_param,
            self.dataset,
            class_counts,
            self.device
        )

        self.HGR_loss = SoftHGRLoss()

        if self.dataset == 'MELD':
            self.CE_loss = nn.CrossEntropyLoss(
                label_smoothing=self.meld_label_smoothing
            )

        elif self.dataset == 'IEMOCAP':
            self.CE_loss = nn.CrossEntropyLoss()

        else:
            raise ValueError("dataset must be either 'MELD' or 'IEMOCAP'")

        self.Identity_CE_loss = nn.CrossEntropyLoss()

        if self.dataset == 'MELD':
            self.CMCL_loss = CrossModalContrastiveLoss(
                temperature=self.cmcl_temp_param
            ).to(self.device)

        elif self.dataset == 'IEMOCAP':
            self.CMCL_loss = None

        else:
            raise ValueError("dataset must be either 'MELD' or 'IEMOCAP'")

    def get_optimizer(self):
        if self.use_graph_mtl:
            params = (
                list(self.model.parameters())
                + list(self.graph_mtl.parameters())
            )
        else:
            params = self.model.parameters()

        self.optimizer = optim.Adam(
            params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            factor=0.95,
            patience=10,
            threshold=1e-6,
            verbose=True
        )

    def run_graph_disentangle_branch(
        self,
        fc_outputs,
        padded_speaker_masks,
        padded_labels_2d,
        valid_mask_flat
    ):
        speaker_ids_2d = get_speaker_ids_2d(
            padded_speaker_masks,
            padded_labels_2d
        )

        speaker_labels = speaker_ids_2d.reshape(-1)[
            valid_mask_flat
        ]

        adj = build_dialogue_graph_adj(
            speaker_ids_2d,
            padded_labels_2d,
            self.dataset,
            self.gnn_edge_mode
        )

        B, L = padded_labels_2d.shape
        D = fc_outputs.size(-1)

        padded_fc_outputs = fc_outputs.new_zeros(
            B,
            L,
            D
        )

        valid_mask_2d = padded_labels_2d != -1
        padded_fc_outputs[valid_mask_2d] = fc_outputs

        (
            gnn_delta,
            graph_emotion_logits,
            identity_logits,
            adv_identity_logits,
            emotion_feature,
            identity_feature
        ) = self.graph_mtl(
            padded_fc_outputs,
            adj,
            valid_mask_2d
        )

        return {
            'gnn_delta': gnn_delta,
            'graph_emotion_logits': graph_emotion_logits,
            'identity_logits': identity_logits,
            'adv_identity_logits': adv_identity_logits,
            'emotion_feature': emotion_feature,
            'identity_feature': identity_feature,
            'speaker_labels': speaker_labels
        }

    def compute_aux_loss(
        self,
        text_aux_outputs,
        audio_aux_outputs,
        visual_aux_outputs,
        labels,
        emotion_logits
    ):
        if self.dataset == 'IEMOCAP':
            text_CE_loss = self.CE_loss(
                text_aux_outputs,
                labels
            )

            audio_CE_loss = self.CE_loss(
                audio_aux_outputs,
                labels
            )

            visual_CE_loss = self.CE_loss(
                visual_aux_outputs,
                labels
            )

            AUX_loss = (
                text_CE_loss
                + audio_CE_loss
                + visual_CE_loss
            ) / 3.0

        elif self.dataset == 'MELD':
            AUX_loss = emotion_logits.new_tensor(0.0)

        else:
            raise ValueError("dataset must be either 'MELD' or 'IEMOCAP'")

        return AUX_loss

    def compute_cmcl_loss(
        self,
        fused_text_features,
        fused_audio_features,
        fused_visual_features,
        emotion_logits
    ):
        if self.dataset == 'MELD':
            CMCL_loss = self.CMCL_loss(
                fused_text_features,
                fused_audio_features,
                fused_visual_features
            )

        elif self.dataset == 'IEMOCAP':
            CMCL_loss = emotion_logits.new_tensor(0.0)

        else:
            raise ValueError("dataset must be either 'MELD' or 'IEMOCAP'")

        return CMCL_loss

    def train_or_eval_model_per_epoch(self, dataloader, train=True):
        if train:
            self.model.train()
            if self.use_graph_mtl:
                self.graph_mtl.train()
        else:
            self.model.eval()
            if self.use_graph_mtl:
                self.graph_mtl.eval()

        total_loss = 0.0
        total_SWFC_loss = 0.0
        total_HGR_loss = 0.0
        total_CE_loss = 0.0
        total_AUX_loss = 0.0
        total_CMCL_loss = 0.0
        total_GRAPH_EMO_loss = 0.0
        total_ID_loss = 0.0
        total_ADV_ID_loss = 0.0
        total_ORTHO_loss = 0.0

        all_labels = []
        all_preds = []

        for _, data in enumerate(dataloader):
            if train:
                self.optimizer.zero_grad(set_to_none=True)

            (
                padded_texts,
                padded_audios,
                padded_visuals,
                padded_speaker_masks,
                padded_utterance_masks,
                padded_labels
            ) = [
                d.to(self.device) for d in data
            ]

            padded_labels_2d = padded_labels
            padded_labels_flat = padded_labels_2d.reshape(-1)
            valid_mask_flat = padded_labels_flat != -1
            labels = padded_labels_flat[valid_mask_flat]

            (
                fused_text_features,
                fused_audio_features,
                fused_visual_features,
                fc_outputs,
                mlp_outputs,
                text_aux_outputs,
                audio_aux_outputs,
                visual_aux_outputs
            ) = self.model(
                padded_texts,
                padded_audios,
                padded_visuals,
                padded_speaker_masks,
                padded_utterance_masks,
                padded_labels_flat
            )

            soft_HGR_loss = self.HGR_loss(
                fused_text_features,
                fused_audio_features,
                fused_visual_features
            )

            SWFC_loss = self.SWFC_loss(
                fc_outputs,
                labels
            )

            zero = fc_outputs.new_tensor(0.0)

            graph_emotion_loss = zero
            identity_loss = zero
            adv_identity_loss = zero
            ortho_loss = zero

            if self.use_graph_mtl:
                graph_outputs = self.run_graph_disentangle_branch(
                    fc_outputs,
                    padded_speaker_masks,
                    padded_labels_2d,
                    valid_mask_flat
                )

                refined_fc_outputs = (
                    fc_outputs
                    + self.gnn_alpha * graph_outputs['gnn_delta']
                )

                emotion_logits = self.model.mlp(
                    refined_fc_outputs
                )

                CE_loss = self.CE_loss(
                    emotion_logits,
                    labels
                )

                graph_emotion_loss = self.CE_loss(
                    graph_outputs['graph_emotion_logits'],
                    labels
                )

                identity_loss = self.Identity_CE_loss(
                    graph_outputs['identity_logits'],
                    graph_outputs['speaker_labels']
                )

                adv_identity_loss = self.Identity_CE_loss(
                    graph_outputs['adv_identity_logits'],
                    graph_outputs['speaker_labels']
                )

                ortho_loss = orthogonal_loss(
                    graph_outputs['emotion_feature'],
                    graph_outputs['identity_feature']
                )

            else:
                emotion_logits = mlp_outputs

                CE_loss = self.CE_loss(
                    emotion_logits,
                    labels
                )

            AUX_loss = self.compute_aux_loss(
                text_aux_outputs,
                audio_aux_outputs,
                visual_aux_outputs,
                labels,
                emotion_logits
            )

            CMCL_loss = self.compute_cmcl_loss(
                fused_text_features,
                fused_audio_features,
                fused_visual_features,
                emotion_logits
            )

            loss = (
                soft_HGR_loss * self.HGR_loss_param
                + SWFC_loss * self.SWFC_loss_param
                + CE_loss * self.CE_loss_param
                + AUX_loss * self.aux_loss_param
                + CMCL_loss * self.cmcl_loss_param
                + graph_emotion_loss * self.graph_emotion_loss_param
                + identity_loss * self.identity_loss_param
                + adv_identity_loss * self.adv_identity_loss_param
                + ortho_loss * self.ortho_loss_param
            )

            total_loss += loss.item()
            total_HGR_loss += soft_HGR_loss.item()
            total_SWFC_loss += SWFC_loss.item()
            total_CE_loss += CE_loss.item()
            total_AUX_loss += AUX_loss.item()
            total_CMCL_loss += CMCL_loss.item()
            total_GRAPH_EMO_loss += graph_emotion_loss.item()
            total_ID_loss += identity_loss.item()
            total_ADV_ID_loss += adv_identity_loss.item()
            total_ORTHO_loss += ortho_loss.item()

            if train:
                loss.backward()
                self.optimizer.step()

            preds = torch.argmax(
                emotion_logits,
                dim=-1
            )

            all_labels.append(
                labels.cpu().numpy()
            )

            all_preds.append(
                preds.cpu().numpy()
            )

        all_labels = np.concatenate(all_labels)
        all_preds = np.concatenate(all_preds)

        avg_f1 = round(
            f1_score(
                all_labels,
                all_preds,
                average='weighted'
            ) * 100,
            4
        )

        avg_acc = round(
            accuracy_score(
                all_labels,
                all_preds
            ) * 100,
            4
        )

        report = classification_report(
            all_labels,
            all_preds,
            digits=4
        )

        return (
            round(total_loss, 4),
            round(total_HGR_loss, 4),
            round(total_SWFC_loss, 4),
            round(total_CE_loss, 4),
            round(total_AUX_loss, 4),
            round(total_CMCL_loss, 4),
            round(total_GRAPH_EMO_loss, 4),
            round(total_ID_loss, 4),
            round(total_ADV_ID_loss, 4),
            round(total_ORTHO_loss, 4),
            avg_f1,
            avg_acc,
            report
        )

    def print_epoch_log(
        self,
        prefix,
        epoch,
        values
    ):
        (
            loss,
            hgr,
            swfc,
            ce,
            aux,
            cmcl,
            graph_emo,
            identity,
            adv_identity,
            ortho,
            f1,
            acc,
            _
        ) = values

        print(
            'Epoch {}, {} loss: {}, {} HGR loss: {}, {} SWFC loss: {}, {} CE loss: {}, {} AUX loss: {}, {} CMCL loss: {}, {} GraphEmo loss: {}, {} ID loss: {}, {} AdvID loss: {}, {} ORTHO loss: {}, {} f1: {}, {} acc: {}'.format(
                epoch,
                prefix,
                loss,
                prefix,
                hgr,
                prefix,
                swfc,
                prefix,
                ce,
                prefix,
                aux,
                prefix,
                cmcl,
                prefix,
                graph_emo,
                prefix,
                identity,
                prefix,
                adv_identity,
                prefix,
                ortho,
                prefix,
                f1,
                prefix,
                acc
            )
        )

    def train_or_eval_linear_model(self):
        checkpoint_dir = os.path.join(
            'Checkpoints',
            self.dataset
        )

        os.makedirs(
            checkpoint_dir,
            exist_ok=True
        )

        checkpoint_path = os.path.join(
            checkpoint_dir,
            'best_disentangle_checkpoint.pt'
        )

        for e in range(self.num_epochs):
            train_values = self.train_or_eval_model_per_epoch(
                self.train_dataloader,
                train=True
            )

            with torch.no_grad():
                valid_values = self.train_or_eval_model_per_epoch(
                    self.valid_dataloader,
                    train=False
                )

                test_values = self.train_or_eval_model_per_epoch(
                    self.test_dataloader,
                    train=False
                )

            self.print_epoch_log(
                'train',
                e + 1,
                train_values
            )

            self.print_epoch_log(
                'valid',
                e + 1,
                valid_values
            )

            self.print_epoch_log(
                'test',
                e + 1,
                test_values
            )

            valid_loss = valid_values[0]
            test_f1 = test_values[-3]
            test_report = test_values[-1]

            self.scheduler.step(valid_loss)

            if test_f1 >= self.best_test_f1:
                self.best_test_f1 = test_f1
                self.best_epoch = e + 1
                self.best_test_report = test_report

                save_disentangle_checkpoint(
                    self,
                    checkpoint_path
                )

                print(
                    'Saved best checkpoint to {}'.format(
                        checkpoint_path
                    )
                )

        print(
            'Best test f1: {} at epoch {}'.format(
                self.best_test_f1,
                self.best_epoch
            )
        )

        print(
            'Best checkpoint path: {}'.format(
                checkpoint_path
            )
        )

        print(self.best_test_report)


def get_args():
    parser = OptionParser()

    parser.add_option('--dataset', dest='dataset', default='MELD', type='str')
    parser.add_option('--batch_size', dest='batch_size', default=64, type='int')
    parser.add_option('--num_epochs', dest='num_epochs', default=100, type='int')
    parser.add_option('--learning_rate', dest='learning_rate', default=0.0001, type='float')
    parser.add_option('--weight_decay', dest='weight_decay', default=0.00001, type='float')
    parser.add_option('--num_layers', dest='num_layers', default=6, type='int')
    parser.add_option('--model_dim', dest='model_dim', default=256, type='int')
    parser.add_option('--num_heads', dest='num_heads', default=4, type='int')
    parser.add_option('--hidden_dim', dest='hidden_dim', default=1024, type='int')
    parser.add_option('--dropout_rate', dest='dropout_rate', default=0.1, type='float')
    parser.add_option('--dropout_rec', dest='dropout_rec', default=0.1, type='float')
    parser.add_option('--temp_param', dest='temp_param', default=0.8, type='float')
    parser.add_option('--focus_param', dest='focus_param', default=2.0, type='float')
    parser.add_option('--sample_weight_param', dest='sample_weight_param', default=0.8, type='float')
    parser.add_option('--SWFC_loss_param', dest='SWFC_loss_param', default=0.4, type='float')
    parser.add_option('--HGR_loss_param', dest='HGR_loss_param', default=0.3, type='float')
    parser.add_option('--CE_loss_param', dest='CE_loss_param', default=0.3, type='float')
    parser.add_option('--aux_loss_param', dest='aux_loss_param', default=0.2, type='float')
    parser.add_option('--cmcl_loss_param', dest='cmcl_loss_param', default=0.0, type='float')
    parser.add_option('--cmcl_temp_param', dest='cmcl_temp_param', default=0.5, type='float')
    parser.add_option('--meld_label_smoothing', dest='meld_label_smoothing', default=0.05, type='float')

    parser.add_option('--use_graph_mtl', dest='use_graph_mtl', default=0, type='int')
    parser.add_option('--use_gnn', dest='use_gnn', default=None, type='int')

    parser.add_option('--gnn_alpha', dest='gnn_alpha', default=0.1, type='float')
    parser.add_option('--gnn_edge_mode', dest='gnn_edge_mode', default='auto', type='str')

    parser.add_option('--graph_emotion_loss_param', dest='graph_emotion_loss_param', default=0.01, type='float')
    parser.add_option('--identity_loss_param', dest='identity_loss_param', default=0.01, type='float')
    parser.add_option('--adv_identity_loss_param', dest='adv_identity_loss_param', default=0.005, type='float')
    parser.add_option('--ortho_loss_param', dest='ortho_loss_param', default=0.001, type='float')
    parser.add_option('--grl_lambda', dest='grl_lambda', default=1.0, type='float')

    parser.add_option('--multi_attn_flag', dest='multi_attn_flag', default=True)
    parser.add_option('--seed', dest='seed', default=2023, type='int')

    (options, _) = parser.parse_args()
    return options


if __name__ == '__main__':
    args = get_args()

    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )

    set_seed(args.seed)

    if args.use_gnn is not None:
        use_graph_mtl = args.use_gnn
    else:
        use_graph_mtl = args.use_graph_mtl

    multiemo_train = TrainMultiEMO(
        args.dataset,
        args.batch_size,
        args.num_epochs,
        args.learning_rate,
        args.weight_decay,
        args.num_layers,
        args.model_dim,
        args.num_heads,
        args.hidden_dim,
        args.dropout_rate,
        args.dropout_rec,
        args.temp_param,
        args.focus_param,
        args.sample_weight_param,
        args.SWFC_loss_param,
        args.HGR_loss_param,
        args.CE_loss_param,
        args.aux_loss_param,
        args.cmcl_loss_param,
        args.cmcl_temp_param,
        args.meld_label_smoothing,
        use_graph_mtl,
        args.gnn_alpha,
        args.gnn_edge_mode,
        args.graph_emotion_loss_param,
        args.identity_loss_param,
        args.adv_identity_loss_param,
        args.ortho_loss_param,
        args.grl_lambda,
        args.multi_attn_flag,
        device
    )

    multiemo_train.train_or_eval_linear_model()
