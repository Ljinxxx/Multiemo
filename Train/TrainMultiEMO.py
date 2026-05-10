import sys
sys.path.append('Loss')
sys.path.append('Model')
sys.path.append('Dataset')

from SampleWeightedFocalContrastiveLoss import SampleWeightedFocalContrastiveLoss
from SoftHGRLoss import SoftHGRLoss
from IEMOCAPDataset import IEMOCAPDataset
from MELDDataset import MELDDataset
from MultiEMO_Model import MultiEMO

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from optparse import OptionParser
import torch.optim as optim
import numpy as np
from sklearn.metrics import classification_report, f1_score, accuracy_score
import warnings
warnings.filterwarnings('ignore')
from torch.utils.data.sampler import SubsetRandomSampler
import random


class CrossModalContrastiveLoss(nn.Module):
    """
    Ordinary three-pair Cross-modal Contrastive Loss.

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

    def forward(
        self,
        text_features,
        audio_features,
        visual_features
    ):
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
        multi_attn_flag,
        device,
        use_line_graph=False,
        line_graph_dropout=0.1,
        line_graph_gate_init=-3.0,
        line_graph_use_vector_gate=False,
        line_graph_use_confidence_gate=True,
        line_graph_uncertainty_gamma=1.0
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

        self.multi_attn_flag = multi_attn_flag
        self.device = device

        self.use_line_graph = use_line_graph
        self.line_graph_dropout = line_graph_dropout
        self.line_graph_gate_init = line_graph_gate_init
        self.line_graph_use_vector_gate = line_graph_use_vector_gate
        self.line_graph_use_confidence_gate = line_graph_use_confidence_gate
        self.line_graph_uncertainty_gamma = line_graph_uncertainty_gamma

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

        return SubsetRandomSampler(idx[split:]), SubsetRandomSampler(idx[:split])

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
            self.device,
            use_line_graph=self.use_line_graph,
            line_graph_dropout=self.line_graph_dropout,
            line_graph_gate_init=self.line_graph_gate_init,
            line_graph_use_vector_gate=self.line_graph_use_vector_gate,
            line_graph_use_confidence_gate=self.line_graph_use_confidence_gate,
            line_graph_uncertainty_gamma=self.line_graph_uncertainty_gamma
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

        if self.dataset == 'MELD':
            self.CMCL_loss = CrossModalContrastiveLoss(
                temperature=self.cmcl_temp_param
            ).to(self.device)

        elif self.dataset == 'IEMOCAP':
            self.CMCL_loss = None

        else:
            raise ValueError("dataset must be either 'MELD' or 'IEMOCAP'")

    def get_optimizer(self):
        self.optimizer = optim.Adam(
            self.model.parameters(),
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

    def train_or_eval_model_per_epoch(self, dataloader, train=True):
        if train:
            self.model.train()
        else:
            self.model.eval()

        total_loss = 0.0
        total_SWFC_loss = 0.0
        total_HGR_loss = 0.0
        total_CE_loss = 0.0
        total_AUX_loss = 0.0
        total_CMCL_loss = 0.0

        all_labels = []
        all_preds = []

        for _, data in enumerate(dataloader):
            if train:
                self.optimizer.zero_grad()

            padded_texts, padded_audios, padded_visuals, padded_speaker_masks, padded_utterance_masks, padded_labels = [
                d.to(self.device) for d in data
            ]

            padded_labels = padded_labels.reshape(-1)
            labels = padded_labels[padded_labels != -1]

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
                padded_labels
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

            CE_loss = self.CE_loss(
                mlp_outputs,
                labels
            )

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
                AUX_loss = mlp_outputs.new_tensor(0.0)

            else:
                raise ValueError("dataset must be either 'MELD' or 'IEMOCAP'")

            if self.dataset == 'MELD':
                CMCL_loss = self.CMCL_loss(
                    fused_text_features,
                    fused_audio_features,
                    fused_visual_features
                )

            elif self.dataset == 'IEMOCAP':
                CMCL_loss = mlp_outputs.new_tensor(0.0)

            else:
                raise ValueError("dataset must be either 'MELD' or 'IEMOCAP'")

            loss = (
                soft_HGR_loss * self.HGR_loss_param
                + SWFC_loss * self.SWFC_loss_param
                + CE_loss * self.CE_loss_param
                + AUX_loss * self.aux_loss_param
                + CMCL_loss * self.cmcl_loss_param
            )

            total_loss += loss.item()
            total_HGR_loss += soft_HGR_loss.item()
            total_SWFC_loss += SWFC_loss.item()
            total_CE_loss += CE_loss.item()
            total_AUX_loss += AUX_loss.item()
            total_CMCL_loss += CMCL_loss.item()

            if train:
                loss.backward()
                self.optimizer.step()

            preds = torch.argmax(mlp_outputs, dim=-1)

            all_labels.append(labels.cpu().numpy())
            all_preds.append(preds.cpu().numpy())

        all_labels = np.concatenate(all_labels)
        all_preds = np.concatenate(all_preds)

        avg_f1 = round(
            f1_score(all_labels, all_preds, average='weighted') * 100,
            4
        )

        avg_acc = round(
            accuracy_score(all_labels, all_preds) * 100,
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
            avg_f1,
            avg_acc,
            report
        )

    def train_or_eval_linear_model(self):
        for e in range(self.num_epochs):
            (
                train_loss,
                train_HGR_loss,
                train_SWFC_loss,
                train_CE_loss,
                train_AUX_loss,
                train_CMCL_loss,
                train_f1,
                train_acc,
                _
            ) = self.train_or_eval_model_per_epoch(
                self.train_dataloader,
                train=True
            )

            with torch.no_grad():
                (
                    valid_loss,
                    valid_HGR_loss,
                    valid_SWFC_loss,
                    valid_CE_loss,
                    valid_AUX_loss,
                    valid_CMCL_loss,
                    valid_f1,
                    valid_acc,
                    _
                ) = self.train_or_eval_model_per_epoch(
                    self.valid_dataloader,
                    train=False
                )

                (
                    test_loss,
                    test_HGR_loss,
                    test_SWFC_loss,
                    test_CE_loss,
                    test_AUX_loss,
                    test_CMCL_loss,
                    test_f1,
                    test_acc,
                    test_report
                ) = self.train_or_eval_model_per_epoch(
                    self.test_dataloader,
                    train=False
                )

            print(
                'Epoch {}, train loss: {}, train HGR loss: {}, train SWFC loss: {}, train CE loss: {}, train AUX loss: {}, train CMCL loss: {}, train f1: {}, train acc: {}'.format(
                    e + 1,
                    train_loss,
                    train_HGR_loss,
                    train_SWFC_loss,
                    train_CE_loss,
                    train_AUX_loss,
                    train_CMCL_loss,
                    train_f1,
                    train_acc
                )
            )

            print(
                'Epoch {}, valid loss: {}, valid HGR loss: {}, valid SWFC loss: {}, valid CE loss: {}, valid AUX loss: {}, valid CMCL loss: {}, valid f1: {}, valid acc: {}'.format(
                    e + 1,
                    valid_loss,
                    valid_HGR_loss,
                    valid_SWFC_loss,
                    valid_CE_loss,
                    valid_AUX_loss,
                    valid_CMCL_loss,
                    valid_f1,
                    valid_acc
                )
            )

            print(
                'Epoch {}, test loss: {}, test HGR loss: {}, test SWFC loss: {}, test CE loss: {}, test AUX loss: {}, test CMCL loss: {}, test f1: {}, test acc: {}, '.format(
                    e + 1,
                    test_loss,
                    test_HGR_loss,
                    test_SWFC_loss,
                    test_CE_loss,
                    test_AUX_loss,
                    test_CMCL_loss,
                    test_f1,
                    test_acc
                )
            )

            # Large-scale experiments produce very large logs.
            # The best report will still be printed at the end.
            # print(test_report)

            self.scheduler.step(valid_loss)

            if test_f1 >= self.best_test_f1:
                self.best_test_f1 = test_f1
                self.best_epoch = e + 1
                self.best_test_report = test_report

        print(
            'Best test f1: {} at epoch {}'.format(
                self.best_test_f1,
                self.best_epoch
            )
        )
        print(self.best_test_report)


def get_args():
    parser = OptionParser()

    parser.add_option(
        '--dataset',
        dest='dataset',
        default='MELD',
        type='str',
        help='MELD or IEMOCAP'
    )

    parser.add_option(
        '--batch_size',
        dest='batch_size',
        default=64,
        type='int',
        help='64 for IEMOCAP and 100 for MELD'
    )

    parser.add_option(
        '--num_epochs',
        dest='num_epochs',
        default=100,
        type='int',
        help='number of epochs'
    )

    parser.add_option(
        '--learning_rate',
        dest='learning_rate',
        default=0.0001,
        type='float',
        help='learning rate'
    )

    parser.add_option(
        '--weight_decay',
        dest='weight_decay',
        default=0.00001,
        type='float',
        help='weight decay parameter'
    )

    parser.add_option(
        '--num_layers',
        dest='num_layers',
        default=6,
        type='int',
        help='number of layers in MultiAttn'
    )

    parser.add_option(
        '--model_dim',
        dest='model_dim',
        default=256,
        type='int',
        help='model dimension in MultiAttn'
    )

    parser.add_option(
        '--num_heads',
        dest='num_heads',
        default=4,
        type='int',
        help='number of heads in MultiAttn'
    )

    parser.add_option(
        '--hidden_dim',
        dest='hidden_dim',
        default=1024,
        type='int',
        help='hidden dimension in MultiAttn'
    )

    parser.add_option(
        '--dropout_rate',
        dest='dropout_rate',
        default=0.1,
        type='float',
        help='dropout rate'
    )

    parser.add_option(
        '--dropout_rec',
        dest='dropout_rec',
        default=0.1,
        type='float',
        help='dropout rec'
    )

    parser.add_option(
        '--temp_param',
        dest='temp_param',
        default=0.8,
        type='float',
        help='temperature parameter of SWFC loss'
    )

    parser.add_option(
        '--focus_param',
        dest='focus_param',
        default=2.0,
        type='float',
        help='focusing parameter of SWFC loss'
    )

    parser.add_option(
        '--sample_weight_param',
        dest='sample_weight_param',
        default=0.8,
        type='float',
        help='sample-weight parameter of SWFC loss'
    )

    parser.add_option(
        '--SWFC_loss_param',
        dest='SWFC_loss_param',
        default=0.4,
        type='float',
        help='coefficient of SWFC loss'
    )

    parser.add_option(
        '--HGR_loss_param',
        dest='HGR_loss_param',
        default=0.3,
        type='float',
        help='coefficient of Soft-HGR loss'
    )

    parser.add_option(
        '--CE_loss_param',
        dest='CE_loss_param',
        default=0.3,
        type='float',
        help='coefficient of Cross Entropy loss'
    )

    parser.add_option(
        '--aux_loss_param',
        dest='aux_loss_param',
        default=0.2,
        type='float',
        help='coefficient of auxiliary unimodal CE loss'
    )

    parser.add_option(
        '--cmcl_loss_param',
        dest='cmcl_loss_param',
        default=0.0,
        type='float',
        help='coefficient of MELD-only ordinary CMCL'
    )

    parser.add_option(
        '--cmcl_temp_param',
        dest='cmcl_temp_param',
        default=0.5,
        type='float',
        help='temperature of MELD-only ordinary CMCL'
    )

    parser.add_option(
        '--meld_label_smoothing',
        dest='meld_label_smoothing',
        default=0.05,
        type='float',
        help='label smoothing value for MELD CE loss'
    )

    parser.add_option(
        '--multi_attn_flag',
        dest='multi_attn_flag',
        default=True,
        help='Multimodal fusion'
    )

    parser.add_option(
        '--use_line_graph',
        dest='use_line_graph',
        default=0,
        type='int',
        help='whether to enable Local Residual LineGraph Adapter: 0 or 1'
    )

    parser.add_option(
        '--line_graph_dropout',
        dest='line_graph_dropout',
        default=0.1,
        type='float',
        help='dropout rate in LineGraph adapter'
    )

    parser.add_option(
        '--line_graph_gate_init',
        dest='line_graph_gate_init',
        default=-5.0,
        type='float',
        help='initial logit for residual graph gate; -5.0 means graph branch starts nearly closed'
    )

    parser.add_option(
        '--line_graph_use_vector_gate',
        dest='line_graph_use_vector_gate',
        default=0,
        type='int',
        help='whether to use vector gate instead of scalar gate: 0 or 1'
    )

    parser.add_option(
        '--line_graph_use_confidence_gate',
        dest='line_graph_use_confidence_gate',
        default=1,
        type='int',
        help='whether to use confidence-aware uncertainty gate: 0 or 1'
    )

    parser.add_option(
        '--line_graph_uncertainty_gamma',
        dest='line_graph_uncertainty_gamma',
        default=1.0,
        type='float',
        help='exponent for uncertainty gate: (1 - confidence)^gamma'
    )

    parser.add_option(
        '--seed',
        dest='seed',
        default=2023,
        type='int',
        help='random seed'
    )

    (options, _) = parser.parse_args()

    return options


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    args = get_args()

    dataset = args.dataset
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    learning_rate = args.learning_rate
    weight_decay = args.weight_decay
    num_layers = args.num_layers
    model_dim = args.model_dim
    num_heads = args.num_heads
    hidden_dim = args.hidden_dim
    dropout_rate = args.dropout_rate
    dropout_rec = args.dropout_rec
    temp_param = args.temp_param
    focus_param = args.focus_param
    sample_weight_param = args.sample_weight_param
    SWFC_loss_param = args.SWFC_loss_param
    HGR_loss_param = args.HGR_loss_param
    CE_loss_param = args.CE_loss_param
    aux_loss_param = args.aux_loss_param
    cmcl_loss_param = args.cmcl_loss_param
    cmcl_temp_param = args.cmcl_temp_param
    meld_label_smoothing = args.meld_label_smoothing
    multi_attn_flag = args.multi_attn_flag
    seed = args.seed

    use_line_graph = bool(args.use_line_graph)
    line_graph_dropout = args.line_graph_dropout
    line_graph_gate_init = args.line_graph_gate_init
    line_graph_use_vector_gate = bool(args.line_graph_use_vector_gate)
    line_graph_use_confidence_gate = bool(args.line_graph_use_confidence_gate)
    line_graph_uncertainty_gamma = args.line_graph_uncertainty_gamma

    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )

    set_seed(seed)

    multiemo_train = TrainMultiEMO(
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
        multi_attn_flag,
        device,
        use_line_graph=use_line_graph,
        line_graph_dropout=line_graph_dropout,
        line_graph_gate_init=line_graph_gate_init,
        line_graph_use_vector_gate=line_graph_use_vector_gate,
        line_graph_use_confidence_gate=line_graph_use_confidence_gate,
        line_graph_uncertainty_gamma=line_graph_uncertainty_gamma
    )

    multiemo_train.train_or_eval_linear_model()