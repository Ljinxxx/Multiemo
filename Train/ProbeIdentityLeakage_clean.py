import os
import sys
import argparse
import random
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        '..'
    )
)

sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, 'Train'))
sys.path.append(os.path.join(PROJECT_ROOT, 'Model'))
sys.path.append(os.path.join(PROJECT_ROOT, 'Dataset'))
sys.path.append(os.path.join(PROJECT_ROOT, 'Loss'))
sys.path.append(os.path.join(PROJECT_ROOT, 'Utils'))

from TrainMultiEMO_clean import TrainMultiEMO
from graph_utils import (
    get_speaker_ids_2d,
    build_dialogue_graph_adj
)
from loss_utils import (
    compute_macro_f1,
    majority_baseline
)
from seed_utils import set_seed


class LinearProbe(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.classifier = nn.Linear(
            input_dim,
            num_classes
        )

    def forward(self, x):
        return self.classifier(x)


def train_probe(
    train_x,
    train_y,
    test_x,
    test_y,
    num_classes,
    device,
    probe_epochs,
    probe_lr,
    probe_batch_size
):
    train_x = train_x.float()
    test_x = test_x.float()
    train_y = train_y.long()
    test_y = test_y.long()

    train_dataset = TensorDataset(
        train_x,
        train_y
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=probe_batch_size,
        shuffle=True
    )

    probe = LinearProbe(
        input_dim=train_x.size(-1),
        num_classes=num_classes
    ).to(device)

    optimizer = torch.optim.Adam(
        probe.parameters(),
        lr=probe_lr,
        weight_decay=1e-5
    )

    criterion = nn.CrossEntropyLoss()

    for _ in range(probe_epochs):
        probe.train()

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = probe(batch_x)

            loss = criterion(
                logits,
                batch_y
            )

            loss.backward()
            optimizer.step()

    probe.eval()

    with torch.no_grad():
        logits = probe(
            test_x.to(device)
        )

        preds = torch.argmax(
            logits,
            dim=-1
        ).cpu()

    acc = (
        preds.eq(test_y).float().mean().item()
        * 100.0
    )

    macro_f1 = compute_macro_f1(
        preds,
        test_y,
        num_classes
    )

    return acc, macro_f1


def build_trainer_from_checkpoint(
    checkpoint,
    device,
    data_batch_size
):
    cfg = checkpoint['config']

    trainer = TrainMultiEMO(
        cfg['dataset'],
        data_batch_size,
        cfg['num_epochs'],
        cfg['learning_rate'],
        cfg['weight_decay'],
        cfg['num_layers'],
        cfg['model_dim'],
        cfg['num_heads'],
        cfg['hidden_dim'],
        cfg['dropout_rate'],
        cfg['dropout_rec'],
        cfg['temp_param'],
        cfg['focus_param'],
        cfg['sample_weight_param'],
        cfg['SWFC_loss_param'],
        cfg['HGR_loss_param'],
        cfg['CE_loss_param'],
        cfg['aux_loss_param'],
        cfg['cmcl_loss_param'],
        cfg['cmcl_temp_param'],
        cfg['meld_label_smoothing'],
        cfg['use_graph_mtl'],
        cfg['gnn_alpha'],
        cfg['gnn_edge_mode'],
        cfg['graph_emotion_loss_param'],
        cfg['identity_loss_param'],
        cfg['adv_identity_loss_param'],
        cfg['ortho_loss_param'],
        cfg['grl_lambda'],
        cfg['multi_attn_flag'],
        device
    )

    trainer.model.load_state_dict(
        checkpoint['model_state_dict']
    )

    trainer.graph_mtl.load_state_dict(
        checkpoint['graph_mtl_state_dict']
    )

    trainer.model.eval()
    trainer.graph_mtl.eval()

    return trainer


@torch.no_grad()
def extract_features(
    trainer,
    dataloader,
    device
):
    fc_feature_list = []
    emotion_feature_list = []
    identity_feature_list = []
    speaker_label_list = []
    emotion_label_list = []

    trainer.model.eval()
    trainer.graph_mtl.eval()

    for _, data in enumerate(dataloader):
        (
            padded_texts,
            padded_audios,
            padded_visuals,
            padded_speaker_masks,
            padded_utterance_masks,
            padded_labels
        ) = [
            d.to(device) for d in data
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
        ) = trainer.model(
            padded_texts,
            padded_audios,
            padded_visuals,
            padded_speaker_masks,
            padded_utterance_masks,
            padded_labels_flat
        )

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
            trainer.dataset,
            trainer.gnn_edge_mode
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
        ) = trainer.graph_mtl(
            padded_fc_outputs,
            adj,
            valid_mask_2d
        )

        fc_feature_list.append(fc_outputs.detach().cpu())
        emotion_feature_list.append(emotion_feature.detach().cpu())
        identity_feature_list.append(identity_feature.detach().cpu())
        speaker_label_list.append(speaker_labels.detach().cpu())
        emotion_label_list.append(labels.detach().cpu())

    return {
        'fc_features': torch.cat(fc_feature_list, dim=0),
        'emotion_features': torch.cat(emotion_feature_list, dim=0),
        'identity_features': torch.cat(identity_feature_list, dim=0),
        'speaker_labels': torch.cat(speaker_label_list, dim=0),
        'emotion_labels': torch.cat(emotion_label_list, dim=0)
    }


def run_probe_experiment(
    name,
    train_features,
    test_features,
    train_speaker_labels,
    test_speaker_labels,
    train_emotion_labels,
    test_emotion_labels,
    num_speakers,
    num_emotions,
    device,
    args
):
    speaker_acc, speaker_f1 = train_probe(
        train_features,
        train_speaker_labels,
        test_features,
        test_speaker_labels,
        num_speakers,
        device,
        args.probe_epochs,
        args.probe_lr,
        args.probe_batch_size
    )

    emotion_acc, emotion_f1 = train_probe(
        train_features,
        train_emotion_labels,
        test_features,
        test_emotion_labels,
        num_emotions,
        device,
        args.probe_epochs,
        args.probe_lr,
        args.probe_batch_size
    )

    return {
        'name': name,
        'speaker_acc': speaker_acc,
        'speaker_macro_f1': speaker_f1,
        'emotion_acc': emotion_acc,
        'emotion_macro_f1': emotion_f1
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True
    )

    parser.add_argument(
        '--data_batch_size',
        type=int,
        default=64
    )

    parser.add_argument(
        '--probe_epochs',
        type=int,
        default=100
    )

    parser.add_argument(
        '--probe_lr',
        type=float,
        default=1e-3
    )

    parser.add_argument(
        '--probe_batch_size',
        type=int,
        default=512
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=2023
    )

    parser.add_argument(
        '--output',
        type=str,
        default=''
    )

    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )

    checkpoint = torch.load(
        args.checkpoint,
        map_location=device
    )

    dataset = checkpoint['dataset']

    trainer = build_trainer_from_checkpoint(
        checkpoint,
        device,
        args.data_batch_size
    )

    print('==========================================')
    print('Loaded checkpoint:', args.checkpoint)
    print('Dataset:', dataset)
    print('Best test f1:', checkpoint.get('best_test_f1'))
    print('Best epoch:', checkpoint.get('best_epoch'))
    print('==========================================')

    print('Extracting train features...')
    train_data = extract_features(
        trainer,
        trainer.train_dataloader,
        device
    )

    print('Extracting test features...')
    test_data = extract_features(
        trainer,
        trainer.test_dataloader,
        device
    )

    num_speakers = trainer.n_speakers
    num_emotions = trainer.num_classes

    speaker_majority_acc = majority_baseline(
        test_data['speaker_labels'],
        num_speakers
    )

    emotion_majority_acc = majority_baseline(
        test_data['emotion_labels'],
        num_emotions
    )

    results = []

    for feature_name in [
        'fc_features',
        'emotion_features',
        'identity_features'
    ]:
        result = run_probe_experiment(
            feature_name,
            train_data[feature_name],
            test_data[feature_name],
            train_data['speaker_labels'],
            test_data['speaker_labels'],
            train_data['emotion_labels'],
            test_data['emotion_labels'],
            num_speakers,
            num_emotions,
            device,
            args
        )

        results.append(result)

    lines = []

    lines.append('==========================================')
    lines.append('Identity Leakage Probe Results')
    lines.append('==========================================')
    lines.append('Checkpoint: {}'.format(args.checkpoint))
    lines.append('Dataset: {}'.format(dataset))
    lines.append('Checkpoint best test f1: {}'.format(checkpoint.get('best_test_f1')))
    lines.append('Checkpoint best epoch: {}'.format(checkpoint.get('best_epoch')))
    lines.append('')
    lines.append('Speaker majority baseline acc: {:.4f}'.format(speaker_majority_acc))
    lines.append('Emotion majority baseline acc: {:.4f}'.format(emotion_majority_acc))
    lines.append('')
    lines.append('Feature\tSpeakerAcc\tSpeakerMacroF1\tEmotionAcc\tEmotionMacroF1')

    for r in results:
        lines.append(
            '{}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}'.format(
                r['name'],
                r['speaker_acc'],
                r['speaker_macro_f1'],
                r['emotion_acc'],
                r['emotion_macro_f1']
            )
        )

    lines.append('')
    lines.append('Interpretation:')
    lines.append('1. If emotion_features SpeakerAcc is much lower than fc_features SpeakerAcc, identity leakage is reduced.')
    lines.append('2. If identity_features SpeakerAcc is high, identity branch successfully captures speaker information.')
    lines.append('3. Good disentanglement means: emotion_features low SpeakerAcc + identity_features high SpeakerAcc + emotion_features keep reasonable EmotionAcc.')

    output_text = '\n'.join(lines)
    print(output_text)

    if args.output == '':
        os.makedirs(
            'Log/Summary',
            exist_ok=True
        )

        args.output = os.path.join(
            'Log',
            'Summary',
            '{}_identity_leakage_probe.txt'.format(dataset)
        )

    with open(args.output, 'w', encoding='utf-8') as f:
        f.write(output_text)

    print('Saved probe result to {}'.format(args.output))


if __name__ == '__main__':
    main()
