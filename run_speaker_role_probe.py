"""
Post-hoc Pairwise Speaker-Role Leakage Probe.

Loads a trained MultiEMO checkpoint, freezes the model, extracts fc_outputs_seq,
constructs balanced same/different speaker-role pairs within each dialogue,
trains a fresh pairwise probe, and reports AUC / balanced accuracy / F1.

Usage:
    python run_speaker_role_probe.py \
        --checkpoint checkpoints/meld_baseline_seed2023.pt \
        --dataset MELD \
        --batch_size 100 \
        --probe_epochs 30 \
        --max_pairs_per_dialogue 64 \
        --output_json probe_meld_baseline_seed2023.json
"""

import os
import sys
import json
import random
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import (
    roc_auc_score,
    balanced_accuracy_score,
    accuracy_score,
    f1_score,
    classification_report
)

sys.path.append('Model')
sys.path.append('Dataset')

from MultiEMO_Model import MultiEMO
from IEMOCAPDataset import IEMOCAPDataset
from MELDDataset import MELDDataset


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_model_from_checkpoint(checkpoint, device):
    """Reconstruct MultiEMO from checkpoint metadata and load weights."""
    dataset = checkpoint['dataset']

    if dataset == 'IEMOCAP':
        num_classes = 6
        n_speakers = 2
    elif dataset == 'MELD':
        num_classes = 7
        n_speakers = 9
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    model_dim = checkpoint['model_dim']

    model = MultiEMO(
        dataset=dataset,
        multi_attn_flag=checkpoint['multi_attn_flag'],
        roberta_dim=768,
        hidden_dim=checkpoint['hidden_dim'],
        dropout=checkpoint['dropout_rate'],
        num_layers=checkpoint['num_layers'],
        model_dim=model_dim,
        num_heads=checkpoint['num_heads'],
        D_m_audio=512,
        D_m_visual=1000,
        D_g=model_dim,
        D_p=model_dim,
        D_e=model_dim,
        D_h=model_dim,
        n_classes=num_classes,
        n_speakers=n_speakers,
        listener_state=False,
        context_attention='simple',
        D_a=model_dim,
        dropout_rec=checkpoint['dropout_rec'],
        device=device,
        use_line_graph=checkpoint.get('use_line_graph', False),
        line_graph_dropout=checkpoint.get('line_graph_dropout', 0.1),
        line_graph_gate_init=checkpoint.get('line_graph_gate_init', -5.0),
        line_graph_use_vector_gate=checkpoint.get('line_graph_use_vector_gate', False),
        line_graph_use_confidence_gate=checkpoint.get('line_graph_use_confidence_gate', True),
        line_graph_uncertainty_gamma=checkpoint.get('line_graph_uncertainty_gamma', 1.0),
        use_speaker_role_adv=checkpoint.get('use_speaker_role_adv', False),
        speaker_adv_dropout=checkpoint.get('speaker_adv_dropout', 0.1),
        speaker_adv_hidden_dim=checkpoint.get('speaker_adv_hidden_dim', None),
        speaker_adv_max_pairs_per_dialogue=checkpoint.get('speaker_adv_max_pairs_per_dialogue', 32),
    )

    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    model.to(device)
    model.eval()

    return model


def get_dataloader(dataset, train, batch_size):
    """Create a DataLoader for the specified dataset split."""
    if dataset == 'IEMOCAP':
        ds = IEMOCAPDataset(train=train)
    elif dataset == 'MELD':
        ds = MELDDataset(train=train)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    return DataLoader(
        dataset=ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=ds.collate_fn,
        num_workers=0
    )


def extract_dialogue_features(model, dataloader, device):
    """
    Extract fc_outputs_seq for each dialogue.

    Returns:
        features_list: list of tensors, each [n_valid, D]
        speaker_slots_list: list of tensors, each [n_valid]
    """
    features_list = []
    speaker_slots_list = []

    with torch.no_grad():
        for data in dataloader:
            (
                padded_texts,
                padded_audios,
                padded_visuals,
                padded_speaker_masks,
                padded_utterance_masks,
                padded_labels
            ) = [d.to(device) for d in data]

            flat_padded_labels = padded_labels.reshape(-1)

            outputs = model(
                padded_texts,
                padded_audios,
                padded_visuals,
                padded_speaker_masks,
                padded_utterance_masks,
                flat_padded_labels,
                compute_speaker_relation=False,
                speaker_grl_lambda=0.0,
                return_fc_outputs_seq=True
            )

            # Last element is fc_outputs_seq [B, T, D]
            fc_outputs_seq = outputs[-1]

            # valid_mask from utterance_masks [B, T]
            valid_mask = padded_utterance_masks.bool()

            # speaker_masks: could be [T, B, S] from collate_fn, need [B, T, S]
            sm = padded_speaker_masks
            if sm.shape[0] != valid_mask.shape[0] or sm.shape[1] != valid_mask.shape[1]:
                if sm.shape[0] == valid_mask.shape[1] and sm.shape[1] == valid_mask.shape[0]:
                    sm = sm.transpose(0, 1)
                else:
                    raise ValueError(
                        f"speaker_masks shape {sm.shape} incompatible with valid_mask {valid_mask.shape}"
                    )

            speaker_slots = torch.argmax(sm, dim=-1)  # [B, T]

            B = fc_outputs_seq.shape[0]
            for b in range(B):
                valid_positions = torch.nonzero(valid_mask[b], as_tuple=False).view(-1)
                if valid_positions.numel() == 0:
                    continue
                features_list.append(
                    fc_outputs_seq[b, valid_positions].detach().cpu()
                )
                speaker_slots_list.append(
                    speaker_slots[b, valid_positions].detach().cpu()
                )

    return features_list, speaker_slots_list


def build_pair_dataset(features_list, speaker_slots_list, max_pairs_per_dialogue=64, seed=2023):
    """
    Build balanced pairwise same/different speaker-role dataset.

    Returns:
        X: [P, 4*D]
        y: [P]
    """
    rng = random.Random(seed)

    all_pair_feats = []
    all_labels = []

    for feats, slots in zip(features_list, speaker_slots_list):
        n = feats.shape[0]
        if n < 2:
            continue

        same_pairs = []
        diff_pairs = []

        for i in range(n):
            for j in range(i + 1, n):
                if slots[i].item() == slots[j].item():
                    same_pairs.append((i, j))
                else:
                    diff_pairs.append((i, j))

        if len(same_pairs) == 0 or len(diff_pairs) == 0:
            continue

        num_each = min(len(same_pairs), len(diff_pairs), max_pairs_per_dialogue // 2)

        sampled_same = rng.sample(same_pairs, num_each)
        sampled_diff = rng.sample(diff_pairs, num_each)

        for (i, j) in sampled_same:
            hi, hj = feats[i], feats[j]
            pair_feat = torch.cat([hi, hj, torch.abs(hi - hj), hi * hj], dim=-1)
            all_pair_feats.append(pair_feat)
            all_labels.append(1.0)

        for (i, j) in sampled_diff:
            hi, hj = feats[i], feats[j]
            pair_feat = torch.cat([hi, hj, torch.abs(hi - hj), hi * hj], dim=-1)
            all_pair_feats.append(pair_feat)
            all_labels.append(0.0)

    X = torch.stack(all_pair_feats, dim=0)
    y = torch.tensor(all_labels, dtype=torch.float)

    return X, y


class PairwiseSpeakerProbe(nn.Module):
    """Fresh pairwise speaker-role probe (not connected to main model)."""

    def __init__(self, input_dim, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_probe(X_train, y_train, X_test, y_test, probe_epochs=30, device='cpu'):
    """Train a fresh probe and return test predictions."""
    input_dim = X_train.shape[1]
    probe = PairwiseSpeakerProbe(input_dim, hidden_dim=256, dropout=0.1).to(device)

    optimizer = torch.optim.Adam(probe.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss()

    train_dataset = TensorDataset(X_train.to(device), y_train.to(device))
    batch_size = 512

    for epoch in range(probe_epochs):
        probe.train()
        indices = torch.randperm(len(train_dataset))

        total_loss = 0.0
        num_batches = 0

        for start in range(0, len(indices), batch_size):
            batch_idx = indices[start:start + batch_size]
            batch_x = X_train[batch_idx].to(device)
            batch_y = y_train[batch_idx].to(device)

            optimizer.zero_grad()
            logits = probe(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

    # Evaluate on test
    probe.eval()
    with torch.no_grad():
        test_logits = probe(X_test.to(device))
        test_probs = torch.sigmoid(test_logits).cpu().numpy()

    return test_probs


def evaluate_probe(y_true, probs):
    """Compute probe metrics."""
    preds = (probs >= 0.5).astype(float)

    metrics = {}

    try:
        metrics['auc'] = round(float(roc_auc_score(y_true, probs)), 4)
    except ValueError:
        metrics['auc'] = 0.0

    metrics['acc'] = round(float(accuracy_score(y_true, preds)) * 100, 4)
    metrics['balanced_acc'] = round(float(balanced_accuracy_score(y_true, preds)) * 100, 4)
    metrics['f1'] = round(float(f1_score(y_true, preds)) * 100, 4)
    metrics['report'] = classification_report(y_true, preds, digits=4)

    return metrics


def main():
    parser = argparse.ArgumentParser(description='Post-hoc Pairwise Speaker-Role Leakage Probe')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--dataset', type=str, default='', help='Dataset override (MELD or IEMOCAP)')
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size for feature extraction')
    parser.add_argument('--probe_epochs', type=int, default=30, help='Number of probe training epochs')
    parser.add_argument('--max_pairs_per_dialogue', type=int, default=64, help='Max pairs per dialogue')
    parser.add_argument('--output_json', type=str, default='', help='Path to save probe results JSON')
    parser.add_argument('--seed', type=int, default=2023, help='Random seed')
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # Load checkpoint
    print(f'Loading checkpoint: {args.checkpoint}')
    checkpoint = torch.load(args.checkpoint, map_location=device)

    dataset = args.dataset if args.dataset else checkpoint['dataset']
    print(f'Dataset: {dataset}')
    print(f'Best epoch: {checkpoint.get("best_epoch", "N/A")}')
    print(f'Best test weighted F1: {checkpoint.get("best_test_weighted_f1", "N/A")}')
    print(f'use_speaker_role_adv: {checkpoint.get("use_speaker_role_adv", False)}')
    print(f'speaker_adv_lambda: {checkpoint.get("speaker_adv_lambda", 0.0)}')

    # Build and load model
    model = build_model_from_checkpoint(checkpoint, device)
    print('Model loaded and frozen (eval mode).')

    # Extract features from train and test splits
    print('Extracting train features...')
    train_loader = get_dataloader(dataset, train=True, batch_size=args.batch_size)
    train_features, train_slots = extract_dialogue_features(model, train_loader, device)
    print(f'  Train dialogues: {len(train_features)}')

    print('Extracting test features...')
    test_loader = get_dataloader(dataset, train=False, batch_size=args.batch_size)
    test_features, test_slots = extract_dialogue_features(model, test_loader, device)
    print(f'  Test dialogues: {len(test_features)}')

    # Build pair datasets
    print('Building train pair dataset...')
    X_train, y_train = build_pair_dataset(
        train_features, train_slots,
        max_pairs_per_dialogue=args.max_pairs_per_dialogue,
        seed=args.seed
    )
    train_pos = int((y_train == 1).sum().item())
    train_neg = int((y_train == 0).sum().item())
    print(f'  Train pairs: {len(y_train)}, pos: {train_pos}, neg: {train_neg}')

    print('Building test pair dataset...')
    X_test, y_test = build_pair_dataset(
        test_features, test_slots,
        max_pairs_per_dialogue=args.max_pairs_per_dialogue,
        seed=args.seed + 1
    )
    test_pos = int((y_test == 1).sum().item())
    test_neg = int((y_test == 0).sum().item())
    print(f'  Test pairs: {len(y_test)}, pos: {test_pos}, neg: {test_neg}')

    # Train probe
    print(f'Training probe for {args.probe_epochs} epochs...')
    test_probs = train_probe(
        X_train, y_train, X_test, y_test,
        probe_epochs=args.probe_epochs,
        device=device
    )

    # Evaluate
    y_test_np = y_test.numpy()
    metrics = evaluate_probe(y_test_np, test_probs)

    print('\n' + '=' * 60)
    print('PROBE RESULTS')
    print('=' * 60)
    print(f'  AUC:          {metrics["auc"]}')
    print(f'  Accuracy:     {metrics["acc"]}%')
    print(f'  Balanced Acc: {metrics["balanced_acc"]}%')
    print(f'  F1:           {metrics["f1"]}%')
    print(f'\n{metrics["report"]}')

    # Save JSON
    if args.output_json:
        output_dir = os.path.dirname(args.output_json)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        result = {
            'checkpoint': args.checkpoint,
            'dataset': dataset,
            'use_speaker_role_adv': checkpoint.get('use_speaker_role_adv', False),
            'speaker_adv_lambda': checkpoint.get('speaker_adv_lambda', 0.0),
            'best_test_weighted_f1': checkpoint.get('best_test_weighted_f1', None),
            'best_test_macro_f1': checkpoint.get('best_test_macro_f1', None),
            'best_test_acc': checkpoint.get('best_test_acc', None),
            'train_pairs': len(y_train),
            'train_pos': train_pos,
            'train_neg': train_neg,
            'test_pairs': len(y_test),
            'test_pos': test_pos,
            'test_neg': test_neg,
            'probe_auc': metrics['auc'],
            'probe_acc': metrics['acc'],
            'probe_balanced_acc': metrics['balanced_acc'],
            'probe_f1': metrics['f1'],
            'probe_report': metrics['report'],
        }

        with open(args.output_json, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f'\nResults saved to {args.output_json}')


if __name__ == '__main__':
    main()
