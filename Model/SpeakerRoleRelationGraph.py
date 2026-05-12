import torch
import torch.nn as nn
import torch.nn.functional as F


class GradReverse(torch.autograd.Function):
    """Gradient Reversal Layer."""

    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


def grad_reverse(x, lambd):
    return GradReverse.apply(x, lambd)


class SpeakerRoleRelationGraphDiscriminator(nn.Module):
    """
    Speaker-Role Relation Graph Discriminator.

    Uses a local temporal graph encoder (self/prev/next) on GRL-reversed
    fc_outputs_seq, then samples balanced same/different speaker-role pairs
    within each dialogue for binary classification.

    This module does NOT modify emotion logits. It only provides an
    adversarial signal to encourage the backbone to disentangle
    speaker-role identity from emotion representations.
    """

    def __init__(
        self,
        model_dim,
        hidden_dim=None,
        dropout=0.1,
        max_pairs_per_dialogue=32
    ):
        super().__init__()

        if hidden_dim is None:
            hidden_dim = model_dim

        self.model_dim = model_dim
        self.max_pairs_per_dialogue = max_pairs_per_dialogue

        # Local temporal relation graph encoder
        # Input: [h_i, neighbor_mean - h_i] -> 2 * model_dim
        self.node_mlp = nn.Sequential(
            nn.Linear(model_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, model_dim),
            nn.ReLU()
        )

        # Pair classifier
        # Input: [z_i, z_j, abs(z_i - z_j), z_i * z_j] -> 4 * model_dim
        self.pair_classifier = nn.Sequential(
            nn.Linear(model_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, h_seq, valid_mask, speaker_masks, grl_lambda=0.0):
        """
        Args:
            h_seq: [B, T, D] - fc_outputs_seq
            valid_mask: [B, T] - bool or 0/1
            speaker_masks: [B, T, S] - one-hot speaker slot
            grl_lambda: float - gradient reversal strength

        Returns:
            relation_logits: [P] - predicted same/diff logits
            relation_labels: [P] - ground truth (1=same, 0=diff)
        """
        # 1. Ensure valid_mask is bool
        valid_mask = valid_mask.bool()

        # 2. Apply Gradient Reversal Layer
        h_seq = grad_reverse(h_seq, grl_lambda)

        # 3. Zero out padding
        h_seq = h_seq * valid_mask.unsqueeze(-1).float()

        # 4. Local temporal graph encoder
        prev_h = torch.zeros_like(h_seq)
        next_h = torch.zeros_like(h_seq)
        prev_h[:, 1:, :] = h_seq[:, :-1, :]
        next_h[:, :-1, :] = h_seq[:, 1:, :]

        prev_mask = torch.zeros_like(valid_mask)
        next_mask = torch.zeros_like(valid_mask)
        prev_mask[:, 1:] = valid_mask[:, :-1]
        next_mask[:, :-1] = valid_mask[:, 1:]

        neighbor_sum = (
            prev_h * prev_mask.unsqueeze(-1).float()
            + next_h * next_mask.unsqueeze(-1).float()
        )
        neighbor_count = (
            prev_mask.float() + next_mask.float()
        ).clamp(min=1.0)
        neighbor_mean = neighbor_sum / neighbor_count.unsqueeze(-1)

        node_input = torch.cat([h_seq, neighbor_mean - h_seq], dim=-1)
        z_seq = self.node_mlp(node_input)
        z_seq = z_seq * valid_mask.unsqueeze(-1).float()

        # speaker_masks should be [B, T, S].
        # If it is [T, B, S], transpose it to [B, T, S].
        if speaker_masks.dim() != 3:
            raise ValueError(f"speaker_masks must be 3D, got shape {speaker_masks.shape}")

        if speaker_masks.shape[0] != valid_mask.shape[0] or speaker_masks.shape[1] != valid_mask.shape[1]:
            if speaker_masks.shape[0] == valid_mask.shape[1] and speaker_masks.shape[1] == valid_mask.shape[0]:
                speaker_masks = speaker_masks.transpose(0, 1)
            else:
                raise ValueError(
                    f"speaker_masks shape {speaker_masks.shape} does not match valid_mask shape {valid_mask.shape}"
                )

        # 5. Speaker slot
        speaker_slots = torch.argmax(speaker_masks, dim=-1)  # [B, T]

        # 6. Sample balanced pairs within each dialogue
        B = h_seq.shape[0]
        all_b_indices = []
        all_i_indices = []
        all_j_indices = []
        all_labels = []

        for b in range(B):
            valid_positions = torch.nonzero(valid_mask[b], as_tuple=False).view(-1)
            n_valid = valid_positions.shape[0]

            if n_valid < 2:
                continue

            # Get speaker slots for valid positions
            slots = speaker_slots[b][valid_positions]  # [n_valid]

            # Build all i < j pairs
            # Use indices into valid_positions
            idx_i = []
            idx_j = []
            same_pairs = []
            diff_pairs = []

            for ii in range(n_valid):
                for jj in range(ii + 1, n_valid):
                    pos_i = valid_positions[ii].item()
                    pos_j = valid_positions[jj].item()
                    if slots[ii].item() == slots[jj].item():
                        same_pairs.append((pos_i, pos_j))
                    else:
                        diff_pairs.append((pos_i, pos_j))

            if len(same_pairs) == 0 or len(diff_pairs) == 0:
                continue

            # Balanced sampling
            num_each = min(
                len(same_pairs),
                len(diff_pairs),
                self.max_pairs_per_dialogue // 2
            )

            # Random permutation for sampling
            same_perm = torch.randperm(
                len(same_pairs), device=h_seq.device
            )[:num_each]
            diff_perm = torch.randperm(
                len(diff_pairs), device=h_seq.device
            )[:num_each]

            for idx in same_perm:
                pi, pj = same_pairs[idx.item()]
                all_b_indices.append(b)
                all_i_indices.append(pi)
                all_j_indices.append(pj)
                all_labels.append(1.0)

            for idx in diff_perm:
                pi, pj = diff_pairs[idx.item()]
                all_b_indices.append(b)
                all_i_indices.append(pi)
                all_j_indices.append(pj)
                all_labels.append(0.0)

        # 7. If no pairs found
        if len(all_labels) == 0:
            return h_seq.new_zeros((0,)), h_seq.new_zeros((0,))

        # 8. Construct pair features and classify
        b_indices = torch.tensor(all_b_indices, device=h_seq.device, dtype=torch.long)
        i_indices = torch.tensor(all_i_indices, device=h_seq.device, dtype=torch.long)
        j_indices = torch.tensor(all_j_indices, device=h_seq.device, dtype=torch.long)
        labels = torch.tensor(all_labels, device=h_seq.device, dtype=torch.float)

        zi = z_seq[b_indices, i_indices]  # [P, D]
        zj = z_seq[b_indices, j_indices]  # [P, D]

        pair_feat = torch.cat(
            [zi, zj, torch.abs(zi - zj), zi * zj],
            dim=-1
        )

        relation_logits = self.pair_classifier(pair_feat).squeeze(-1)  # [P]

        return relation_logits, labels
