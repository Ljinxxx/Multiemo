import torch
import torch.nn as nn
import torch.nn.functional as F


class ParallelLineGraphLogitHead(nn.Module):
    """
    Parallel Line Graph Logit Head (Graph-v1b/v1c).

    A parallel branch that produces graph_logits_delta based on local
    neighbor context (prev/next utterance). The final classification is:
        mlp_outputs = base_logits + sigmoid(gate) * [uncertainty_gate *] graph_logits_delta

    Key design choices:
    - Does NOT replace or modify fc_outputs (SWFC loss stays clean).
    - graph_classifier is zero-initialized so initial output is exactly 0.
    - gate_init=-5.0 -> sigmoid ~ 0.0067, ensuring near-zero contribution at init.
    - Uses (neighbor_mean - h_i) as relational signal, not self-smoothing.
    - No LayerNorm.
    """

    def __init__(
        self,
        model_dim,
        num_classes,
        hidden_dim=None,
        dropout=0.1,
        gate_init=-5.0
    ):
        super().__init__()

        if hidden_dim is None:
            hidden_dim = model_dim

        self.model_dim = model_dim
        self.num_classes = num_classes

        # Input: [h_i, neighbor_mean - h_i] -> 2 * model_dim
        self.message_mlp = nn.Sequential(
            nn.Linear(model_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, model_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Zero-initialized classifier ensures initial graph_logits_delta = 0
        self.graph_classifier = nn.Linear(model_dim, num_classes)
        nn.init.zeros_(self.graph_classifier.weight)
        nn.init.zeros_(self.graph_classifier.bias)

        # Scalar gate, initialized very small
        self.gate_logit = nn.Parameter(torch.tensor(float(gate_init)))

    def forward(self, h_seq, valid_mask):
        """
        Args:
            h_seq: [B, T, D] - fc_outputs at sequence level
            valid_mask: [B, T] - bool or 0/1, True for valid utterances

        Returns:
            graph_logits_delta: [B, T, num_classes]
            graph_gate: scalar (sigmoid of gate_logit)
        """
        # 1. Ensure valid_mask is bool
        valid_mask = valid_mask.bool()

        # 2. Zero out padding positions
        h_seq = h_seq * valid_mask.unsqueeze(-1).float()

        # 3. Construct shifted features
        prev_h = torch.zeros_like(h_seq)
        next_h = torch.zeros_like(h_seq)
        prev_h[:, 1:, :] = h_seq[:, :-1, :]
        next_h[:, :-1, :] = h_seq[:, 1:, :]

        # 4. Construct shifted masks
        prev_mask = torch.zeros_like(valid_mask)
        next_mask = torch.zeros_like(valid_mask)
        prev_mask[:, 1:] = valid_mask[:, :-1]
        next_mask[:, :-1] = valid_mask[:, 1:]

        # 5. Compute neighbor_mean (only prev and next, NOT self)
        neighbor_sum = (
            prev_h * prev_mask.unsqueeze(-1).float()
            + next_h * next_mask.unsqueeze(-1).float()
        )
        neighbor_count = (
            prev_mask.float() + next_mask.float()
        ).clamp(min=1.0)
        neighbor_mean = neighbor_sum / neighbor_count.unsqueeze(-1)

        # 6. For nodes with no valid neighbors, neighbor_mean is zero
        #    (already handled by clamp and zero init of prev_h/next_h)

        # 7. Construct delta input: [h_i, neighbor_mean - h_i]
        delta_input = torch.cat(
            [h_seq, neighbor_mean - h_seq],
            dim=-1
        )

        # 8. Message MLP
        graph_hidden = self.message_mlp(delta_input)

        # 9. Graph classifier (zero-initialized)
        graph_logits_delta = self.graph_classifier(graph_hidden)

        # 10. Zero out padding positions and no-neighbor positions
        has_neighbor_mask = prev_mask | next_mask

        graph_logits_delta = graph_logits_delta * valid_mask.unsqueeze(-1).float()
        graph_logits_delta = graph_logits_delta * has_neighbor_mask.unsqueeze(-1).float()

        # 11. Return delta logits and gate value
        return graph_logits_delta, torch.sigmoid(self.gate_logit)
