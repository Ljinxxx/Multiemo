import torch


def get_speaker_ids_2d(padded_speaker_masks, padded_labels_2d):
    """
    Convert padded_speaker_masks into speaker id matrix [B, L].

    Compatible input shapes:
        [B, L, S]
        [L, B, S]
        [B, L]
        [L, B]
    """

    if len(padded_labels_2d.shape) != 2:
        raise ValueError("padded_labels should have shape [B, L]")

    B, L = padded_labels_2d.shape

    if padded_speaker_masks.dim() == 3:
        if (
            padded_speaker_masks.size(0) == B
            and padded_speaker_masks.size(1) == L
        ):
            speaker_ids = torch.argmax(
                padded_speaker_masks,
                dim=-1
            )

        elif (
            padded_speaker_masks.size(0) == L
            and padded_speaker_masks.size(1) == B
        ):
            speaker_ids = torch.argmax(
                padded_speaker_masks,
                dim=-1
            ).transpose(0, 1)

        else:
            raise ValueError(
                "padded_speaker_masks shape does not match padded_labels"
            )

    elif padded_speaker_masks.dim() == 2:
        if (
            padded_speaker_masks.size(0) == B
            and padded_speaker_masks.size(1) == L
        ):
            speaker_ids = padded_speaker_masks.long()

        elif (
            padded_speaker_masks.size(0) == L
            and padded_speaker_masks.size(1) == B
        ):
            speaker_ids = padded_speaker_masks.transpose(0, 1).long()

        else:
            raise ValueError(
                "padded_speaker_masks shape does not match padded_labels"
            )

    else:
        raise ValueError("Unsupported padded_speaker_masks shape")

    return speaker_ids.long()


def get_effective_gnn_edge_mode(dataset, gnn_edge_mode):
    """
    auto:
        IEMOCAP -> speaker_temporal
        MELD    -> temporal
    """

    if gnn_edge_mode == 'auto':
        if dataset == 'IEMOCAP':
            return 'speaker_temporal'

        if dataset == 'MELD':
            return 'temporal'

        raise ValueError("dataset must be either 'MELD' or 'IEMOCAP'")

    return gnn_edge_mode


def build_dialogue_graph_adj(
    speaker_ids_2d,
    padded_labels_2d,
    dataset,
    gnn_edge_mode
):
    """
    Memory-safe batched dialogue graph.

    Instead of building one huge graph:
        adj: [1, N, N]

    We build dialogue-level graphs:
        adj: [B, L, L]

    Edges:
        self-loop
        adjacent temporal edge
        optional same-speaker edge
    """

    B, L = padded_labels_2d.shape
    device = padded_labels_2d.device

    valid = padded_labels_2d != -1
    valid_pair = valid.unsqueeze(2) & valid.unsqueeze(1)

    same_speaker = speaker_ids_2d.unsqueeze(2).eq(
        speaker_ids_2d.unsqueeze(1)
    )

    speaker_edge = same_speaker & valid_pair

    position_ids = torch.arange(
        L,
        device=device
    )

    position_distance = torch.abs(
        position_ids.unsqueeze(0)
        - position_ids.unsqueeze(1)
    )

    temporal_edge = (
        position_distance <= 1
    ).unsqueeze(0).expand(B, L, L)

    temporal_edge = temporal_edge & valid_pair

    self_loop = torch.eye(
        L,
        device=device,
        dtype=torch.bool
    ).unsqueeze(0).expand(B, L, L)

    self_loop = self_loop & valid_pair

    edge_mode = get_effective_gnn_edge_mode(
        dataset,
        gnn_edge_mode
    )

    if edge_mode == 'speaker_temporal':
        adj = temporal_edge | speaker_edge | self_loop

    elif edge_mode == 'temporal':
        adj = temporal_edge | self_loop

    elif edge_mode == 'speaker':
        adj = speaker_edge | self_loop

    else:
        raise ValueError(
            "gnn_edge_mode must be one of: auto, speaker_temporal, temporal, speaker"
        )

    adj = adj.float()

    deg = adj.sum(
        dim=-1,
        keepdim=True
    ).clamp_min(1.0)

    adj = adj / deg

    return adj


def flatten_valid_speaker_labels(
    padded_speaker_masks,
    padded_labels_2d
):
    speaker_ids_2d = get_speaker_ids_2d(
        padded_speaker_masks,
        padded_labels_2d
    )

    valid_mask_flat = padded_labels_2d.reshape(-1) != -1

    speaker_labels = speaker_ids_2d.reshape(-1)[
        valid_mask_flat
    ]

    return speaker_ids_2d, speaker_labels
