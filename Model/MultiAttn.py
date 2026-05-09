import torch
import torch.nn as nn
import torch.nn.functional as F


"""
[MODIFIED]
Masked MultiAttn.

Original problem:
    The previous MultiAttn computes attention over all sequence positions,
    including padded utterances.

Modification:
    Add attention_mask to every cross-attention layer.

attention_mask:
    shape: [batch_size, seq_len]
    value: 1 means valid utterance
    value: 0 means padding utterance
"""


class BidirectionalCrossAttention(nn.Module):
    def __init__(self, model_dim, Q_dim, K_dim, V_dim):
        super().__init__()

        self.query_matrix = nn.Linear(model_dim, Q_dim)
        self.key_matrix = nn.Linear(model_dim, K_dim)
        self.value_matrix = nn.Linear(model_dim, V_dim)

    def bidirectional_scaled_dot_product_attention(
        self,
        Q,
        K,
        V,
        attention_mask=None
    ):
        """
        Q: [batch_size, seq_len, Q_dim]
        K: [batch_size, seq_len, K_dim]
        V: [batch_size, seq_len, V_dim]

        attention_mask:
            [batch_size, seq_len]
            1 = valid utterance
            0 = padding utterance
        """

        score = torch.bmm(Q, K.transpose(-1, -2))
        scaled_score = score / (K.shape[-1] ** 0.5)

        # ============================================================
        # [MODIFIED-1]
        # Mask padded key/value positions before softmax.
        #
        # scaled_score shape:
        #     [batch_size, query_len, key_len]
        #
        # attention_mask shape:
        #     [batch_size, key_len]
        #
        # invalid positions are filled with a very negative value,
        # so softmax gives them probability close to 0.
        # ============================================================
        if attention_mask is not None:
            key_padding_mask = ~attention_mask.bool()
            scaled_score = scaled_score.masked_fill(
                key_padding_mask.unsqueeze(1),
                -1e9
            )

        attention_weights = F.softmax(scaled_score, dim=-1)

        attention = torch.bmm(attention_weights, V)

        # ============================================================
        # [MODIFIED-2]
        # Also zero out padded query positions.
        # This prevents padded utterances from producing non-zero outputs.
        # ============================================================
        if attention_mask is not None:
            attention = attention * attention_mask.unsqueeze(-1).float()

        return attention

    def forward(self, query, key, value, attention_mask=None):
        Q = self.query_matrix(query)
        K = self.key_matrix(key)
        V = self.value_matrix(value)

        attention = self.bidirectional_scaled_dot_product_attention(
            Q,
            K,
            V,
            attention_mask=attention_mask
        )

        return attention


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, model_dim, Q_dim, K_dim, V_dim):
        super().__init__()

        self.num_heads = num_heads

        self.attention_heads = nn.ModuleList(
            [
                BidirectionalCrossAttention(
                    model_dim,
                    Q_dim,
                    K_dim,
                    V_dim
                )
                for _ in range(self.num_heads)
            ]
        )

        self.projection_matrix = nn.Linear(num_heads * V_dim, model_dim)

    def forward(self, query, key, value, attention_mask=None):
        heads = [
            self.attention_heads[i](
                query,
                key,
                value,
                attention_mask=attention_mask
            )
            for i in range(self.num_heads)
        ]

        multihead_attention = self.projection_matrix(
            torch.cat(heads, dim=-1)
        )

        if attention_mask is not None:
            multihead_attention = (
                multihead_attention
                * attention_mask.unsqueeze(-1).float()
            )

        return multihead_attention


class Feedforward(nn.Module):
    def __init__(self, model_dim, hidden_dim, dropout_rate):
        super().__init__()

        self.linear_W1 = nn.Linear(model_dim, hidden_dim)
        self.linear_W2 = nn.Linear(hidden_dim, model_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        return self.dropout(
            self.linear_W2(
                self.relu(
                    self.linear_W1(x)
                )
            )
        )


class AddNorm(nn.Module):
    def __init__(self, model_dim, dropout_rate):
        super().__init__()

        self.layer_norm = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, sublayer_output, attention_mask=None):
        """
        x:              [batch_size, seq_len, model_dim]
        sublayer_output:[batch_size, seq_len, model_dim]
        attention_mask: [batch_size, seq_len]
        """

        output = self.layer_norm(
            x + self.dropout(sublayer_output)
        )

        # ============================================================
        # [MODIFIED-3]
        # Keep padded positions zero after residual + layer norm.
        # ============================================================
        if attention_mask is not None:
            output = output * attention_mask.unsqueeze(-1).float()

        return output


class MultiAttnLayer(nn.Module):
    def __init__(self, num_heads, model_dim, hidden_dim, dropout_rate):
        super().__init__()

        Q_dim = K_dim = V_dim = model_dim // num_heads

        self.attn_1 = MultiHeadAttention(
            num_heads,
            model_dim,
            Q_dim,
            K_dim,
            V_dim
        )
        self.add_norm_1 = AddNorm(model_dim, dropout_rate)

        self.attn_2 = MultiHeadAttention(
            num_heads,
            model_dim,
            Q_dim,
            K_dim,
            V_dim
        )
        self.add_norm_2 = AddNorm(model_dim, dropout_rate)

        self.ff = Feedforward(model_dim, hidden_dim, dropout_rate)
        self.add_norm_3 = AddNorm(model_dim, dropout_rate)

    def forward(
        self,
        query_modality,
        modality_A,
        modality_B,
        attention_mask=None
    ):
        """
        query_modality: [batch_size, seq_len, model_dim]
        modality_A:     [batch_size, seq_len, model_dim]
        modality_B:     [batch_size, seq_len, model_dim]
        attention_mask:  [batch_size, seq_len]
        """

        # query_modality attends to modality_A
        attn_1_output = self.attn_1(
            query_modality,
            modality_A,
            modality_A,
            attention_mask=attention_mask
        )

        attn_output_1 = self.add_norm_1(
            query_modality,
            attn_1_output,
            attention_mask=attention_mask
        )

        # previous output attends to modality_B
        attn_2_output = self.attn_2(
            attn_output_1,
            modality_B,
            modality_B,
            attention_mask=attention_mask
        )

        attn_output_2 = self.add_norm_2(
            attn_output_1,
            attn_2_output,
            attention_mask=attention_mask
        )

        # feed-forward
        ff_output = self.ff(attn_output_2)

        output = self.add_norm_3(
            attn_output_2,
            ff_output,
            attention_mask=attention_mask
        )

        return output


class MultiAttn(nn.Module):
    def __init__(
        self,
        num_layers,
        model_dim,
        num_heads,
        hidden_dim,
        dropout_rate
    ):
        super().__init__()

        self.multiattn_layers = nn.ModuleList(
            [
                MultiAttnLayer(
                    num_heads,
                    model_dim,
                    hidden_dim,
                    dropout_rate
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        query_modality,
        modality_A,
        modality_B,
        attention_mask=None
    ):
        for multiattn_layer in self.multiattn_layers:
            query_modality = multiattn_layer(
                query_modality,
                modality_A,
                modality_B,
                attention_mask=attention_mask
            )

        return query_modality


class MultiAttnModel(nn.Module):
    def __init__(
        self,
        num_layers,
        model_dim,
        num_heads,
        hidden_dim,
        dropout_rate
    ):
        super().__init__()

        self.multiattn_text = MultiAttn(
            num_layers,
            model_dim,
            num_heads,
            hidden_dim,
            dropout_rate
        )

        self.multiattn_audio = MultiAttn(
            num_layers,
            model_dim,
            num_heads,
            hidden_dim,
            dropout_rate
        )

        self.multiattn_visual = MultiAttn(
            num_layers,
            model_dim,
            num_heads,
            hidden_dim,
            dropout_rate
        )

    def forward(
        self,
        text_features,
        audio_features,
        visual_features,
        attention_mask=None
    ):
        """
        text_features:   [batch_size, seq_len, model_dim]
        audio_features:  [batch_size, seq_len, model_dim]
        visual_features: [batch_size, seq_len, model_dim]
        attention_mask:  [batch_size, seq_len]
        """

        f_t = self.multiattn_text(
            text_features,
            audio_features,
            visual_features,
            attention_mask=attention_mask
        )

        f_a = self.multiattn_audio(
            audio_features,
            text_features,
            visual_features,
            attention_mask=attention_mask
        )

        f_v = self.multiattn_visual(
            visual_features,
            text_features,
            audio_features,
            attention_mask=attention_mask
        )

        return f_t, f_a, f_v