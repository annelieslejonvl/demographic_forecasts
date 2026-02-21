"""
Neural network architectures for sequence-based demographic event prediction.

Three encoder variants (LSTM, GRU, Transformer) with shared embedding
and prediction head. All produce per-event, per-horizon logits.
"""
import math
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    """
    Shared embedding layer: token embedding + positional embedding.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        max_seq_len: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(
            vocab_size, embed_dim, padding_idx=0
        )
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.embed_dim = embed_dim

    def forward(self, input_ids: torch.LongTensor) -> torch.Tensor:
        """
        Args:
            input_ids: (batch, seq_len) token IDs
        Returns:
            (batch, seq_len, embed_dim) embeddings
        """
        batch_size, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.layer_norm(x)
        x = self.dropout(x)
        return x


class LSTMEncoder(nn.Module):
    """Bidirectional LSTM encoder."""

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output_dim = hidden_dim * 2  # bidirectional

    def forward(
        self, embeddings: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            embeddings: (batch, seq_len, embed_dim)
            attention_mask: (batch, seq_len) with 1 for real tokens, 0 for padding
        Returns:
            (batch, hidden_dim * 2) — final hidden state
        """
        lengths = attention_mask.sum(dim=1).long().cpu().clamp(min=1)

        packed = nn.utils.rnn.pack_padded_sequence(
            embeddings, lengths, batch_first=True, enforce_sorted=False
        )
        _, (hidden, _) = self.lstm(packed)

        # hidden: (num_layers * 2, batch, hidden_dim)
        # Take last layer forward and backward
        forward_h = hidden[-2]  # (batch, hidden_dim)
        backward_h = hidden[-1]  # (batch, hidden_dim)
        return torch.cat([forward_h, backward_h], dim=1)


class GRUEncoder(nn.Module):
    """Bidirectional GRU encoder."""

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output_dim = hidden_dim * 2

    def forward(
        self, embeddings: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Returns: (batch, hidden_dim * 2) — final hidden state.
        """
        lengths = attention_mask.sum(dim=1).long().cpu().clamp(min=1)

        packed = nn.utils.rnn.pack_padded_sequence(
            embeddings, lengths, batch_first=True, enforce_sorted=False
        )
        _, hidden = self.gru(packed)

        forward_h = hidden[-2]
        backward_h = hidden[-1]
        return torch.cat([forward_h, backward_h], dim=1)


class TransformerSequenceEncoder(nn.Module):
    """
    Standard Transformer encoder with mean-pooling over non-padded positions.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        num_layers: int = 3,
        ff_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation='gelu',
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        self.output_dim = embed_dim

    def forward(
        self, embeddings: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            embeddings: (batch, seq_len, embed_dim)
            attention_mask: (batch, seq_len) — 1 for real, 0 for pad
        Returns:
            (batch, embed_dim) — mean-pooled representation
        """
        # TransformerEncoder expects src_key_padding_mask where True = ignore
        padding_mask = (attention_mask == 0)
        encoded = self.transformer(embeddings, src_key_padding_mask=padding_mask)

        # Mean-pool over non-padded positions
        mask_expanded = attention_mask.unsqueeze(-1)  # (batch, seq_len, 1)
        summed = (encoded * mask_expanded).sum(dim=1)  # (batch, embed_dim)
        counts = mask_expanded.sum(dim=1).clamp(min=1)  # (batch, 1)
        return summed / counts


class HorizonPredictionHead(nn.Module):
    """
    Multi-label classification head.

    Takes encoder output and produces logits for each event at each horizon.
    Output size = n_events * n_horizons.
    """

    def __init__(
        self,
        input_dim: int,
        n_events: int,
        n_horizons: int,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.2,
        output_dim: Optional[int] = None,
    ):
        super().__init__()
        hidden_dims = hidden_dims or [128, 64]
        output_dim = output_dim or (n_events * n_horizons)

        layers: List[nn.Module] = []
        prev_dim = input_dim
        for hdim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hdim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hdim
        layers.append(nn.Linear(prev_dim, output_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(self, encoder_output: torch.Tensor) -> torch.Tensor:
        """
        Args:
            encoder_output: (batch, encoder_dim)
        Returns:
            logits: (batch, n_events * n_horizons)
        """
        return self.mlp(encoder_output)


class MultiEventHead(nn.Module):
    """
    Separate prediction head per event, sharing the encoder backbone.

    Each event gets its own MLP, allowing different events to learn
    independent decision boundaries without competing in a shared head.
    Output layout matches HorizonPredictionHead for compatibility.
    """

    def __init__(
        self,
        input_dim: int,
        n_events: int,
        n_horizons: int,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.2,
        loss_type: str = 'bce',
    ):
        super().__init__()
        hidden_dims = hidden_dims or [128, 64]
        per_event_out = 2 if loss_type == 'aft' else n_horizons

        self.heads = nn.ModuleList()
        for _ in range(n_events):
            layers: List[nn.Module] = []
            prev_dim = input_dim
            for hdim in hidden_dims:
                layers.extend([
                    nn.Linear(prev_dim, hdim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ])
                prev_dim = hdim
            layers.append(nn.Linear(prev_dim, per_event_out))
            self.heads.append(nn.Sequential(*layers))

        self.n_events = n_events
        self.per_event_out = per_event_out
        self.loss_type = loss_type

    def forward(self, encoder_output: torch.Tensor) -> torch.Tensor:
        outputs = [head(encoder_output) for head in self.heads]

        if self.loss_type == 'aft':
            # Each head outputs (batch, 2): [mu_i, log_sigma_i]
            # Rearrange to [mu_0..mu_n, log_sigma_0..log_sigma_n]
            stacked = torch.stack(outputs, dim=1)  # (batch, n_events, 2)
            mu = stacked[:, :, 0]
            log_sigma = stacked[:, :, 1]
            return torch.cat([mu, log_sigma], dim=-1)  # (batch, 2*n_events)
        else:
            # [event0_h0..event0_hn, event1_h0..event1_hn, ...]
            return torch.cat(outputs, dim=-1)  # (batch, n_events*n_horizons)


class SequenceModel(nn.Module):
    """
    Full sequence model: Embedding -> Encoder -> Prediction Head.

    Supports three encoder types: lstm, gru, transformer.
    Supports multi_head=True for separate per-event prediction heads.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        encoder_type: str = 'lstm',
        encoder_config: Optional[Dict[str, Any]] = None,
        n_events: int = 5,
        n_horizons: int = 3,
        max_seq_len: int = 256,
        head_hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.1,
        loss_type: str = 'bce',
        multi_head: bool = False,
    ):
        super().__init__()
        encoder_config = encoder_config or {}

        self.embedding = TokenEmbedding(
            vocab_size, embed_dim, max_seq_len, dropout
        )

        if encoder_type == 'lstm':
            self.encoder = LSTMEncoder(
                embed_dim=embed_dim,
                hidden_dim=encoder_config.get('hidden_dim', 256),
                num_layers=encoder_config.get('num_layers', 2),
                dropout=encoder_config.get('dropout', 0.3),
            )
        elif encoder_type == 'gru':
            self.encoder = GRUEncoder(
                embed_dim=embed_dim,
                hidden_dim=encoder_config.get('hidden_dim', 256),
                num_layers=encoder_config.get('num_layers', 2),
                dropout=encoder_config.get('dropout', 0.3),
            )
        elif encoder_type == 'transformer':
            self.encoder = TransformerSequenceEncoder(
                embed_dim=embed_dim,
                num_heads=encoder_config.get('num_heads', 4),
                num_layers=encoder_config.get('num_layers', 3),
                ff_dim=encoder_config.get('ff_dim', 512),
                dropout=encoder_config.get('dropout', 0.1),
            )
        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}")

        # Build prediction head(s)
        if multi_head:
            self.head = MultiEventHead(
                input_dim=self.encoder.output_dim,
                n_events=n_events,
                n_horizons=n_horizons,
                hidden_dims=head_hidden_dims,
                dropout=dropout,
                loss_type=loss_type,
            )
        else:
            head_output_dim = (n_events * 2) if loss_type == 'aft' else None
            self.head = HorizonPredictionHead(
                input_dim=self.encoder.output_dim,
                n_events=n_events,
                n_horizons=n_horizons,
                hidden_dims=head_hidden_dims,
                dropout=dropout,
                output_dim=head_output_dim,
            )

        self.encoder_type = encoder_type
        self.loss_type = loss_type
        self.multi_head = multi_head
        self.n_events = n_events
        self.n_horizons = n_horizons

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            input_ids: (batch, seq_len) token IDs
            attention_mask: (batch, seq_len) 1=real, 0=pad
        Returns:
            logits: (batch, n_events * n_horizons)
        """
        embeddings = self.embedding(input_ids)
        encoded = self.encoder(embeddings, attention_mask)
        logits = self.head(encoded)
        return logits

    def enable_head_dropout(self):
        """Enable dropout only in the prediction head for MC Dropout inference.

        Sets the whole model to eval mode (disabling BN, encoder dropout etc.),
        then re-enables training mode on the head so its Dropout layers remain
        active during forward passes.
        """
        self.eval()
        self.head.train()

    def disable_head_dropout(self):
        """Fully disable dropout (standard eval mode)."""
        self.eval()
