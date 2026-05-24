"""
BarkSSL Model Components Module
CNN encoder, Transformer encoder, and related components.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class CNNEncoder(nn.Module):
    """
    CNN feature encoder for raw waveform input.

    Inspired by WavLM/HuBERT CNN encoder.
    Converts raw waveform to latent frame sequences.
    """

    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 512,
        kernel_sizes: Tuple[int, ...] = (10, 3, 3, 3),
        strides: Tuple[int, ...] = (5, 2, 2, 2),
        bias: bool = True,
    ):
        """
        Initialize CNN encoder.

        Args:
            input_dim: Input channels (1 for mono audio)
            hidden_dim: Output feature dimension
            kernel_sizes: Tuple of kernel sizes for each conv layer
            strides: Tuple of strides for each conv layer
            bias: Whether to use bias in conv layers
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.kernel_sizes = kernel_sizes
        self.strides = strides

        # Build convolutional layers
        layers = []
        in_channels = input_dim

        for i, (kernel_size, stride) in enumerate(zip(kernel_sizes, strides)):
            out_channels = hidden_dim if i == len(kernel_sizes) - 1 else hidden_dim
            layers.append(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=stride,
                    bias=bias,
                )
            )
            in_channels = out_channels

        self.conv_layers = nn.ModuleList(layers)
        self.num_layers = len(layers)

        # Activation and normalization
        self.activation = nn.GELU()
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input waveform [batch, seq_len] or [batch, 1, seq_len]

        Returns:
            features: [batch, n_frames, hidden_dim]
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add channel dimension

        # Apply conv layers
        for i, conv in enumerate(self.conv_layers):
            x = conv(x)
            if i < self.num_layers - 1:
                x = self.activation(x)

        # Transpose for layer norm: [batch, hidden_dim, n_frames] -> [batch, n_frames, hidden_dim]
        x = x.transpose(1, 2)
        x = self.layer_norm(x)

        return x


class RelativePositionBias(nn.Module):
    """
    Relative position bias for Transformer (WavLM-style).
    """

    def __init__(self, num_heads: int = 8, max_seq_len: int = 512):
        """
        Initialize relative position bias.

        Args:
            num_heads: Number of attention heads
            max_seq_len: Maximum sequence length
        """
        super().__init__()
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len

        # Learnable relative position bias
        self.bias = nn.Parameter(
            torch.zeros(num_heads, 2 * max_seq_len - 1)
        )

        # Initialize
        nn.init.trunc_normal_(self.bias, std=0.02)

    def _relative_position_bucket(self, relative_position: torch.Tensor) -> torch.Tensor:
        """
        Compute relative position bucket for bias lookup.

        Args:
            relative_position: [query_len, key_len]

        Returns:
            buckets: [query_len, key_len]
        """
        num_buckets = 2 * self.max_seq_len - 1
        max_distance = self.max_seq_len - 1

        relative_buckets = (relative_position > 0).long() * num_buckets
        relative_position = torch.abs(relative_position)

        relative_position = torch.max(
            relative_position,
            torch.full_like(relative_position, max_distance)
        )

        max_exact = max_distance // 2
        is_small = relative_position < max_exact

        relative_position_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact) /
            math.log(max_distance / max_exact) *
            (num_buckets - max_exact)
        ).long()
        relative_position_if_large = torch.min(
            relative_position_if_large,
            torch.full_like(relative_position_if_large, num_buckets - 1)
        )

        relative_buckets += torch.where(
            is_small,
            relative_position,
            relative_position_if_large
        )

        return relative_buckets

    def forward(self, seq_len: int, num_buckets: Optional[int] = None) -> torch.Tensor:
        """
        Get relative position bias for attention.

        Args:
            seq_len: Sequence length
            num_buckets: Number of buckets (defaults to seq_len)

        Returns:
            bias: [num_heads, seq_len, seq_len]
        """
        if num_buckets is None:
            num_buckets = seq_len

        # Compute relative position indices
        query_positions = torch.arange(seq_len, device=self.bias.device)
        key_positions = torch.arange(num_buckets, device=self.bias.device)

        relative_position = query_positions.unsqueeze(1) - key_positions.unsqueeze(0)
        relative_position = relative_position + num_buckets - 1

        # Clamp to valid range
        relative_position = torch.clamp(
            relative_position,
            0,
            self.bias.shape[1] - 1
        )

        # Lookup bias
        bias = self.bias[:, relative_position]  # [num_heads, query_len, key_len]

        return bias


class TransformerEncoderLayer(nn.Module):
    """
    Transformer encoder layer with relative position bias.
    """

    def __init__(
        self,
        hidden_dim: int = 384,
        num_heads: int = 6,
        intermediate_size: int = 1536,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-12,
    ):
        """
        Initialize transformer encoder layer.

        Args:
            hidden_dim: Hidden dimension
            num_heads: Number of attention heads
            intermediate_size: FFN intermediate size
            dropout: Dropout rate
            layer_norm_eps: LayerNorm epsilon
        """
        super().__init__()

        self.self_attn = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.relative_bias = RelativePositionBias(
            num_heads=num_heads,
            max_seq_len=512,
        )

        self.linear1 = nn.Linear(hidden_dim, intermediate_size)
        self.linear2 = nn.Linear(intermediate_size, hidden_dim)

        self.norm1 = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)

        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input [batch, seq_len, hidden_dim]
            attention_mask: Optional attention mask

        Returns:
            Output [batch, seq_len, hidden_dim]
        """
        # Self-attention with residual
        residual = x

        attn_output, _ = self.self_attn(
            x, x, x,
            attn_mask=attention_mask,
        )

        x = residual + self.dropout(attn_output)
        x = self.norm1(x)

        # FFN with residual
        residual = x
        x = self.linear2(self.activation(self.linear1(x)))
        x = residual + self.dropout(x)
        x = self.norm2(x)

        return x


class TransformerEncoder(nn.Module):
    """
    Stack of Transformer encoder layers.
    """

    def __init__(
        self,
        num_layers: int = 6,
        hidden_dim: int = 384,
        num_heads: int = 6,
        intermediate_size: int = 1536,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-12,
    ):
        """
        Initialize transformer encoder.

        Args:
            num_layers: Number of encoder layers
            hidden_dim: Hidden dimension
            num_heads: Number of attention heads
            intermediate_size: FFN intermediate size
            dropout: Dropout rate
            layer_norm_eps: LayerNorm epsilon
        """
        super().__init__()

        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                intermediate_size=intermediate_size,
                dropout=dropout,
                layer_norm_eps=layer_norm_eps,
            )
            for _ in range(num_layers)
        ])

        self.num_layers = num_layers

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input [batch, seq_len, hidden_dim]
            attention_mask: Optional attention mask

        Returns:
            Output [batch, seq_len, hidden_dim]
        """
        for layer in self.layers:
            x = layer(x, attention_mask)

        return x


class MeanPooling(nn.Module):
    """
    Mean pooling over time dimension.
    """

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input [batch, seq_len, hidden_dim]
            mask: Optional mask [batch, seq_len]

        Returns:
            pooled: [batch, hidden_dim]
        """
        if mask is not None:
            mask = mask.unsqueeze(-1).float()
            return (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return x.mean(dim=1)


class AttentiveStatisticsPooling(nn.Module):
    """
    Attentive statistics pooling for audio.
    Computes weighted mean and standard deviation.
    """

    def __init__(self, hidden_dim: int = 384):
        """
        Initialize attentive pooling.

        Args:
            hidden_dim: Input hidden dimension
        """
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input [batch, seq_len, hidden_dim]
            mask: Optional mask [batch, seq_len]

        Returns:
            pooled: [batch, 2 * hidden_dim]
        """
        # Compute attention weights
        attention_weights = self.attention(x).squeeze(-1)  # [batch, seq_len]

        if mask is not None:
            attention_weights = attention_weights.masked_fill(~mask, float('-inf'))

        attention_weights = F.softmax(attention_weights, dim=-1)

        # Weighted mean
        mean = (x * attention_weights.unsqueeze(-1)).sum(dim=1)

        # Weighted standard deviation
        variance = ((x - mean.unsqueeze(1)) ** 2 * attention_weights.unsqueeze(-1)).sum(dim=1)
        std = variance.clamp(min=1e-6).sqrt()

        # Concatenate mean and std
        return torch.cat([mean, std], dim=-1)


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding.
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)