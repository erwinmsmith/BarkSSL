"""
BarkSSL Emotion Classifier Module
Downstream emotion classification head with pooling.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from .components import MeanPooling, AttentiveStatisticsPooling
from .canine_encoder import CanineEncoder


class EmotionClassifier(nn.Module):
    """
    Emotion classifier for downstream bark emotion recognition.

    Architecture:
    - Pretrained canine encoder (frozen or fine-tuned)
    - Pooling layer
    - MLP classifier head
    """

    EMOTION_LABELS = ['angry', 'anxious', 'happy', 'lonely', 'sad']

    def __init__(
        self,
        encoder: CanineEncoder,
        num_classes: int = 5,
        pooling_type: str = 'attentive',
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        """
        Initialize emotion classifier.

        Args:
            encoder: Pretrained canine encoder
            num_classes: Number of emotion classes
            pooling_type: Pooling type ('mean' or 'attentive')
            hidden_dim: Hidden dimension (defaults to encoder.hidden_dim)
            dropout: Dropout rate
        """
        super().__init__()

        self.encoder = encoder
        self.num_classes = num_classes
        pooling_dim = hidden_dim or encoder.hidden_dim

        # Pooling layer
        if pooling_type == 'mean':
            self.pooler = MeanPooling()
            classifier_input_dim = pooling_dim
        elif pooling_type == 'attentive':
            self.pooler = AttentiveStatisticsPooling(pooling_dim)
            classifier_input_dim = pooling_dim * 2
        else:
            raise ValueError(f"Unknown pooling type: {pooling_type}")

        # Classifier head
        self.classifier = nn.Sequential(
            nn.LayerNorm(classifier_input_dim),
            nn.Dropout(dropout),
            nn.Linear(classifier_input_dim, pooling_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(pooling_dim, num_classes),
        )

        self.pooling_type = pooling_type

    def forward(
        self,
        waveform: torch.Tensor,
        return_embedding: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            waveform: Input waveform [batch, seq_len]
            return_embedding: Whether to return embedding only

        Returns:
            Dictionary with:
                - logits: [batch, num_classes]
                - embedding: [batch, hidden_dim] (if return_embedding=True)
                - hidden_states: [batch, n_frames, hidden_dim]
        """
        # Extract features from encoder
        encoder_output = self.encoder(waveform)
        hidden_states = encoder_output['hidden_states']

        # Pooling
        embedding = self.pooler(hidden_states)

        if return_embedding:
            return {
                'embedding': embedding,
                'hidden_states': hidden_states,
            }

        # Classification
        logits = self.classifier(embedding)

        return {
            'logits': logits,
            'embedding': embedding,
            'hidden_states': hidden_states,
        }

    def extract_embedding(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Extract embedding for a waveform.

        Args:
            waveform: Input waveform [batch, seq_len]

        Returns:
            embedding: [batch, hidden_dim]
        """
        output = self.forward(waveform, return_embedding=True)
        return output['embedding']

    def get_num_parameters(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())

    def train_step(self, batch: Dict[str, Any], **kwargs) -> Dict[str, torch.Tensor]:
        """Single training step."""
        labels = batch.get('labels')
        output = self.forward(batch.get('waveforms', batch.get('waveform')))
        logits = output['logits']
        embedding = output['embedding']

        return {
            'logits': logits,
            'embedding': embedding,
            'labels': labels,
        }

    def eval_step(self, batch: Dict[str, Any], **kwargs) -> Dict[str, torch.Tensor]:
        """Single evaluation step."""
        return self.train_step(batch)


class MultiLayerClassifier(nn.Module):
    """
    Multi-layer classifier with configurable depth.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 384,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        """
        Initialize multi-layer classifier.

        Args:
            input_dim: Input dimension
            num_classes: Number of output classes
            hidden_dim: Hidden dimension
            num_layers: Number of hidden layers
            dropout: Dropout rate
        """
        super().__init__()

        layers = []
        in_dim = input_dim

        for i in range(num_layers):
            out_dim = hidden_dim if i < num_layers - 1 else num_classes
            layers.append(nn.Linear(in_dim, out_dim))

            if i < num_layers - 1:
                layers.append(nn.LayerNorm(out_dim))
                layers.append(nn.GELU())
                layers.append(nn.Dropout(dropout))

            in_dim = out_dim

        self.classifier = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


class WeightedEmotionClassifier(EmotionClassifier):
    """
    Emotion classifier with class weights support for imbalanced data.
    """

    def __init__(
        self,
        encoder: CanineEncoder,
        num_classes: int = 5,
        pooling_type: str = 'attentive',
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
        class_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__(
            encoder=encoder,
            num_classes=num_classes,
            pooling_type=pooling_type,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        # Store class weights for loss computation
        self.register_buffer('class_weights', class_weights if class_weights is not None else torch.ones(num_classes))

    def update_class_weights(self, weights: torch.Tensor) -> None:
        """
        Update class weights for imbalanced data.

        Args:
            weights: Tensor of shape [num_classes]
        """
        self.class_weights = weights

    def forward(self, waveform: torch.Tensor, return_embedding: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass with weighted logits.

        Args:
            waveform: Input waveform
            return_embedding: Whether to return embedding only

        Returns:
            Dictionary with logits, embedding, hidden_states
        """
        output = super().forward(waveform, return_embedding)

        if return_embedding:
            return output

        # Apply class weights to logits
        logits = output['logits']
        weighted_logits = logits * self.class_weights.unsqueeze(0)

        return {
            'logits': weighted_logits,
            'embedding': output['embedding'],
            'hidden_states': output['hidden_states'],
        }


class TransformerClassifier(nn.Module):
    """
    Transformer-based classifier using pooled encoder outputs.
    """

    def __init__(
        self,
        encoder: CanineEncoder,
        num_classes: int = 5,
        num_transformer_layers: int = 2,
        dropout: float = 0.1,
    ):
        """
        Initialize transformer classifier.

        Args:
            encoder: Pretrained canine encoder
            num_classes: Number of emotion classes
            num_transformer_layers: Additional transformer layers for classification
            dropout: Dropout rate
        """
        super().__init__()

        self.encoder = encoder
        self.num_classes = num_classes

        # Use encoder's transformer layers but freeze them
        self.encoder.transformer_encoder.requires_grad_(False)

        hidden_dim = encoder.hidden_dim

        # Additional transformer layers
        from .components import TransformerEncoderLayer
        self.class_transformer = nn.ModuleList([
            TransformerEncoderLayer(
                hidden_dim=hidden_dim,
                num_heads=encoder.num_heads,
                intermediate_size=encoder.intermediate_size,
                dropout=dropout,
            )
            for _ in range(num_transformer_layers)
        ])

        # Attention pooling for classification
        self.attention_pool = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, waveform: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            waveform: Input waveform [batch, seq_len]

        Returns:
            Dictionary with logits and embeddings
        """
        # Extract features from encoder
        encoder_output = self.encoder(waveform)
        hidden_states = encoder_output['hidden_states']

        # Additional transformer layers
        for layer in self.class_transformer:
            hidden_states = layer(hidden_states)

        # Attention pooling
        attention_weights = self.attention_pool(hidden_states).squeeze(-1)
        attention_weights = torch.softmax(attention_weights, dim=-1)
        embedding = (hidden_states * attention_weights.unsqueeze(-1)).sum(dim=1)

        # Classification
        logits = self.classifier(embedding)

        return {
            'logits': logits,
            'embedding': embedding,
            'hidden_states': hidden_states,
        }


def create_emotion_classifier(
    encoder: CanineEncoder,
    num_classes: int = 5,
    pooling_type: str = 'attentive',
    classifier_type: str = 'standard',
    **kwargs,
) -> EmotionClassifier:
    """
    Factory function to create emotion classifier.

    Args:
        encoder: Pretrained canine encoder
        num_classes: Number of emotion classes
        pooling_type: Pooling type ('mean' or 'attentive')
        classifier_type: Classifier type ('standard', 'weighted', 'transformer')
        **kwargs: Additional arguments

    Returns:
        EmotionClassifier instance
    """
    if classifier_type == 'standard':
        return EmotionClassifier(
            encoder=encoder,
            num_classes=num_classes,
            pooling_type=pooling_type,
            **kwargs,
        )
    elif classifier_type == 'weighted':
        return WeightedEmotionClassifier(
            encoder=encoder,
            num_classes=num_classes,
            pooling_type=pooling_type,
            **kwargs,
        )
    elif classifier_type == 'transformer':
        return TransformerClassifier(
            encoder=encoder,
            num_classes=num_classes,
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown classifier type: {classifier_type}")