"""
BarkSSL Masked Pretraining Module
Stage 2: Self-supervised pretraining with masked unit prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple
import numpy as np
from pathlib import Path

from ..models.canine_encoder import CanineEncoder, CanineWavLMEncoder
from ..data.transforms import BarkDenoising, Compose


class Masking:
    """
    Masking strategy for masked prediction training.
    """

    def __init__(
        self,
        mask_prob: float = 0.075,
        mask_span: int = 10,
        mask_value: float = 0.0,
    ):
        """
        Initialize masking.

        Args:
            mask_prob: Probability of starting a mask span
            mask_span: Average length of mask spans
            mask_value: Value to fill masked positions
        """
        self.mask_prob = mask_prob
        self.mask_span = mask_span
        self.mask_value = mask_value

    def mask(
        self,
        hidden_states: torch.Tensor,
        pseudo_labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply masking to hidden states.

        Args:
            hidden_states: [batch, seq_len, hidden_dim]
            pseudo_labels: Optional pseudo labels [batch, seq_len]

        Returns:
            Tuple of:
                - masked_hidden: Hidden states with mask applied
                - mask_positions: Boolean mask [batch, seq_len]
                - target_labels: Labels for masked positions (only valid where masked)
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape

        # Create mask probability mask
        mask_probs = torch.full((batch_size, seq_len), self.mask_prob, device=hidden_states.device)
        mask_indicator = torch.bernoulli(mask_probs).bool()

        # Expand mask spans
        for b in range(batch_size):
            i = 0
            while i < seq_len:
                if mask_indicator[b, i]:
                    # Random span length centered around mask_span
                    span_len = max(1, int(np.random.normal(self.mask_span, 2)))
                    span_len = min(span_len, seq_len - i)
                    mask_indicator[b, i:i+span_len] = True
                    i += span_len
                else:
                    i += 1

        # Apply mask to hidden states
        masked_hidden = hidden_states.clone()
        masked_hidden[mask_indicator] = self.mask_value

        # Get target labels for masked positions
        target_labels = None
        if pseudo_labels is not None:
            # Set non-masked positions to -100 (ignore in CE loss)
            target_labels = pseudo_labels.clone()
            target_labels[~mask_indicator] = -100

        return masked_hidden, mask_indicator, target_labels


class MaskedPretraining:
    """
    Stage 2: Masked unit prediction pretraining.

    Implements HuBERT-style masked prediction:
    1. Extract features from DogSpeak audio
    2. Apply masking to some frames
    3. Predict pseudo-unit labels from masked positions
    4. Compute masked cross-entropy loss
    """

    def __init__(
        self,
        encoder: CanineEncoder,
        masking: Optional[Masking] = None,
        use_denoising: bool = False,
        acoustic_unit: Optional[Any] = None,
    ):
        """
        Initialize masked pretraining.

        Args:
            encoder: Canine encoder model
            masking: Masking strategy
            use_denoising: Whether to use WavLM-style denoising
            acoustic_unit: Optional acoustic unit discovery for on-the-fly pseudo-labels
        """
        self.encoder = encoder
        self.masking = masking or Masking()
        self.use_denoising = use_denoising
        self.acoustic_unit = acoustic_unit

        # Denoising transform
        self.denoising_transform = BarkDenoising() if use_denoising else None

    def compute_loss(
        self,
        hidden_states: torch.Tensor,
        target_labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Compute masked prediction loss.

        Args:
            hidden_states: [batch, seq_len, hidden_dim]
            target_labels: [batch, seq_len] with -100 for non-masked

        Returns:
            Tuple of (loss, metrics)
        """
        logits = self.encoder.prediction_head(hidden_states)

        # Compute CE loss, ignoring -100 labels
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            target_labels.view(-1),
            ignore_index=-100,
        )

        # Compute accuracy on masked positions only
        mask = target_labels != -100
        if mask.sum() > 0:
            pred_labels = logits.argmax(dim=-1)
            acc = (pred_labels[mask] == target_labels[mask]).float().mean().item()
        else:
            acc = 0.0

        return loss, {'loss': loss.item(), 'accuracy': acc}

    def forward_step(
        self,
        waveform: torch.Tensor,
        pseudo_labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Single forward step.

        Args:
            waveform: Input waveform [batch, seq_len]
            pseudo_labels: Frame-level pseudo labels [batch, seq_len]

        Returns:
            Dictionary with loss, metrics, and outputs
        """
        # Optional: apply denoising augmentation
        if self.use_denoising and self.denoising_transform:
            waveform = self.denoising_transform(waveform.squeeze().cpu().numpy())
            waveform = torch.from_numpy(waveform).float().to(self.encoder.layer_norm.weight.device)
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)

        # Extract features
        encoder_output = self.encoder(waveform)
        hidden_states = encoder_output['hidden_states']

        # Generate pseudo-labels if not provided (on-the-fly)
        if pseudo_labels is None and self.acoustic_unit is not None:
            # Generate pseudo-labels from acoustic unit
            batch_size, seq_len, _ = hidden_states.shape
            pseudo_labels = []
            for i in range(batch_size):
                wave = waveform[i].cpu().numpy()
                labels = self.acoustic_unit.predict(wave)
                # Map mel-feature frames to CNN frames
                # mel frame stride = 160 samples, CNN frame stride = 40 samples
                # CNN frames = mel_frames * 4
                labels = np.repeat(labels, 4)[:seq_len]
                pseudo_labels.append(labels)
            pseudo_labels = torch.LongTensor(np.array(pseudo_labels)).to(hidden_states.device)
        elif pseudo_labels is None:
            # Skip loss computation if no pseudo-labels and no acoustic unit
            return {
                'loss': torch.tensor(0.0, device=hidden_states.device),
                'metrics': {'loss': 0.0, 'skipped': True},
                'hidden_states': hidden_states,
            }

        # Apply masking
        masked_hidden, mask_positions, target_labels = self.masking.mask(
            hidden_states, pseudo_labels
        )

        # Get predictions for masked positions
        # Handle both DDP-wrapped and unwrapped models
        encoder = self.encoder.module if hasattr(self.encoder, 'module') else self.encoder
        logits = encoder.prediction_head(masked_hidden)

        # Compute loss
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            target_labels.view(-1),
            ignore_index=-100,
        )

        # Metrics
        metrics = {'loss': loss.item()}

        mask_count = (target_labels != -100).sum().item()
        if mask_count > 0:
            pred_labels = logits.argmax(dim=-1)
            acc = (pred_labels[target_labels != -100] == target_labels[target_labels != -100]).float().mean().item()
            metrics['accuracy'] = acc
            metrics['mask_count'] = mask_count

        return {
            'loss': loss,
            'metrics': metrics,
            'logits': logits,
            'hidden_states': hidden_states,
            'target_labels': target_labels,
        }


class CanineHuBERTPretraining:
    """
    Complete HuBERT-style pretraining pipeline.
    """

    def __init__(
        self,
        encoder: CanineEncoder,
        acoustic_unit_discovery: Any,
        mask_prob: float = 0.075,
        mask_span: int = 10,
        use_denoising: bool = False,
    ):
        """
        Initialize pretraining.

        Args:
            encoder: Canine encoder
            acoustic_unit_discovery: Fitted AcousticUnitDiscovery for pseudo-labels
            mask_prob: Mask probability
            mask_span: Mask span length
            use_denoising: Use WavLM-style denoising
        """
        self.encoder = encoder
        self.acoustic_unit = acoustic_unit_discovery
        self.masked_pretraining = MaskedPretraining(
            encoder=encoder,
            masking=Masking(mask_prob=mask_prob, mask_span=mask_span),
            use_denoising=use_denoising,
            acoustic_unit=acoustic_unit_discovery,
        )

    def generate_pseudo_labels(
        self,
        waveform: np.ndarray,
    ) -> torch.Tensor:
        """
        Generate pseudo-labels for a waveform.

        Args:
            waveform: Audio waveform as numpy array

        Returns:
            Pseudo labels as tensor
        """
        labels = self.acoustic_unit.predict(waveform)
        return torch.from_numpy(labels).long()

    def train_step(
        self,
        batch: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Single training step.

        Args:
            batch: Dictionary with 'waveform' and 'pseudo_labels'

        Returns:
            Dictionary with loss and metrics
        """
        waveform = batch['waveform']
        pseudo_labels = batch.get('pseudo_labels')

        # Ensure waveform is 2D
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        if pseudo_labels is not None and pseudo_labels.dim() == 1:
            pseudo_labels = pseudo_labels.unsqueeze(0)

        # Forward step
        output = self.masked_pretraining.forward_step(waveform, pseudo_labels)

        return output

    def save(self, path: str) -> None:
        """
        Save pretraining checkpoint.

        Args:
            path: Path to save checkpoint
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        torch.save({
            'encoder_state_dict': self.encoder.state_dict(),
            'mask_prob': self.masked_pretraining.masking.mask_prob,
            'mask_span': self.masked_pretraining.masking.mask_span,
            'use_denoising': self.masked_pretraining.use_denoising,
        }, path)

    def load(self, path: str) -> None:
        """
        Load pretraining checkpoint.

        Args:
            path: Path to checkpoint
        """
        checkpoint = torch.load(path, map_location='cpu')

        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])

    def to(self, device: torch.device) -> 'CanineHuBERTPretraining':
        """Move model to device."""
        self.encoder.to(device)
        return self

    def parameters(self):
        """Return encoder parameters for optimizer."""
        return self.encoder.parameters()

    def train(self, mode: bool = True) -> 'CanineHuBERTPretraining':
        """Set train/eval mode."""
        self.encoder.train(mode)
        return self

    def eval(self) -> 'CanineHuBERTPretraining':
        """Set eval mode."""
        return self.train(False)

    def state_dict(self):
        """Return encoder state dict."""
        return self.encoder.state_dict()

    def load_state_dict(self, state_dict):
        """Load encoder state dict."""
        self.encoder.load_state_dict(state_dict)

    def train_step(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Single training step.

        Args:
            batch: Dictionary with 'waveform' and 'pseudo_labels'

        Returns:
            Dictionary with loss and metrics
        """
        waveform = batch['waveform']
        device = next(self.encoder.parameters()).device

        # Handle batch of variable-length waveforms (list format)
        if isinstance(waveform, list):
            # Find max length
            max_len = max(len(w) for w in waveform)
            # Pad all to max_len (CNN will handle alignment internally)
            padded = []
            for w in waveform:
                if len(w) < max_len:
                    w_padded = np.pad(w, (0, max_len - len(w)), mode='constant')
                else:
                    w_padded = w[:max_len]
                padded.append(w_padded)
            waveform = np.vstack(padded)
            waveform = torch.from_numpy(waveform).float().to(device)
        elif isinstance(waveform, np.ndarray):
            waveform = torch.from_numpy(waveform).float().to(device)

        pseudo_labels = batch.get('pseudo_labels')

        # Ensure waveform is 2D
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        if pseudo_labels is not None and pseudo_labels.dim() == 1:
            pseudo_labels = pseudo_labels.unsqueeze(0)

        # Forward step
        output = self.masked_pretraining.forward_step(waveform, pseudo_labels)

        return output


def create_masked_pretraining(
    encoder: CanineEncoder,
    config: Dict[str, Any],
    acoustic_unit_discovery: Any = None,
) -> CanineHuBERTPretraining:
    """
    Factory function to create masked pretraining.

    Args:
        encoder: Canine encoder
        config: Configuration dictionary
        acoustic_unit_discovery: Optional pre-fitted acoustic unit discovery

    Returns:
        CanineHuBERTPretraining instance
    """
    mask_prob = config.get('mask_prob', 0.075)
    mask_span = config.get('mask_span', 10)
    use_denoising = config.get('use_denoising', False)

    return CanineHuBERTPretraining(
        encoder=encoder,
        acoustic_unit_discovery=acoustic_unit_discovery,
        mask_prob=mask_prob,
        mask_span=mask_span,
        use_denoising=use_denoising,
    )