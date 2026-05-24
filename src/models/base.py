"""
BarkSSL Base Model Module
Abstract base class for all models.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
from pathlib import Path


class BaseModel(ABC, nn.Module):
    """
    Abstract base class for all models.

    Provides common interface for training, evaluation, and inference.
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Returns:
            Dictionary with model outputs
        """
        pass

    def train_step(self, batch: Dict[str, Any], **kwargs) -> Dict[str, torch.Tensor]:
        """
        Single training step.

        Args:
            batch: Input batch

        Returns:
            Dictionary with losses and outputs
        """
        raise NotImplementedError

    def eval_step(self, batch: Dict[str, Any], **kwargs) -> Dict[str, torch.Tensor]:
        """
        Single evaluation step.

        Args:
            batch: Input batch

        Returns:
            Dictionary with outputs
        """
        raise NotImplementedError

    def predict(self, waveform: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Prediction/inference.

        Args:
            waveform: Input waveform

        Returns:
            Dictionary with predictions
        """
        raise NotImplementedError

    def save(self, path: str) -> None:
        """
        Save model to checkpoint file.

        Args:
            path: Path to save checkpoint
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        torch.save({
            'model_state_dict': self.state_dict(),
            'model_config': self.get_config(),
        }, path)

    def load(self, path: str, strict: bool = True) -> None:
        """
        Load model from checkpoint file.

        Args:
            path: Path to checkpoint
            strict: Whether to strictly match keys
        """
        checkpoint = torch.load(path, map_location='cpu')

        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        self.load_state_dict(state_dict, strict=strict)

    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """
        Get model configuration.

        Returns:
            Dictionary with model config
        """
        pass

    def get_num_parameters(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())

    def get_trainable_parameters(self) -> int:
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def freeze(self, layers: Optional[list] = None) -> None:
        """
        Freeze model parameters.

        Args:
            layers: List of layer names to freeze (None = freeze all)
        """
        if layers is None:
            for param in self.parameters():
                param.requires_grad = False
        else:
            for name, param in self.named_parameters():
                if any(layer in name for layer in layers):
                    param.requires_grad = False

    def unfreeze(self, layers: Optional[list] = None) -> None:
        """
        Unfreeze model parameters.

        Args:
            layers: List of layer names to unfreeze (None = unfreeze all)
        """
        if layers is None:
            for param in self.parameters():
                param.requires_grad = True
        else:
            for name, param in self.named_parameters():
                if any(layer in name for layer in layers):
                    param.requires_grad = True


class TrainableModel(BaseModel, ABC):
    """
    Base class for models that require training.
    """

    def __init__(self):
        super().__init__()
        self.training_step = 0
        self.epoch = 0
        self.best_metric = 0.0

    @abstractmethod
    def compute_loss(self, batch: Dict[str, Any], output: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute loss for training.

        Args:
            batch: Input batch
            output: Model output

        Returns:
            Loss tensor
        """
        pass

    def train_step(self, batch: Dict[str, Any], **kwargs) -> Dict[str, torch.Tensor]:
        """Default training step implementation."""
        output = self.forward(**batch)
        loss = self.compute_loss(batch, output)

        self.training_step += 1

        return {
            'loss': loss,
            **output,
        }

    def eval_step(self, batch: Dict[str, Any], **kwargs) -> Dict[str, torch.Tensor]:
        """Default evaluation step implementation."""
        with torch.no_grad():
            output = self.forward(**batch)

        return output


class CompositeModel(BaseModel):
    """
    Composite model that combines multiple sub-models.
    """

    def __init__(self, encoder: BaseModel, head: BaseModel):
        """
        Initialize composite model.

        Args:
            encoder: Encoder model
            head: Classification/regression head
        """
        super().__init__()
        self.encoder = encoder
        self.head = head

    def forward(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward through encoder then head."""
        encoder_output = self.encoder.forward(*args, **kwargs)
        hidden_states = encoder_output.get('hidden_states', encoder_output.get('embedding'))

        head_output = self.head.forward(hidden_states)

        return {
            **encoder_output,
            **head_output,
        }

    def get_config(self) -> Dict[str, Any]:
        return {
            'encoder_config': self.encoder.get_config(),
            'head_config': self.head.get_config(),
        }