"""
BarkSSL Loss Functions Module
Cross-entropy and supervised contrastive loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any


class CrossEntropyLoss(nn.Module):
    """
    Standard cross-entropy loss for classification.
    """

    def __init__(
        self,
        num_classes: int = 5,
        class_weights: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0,
    ):
        """
        Initialize cross-entropy loss.

        Args:
            num_classes: Number of classes
            class_weights: Optional class weights for imbalanced data
            label_smoothing: Label smoothing factor
        """
        super().__init__()

        self.num_classes = num_classes
        self.class_weights = class_weights
        self.label_smoothing = label_smoothing

        if class_weights is not None:
            self.register_buffer('weights', class_weights)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute cross-entropy loss.

        Args:
            logits: Model predictions [batch, num_classes]
            targets: Ground truth labels [batch]

        Returns:
            Loss value
        """
        if self.class_weights is not None:
            return F.cross_entropy(
                logits,
                targets,
                weight=self.weights,
                label_smoothing=self.label_smoothing,
            )
        else:
            return F.cross_entropy(
                logits,
                targets,
                label_smoothing=self.label_smoothing,
            )


class SupervisedContrastiveLoss(nn.Module):
    """
    Supervised Contrastive Learning Loss (Khosla et al.).

    Encourages embeddings of the same class to be close,
    and embeddings of different classes to be far apart.
    """

    def __init__(
        self,
        temperature: float = 0.07,
        base_temperature: float = 0.07,
        lambda_: float = 0.1,
    ):
        """
        Initialize supervised contrastive loss.

        Args:
            temperature: Temperature for scaling similarities
            base_temperature: Base temperature
            lambda_: Weight for combining with CE loss
        """
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.lambda_ = lambda_

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute supervised contrastive loss.

        Args:
            embeddings: Normalized embeddings [batch, embedding_dim]
            labels: Class labels [batch]

        Returns:
            Loss value
        """
        batch_size = embeddings.size(0)

        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)

        # Compute similarity matrix
        similarities = torch.matmul(embeddings, embeddings.T) / self.temperature

        # Create mask for positive pairs (same label)
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float()

        # Remove diagonal (self-similarity)
        mask = mask - torch.eye(batch_size, device=mask.device)

        # For numerical stability
        similarities = similarities - similarities.max(dim=-1, keepdim=True)[0]

        # Compute loss
        exp_sim = torch.exp(similarities)

        # Sum of exp similarities for denominator
        denom = exp_sim.sum(dim=-1, keepdim=True) - torch.exp(torch.tensor(0.0, device=similarities.device))

        # Positive pairs loss
        pos_pairs = (exp_sim * mask).sum(dim=-1)
        pos_pairs = torch.clamp(pos_pairs, min=1e-8)

        loss = -torch.log(pos_pairs / denom.squeeze())

        # Mask out zero-denominator cases
        valid_mask = denom.squeeze() > 0
        loss = loss[valid_mask]

        if len(loss) == 0:
            return torch.tensor(0.0, device=embeddings.device)

        return loss.mean()


class CombinedLoss(nn.Module):
    """
    Combined Cross-Entropy and Supervised Contrastive Loss.

    Total loss = CE_loss + lambda * SupCon_loss
    """

    def __init__(
        self,
        num_classes: int = 5,
        supcon_temperature: float = 0.07,
        supcon_lambda: float = 0.1,
        class_weights: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0,
    ):
        """
        Initialize combined loss.

        Args:
            num_classes: Number of emotion classes
            supcon_temperature: Temperature for SupCon loss
            supcon_lambda: Weight for SupCon loss
            class_weights: Optional class weights for CE
            label_smoothing: Label smoothing for CE
        """
        super().__init__()

        self.ce_loss = CrossEntropyLoss(
            num_classes=num_classes,
            class_weights=class_weights,
            label_smoothing=label_smoothing,
        )

        self.supcon_loss = SupervisedContrastiveLoss(
            temperature=supcon_temperature,
        )

        self.lambda_ = supcon_lambda

    def forward(
        self,
        logits: torch.Tensor,
        embeddings: torch.Tensor,
        targets: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.

        Args:
            logits: Model predictions [batch, num_classes]
            embeddings: Learned embeddings [batch, embedding_dim]
            targets: Class labels [batch]

        Returns:
            Dictionary with total_loss and component losses
        """
        ce_loss = self.ce_loss(logits, targets)
        supcon_loss = self.supcon_loss(embeddings, targets)

        total_loss = ce_loss + self.lambda_ * supcon_loss

        return {
            'total_loss': total_loss,
            'ce_loss': ce_loss,
            'supcon_loss': supcon_loss,
            'ce_loss_value': ce_loss.item(),
            'supcon_loss_value': supcon_loss.item(),
        }


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.

    Focuses on hard examples by down-weighting easy examples.
    """

    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = 'mean',
    ):
        """
        Initialize focal loss.

        Args:
            alpha: Class weights [num_classes]
            gamma: Focusing parameter
            reduction: Reduction method ('mean', 'sum', 'none')
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute focal loss.

        Args:
            logits: [batch, num_classes]
            targets: [batch]

        Returns:
            Loss value
        """
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingCrossEntropyLoss(nn.Module):
    """
    Cross-entropy loss with label smoothing.
    """

    def __init__(
        self,
        num_classes: int = 5,
        smoothing: float = 0.1,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute label smoothing cross-entropy loss.

        Args:
            logits: [batch, num_classes]
            targets: [batch]

        Returns:
            Loss value
        """
        return F.cross_entropy(logits, targets, label_smoothing=self.smoothing)


def create_loss_function(
    loss_type: str = 'ce',
    **kwargs,
) -> nn.Module:
    """
    Factory function to create loss functions.

    Args:
        loss_type: Type of loss ('ce', 'supcon', 'combined', 'focal')
        **kwargs: Additional arguments

    Returns:
        Loss function
    """
    if loss_type == 'ce':
        return CrossEntropyLoss(
            num_classes=kwargs.get('num_classes', 5),
            class_weights=kwargs.get('class_weights'),
            label_smoothing=kwargs.get('label_smoothing', 0.0),
        )
    elif loss_type == 'supcon':
        return SupervisedContrastiveLoss(
            temperature=kwargs.get('temperature', 0.07),
            lambda_=kwargs.get('lambda_', 0.1),
        )
    elif loss_type == 'combined':
        return CombinedLoss(
            num_classes=kwargs.get('num_classes', 5),
            supcon_temperature=kwargs.get('temperature', 0.07),
            supcon_lambda=kwargs.get('lambda_', 0.1),
            class_weights=kwargs.get('class_weights'),
        )
    elif loss_type == 'focal':
        return FocalLoss(
            alpha=kwargs.get('alpha'),
            gamma=kwargs.get('gamma', 2.0),
        )
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")