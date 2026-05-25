"""
BarkSSL Evaluator Module
Model evaluation with metrics computation.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, List
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)
from tqdm import tqdm


class Evaluator:
    """
    Model evaluator with comprehensive metrics.

    Computes accuracy, F1, precision, recall, confusion matrix.
    """

    def __init__(
        self,
        model: nn.Module,
        device: Optional[torch.device] = None,
        num_classes: int = 5,
        class_names: Optional[List[str]] = None,
    ):
        """
        Initialize evaluator.

        Args:
            model: Model to evaluate
            device: Device to run on
            num_classes: Number of classes
            class_names: List of class names
        """
        self.model = model
        self.device = device or torch.device('cpu')
        self.num_classes = num_classes
        self.class_names = class_names or ['angry', 'anxious', 'happy', 'lonely', 'sad']

    @torch.no_grad()
    def predict_batch(self, batch: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """
        Predict on a batch.

        Args:
            batch: Input batch

        Returns:
            Dictionary with predictions and labels
        """
        # Move batch to device
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()}

        # Forward pass
        output = self.model.eval_step(batch)

        logits = output['logits']
        predictions = logits.argmax(dim=-1).cpu().numpy()
        labels = batch['labels'].cpu().numpy()

        return {
            'predictions': predictions,
            'labels': labels,
            'logits': logits.cpu().numpy(),
        }

    def evaluate(
        self,
        dataloader: torch.utils.data.DataLoader,
    ) -> Dict[str, float]:
        """
        Evaluate model on entire dataset.

        Args:
            dataloader: Data loader

        Returns:
            Dictionary with metrics
        """
        self.model.eval()

        all_predictions = []
        all_labels = []
        all_logits = []

        for batch in tqdm(dataloader, desc="Evaluating"):
            result = self.predict_batch(batch)

            all_predictions.extend(result['predictions'])
            all_labels.extend(result['labels'])
            all_logits.append(result['logits'])

        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        all_logits = np.vstack(all_logits)

        # Compute metrics
        metrics = self.compute_metrics(all_predictions, all_labels)

        # Compute loss using cross entropy on logits
        logits_tensor = torch.FloatTensor(all_logits)
        labels_tensor = torch.LongTensor(all_labels)
        loss = nn.CrossEntropyLoss()(logits_tensor, labels_tensor).item()
        metrics['loss'] = loss

        return metrics

    def compute_metrics(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
    ) -> Dict[str, float]:
        """
        Compute metrics from predictions and labels.

        Args:
            predictions: Predicted labels
            labels: Ground truth labels

        Returns:
            Dictionary with metrics
        """
        metrics = {}

        # Basic accuracy
        metrics['accuracy'] = accuracy_score(labels, predictions)

        # F1 scores
        metrics['macro_f1'] = f1_score(labels, predictions, average='macro', zero_division=0)
        metrics['micro_f1'] = f1_score(labels, predictions, average='micro', zero_division=0)
        metrics['weighted_f1'] = f1_score(labels, predictions, average='weighted', zero_division=0)

        # Precision and recall
        metrics['macro_precision'] = precision_score(labels, predictions, average='macro', zero_division=0)
        metrics['macro_recall'] = recall_score(labels, predictions, average='macro', zero_division=0)

        # Per-class metrics
        per_class_f1 = f1_score(labels, predictions, average=None, zero_division=0)
        per_class_precision = precision_score(labels, predictions, average=None, zero_division=0)
        per_class_recall = recall_score(labels, predictions, average=None, zero_division=0)

        for i, class_name in enumerate(self.class_names):
            metrics[f'{class_name}_f1'] = per_class_f1[i]
            metrics[f'{class_name}_precision'] = per_class_precision[i]
            metrics[f'{class_name}_recall'] = per_class_recall[i]

        return metrics

    def compute_confusion_matrix(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
    ) -> np.ndarray:
        """
        Compute confusion matrix.

        Args:
            predictions: Predicted labels
            labels: Ground truth labels

        Returns:
            Confusion matrix
        """
        return confusion_matrix(labels, predictions)

    def print_report(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
    ) -> str:
        """
        Print classification report.

        Args:
            predictions: Predicted labels
            labels: Ground truth labels

        Returns:
            Report string
        """
        return classification_report(
            labels,
            predictions,
            target_names=self.class_names,
            zero_division=0,
        )


class PretrainingEvaluator:
    """
    Evaluator for self-supervised pretraining.
    """

    def __init__(
        self,
        model: Any,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize pretraining evaluator.

        Args:
            model: Pretraining model
            device: Device to run on
        """
        self.model = model
        self.device = device or torch.device('cpu')

    @torch.no_grad()
    def evaluate(
        self,
        dataloader: torch.utils.data.DataLoader,
    ) -> Dict[str, float]:
        """
        Evaluate pretraining model.

        Args:
            dataloader: Data loader

        Returns:
            Dictionary with metrics
        """
        self.model.eval()

        total_loss = 0.0
        total_acc = 0.0
        total_samples = 0

        for batch in tqdm(dataloader, desc="Evaluating pretraining"):
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}

            output = self.model.train_step(batch)

            metrics = output.get('metrics', {})
            total_loss += metrics.get('loss', 0)
            total_acc += metrics.get('accuracy', 0)
            total_samples += 1

        return {
            'loss': total_loss / max(total_samples, 1),
            'accuracy': total_acc / max(total_samples, 1),
        }


def create_evaluator(
    model: nn.Module,
    config: Dict[str, Any],
    **kwargs,
) -> Evaluator:
    """
    Factory function to create evaluator.

    Args:
        model: Model to evaluate
        config: Configuration dictionary
        **kwargs: Additional arguments

    Returns:
        Evaluator instance
    """
    num_classes = config.get('num_classes', 5)

    return Evaluator(
        model=model,
        num_classes=num_classes,
        class_names=config.get('emotion_labels', None),
        **kwargs,
    )