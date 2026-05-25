"""
BarkSSL Trainer Module
Training loop with logging, checkpointing, and evaluation.
"""

import os
import time
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Dict, Any, Optional, Callable
from pathlib import Path
from tqdm import tqdm

from ..utils.logger import Logger, TensorBoardLogger
from ..utils.device import get_device
from .evaluator import Evaluator


class Trainer:
    """
    Generic trainer for BarkSSL models.

    Handles training loop, validation, checkpointing, and logging.
    """

    def __init__(
        self,
        model: nn.Module,
        train_dataloader: torch.utils.data.DataLoader,
        val_dataloader: Optional[torch.utils.data.DataLoader] = None,
        loss_fn: Optional[nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: Optional[torch.device] = None,
        config: Optional[Dict[str, Any]] = None,
        logger: Optional[Logger] = None,
        tb_logger: Optional[TensorBoardLogger] = None,
    ):
        """
        Initialize trainer.

        Args:
            model: Model to train
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            loss_fn: Loss function
            optimizer: Optimizer
            device: Device to train on
            config: Configuration dictionary
            logger: Logger for console/file output
            tb_logger: TensorBoard logger
        """
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.loss_fn = loss_fn
        self.device = device or get_device()
        self.config = config or {}
        self.logger = logger
        self.tb_logger = tb_logger

        # Move model to device
        self.model.to(self.device)

        # Optimizer
        if optimizer is None:
            lr = self.config.get('learning_rate', 1e-4)
            weight_decay = self.config.get('weight_decay', 0.01)
            self.optimizer = AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
            )
        else:
            self.optimizer = optimizer

        # Learning rate scheduler
        scheduler_type = self.config.get('scheduler', 'cosine')
        if scheduler_type == 'cosine':
            max_epochs = self.config.get('finetune_epochs', 30)
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=max_epochs * len(train_dataloader),
            )
        else:
            self.scheduler = None

        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_metric = 0.0

        # Paths
        self.checkpoint_dir = Path(self.config.get('checkpoint_dir', 'checkpoints'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.

        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        metrics = {}

        pbar = tqdm(self.train_dataloader, desc=f"Epoch {self.epoch}")

        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}

            # Forward pass
            self.optimizer.zero_grad()
            output = self.model.train_step(batch)

            # Debug: check output keys
            if self.global_step == 0:
                print(f"DEBUG output keys: {output.keys()}")

            # Compute loss
            if self.loss_fn:
                loss = self.loss_fn(**{k: output[k] for k in ['logits', 'embedding', 'labels']
                                     if k in output and k in ['logits', 'embedding'] or k == 'labels'})
                if isinstance(loss, dict):
                    train_loss = loss.get('total_loss', loss.get('ce_loss', output.get('loss', 0)))
                else:
                    train_loss = loss
            else:
                train_loss = output.get('loss', 0)

            # Backward pass
            if isinstance(train_loss, torch.Tensor):
                train_loss.backward()

                # Gradient clipping
                max_grad = self.config.get('gradient_clip', 1.0)
                if max_grad > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad)

                self.optimizer.step()

                if self.scheduler:
                    self.scheduler.step()

            # Metrics
            batch_size = batch['labels'].size(0) if 'labels' in batch else len(batch)
            total_loss += train_loss.item() if isinstance(train_loss, torch.Tensor) else train_loss

            # Get accuracy from output if available
            if 'accuracy' in output:
                train_acc = output['accuracy'].item() if isinstance(output['accuracy'], torch.Tensor) else output['accuracy']
            else:
                train_acc = 0.0

            total_samples += batch_size

            # Update progress bar
            avg_loss = total_loss / max(total_samples, 1)
            pbar.set_postfix({'loss': f'{avg_loss:.4f}', 'acc': f'{train_acc:.4f}'})

            # Log to TensorBoard
            if self.tb_logger:
                self.tb_logger.log_scalar('loss/train', avg_loss, self.global_step)
                if train_acc > 0:
                    self.tb_logger.log_scalar('acc/train', train_acc, self.global_step)

            self.global_step += 1

        metrics['loss'] = total_loss / max(total_samples, 1)
        metrics['accuracy'] = train_acc if 'accuracy' in locals() else 0.0

        return metrics

    def validate(self) -> Dict[str, float]:
        """
        Validate the model.

        Returns:
            Dictionary with validation metrics
        """
        if self.val_dataloader is None:
            return {}

        evaluator = Evaluator(self.model, self.device)
        metrics = evaluator.evaluate(self.val_dataloader)

        return metrics

    def train(
        self,
        num_epochs: int,
        eval_every: int = 1,
        save_every: int = 5,
    ) -> Dict[str, Any]:
        """
        Train for multiple epochs.

        Args:
            num_epochs: Number of epochs to train
            eval_every: Evaluate every N epochs
            save_every: Save checkpoint every N epochs

        Returns:
            Dictionary with training history
        """
        history = {
            'train_loss': [],
            'val_metrics': [],
        }

        for epoch in range(num_epochs):
            self.epoch = epoch

            # Train
            train_metrics = self.train_epoch()
            history['train_loss'].append(train_metrics['loss'])

            if self.logger:
                self.logger.info(f"Epoch {epoch}: train_loss={train_metrics['loss']:.4f}")

            # Validate
            if epoch % eval_every == 0 and self.val_dataloader:
                val_metrics = self.validate()
                history['val_metrics'].append(val_metrics)

                if self.logger:
                    self.logger.info(f"Epoch {epoch}: val_metrics={val_metrics}")

                # Check if best model
                current_metric = val_metrics.get('accuracy', val_metrics.get('macro_f1', 0))
                if current_metric > self.best_metric:
                    self.best_metric = current_metric
                    self.save_checkpoint('best_model.pt')
                    if self.logger:
                        self.logger.info(f"New best model! accuracy={current_metric:.4f}")

            # Save checkpoint
            if epoch % save_every == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch}.pt')

        return history

    def save_checkpoint(self, filename: str) -> str:
        """
        Save model checkpoint.

        Args:
            filename: Checkpoint filename

        Returns:
            Path to saved checkpoint
        """
        path = self.checkpoint_dir / filename

        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_metric': self.best_metric,
        }

        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        torch.save(checkpoint, path)

        return str(path)

    def load_checkpoint(self, path: str) -> None:
        """
        Load model checkpoint.

        Args:
            path: Path to checkpoint
        """
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_metric = checkpoint.get('best_metric', 0)

        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])


class PretrainingTrainer(Trainer):
    """
    Trainer specialized for self-supervised pretraining.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch with masked prediction."""
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        total_acc = 0.0

        pbar = tqdm(self.train_dataloader, desc=f"Pretrain Epoch {self.epoch}")

        for batch in pbar:
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}

            # Forward pass
            self.optimizer.zero_grad()
            output = self.model.train_step(batch)

            loss = output['loss']

            # Backward pass
            loss.backward()

            # Gradient clipping
            max_grad = self.config.get('gradient_clip', 1.0)
            if max_grad > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad)

            self.optimizer.step()

            if self.scheduler:
                self.scheduler.step()

            # Metrics
            metrics = output.get('metrics', {})
            total_loss += metrics.get('loss', 0)
            total_acc += metrics.get('accuracy', 0)
            total_samples += 1

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{metrics.get("loss", 0):.4f}',
                'acc': f'{metrics.get("accuracy", 0):.4f}',
            })

            if self.tb_logger:
                self.tb_logger.log_scalar('loss/pretrain', metrics.get('loss', 0), self.global_step)
                self.tb_logger.log_scalar('acc/pretrain', metrics.get('accuracy', 0), self.global_step)

            self.global_step += 1

        metrics = {
            'loss': total_loss / max(total_samples, 1),
            'accuracy': total_acc / max(total_samples, 1),
        }

        return metrics


def create_trainer(
    model: nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    config: Dict[str, Any],
    loss_fn: Optional[nn.Module] = None,
    val_dataloader: Optional[torch.utils.data.DataLoader] = None,
    logger: Optional[Logger] = None,
) -> Trainer:
    """
    Factory function to create trainer.

    Args:
        model: Model to train
        train_dataloader: Training data loader
        config: Configuration dictionary
        loss_fn: Loss function
        val_dataloader: Validation data loader
        logger: Logger

    Returns:
        Trainer instance
    """
    device = get_device(cuda_if_available=config.get('cuda_if_available', True))

    return Trainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        loss_fn=loss_fn,
        device=device,
        config=config,
        logger=logger,
    )