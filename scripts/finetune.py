#!/usr/bin/env python3
"""
BarkSSL Fine-tuning Script
Fine-tune pretrained encoder on DogEmotionSound dataset.
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.distributed as dist
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.canine_encoder import CanineEncoder
from src.models.classifier import EmotionClassifier
from src.data.preprocessed_dataset import PreprocessedEmotionDataset
from src.training.trainer import Trainer
from src.utils.logger import Logger, TensorBoardLogger


def parse_args():
    parser = argparse.ArgumentParser(description='BarkSSL Fine-tuning')

    # Checkpoint
    parser.add_argument('--encoder', type=str, required=True,
                       help='Path to pretrained encoder checkpoint')
    parser.add_argument('--config', type=str, default=None,
                       help='Model config (if encoder is plain state_dict)')

    # Model
    parser.add_argument('--num-classes', type=int, default=5,
                       help='Number of emotion classes')
    parser.add_argument('--pooling', type=str, default='attentive',
                       choices=['mean', 'max', 'attentive'],
                       help='Pooling type for classifier')

    # Data
    parser.add_argument('--data-dir', type=str,
                       default='data/emotion_preprocessed',
                       help='Emotion preprocessed data directory')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                       help='Training set ratio')
    parser.add_argument('--val-ratio', type=float, default=0.1,
                       help='Validation set ratio')

    # Training
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.01,
                       help='Weight decay')
    parser.add_argument('--gradient-clip', type=float, default=1.0,
                       help='Gradient clipping norm')

    # Output
    parser.add_argument('--output-dir', type=str,
                       default='outputs/finetune',
                       help='Output directory')

    # Misc
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint')

    return parser.parse_args()


def collate_fixed_length(batch):
    """Collate function for fixed-length emotion samples."""
    import numpy as np
    waveforms = torch.FloatTensor(np.array([b['waveform'] for b in batch]))
    labels = torch.LongTensor([b['label_id'] for b in batch])

    return {
        'waveforms': waveforms,
        'labels': labels,
    }


class EmotionFinetuneModel(nn.Module):
    """
    Emotion classifier that wraps encoder + classifier.
    """

    def __init__(self, encoder, classifier):
        super().__init__()
        self.encoder = encoder
        self.classifier = classifier

    def forward(self, waveforms):
        return self.classifier(waveforms)

    def train_step(self, batch):
        waveforms = batch['waveforms']
        labels = batch['labels']

        output = self.classifier(waveforms)
        logits = output['logits']

        loss = nn.CrossEntropyLoss()(logits, labels)

        preds = logits.argmax(dim=-1)
        acc = (preds == labels).float().mean()

        return {
            'loss': loss,
            'accuracy': acc,
            'logits': logits,
            'labels': labels,
        }

    def eval_step(self, batch):
        """Alias for train_step during evaluation."""
        return self.train_step(batch)

    def state_dict(self):
        """Return combined state dict."""
        return {
            'encoder': self.encoder.state_dict(),
            'classifier': self.classifier.state_dict(),
        }

    def load_state_dict(self, state_dict):
        self.encoder.load_state_dict(state_dict['encoder'])
        self.classifier.load_state_dict(state_dict['classifier'])


def main():
    args = parse_args()

    torch.manual_seed(args.seed)

    # Output dirs
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Logger
    logger = Logger('finetune', str(output_dir))
    logger.info("=" * 60)
    logger.info("BarkSSL Fine-tuning")
    logger.info("=" * 60)
    logger.info(f"Args: {args}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")

    # Load pretrained encoder
    logger.info(f"Loading encoder from: {args.encoder}")
    checkpoint = torch.load(args.encoder, map_location='cpu')

    if 'config' in checkpoint:
        encoder_config = checkpoint['config']
    else:
        encoder_config = {
            'scale': 'small',
            'hidden_dim': 384,
            'num_layers': 6,
            'num_heads': 6,
            'kmeans_k': 100,
        }

    encoder = CanineEncoder(
        scale=encoder_config.get('scale', 'small'),
        hidden_dim=encoder_config.get('hidden_dim', 384),
        num_layers=encoder_config.get('num_layers', 6),
        num_heads=encoder_config.get('num_heads', 6),
        kmeans_k=encoder_config.get('kmeans_k', 100),
    )

    if 'encoder_state_dict' in checkpoint:
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
    elif 'model_state_dict' in checkpoint:
        # Handle DDP-wrapped model
        state_dict = checkpoint['model_state_dict']
        if any(k.startswith('module.') for k in state_dict.keys()):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        encoder.load_state_dict(state_dict)

    encoder.to(device)
    logger.info(f"Encoder loaded: {sum(p.numel() for p in encoder.parameters()):,} parameters")

    # Create classifier
    classifier = EmotionClassifier(
        encoder=encoder,
        num_classes=args.num_classes,
        pooling_type=args.pooling,
    )
    classifier.to(device)

    # Create combined model
    model = EmotionFinetuneModel(encoder, classifier)

    # Load dataset
    logger.info(f"Loading emotion dataset from: {args.data_dir}")
    dataset = PreprocessedEmotionDataset(
        root_dir=args.data_dir,
        target_sr=16000,
        target_duration=4.0,
    )
    logger.info(f"Dataset size: {len(dataset)}")
    logger.info(f"Class distribution: {dataset.get_class_distribution()}")

    # Split dataset
    total = len(dataset)
    train_size = int(total * args.train_ratio)
    val_size = int(total * args.val_ratio)
    test_size = total - train_size - val_size

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fixed_length,
        pin_memory=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fixed_length,
        pin_memory=True,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fixed_length,
        pin_memory=True,
    )

    # TensorBoard
    tb_logger = TensorBoardLogger(str(output_dir / 'tensorboard'))
    logger.info(f"TensorBoard logging enabled: {output_dir / 'tensorboard'}")

    # Config
    config = {
        'learning_rate': args.lr,
        'weight_decay': args.weight_decay,
        'gradient_clip': args.gradient_clip,
        'checkpoint_dir': str(output_dir / 'checkpoints'),
        'finetune_epochs': args.epochs,
    }

    # Create trainer
    trainer = Trainer(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        device=device,
        config=config,
        logger=logger,
        tb_logger=tb_logger,
    )

    # Resume if specified
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)

    # Training loop
    logger.info("\nStarting fine-tuning...")
    best_acc = 0.0

    for epoch in range(args.epochs):
        trainer.epoch = epoch
        train_metrics = trainer.train_epoch()

        logger.info(f"Epoch {epoch}: train_loss={train_metrics['loss']:.4f}, train_acc={train_metrics.get('accuracy', 0):.4f}")

        # Validate
        if trainer.val_dataloader:
            val_metrics = trainer.validate()
            logger.info(f"  Val: loss={val_metrics.get('loss', 0):.4f}, acc={val_metrics.get('accuracy', 0):.4f}")

            # Track best
            val_acc = val_metrics.get('accuracy', 0)
            if val_acc > best_acc:
                best_acc = val_acc
                trainer.save_checkpoint('best_model.pt')
                logger.info(f"  New best! acc={best_acc:.4f}")

        # Save checkpoint
        if (epoch + 1) % 5 == 0:
            trainer.save_checkpoint(f'checkpoint_epoch_{epoch}.pt')

    # Save final model
    final_path = output_dir / 'checkpoints' / 'final_model.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'best_acc': best_acc,
        'config': encoder_config,
        'num_classes': args.num_classes,
    }, final_path)
    logger.info(f"\nFine-tuning complete! Model saved to {final_path}")
    logger.info(f"Best validation accuracy: {best_acc:.4f}")


if __name__ == '__main__':
    main()