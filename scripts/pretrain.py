#!/usr/bin/env python3
"""
BarkSSL Pretraining Script
Run self-supervised pretraining on DogSpeak dataset.

Usage with torchrun (recommended):
    torchrun --nproc_per_node=4 scripts/pretrain.py --epochs 100

Usage with python + rank:
    python -m torch.distributed.launch --nproc_per_node=4 scripts/pretrain.py --epochs 100
"""

import argparse
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.canine_encoder import CanineEncoder
from src.pretraining.acoustic_unit import AcousticUnitDiscovery, PseudoLabelGenerator
from src.pretraining.masked_pretrain import CanineHuBERTPretraining
from src.data.preprocessed_dataset import PreprocessedDogSpeakDataset
from src.training.trainer import PretrainingTrainer
from src.utils.logger import Logger
from src.utils.device import get_device


def parse_args():
    parser = argparse.ArgumentParser(description='BarkSSL Pretraining')

    # Model
    parser.add_argument('--scale', type=str, default='small',
                       choices=['tiny', 'small', 'base', 'large'],
                       help='Model scale')
    parser.add_argument('--hidden-dim', type=int, default=384,
                       help='Hidden dimension')
    parser.add_argument('--num-layers', type=int, default=6,
                       help='Number of transformer layers')
    parser.add_argument('--num-heads', type=int, default=6,
                       help='Number of attention heads')
    parser.add_argument('--kmeans-k', type=int, default=100,
                       help='Number of k-means clusters')

    # Masking
    parser.add_argument('--mask-prob', type=float, default=0.075,
                       help='Mask probability')
    parser.add_argument('--mask-span', type=int, default=10,
                       help='Mask span length')

    # Data
    parser.add_argument('--data-dir', type=str,
                       default='data/dogspeak_preprocessed',
                       help='DogSpeak preprocessed data directory')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Max samples for debugging (None for all)')

    # Training
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size per GPU')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loading workers per GPU')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of pretraining epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.01,
                       help='Weight decay')
    parser.add_argument('--gradient-clip', type=float, default=1.0,
                       help='Gradient clipping norm')
    parser.add_argument('--warmup-steps', type=int, default=1000,
                       help='Warmup steps for LR scheduler')

    # Output
    parser.add_argument('--output-dir', type=str,
                       default='outputs/pretrain',
                       help='Output directory')
    parser.add_argument('--checkpoint-dir', type=str,
                       default=None,
                       help='Checkpoint directory (default: output_dir/checkpoints)')
    parser.add_argument('--log-interval', type=int, default=100,
                       help='Logging interval')

    # Distributed training (auto-detected by torchrun)
    parser.add_argument('--local-rank', type=int, default=-1,
                       help='Local rank for distributed training')

    # Misc
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint')

    return parser.parse_args()


def collate_variable_length(batch):
    """Collate function for variable-length DogSpeak samples."""
    waveforms = [b['waveform'] for b in batch]
    paths = [b['path'] for b in batch]
    lengths = [len(w) for w in waveforms]

    return {
        'waveform': waveforms,
        'waveforms': waveforms,
        'path': paths,
        'length': lengths,
    }


def is_distributed():
    """Check if running in distributed mode."""
    return 'RANK' in os.environ and 'WORLD_SIZE' in os.environ


def get_rank_and_world_size():
    """Get rank and world size for distributed training."""
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    return rank, world_size, local_rank


def setup_distributed(rank, world_size):
    """Initialize distributed training."""
    os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'localhost')
    os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '12355')

    dist.init_process_group(
        backend='nccl' if torch.cuda.is_available() else 'gloo',
        init_method='env://',
        world_size=world_size,
        rank=rank,
    )
    dist.barrier()


def cleanup_distributed():
    """Cleanup distributed training."""
    dist.destroy_process_group()


def train_worker(rank, world_size, args, output_dir, checkpoint_dir):
    """Training worker function for distributed training."""

    # Setup distributed
    if world_size > 1:
        setup_distributed(rank, world_size)
        device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    is_main = rank == 0

    # Logger (only on main process)
    logger = Logger('pretrain', str(output_dir)) if is_main else None

    if is_main:
        logger.info("=" * 60)
        logger.info("BarkSSL Pretraining (Distributed)")
        logger.info(f"World size: {world_size}")
        logger.info("=" * 60)
        logger.info(f"Args: {args}")

    # Set seed
    torch.manual_seed(args.seed + rank)

    # ============ Stage 1: Acoustic Unit Discovery ============
    if is_main:
        logger.info("\n" + "=" * 40)
        logger.info("Stage 1: Acoustic Unit Discovery (k-means)")
        logger.info("=" * 40)

    acoustic_unit = AcousticUnitDiscovery(
        k=args.kmeans_k,
        sample_rate=16000,
        n_mels=80,
    )

    # Only main process fits k-means (others wait)
    if is_main:
        logger.info("Fitting k-means on DogSpeak data...")
        dataset_for_kmeans = PreprocessedDogSpeakDataset(
            root_dir=args.data_dir,
            target_sr=16000,
            max_samples=50000,
        )
        acoustic_unit.fit(dataset_for_kmeans)
        logger.info(f"k-means fitted with {args.kmeans_k} clusters")
        acoustic_unit.save(output_dir / 'acoustic_unit.pt')
    else:
        # Load from file on non-main ranks
        acoustic_unit = AcousticUnitDiscovery.load(str(output_dir / 'acoustic_unit.pt'))

    # Sync across processes
    if world_size > 1:
        dist.barrier()

    # ============ Stage 2: Masked Pretraining ============
    if is_main:
        logger.info("\n" + "=" * 40)
        logger.info("Stage 2: Masked Pretraining")
        logger.info("=" * 40)

    # Create model
    encoder = CanineEncoder(
        scale=args.scale,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        kmeans_k=args.kmeans_k,
    )
    encoder.to(device)

    # Wrap for distributed
    if world_size > 1:
        encoder = torch.nn.parallel.DistributedDataParallel(encoder, device_ids=[rank])

    if is_main:
        num_params = sum(p.numel() for p in encoder.parameters())
        logger.info(f"Model parameters: {num_params:,}")

    pretrain = CanineHuBERTPretraining(
        encoder=encoder,
        acoustic_unit_discovery=acoustic_unit,
        mask_prob=args.mask_prob,
        mask_span=args.mask_span,
    )

    # Create dataset with distributed sampler
    dataset = PreprocessedDogSpeakDataset(
        root_dir=args.data_dir,
        target_sr=16000,
        max_samples=args.max_samples,
    )

    sampler = torch.utils.data.DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
    ) if world_size > 1 else None

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=collate_variable_length,
        pin_memory=True,
    )

    if is_main:
        logger.info(f"Dataset size: {len(dataset)}")
        logger.info(f"Batch size per GPU: {args.batch_size}")
        logger.info(f"Effective batch size: {args.batch_size * world_size}")
        logger.info(f"Steps per epoch: {len(dataloader)}")

    # Create trainer
    config = {
        'learning_rate': args.lr,
        'weight_decay': args.weight_decay,
        'gradient_clip': args.gradient_clip,
        'checkpoint_dir': str(checkpoint_dir),
        'finetune_epochs': args.epochs,
    }

    trainer = PretrainingTrainer(
        model=pretrain,
        train_dataloader=dataloader,
        device=device,
        config=config,
        logger=logger,
    )

    # Training loop
    if is_main:
        logger.info("\nStarting pretraining...")

    for epoch in range(args.epochs):
        if sampler:
            sampler.set_epoch(epoch)

        trainer.epoch = epoch
        metrics = trainer.train_epoch()

        if is_main:
            logger.info(f"Epoch {epoch}: loss={metrics['loss']:.4f}, acc={metrics.get('accuracy', 0):.4f}")

            # Save checkpoint
            if (epoch + 1) % 10 == 0:
                trainer.save_checkpoint(f'checkpoint_epoch_{epoch}.pt')

    # Save final model
    if is_main:
        encoder_path = checkpoint_dir / 'final_encoder.pt'
        torch.save({
            'encoder_state_dict': encoder.module.state_dict() if hasattr(encoder, 'module') else encoder.state_dict(),
            'config': {
                'scale': args.scale,
                'hidden_dim': args.hidden_dim,
                'num_layers': args.num_layers,
                'num_heads': args.num_heads,
                'kmeans_k': args.kmeans_k,
            },
        }, encoder_path)
        logger.info(f"\nPretraining complete! Model saved to {encoder_path}")

    # Cleanup
    if world_size > 1:
        cleanup_distributed()


def main():
    args = parse_args()

    # Check distributed mode
    if is_distributed():
        rank, world_size, local_rank = get_rank_and_world_size()
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else output_dir / 'checkpoints'
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        train_worker(rank, world_size, args, output_dir, checkpoint_dir)
    else:
        # Single GPU / CPU mode
        torch.manual_seed(args.seed)

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else output_dir / 'checkpoints'
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Logger
        logger = Logger('pretrain', str(output_dir))
        logger.info("=" * 60)
        logger.info("BarkSSL Pretraining (Single GPU/CPU)")
        logger.info("=" * 60)
        logger.info(f"Args: {args}")

        device = get_device()
        logger.info(f"Device: {device}")

        # ============ Stage 1: Acoustic Unit Discovery ============
        logger.info("\n" + "=" * 40)
        logger.info("Stage 1: Acoustic Unit Discovery (k-means)")
        logger.info("=" * 40)

        acoustic_unit = AcousticUnitDiscovery(
            k=args.kmeans_k,
            sample_rate=16000,
            n_mels=80,
        )

        logger.info("Fitting k-means on DogSpeak data...")
        dataset_for_kmeans = PreprocessedDogSpeakDataset(
            root_dir=args.data_dir,
            target_sr=16000,
            max_samples=50000,
        )
        acoustic_unit.fit(dataset_for_kmeans)
        logger.info(f"k-means fitted with {args.kmeans_k} clusters")
        acoustic_unit.save(output_dir / 'acoustic_unit.pt')

        # ============ Stage 2: Masked Pretraining ============
        logger.info("\n" + "=" * 40)
        logger.info("Stage 2: Masked Pretraining")
        logger.info("=" * 40)

        encoder = CanineEncoder(
            scale=args.scale,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            kmeans_k=args.kmeans_k,
        )
        encoder.to(device)

        num_params = sum(p.numel() for p in encoder.parameters())
        logger.info(f"Model parameters: {num_params:,}")

        pretrain = CanineHuBERTPretraining(
            encoder=encoder,
            acoustic_unit_discovery=acoustic_unit,
            mask_prob=args.mask_prob,
            mask_span=args.mask_span,
        )

        dataset = PreprocessedDogSpeakDataset(
            root_dir=args.data_dir,
            target_sr=16000,
            max_samples=args.max_samples,
        )

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=collate_variable_length,
            pin_memory=True,
        )

        logger.info(f"Dataset size: {len(dataset)}")
        logger.info(f"Batch size: {args.batch_size}")
        logger.info(f"Steps per epoch: {len(dataloader)}")

        config = {
            'learning_rate': args.lr,
            'weight_decay': args.weight_decay,
            'gradient_clip': args.gradient_clip,
            'checkpoint_dir': str(checkpoint_dir),
            'finetune_epochs': args.epochs,
        }

        trainer = PretrainingTrainer(
            model=pretrain,
            train_dataloader=dataloader,
            device=device,
            config=config,
            logger=logger,
        )

        if args.resume:
            logger.info(f"Resuming from checkpoint: {args.resume}")
            trainer.load_checkpoint(args.resume)

        logger.info("\nStarting pretraining...")
        for epoch in range(args.epochs):
            trainer.epoch = epoch
            metrics = trainer.train_epoch()
            logger.info(f"Epoch {epoch}: loss={metrics['loss']:.4f}, acc={metrics.get('accuracy', 0):.4f}")

            if (epoch + 1) % 10 == 0:
                trainer.save_checkpoint(f'checkpoint_epoch_{epoch}.pt')

        encoder_path = checkpoint_dir / 'final_encoder.pt'
        torch.save({
            'encoder_state_dict': encoder.state_dict(),
            'config': {
                'scale': args.scale,
                'hidden_dim': args.hidden_dim,
                'num_layers': args.num_layers,
                'num_heads': args.num_heads,
                'kmeans_k': args.kmeans_k,
            },
        }, encoder_path)
        logger.info(f"\nPretraining complete! Model saved to {encoder_path}")


if __name__ == '__main__':
    main()