"""
BarkSSL DataLoader Module
DataLoader creation and utilities.
"""

import torch
from torch.utils.data import DataLoader
from typing import Optional, Callable, Dict, Any
from .dataset import DogSpeakDataset, EmotionDataset, collate_fn


def create_dataloader(
    dataset: torch.utils.data.Dataset,
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = False,
    collate_fn: Optional[Callable] = None,
) -> DataLoader:
    """
    Create a DataLoader with standard settings.

    Args:
        dataset: Dataset to load from
        batch_size: Batch size
        shuffle: Whether to shuffle
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory for GPU transfer
        drop_last: Whether to drop last incomplete batch
        collate_fn: Optional collate function

    Returns:
        DataLoader instance
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=collate_fn,
    )


def create_dogspeak_dataloader(
    root_dir: str,
    metadata_path: Optional[str] = None,
    batch_size: int = 32,
    target_sr: int = 16000,
    target_duration: Optional[float] = None,
    num_workers: int = 4,
    max_samples: Optional[int] = None,
) -> DataLoader:
    """
    Create DataLoader for DogSpeak dataset.

    Args:
        root_dir: Path to dogspeak_released directory
        metadata_path: Path to metadata.csv
        batch_size: Batch size
        target_sr: Target sampling rate
        target_duration: Target duration in seconds
        num_workers: Number of workers
        max_samples: Maximum samples (for debugging)

    Returns:
        DataLoader for DogSpeak
    """
    dataset = DogSpeakDataset(
        root_dir=root_dir,
        metadata_path=metadata_path,
        target_sr=target_sr,
        target_duration=target_duration,
        max_samples=max_samples,
    )

    return create_dataloader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=None,  # DogSpeak uses variable length
    )


def create_emotion_dataloader(
    root_dir: str,
    batch_size: int = 16,
    target_sr: int = 16000,
    target_duration: float = 4.0,
    labels: Optional[list] = None,
    max_samples_per_label: Optional[Dict[str, int]] = None,
    num_workers: int = 4,
    shuffle: bool = True,
) -> DataLoader:
    """
    Create DataLoader for Emotion dataset.

    Args:
        root_dir: Path to emotion directory
        batch_size: Batch size
        target_sr: Target sampling rate
        target_duration: Target duration in seconds
        labels: Labels to include
        max_samples_per_label: Max samples per label
        num_workers: Number of workers
        shuffle: Whether to shuffle

    Returns:
        DataLoader for Emotion dataset
    """
    dataset = EmotionDataset(
        root_dir=root_dir,
        target_sr=target_sr,
        target_duration=target_duration,
        labels=labels,
        max_samples_per_label=max_samples_per_label,
    )

    return create_dataloader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )


def split_dataset(
    dataset: torch.utils.data.Dataset,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> tuple:
    """
    Split dataset into train/val/test sets.

    Args:
        dataset: Dataset to split
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        seed: Random seed

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"

    total_size = len(dataset)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size

    return torch.utils.data.random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed),
    )


def create_split_dataloaders(
    root_dir: str,
    batch_size: int = 16,
    target_sr: int = 16000,
    target_duration: float = 4.0,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    num_workers: int = 4,
    seed: int = 42,
) -> Dict[str, DataLoader]:
    """
    Create train/val/test dataloaders for Emotion dataset.

    Args:
        root_dir: Path to emotion directory
        batch_size: Batch size
        target_sr: Target sampling rate
        target_duration: Target duration
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        num_workers: Number of workers
        seed: Random seed

    Returns:
        Dictionary with 'train', 'val', 'test' dataloaders
    """
    # Create full dataset
    full_dataset = EmotionDataset(
        root_dir=root_dir,
        target_sr=target_sr,
        target_duration=target_duration,
    )

    # Split dataset
    train_dataset, val_dataset, test_dataset = split_dataset(
        full_dataset,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    # Create dataloaders
    dataloaders = {
        'train': create_dataloader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate_fn,
        ),
        'val': create_dataloader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
        ),
        'test': create_dataloader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
        ),
    }

    return dataloaders