"""
BarkSSL Preprocessed Dataset Module
Dataset classes for preprocessed DogSpeak and Emotion data.
"""

import numpy as np
import soundfile as sf
import torch
from pathlib import Path
from typing import Optional, List, Dict, Any
from torch.utils.data import Dataset


class PreprocessedDataset(Dataset):
    """
    Base class for preprocessed audio datasets.

    Assumes audio files are already:
    - Resampled to target_sr
    - Converted to mono
    - Aligned to CNN frame stride
    """

    def __init__(
        self,
        root_dir: str,
        target_sr: int = 16000,
        target_duration: Optional[float] = None,
        transform: Optional[callable] = None,
    ):
        """
        Initialize preprocessed dataset.

        Args:
            root_dir: Root directory containing preprocessed audio
            target_sr: Expected sample rate
            target_duration: Expected duration (for fixed-length datasets)
            transform: Optional transform
        """
        self.root_dir = Path(root_dir)
        self.target_sr = target_sr
        self.target_duration = target_duration
        self.target_samples = int(target_sr * target_duration) if target_duration else None
        self.transform = transform
        self.file_paths = []
        self.labels = []  # For base class compatibility
        self.labels = []

    def load_audio(self, filepath: str) -> np.ndarray:
        """
        Load preprocessed audio file.

        Args:
            filepath: Path to audio file

        Returns:
            waveform: Audio array
        """
        waveform, sr = sf.read(filepath)
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)
        return waveform

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        raise NotImplementedError


class PreprocessedEmotionDataset(PreprocessedDataset):
    """
    Preprocessed Emotion dataset with fixed-length audio.

    All files are expected to be 4 seconds (64000 samples @ 16kHz).
    """

    EMOTION_LABELS = ['angry', 'anxious', 'happy', 'lonely', 'sad']
    EMOTION_TO_ID = {label: idx for idx, label in enumerate(EMOTION_LABELS)}

    def __init__(
        self,
        root_dir: str,
        target_sr: int = 16000,
        target_duration: float = 4.0,
        labels: Optional[List[str]] = None,
        transform: Optional[callable] = None,
    ):
        """
        Initialize preprocessed emotion dataset.

        Args:
            root_dir: Path to preprocessed emotion directory
            target_sr: Sample rate
            target_duration: Fixed duration in seconds
            labels: Labels to include
            transform: Optional transform
        """
        super().__init__(
            root_dir=root_dir,
            target_sr=target_sr,
            target_duration=target_duration,
            transform=transform,
        )

        self.labels_filter = labels or self.EMOTION_LABELS
        self.emotion_labels = []
        self._build_file_list()

    def _build_file_list(self) -> None:
        """Build file list from preprocessed directory structure."""
        for label in self.labels_filter:
            label_dir = self.root_dir / label
            if not label_dir.exists():
                continue

            for filepath in sorted(label_dir.glob('*.wav')):
                self.file_paths.append(filepath)
                self.emotion_labels.append(label)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get audio sample.

        Returns:
            Dictionary with waveform, label, etc.
        """
        filepath = self.file_paths[idx]
        label = self.emotion_labels[idx]

        waveform = self.load_audio(str(filepath))

        if self.transform:
            waveform = self.transform(waveform)

        return {
            'waveform': waveform,
            'waveforms': waveform,
            'label': label,
            'label_id': self.EMOTION_TO_ID[label],
            'labels': self.EMOTION_TO_ID[label],
            'path': str(filepath),
        }

    def get_class_distribution(self) -> Dict[str, int]:
        """Get class distribution."""
        from collections import Counter
        return dict(Counter(self.emotion_labels))


class PreprocessedDogSpeakDataset(PreprocessedDataset):
    """
    Preprocessed DogSpeak dataset with variable-length audio.

    Used for self-supervised pretraining.
    """

    def __init__(
        self,
        root_dir: str,
        target_sr: int = 16000,
        transform: Optional[callable] = None,
        max_samples: Optional[int] = None,
        metadata_path: Optional[str] = None,
    ):
        """
        Initialize preprocessed DogSpeak dataset.

        Args:
            root_dir: Path to preprocessed dogspeak directory
            target_sr: Sample rate
            transform: Optional transform
            max_samples: Max number of samples (for debugging)
            metadata_path: Optional path to metadata.csv
        """
        super().__init__(
            root_dir=root_dir,
            target_sr=target_sr,
            target_duration=None,  # Variable length
            transform=transform,
        )

        self.metadata = None
        if metadata_path and Path(metadata_path).exists():
            import pandas as pd
            self.metadata = pd.read_csv(metadata_path)

        self._build_file_list(max_samples)

    def _build_file_list(self, max_samples: Optional[int] = None) -> None:
        """Build file list from dog_* subdirectories."""
        for dog_dir in sorted(self.root_dir.iterdir()):
            if not dog_dir.is_dir() or not dog_dir.name.startswith('dog_'):
                continue

            for filepath in sorted(dog_dir.glob('*.wav')):
                self.file_paths.append(filepath)

                if max_samples and len(self.file_paths) >= max_samples:
                    return

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get audio sample.

        Returns:
            Dictionary with variable-length waveform.
        """
        filepath = self.file_paths[idx]
        waveform = self.load_audio(str(filepath))

        if self.transform:
            waveform = self.transform(waveform)

        return {
            'waveform': waveform,
            'path': str(filepath),
            'length': len(waveform),
        }


def create_preprocessed_dataloaders(
    emotion_dir: str = 'data/emotion_preprocessed',
    dogspeak_dir: str = 'data/dogspeak_preprocessed',
    target_sr: int = 16000,
    target_duration: float = 4.0,
    batch_size: int = 16,
    num_workers: int = 4,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> Dict[str, torch.utils.data.DataLoader]:
    """
    Create dataloaders for preprocessed datasets.

    Args:
        emotion_dir: Path to preprocessed emotion directory
        dogspeak_dir: Path to preprocessed dogspeak directory
        target_sr: Sample rate
        target_duration: Fixed duration for emotion dataset
        batch_size: Batch size
        num_workers: Number of workers
        train_ratio: Training set ratio
        val_ratio: Validation set ratio

    Returns:
        Dictionary with train/val/test dataloaders for emotion,
        and train dataloader for dogspeak.
    """
    from torch.utils.data import DataLoader, random_split

    # Create emotion dataset
    emotion_dataset = PreprocessedEmotionDataset(
        root_dir=emotion_dir,
        target_sr=target_sr,
        target_duration=target_duration,
    )

    # Split emotion dataset
    total = len(emotion_dataset)
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)
    test_size = total - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        emotion_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42),
    )

    def collate_fn(batch):
        """Collate function for emotion dataset."""
        waveforms = []
        labels = []
        for b in batch:
            waveforms.append(b['waveform'])
            labels.append(b['label_id'])

        return {
            'waveforms': torch.FloatTensor(np.array(waveforms)),
            'labels': torch.LongTensor(labels),
        }

    def collate_fn_dogspeak(batch):
        """Collate function for DogSpeak dataset (variable length)."""
        # Find max length and pad
        waveforms = []
        for b in batch:
            waveforms.append(torch.from_numpy(b['waveform']).float())

        # Stack without padding for now (variable length)
        return {
            'waveform': waveforms,  # List of tensors
            'length': [len(w) for w in waveforms],
        }

    dataloaders = {
        'emotion_train': DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate_fn,
        ),
        'emotion_val': DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
        ),
        'emotion_test': DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
        ),
    }

    # Create dogspeak dataloader (for pretraining)
    dogspeak_dataset = PreprocessedDogSpeakDataset(
        root_dir=dogspeak_dir,
        target_sr=target_sr,
    )

    dataloaders['dogspeak'] = DataLoader(
        dogspeak_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn_dogspeak,
    )

    print(f"Dataset sizes:")
    print(f"  Emotion train: {len(train_dataset)}")
    print(f"  Emotion val: {len(val_dataset)}")
    print(f"  Emotion test: {len(test_dataset)}")
    print(f"  DogSpeak: {len(dogspeak_dataset)}")

    return dataloaders