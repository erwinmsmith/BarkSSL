"""
BarkSSL Dataset Module
Dataset classes for DogSpeak and Emotion datasets.
"""

import os
import numpy as np
import pandas as pd
import torch
import torchaudio
import librosa
from pathlib import Path
from typing import Optional, List, Tuple, Callable, Dict, Any
from torch.utils.data import Dataset


class BaseAudioDataset(Dataset):
    """
    Base class for audio datasets.

    Provides common functionality for loading, resampling, and normalizing audio.
    """

    def __init__(
        self,
        root_dir: str,
        target_sr: int = 16000,
        target_duration: Optional[float] = None,
        transform: Optional[Callable] = None,
    ):
        """
        Initialize base audio dataset.

        Args:
            root_dir: Root directory containing audio files
            target_sr: Target sampling rate
            target_duration: Target duration in seconds (None = variable length)
            transform: Optional transform to apply to audio
        """
        self.root_dir = Path(root_dir)
        self.target_sr = target_sr
        self.target_duration = target_duration
        self.target_samples = int(target_sr * target_duration) if target_duration else None
        self.transform = transform

    def load_audio(self, filepath: str) -> Tuple[np.ndarray, int]:
        """
        Load audio file with resampling.

        Args:
            filepath: Path to audio file

        Returns:
            Tuple of (waveform, sample_rate)
        """
        try:
            # Try librosa first for reliable mono conversion
            waveform, sr = librosa.load(
                filepath,
                sr=self.target_sr,
                mono=True
            )
            return waveform, sr

        except Exception as e:
            # Fallback to torchaudio
            try:
                waveform, sr = torchaudio.load(filepath)
                # Convert to numpy and handle multi-channel
                waveform = waveform.numpy()
                if waveform.ndim > 1:
                    # Average channels to mono
                    waveform = waveform.mean(axis=0)
                else:
                    waveform = waveform.squeeze()

                # Resample if necessary
                if sr != self.target_sr:
                    waveform = librosa.resample(
                        waveform,
                        orig_sr=sr,
                        target_sr=self.target_sr
                    )

                return waveform, self.target_sr

            except Exception as e2:
                raise RuntimeError(f"Failed to load audio {filepath}: {e2}")

    def normalize_audio(self, waveform: np.ndarray) -> np.ndarray:
        """
        Normalize audio waveform to [-1, 1] range.

        Args:
            waveform: Input waveform

        Returns:
            Normalized waveform
        """
        max_val = np.abs(waveform).max()
        if max_val > 0:
            waveform = waveform / max_val
        return waveform

    def pad_or_truncate(
        self,
        waveform: np.ndarray,
        target_length: Optional[int] = None
    ) -> np.ndarray:
        """
        Pad or truncate waveform to target length.

        Args:
            waveform: Input waveform
            target_length: Target length in samples

        Returns:
            Processed waveform
        """
        if target_length is None:
            target_length = self.target_samples

        if target_length is None:
            return waveform

        current_length = len(waveform)

        if current_length > target_length:
            # Truncate
            return waveform[:target_length]
        elif current_length < target_length:
            # Pad with zeros
            padding = target_length - current_length
            return np.pad(waveform, (0, padding), mode='constant')
        else:
            return waveform

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        raise NotImplementedError


class DogSpeakDataset(BaseAudioDataset):
    """
    Dataset for DogSpeak unlabeled canine vocalization corpus.

    DogSpeak contains ~77,202 bark sequences from 156 dogs.
    This dataset is used for self-supervised pretraining.

    Note: metadata (dog_id, breed, sex) is NOT used as supervision.
    """

    def __init__(
        self,
        root_dir: str,
        metadata_path: Optional[str] = None,
        target_sr: int = 16000,
        target_duration: Optional[float] = None,
        transform: Optional[Callable] = None,
        max_samples: Optional[int] = None,
    ):
        """
        Initialize DogSpeak dataset.

        Args:
            root_dir: Path to dogspeak_released directory
            metadata_path: Optional path to metadata.csv
            target_sr: Target sampling rate
            target_duration: Target duration in seconds
            transform: Optional audio transform
            max_samples: Maximum number of samples to use (for debugging)
        """
        super().__init__(root_dir, target_sr, target_duration, transform)

        self.metadata = None
        self.file_paths = []

        # Load metadata if available
        if metadata_path and os.path.exists(metadata_path):
            self.metadata = pd.read_csv(metadata_path)

        # Build file list
        self._build_file_list(max_samples)

    def _build_file_list(self, max_samples: Optional[int] = None) -> None:
        """Build list of audio file paths."""
        for dog_dir in sorted(self.root_dir.iterdir()):
            if dog_dir.is_dir() and dog_dir.name.startswith('dog_'):
                for audio_file in sorted(dog_dir.glob('*.wav')):
                    self.file_paths.append(audio_file)

                    if max_samples and len(self.file_paths) >= max_samples:
                        return

        if not self.file_paths:
            raise ValueError(f"No audio files found in {self.root_dir}")

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get audio sample.

        Returns:
            Dictionary with:
                - waveform: Audio waveform as numpy array
                - path: File path
                - length: Actual audio length in samples
        """
        filepath = self.file_paths[idx]

        try:
            waveform, sr = self.load_audio(str(filepath))
            waveform = self.normalize_audio(waveform)

            if self.transform:
                waveform = self.transform(waveform)

            return {
                'waveform': waveform,
                'path': str(filepath),
                'length': len(waveform),
            }

        except Exception as e:
            # Return silent audio on error
            target_len = self.target_samples or 16000
            return {
                'waveform': np.zeros(target_len),
                'path': str(filepath),
                'length': target_len,
                'error': str(e),
            }


class EmotionDataset(BaseAudioDataset):
    """
    Dataset for DogEmotionSound emotion classification.

    Contains 5 emotion categories:
    - angry, anxious, happy, lonely, sad

    Total: 4,229 samples
    """

    EMOTION_LABELS = ['angry', 'anxious', 'happy', 'lonely', 'sad']
    EMOTION_TO_ID = {label: idx for idx, label in enumerate(EMOTION_LABELS)}
    ID_TO_EMOTION = {idx: label for idx, label in enumerate(EMOTION_LABELS)}

    def __init__(
        self,
        root_dir: str,
        target_sr: int = 16000,
        target_duration: Optional[float] = 4.0,
        transform: Optional[Callable] = None,
        labels: Optional[List[str]] = None,
        max_samples_per_label: Optional[Dict[str, int]] = None,
    ):
        """
        Initialize Emotion dataset.

        Args:
            root_dir: Path to emotion directory containing label subdirectories
            target_sr: Target sampling rate
            target_duration: Target duration in seconds (default: 4.0)
            transform: Optional audio transform
            labels: List of labels to include (default: all 5)
            max_samples_per_label: Max samples per label for balancing
        """
        super().__init__(root_dir, target_sr, target_duration, transform)

        self.labels = labels or self.EMOTION_LABELS
        self.file_paths = []
        self.emotion_labels = []
        self.max_samples_per_label = max_samples_per_label

        self._build_file_list()

    def _build_file_list(self) -> None:
        """Build list of audio file paths with labels."""
        for label in self.labels:
            label_dir = self.root_dir / label
            if not label_dir.exists():
                continue

            files = sorted(label_dir.glob('*.wav'))

            # Limit samples per label if specified
            if self.max_samples_per_label and label in self.max_samples_per_label:
                max_n = self.max_samples_per_label[label]
                files = files[:max_n]

            for audio_file in files:
                self.file_paths.append(audio_file)
                self.emotion_labels.append(label)

        if not self.file_paths:
            raise ValueError(f"No audio files found in {self.root_dir}")

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get audio sample with emotion label.

        Returns:
            Dictionary with:
                - waveform: Audio waveform as numpy array
                - label: Emotion label string
                - label_id: Emotion label as integer
                - path: File path
        """
        filepath = self.file_paths[idx]
        label = self.emotion_labels[idx]

        try:
            waveform, sr = self.load_audio(str(filepath))
            waveform = self.normalize_audio(waveform)
            waveform = self.pad_or_truncate(waveform)

            if self.transform:
                waveform = self.transform(waveform)

            return {
                'waveform': waveform,
                'waveforms': waveform,  # Alias for dataloader compatibility
                'label': label,
                'label_id': self.EMOTION_TO_ID[label],
                'labels': self.EMOTION_TO_ID[label],  # Alias for dataloader compatibility
                'path': str(filepath),
            }

        except Exception as e:
            # Return silent audio on error
            target_len = self.target_samples or int(self.target_sr * self.target_duration)
            return {
                'waveform': np.zeros(target_len),
                'waveforms': np.zeros(target_len),
                'label': label,
                'label_id': self.EMOTION_TO_ID[label],
                'labels': self.EMOTION_TO_ID[label],
                'path': str(filepath),
                'error': str(e),
            }

    def get_class_distribution(self) -> Dict[str, int]:
        """Get distribution of classes in the dataset."""
        from collections import Counter
        return dict(Counter(self.emotion_labels))


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function for emotion dataset.

    Pads variable-length waveforms and stacks labels.

    Args:
        batch: List of samples from __getitem__

    Returns:
        Batched dictionary
    """
    # Filter out samples with errors
    batch = [b for b in batch if 'error' not in b]

    if not batch:
        return {
            'waveforms': torch.zeros(1, 16000),
            'labels': torch.zeros(1, dtype=torch.long),
            'paths': [],
        }

    # Get max length for padding
    max_length = max(len(b['waveform']) for b in batch)

    waveforms = []
    labels = []
    paths = []

    for b in batch:
        waveform = b['waveform']
        # Pad if necessary
        if len(waveform) < max_length:
            waveform = np.pad(waveform, (0, max_length - len(waveform)), mode='constant')

        waveforms.append(waveform)
        labels.append(b['label_id'])
        paths.append(b['path'])

    return {
        'waveforms': torch.FloatTensor(np.array(waveforms)),
        'labels': torch.LongTensor(labels),
        'paths': paths,
    }