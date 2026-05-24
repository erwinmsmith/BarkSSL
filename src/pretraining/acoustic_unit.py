"""
BarkSSL Acoustic Unit Discovery Module
Stage 1: k-means clustering for pseudo-label generation.
"""

import os
import numpy as np
import torch
import librosa
from sklearn.cluster import MiniBatchKMeans
from typing import Optional, List, Tuple, Dict, Any
from pathlib import Path
import pickle


class AcousticUnitDiscovery:
    """
    Stage 1: Discover acoustic units via k-means clustering.

    This implements the HuBERT-style pseudo-label generation:
    1. Extract features (MFCC / log-mel / SSL)
    2. Run k-means to get cluster assignments
    3. Use cluster IDs as pseudo-labels
    """

    FEATURE_TYPES = ['mfcc', 'log_mel', 'hidden']

    def __init__(
        self,
        k: int = 100,
        feature_type: str = 'log_mel',
        n_fft: int = 400,
        hop_length: int = 160,
        n_mels: int = 80,
        n_mfcc: int = 40,
        sample_rate: int = 16000,
    ):
        """
        Initialize acoustic unit discovery.

        Args:
            k: Number of clusters (acoustic units)
            feature_type: Feature type ('mfcc', 'log_mel', 'hidden')
            n_fft: FFT window size
            hop_length: Hop length
            n_mels: Number of mel bins
            n_mfcc: Number of MFCCs
            sample_rate: Sample rate
        """
        self.k = k
        self.feature_type = feature_type
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        self.sample_rate = sample_rate

        self.kmeans = MiniBatchKMeans(
            n_clusters=k,
            batch_size=1024,
            n_init=3,
            max_iter=300,
            random_state=42,
        )

        self.is_fitted = False

    def extract_features(self, waveform: np.ndarray) -> np.ndarray:
        """
        Extract features from waveform.

        Args:
            waveform: Audio waveform

        Returns:
            features: Feature array [n_frames, feature_dim]
        """
        if self.feature_type == 'mfcc':
            # Extract MFCC
            mfcc = librosa.feature.mfcc(
                y=waveform,
                sr=self.sample_rate,
                n_mfcc=self.n_mfcc,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
            )
            # Transpose: [n_mfcc, n_frames] -> [n_frames, n_mfcc]
            features = mfcc.T

        elif self.feature_type == 'log_mel':
            # Extract log mel spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=waveform,
                sr=self.sample_rate,
                n_mels=self.n_mels,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
            )
            log_mel = librosa.power_to_db(mel_spec, ref=np.max)
            features = log_mel.T

        else:
            raise ValueError(f"Unknown feature type: {self.feature_type}")

        return features

    def fit(
        self,
        dataset_or_paths,
        max_samples: Optional[int] = None,
        progress_callback: Optional[callable] = None,
    ) -> 'AcousticUnitDiscovery':
        """
        Fit k-means model on audio files.

        Args:
            dataset_or_paths: List of audio file paths or dataset with file_paths attribute
            max_samples: Maximum number of files to process
            progress_callback: Optional callback for progress updates

        Returns:
            Self for chaining
        """
        # Support both file paths list and dataset with file_paths attribute
        if hasattr(dataset_or_paths, 'file_paths'):
            file_paths = dataset_or_paths.file_paths
        else:
            file_paths = list(dataset_or_paths)

        if max_samples:
            file_paths = file_paths[:max_samples]

        all_features = []
        total_files = len(file_paths)

        for i, filepath in enumerate(file_paths):
            try:
                # Convert Path objects to strings
                filepath = str(filepath)

                # Load audio
                waveform, sr = librosa.load(filepath, sr=self.sample_rate, mono=True)

                # Extract features
                features = self.extract_features(waveform)
                all_features.append(features)

                if progress_callback:
                    progress_callback(i + 1, total_files)

            except Exception as e:
                print(f"Warning: Failed to process {filepath}: {e}")
                continue

        if not all_features:
            raise ValueError("No valid features extracted")

        # Concatenate all features
        all_features = np.vstack(all_features)

        # Subsample if too large (for memory efficiency)
        if len(all_features) > 500000:
            indices = np.random.choice(len(all_features), 500000, replace=False)
            all_features = all_features[indices]

        # Fit k-means
        print(f"Fitting k-means with {self.k} clusters on {len(all_features)} samples...")
        self.kmeans.fit(all_features)
        self.is_fitted = True

        return self

    def predict(self, waveform: np.ndarray) -> np.ndarray:
        """
        Predict cluster assignments for a waveform.

        Args:
            waveform: Audio waveform

        Returns:
            labels: Cluster labels for each frame [n_frames]
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")

        features = self.extract_features(waveform)
        labels = self.kmeans.predict(features)

        return labels

    def predict_proba(self, waveform: np.ndarray) -> np.ndarray:
        """
        Predict cluster probabilities for a waveform.

        Args:
            waveform: Audio waveform

        Returns:
            probabilities: [n_frames, k]
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")

        features = self.extract_features(waveform)
        # MiniBatchKMeans doesn't support predict_proba,
        # so we use cluster distances
        distances = self.kmeans.transform(features)
        # Convert distances to pseudo-probabilities (softmax-like)
        probs = np.exp(-distances / distances.mean())
        probs = probs / probs.sum(axis=1, keepdims=True)

        return probs

    def save(self, path: str) -> None:
        """
        Save k-means model to file.

        Args:
            path: Path to save model
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            'kmeans': self.kmeans,
            'k': self.k,
            'feature_type': self.feature_type,
            'n_fft': self.n_fft,
            'hop_length': self.hop_length,
            'n_mels': self.n_mels,
            'n_mfcc': self.n_mfcc,
            'sample_rate': self.sample_rate,
        }

        with open(path, 'wb') as f:
            pickle.dump(model_data, f)

    @classmethod
    def load(cls, path: str) -> 'AcousticUnitDiscovery':
        """
        Load k-means model from file.

        Args:
            path: Path to saved model

        Returns:
            AcousticUnitDiscovery instance
        """
        with open(path, 'rb') as f:
            model_data = pickle.load(f)

        instance = cls(
            k=model_data['k'],
            feature_type=model_data['feature_type'],
            n_fft=model_data['n_fft'],
            hop_length=model_data['hop_length'],
            n_mels=model_data['n_mels'],
            n_mfcc=model_data['n_mfcc'],
            sample_rate=model_data['sample_rate'],
        )

        instance.kmeans = model_data['kmeans']
        instance.is_fitted = True

        return instance


class PseudoLabelGenerator:
    """
    Generate pseudo-labels for DogSpeak dataset.
    """

    def __init__(
        self,
        acoustic_unit_discovery: AcousticUnitDiscovery,
    ):
        """
        Initialize pseudo-label generator.

        Args:
            acoustic_unit_discovery: Fitted AcousticUnitDiscovery model
        """
        self.acoustic_unit = acoustic_unit_discovery

    def generate_labels(
        self,
        file_path: str,
    ) -> Dict[str, Any]:
        """
        Generate pseudo-labels for a single file.

        Args:
            file_path: Path to audio file

        Returns:
            Dictionary with:
                - labels: Frame-level cluster labels
                - features: Frame-level features
                - n_frames: Number of frames
        """
        waveform, sr = librosa.load(file_path, sr=self.acoustic_unit.sample_rate, mono=True)
        features = self.acoustic_unit.extract_features(waveform)
        labels = self.acoustic_unit.kmeans.predict(features)

        return {
            'labels': labels,
            'features': features,
            'n_frames': len(labels),
            'file_path': file_path,
        }

    def generate_labels_batch(
        self,
        file_paths: List[str],
        save_dir: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Generate pseudo-labels for batch of files.

        Args:
            file_paths: List of audio file paths
            save_dir: Optional directory to save labels

        Returns:
            List of label dictionaries
        """
        all_labels = []

        for filepath in file_paths:
            try:
                labels = self.generate_labels(filepath)

                if save_dir:
                    # Save to disk
                    import numpy as np
                    save_path = Path(save_dir) / f"{Path(filepath).stem}.npy"
                    np.save(save_path, labels['labels'])

                all_labels.append(labels)

            except Exception as e:
                print(f"Warning: Failed to generate labels for {filepath}: {e}")

        return all_labels


def create_acoustic_unit_discovery(config: Dict[str, Any]) -> AcousticUnitDiscovery:
    """
    Factory function to create acoustic unit discovery from config.

    Args:
        config: Configuration dictionary

    Returns:
        AcousticUnitDiscovery instance
    """
    return AcousticUnitDiscovery(
        k=config.get('kmeans_k', 100),
        feature_type=config.get('feature_type', 'log_mel'),
        n_fft=config.get('n_fft', 400),
        hop_length=config.get('hop_length', 160),
        n_mels=config.get('n_mels', 80),
        n_mfcc=config.get('n_mfcc', 40),
        sample_rate=config.get('target_sr', 16000),
    )