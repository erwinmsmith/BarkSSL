"""
BarkSSL Audio Transforms Module
Audio preprocessing and augmentation transforms.
"""

import numpy as np
import torch
import librosa
from typing import Optional, Callable, List
from abc import ABC, abstractmethod


class AudioTransform(ABC):
    """
    Abstract base class for audio transforms.
    """

    @abstractmethod
    def __call__(self, waveform: np.ndarray, sr: int = 16000) -> np.ndarray:
        """
        Apply transform to waveform.

        Args:
            waveform: Input waveform
            sr: Sample rate

        Returns:
            Transformed waveform
        """
        pass


class TimeShift(AudioTransform):
    """
    Time shift augmentation - shift audio by random amount.
    """

    def __init__(self, max_shift: float = 0.2):
        """
        Initialize time shift.

        Args:
            max_shift: Maximum shift as fraction of audio length
        """
        self.max_shift = max_shift

    def __call__(self, waveform: np.ndarray, sr: int = 16000) -> np.ndarray:
        shift_samples = int(len(waveform) * np.random.uniform(-self.max_shift, self.max_shift))
        return np.roll(waveform, shift_samples)


class PitchShift(AudioTransform):
    """
    Pitch shift augmentation.
    """

    def __init__(self, n_steps: float = 2.0):
        """
        Initialize pitch shift.

        Args:
            n_steps: Range of pitch shift in semitones
        """
        self.n_steps = n_steps

    def __call__(self, waveform: np.ndarray, sr: int = 16000) -> np.ndarray:
        n_steps = np.random.uniform(-self.n_steps, self.n_steps)
        return librosa.effects.pitch_shift(waveform, sr=sr, n_steps=n_steps)


class AddNoise(AudioTransform):
    """
    Add Gaussian noise.
    """

    def __init__(self, noise_level: float = 0.005):
        """
        Initialize noise.

        Args:
            noise_level: Standard deviation of noise
        """
        self.noise_level = noise_level

    def __call__(self, waveform: np.ndarray, sr: int = 16000) -> np.ndarray:
        noise = np.random.normal(0, self.noise_level, waveform.shape)
        return waveform + noise


class TimeStretch(AudioTransform):
    """
    Time stretch augmentation.
    """

    def __init__(self, rate_range: tuple = (0.8, 1.2)):
        """
        Initialize time stretch.

        Args:
            rate_range: Range of stretch rates
        """
        self.rate_range = rate_range

    def __call__(self, waveform: np.ndarray, sr: int = 16000) -> np.ndarray:
        rate = np.random.uniform(*self.rate_range)
        waveform_stretched = librosa.effects.time_stretch(waveform, rate=rate)

        # Ensure same length
        if len(waveform_stretched) > len(waveform):
            return waveform_stretched[:len(waveform)]
        elif len(waveform_stretched) < len(waveform):
            return np.pad(waveform_stretched, (0, len(waveform) - len(waveform_stretched)), mode='constant')
        return waveform_stretched


class SpecAugment(AudioTransform):
    """
    SpecAugment-style masking on spectrogram.
    """

    def __init__(
        self,
        freq_mask_param: int = 15,
        time_mask_param: int = 35,
        n_mels: int = 80,
        n_fft: int = 400,
        hop_length: int = 160,
    ):
        """
        Initialize SpecAugment.

        Args:
            freq_mask_param: Maximum frequency mask width
            time_mask_param: Maximum time mask width
            n_mels: Number of mel bins
            n_fft: FFT size
            hop_length: Hop length
        """
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length

    def __call__(self, waveform: np.ndarray, sr: int = 16000) -> np.ndarray:
        # Compute mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=waveform, sr=sr, n_mels=self.n_mels,
            n_fft=self.n_fft, hop_length=self.hop_length
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        # Frequency masking
        n_freq_masks = 1
        for _ in range(n_freq_masks):
            f = np.random.randint(0, self.freq_mask_param)
            f0 = np.random.randint(0, mel_spec_db.shape[0] - f)
            mel_spec_db[f0:f0 + f, :] = 0

        # Time masking
        n_time_masks = 1
        for _ in range(n_time_masks):
            t = np.random.randint(0, self.time_mask_param)
            t0 = np.random.randint(0, mel_spec_db.shape[1] - t)
            mel_spec_db[:, t0:t0 + t] = 0

        # Inverse mel spectrogram
        inv = librosa.feature.inverse.mel_to_audio(
            librosa.db_to_power(mel_spec_db),
            sr=sr, n_fft=self.n_fft, hop_length=self.hop_length
        )

        # Ensure same length
        if len(inv) > len(waveform):
            return inv[:len(waveform)]
        elif len(inv) < len(waveform):
            return np.pad(inv, (0, len(waveform) - len(inv)), mode='constant')
        return inv


class BarkDenoising(AudioTransform):
    """
    WavLM-style noise/dog bark mixing for denoising training.

    Simulates real-world noisy recordings from social media.
    """

    def __init__(
        self,
        noise_prob: float = 0.5,
        noise_dir: Optional[str] = None,
        bark_dir: Optional[str] = None,
        snr_range: tuple = (0, 20),
    ):
        """
        Initialize bark denoising.

        Args:
            noise_prob: Probability of adding noise
            noise_dir: Directory with noise files (optional)
            bark_dir: Directory with other dog bark files (optional)
            snr_range: Signal-to-noise ratio range in dB
        """
        self.noise_prob = noise_prob
        self.noise_dir = noise_dir
        self.bark_dir = bark_dir
        self.snr_range = snr_range
        self.noise_files = []
        self.bark_files = []

        # Load noise file list if directory provided
        import os
        if noise_dir and os.path.exists(noise_dir):
            self.noise_files = list(Path(noise_dir).glob('*.wav'))[:1000]
        if bark_dir and os.path.exists(bark_dir):
            self.bark_files = list(Path(bark_dir).glob('*.wav'))[:1000]

    def _mix_signals(
        self,
        signal: np.ndarray,
        noise: np.ndarray,
        target_length: int,
    ) -> np.ndarray:
        """Mix signal with noise at given SNR."""
        # Truncate/pad noise to target length
        if len(noise) > target_length:
            start = np.random.randint(0, len(noise) - target_length)
            noise = noise[start:start + target_length]
        elif len(noise) < target_length:
            noise = np.pad(noise, (0, target_length - len(noise)), mode='constant')

        # Calculate SNR
        signal_power = np.mean(signal ** 2)
        noise_power = np.mean(noise ** 2)

        if noise_power > 0:
            snr_db = np.random.uniform(*self.snr_range)
            snr_linear = 10 ** (snr_db / 10)
            noise_scale = np.sqrt(signal_power / (snr_linear * noise_power))
            noise = noise * noise_scale

        return signal + noise

    def __call__(self, waveform: np.ndarray, sr: int = 16000) -> np.ndarray:
        """Apply denoising augmentation."""
        if np.random.random() > self.noise_prob:
            return waveform

        target_length = len(waveform)

        # Randomly choose noise type
        noise_type = np.random.choice(['noise', 'bark', 'gaussian', 'none'])

        if noise_type == 'gaussian':
            # Add Gaussian noise
            noise_level = np.random.uniform(0.001, 0.05)
            noise = np.random.normal(0, noise_level, waveform.shape)
            return waveform + noise

        elif noise_type == 'noise' and self.noise_files:
            # Mix with environmental noise
            noise_file = np.random.choice(self.noise_files)
            try:
                noise, _ = librosa.load(str(noise_file), sr=sr, mono=True)
                return self._mix_signals(waveform, noise, target_length)
            except:
                return waveform

        elif noise_type == 'bark' and self.bark_files:
            # Mix with another dog bark
            bark_file = np.random.choice(self.bark_files)
            try:
                bark, _ = librosa.load(str(bark_file), sr=sr, mono=True)
                return self._mix_signals(waveform, bark, target_length)
            except:
                return waveform

        return waveform


class Compose:
    """
    Compose multiple transforms.
    """

    def __init__(self, transforms: List[AudioTransform]):
        """
        Args:
            transforms: List of transforms to compose
        """
        self.transforms = transforms

    def __call__(self, waveform: np.ndarray, sr: int = 16000) -> np.ndarray:
        for transform in self.transforms:
            waveform = transform(waveform, sr)
        return waveform


def get_augmentation_transform(
    use_denoising: bool = True,
    use_spec_augment: bool = True,
) -> Optional[Compose]:
    """
    Get standard augmentation pipeline.

    Args:
        use_denoising: Include WavLM-style denoising
        use_spec_augment: Include SpecAugment

    Returns:
        Compose object with transforms, or None
    """
    transforms = [
        TimeShift(max_shift=0.1),
        AddNoise(noise_level=0.005),
        TimeStretch(rate_range=(0.9, 1.1)),
    ]

    if use_spec_augment:
        transforms.append(SpecAugment())

    return Compose(transforms)


def get_denoising_transform(
    noise_dir: Optional[str] = None,
    bark_dir: Optional[str] = None,
) -> BarkDenoising:
    """
    Get WavLM-style denoising transform.

    Args:
        noise_dir: Directory with noise files
        bark_dir: Directory with dog bark files

    Returns:
        BarkDenoising transform
    """
    return BarkDenoising(
        noise_prob=0.5,
        noise_dir=noise_dir,
        bark_dir=bark_dir,
        snr_range=(0, 20),
    )