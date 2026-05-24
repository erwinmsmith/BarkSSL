#!/usr/bin/env python3
"""
BarkSSL Data Preprocessing Script

Preprocesses DogSpeak and Emotion datasets for:
1. Consistent sampling rate (16kHz)
2. Mono conversion
3. Fixed duration for fine-tuning (4s)
4. Compatible frame alignment for k-means and CNN

Key Alignment:
- CNN encoder stride: 40 samples
- 4s @ 16kHz = 64000 samples = 1600 CNN frames
- DogSpeak variable length (padded to multiple of 40)
"""

import os
import sys
import argparse
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from tqdm import tqdm
import json

# Config
TARGET_SR = 16000
TARGET_DURATION = 4.0
TARGET_SAMPLES = int(TARGET_SR * TARGET_DURATION)  # 64000 samples
CNN_STRIDE = 40  # Must divide TARGET_SAMPLES

# Ensure TARGET_SAMPLES is divisible by CNN_STRIDE
assert TARGET_SAMPLES % CNN_STRIDE == 0, f"{TARGET_SAMPLES} must be divisible by {CNN_STRIDE}"


def load_audio_mono(filepath, target_sr=TARGET_SR):
    """
    Load audio file and convert to mono at target sample rate.

    Args:
        filepath: Path to audio file
        target_sr: Target sample rate

    Returns:
        waveform: Mono audio array
        sr: Actual sample rate used
    """
    try:
        # Try soundfile first (handles various formats better)
        waveform, sr = sf.read(filepath, dtype='float32')
    except Exception:
        # Fallback to librosa
        waveform, sr = librosa.load(filepath, sr=target_sr, mono=True)
        return waveform, sr

    # Convert to mono if stereo
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)

    # Resample if needed
    if sr != target_sr:
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    return waveform, sr


def pad_or_truncate(waveform, target_samples):
    """
    Pad or truncate waveform to target length.

    Args:
        waveform: Audio array
        target_samples: Target number of samples

    Returns:
        Processed waveform
    """
    current_len = len(waveform)

    if current_len > target_samples:
        return waveform[:target_samples]
    elif current_len < target_samples:
        # Pad with zeros
        return np.pad(waveform, (0, target_samples - current_len), mode='constant')
    else:
        return waveform


def pad_to_multiple(waveform, multiple):
    """
    Pad waveform to multiple of given length.
    Used for CNN frame alignment.
    """
    remainder = len(waveform) % multiple
    if remainder > 0:
        waveform = np.pad(waveform, (0, multiple - remainder), mode='constant')
    return waveform


def preprocess_emotion_dataset(input_dir, output_dir, target_sr=TARGET_SR, target_duration=TARGET_DURATION):
    """
    Preprocess Emotion dataset.

    All files are padded/truncated to fixed duration for fine-tuning.
    """
    print(f"\n{'='*60}")
    print("Preprocessing Emotion Dataset")
    print(f"{'='*60}")

    target_samples = int(target_sr * target_duration)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    stats = {
        'total': 0,
        'success': 0,
        'failed': 0,
        'sample_rates': {},
        'durations': {},
        'channels': {},
    }

    for emotion in ['angry', 'anxious', 'happy', 'lonely', 'sad']:
        emotion_input = Path(input_dir) / emotion
        emotion_output = output_path / emotion
        emotion_output.mkdir(exist_ok=True)

        if not emotion_input.exists():
            print(f"Warning: {emotion_input} not found, skipping")
            continue

        files = list(emotion_input.glob('*.wav'))
        print(f"\nProcessing {emotion}: {len(files)} files")

        for filepath in tqdm(files, desc=emotion):
            try:
                # Load and preprocess
                waveform, sr = load_audio_mono(str(filepath), target_sr)
                waveform = pad_or_truncate(waveform, target_samples)

                # Save
                output_file = emotion_output / filepath.name
                sf.write(output_file, waveform, target_sr)

                stats['success'] += 1
                stats['sample_rates'][sr] = stats['sample_rates'].get(sr, 0) + 1

            except Exception as e:
                stats['failed'] += 1
                print(f"Error: {filepath.name}: {e}")

        stats['total'] += len(files)

    print(f"\nEmotion Preprocessing Summary:")
    print(f"  Total: {stats['total']}")
    print(f"  Success: {stats['success']}")
    print(f"  Failed: {stats['failed']}")
    print(f"  Sample rates: {stats['sample_rates']}")

    return stats


def preprocess_dogspeak_dataset(input_dir, output_dir, target_sr=TARGET_SR):
    """
    Preprocess DogSpeak dataset.

    Files are NOT padded to fixed length - kept variable for pretraining.
    Just resampled and converted to mono.
    """
    print(f"\n{'='*60}")
    print("Preprocessing DogSpeak Dataset")
    print(f"{'='*60}")

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    stats = {
        'total': 0,
        'success': 0,
        'failed': 0,
        'sample_rates': {},
        'durations': {'min': float('inf'), 'max': 0, 'avg': 0},
    }

    total_duration = 0

    for dog_dir in sorted(input_path.iterdir()):
        if not dog_dir.is_dir() or not dog_dir.name.startswith('dog_'):
            continue

        dog_output = output_path / dog_dir.name
        dog_output.mkdir(exist_ok=True)

        files = list(dog_dir.glob('*.wav'))
        stats['total'] += len(files)

        for filepath in tqdm(files, desc=dog_dir.name, leave=False):
            try:
                # Load and preprocess
                waveform, sr = load_audio_mono(str(filepath), target_sr)

                # Pad to multiple of CNN_STRIDE for frame alignment
                waveform_padded = pad_to_multiple(waveform, CNN_STRIDE)

                # Save
                output_file = dog_output / filepath.name
                sf.write(output_file, waveform_padded, target_sr)

                stats['success'] += 1
                stats['sample_rates'][sr] = stats['sample_rates'].get(sr, 0) + 1

                # Track duration stats
                dur = len(waveform) / sr
                stats['durations']['min'] = min(stats['durations']['min'], dur)
                stats['durations']['max'] = max(stats['durations']['max'], dur)
                total_duration += dur

            except Exception as e:
                stats['failed'] += 1

    if stats['success'] > 0:
        stats['durations']['avg'] = total_duration / stats['success']

    print(f"\nDogSpeak Preprocessing Summary:")
    print(f"  Total files: {stats['total']}")
    print(f"  Success: {stats['success']}")
    print(f"  Failed: {stats['failed']}")
    print(f"  Duration: min={stats['durations']['min']:.2f}s, "
          f"max={stats['durations']['max']:.2f}s, "
          f"avg={stats['durations']['avg']:.2f}s")
    print(f"  Sample rates: {stats['sample_rates']}")

    return stats


def verify_alignment(preprocessed_emotion_dir, preprocessed_dogspeak_dir):
    """
    Verify that preprocessed data is compatible for pretraining and fine-tuning.
    """
    print(f"\n{'='*60}")
    print("Verifying Data Alignment")
    print(f"{'='*60}")

    # Check emotion dataset
    emotion_sample = list(Path(preprocessed_emotion_dir).glob('angry/*.wav'))[0]
    waveform, sr = sf.read(emotion_sample)

    print(f"\nEmotion Dataset:")
    print(f"  Sample: {emotion_sample.name}")
    print(f"  Sample rate: {sr}")
    print(f"  Duration: {len(waveform)/sr:.2f}s")
    print(f"  Samples: {len(waveform)}")
    print(f"  CNN frames: {len(waveform) // CNN_STRIDE}")
    print(f"  Aligned: {len(waveform) % CNN_STRIDE == 0}")

    # Check DogSpeak dataset
    dogspeak_sample = list(Path(preprocessed_dogspeak_dir).glob('dog_1/*.wav'))[0]
    waveform, sr = sf.read(dogspeak_sample)

    print(f"\nDogSpeak Dataset:")
    print(f"  Sample: {dogspeak_sample.name}")
    print(f"  Sample rate: {sr}")
    print(f"  Duration: {len(waveform)/sr:.2f}s")
    print(f"  Samples: {len(waveform)}")
    print(f"  CNN frames: {len(waveform) // CNN_STRIDE}")
    print(f"  Aligned: {len(waveform) % CNN_STRIDE == 0}")

    # Verify compatibility
    print(f"\nCompatibility Check:")
    print(f"  Sample rate match: {sr == TARGET_SR}")
    print(f"  Emotion fixed length: {len(waveform) == TARGET_SAMPLES}")
    print(f"  Both aligned to CNN stride: {len(waveform) % CNN_STRIDE == 0}")

    return sr == TARGET_SR


def main():
    parser = argparse.ArgumentParser(description='BarkSSL Data Preprocessing')
    parser.add_argument('--input-emotion', default='data/emotion', help='Input emotion directory')
    parser.add_argument('--input-dogspeak', default='data/dogspeak/dogspeak_released',
                        help='Input dogspeak directory')
    parser.add_argument('--output-emotion', default='data/emotion_preprocessed',
                        help='Output emotion directory')
    parser.add_argument('--output-dogspeak', default='data/dogspeak_preprocessed',
                        help='Output dogspeak directory')
    parser.add_argument('--sr', type=int, default=TARGET_SR, help='Target sample rate')
    parser.add_argument('--duration', type=float, default=TARGET_DURATION,
                        help='Target duration for emotion dataset (seconds)')
    parser.add_argument('--verify', action='store_true', help='Verify alignment after preprocessing')

    args = parser.parse_args()

    print("BarkSSL Data Preprocessing")
    print(f"Target sample rate: {args.sr} Hz")
    print(f"Target duration: {args.duration} s")
    print(f"CNN stride: {CNN_STRIDE}")
    print(f"Emotion fixed samples: {args.sr * args.duration}")

    # Preprocess Emotion dataset
    emotion_stats = preprocess_emotion_dataset(
        args.input_emotion,
        args.output_emotion,
        args.sr,
        args.duration
    )

    # Preprocess DogSpeak dataset
    dogspeak_stats = preprocess_dogspeak_dataset(
        args.input_dogspeak,
        args.output_dogspeak,
        args.sr
    )

    # Save preprocessing stats
    stats = {
        'target_sr': args.sr,
        'target_duration': args.duration,
        'target_samples': args.sr * args.duration,
        'cnn_stride': CNN_STRIDE,
        'emotion_stats': emotion_stats,
        'dogspeak_stats': dogspeak_stats,
    }

    with open('data/preprocessing_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"\nStats saved to data/preprocessing_stats.json")

    # Verify if requested
    if args.verify:
        verify_alignment(args.output_emotion, args.output_dogspeak)

    print("\nPreprocessing complete!")


if __name__ == '__main__':
    main()