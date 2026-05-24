# PetEmotion Dataset Description

## Directory Structure

```
data/
├── dogspeak/              # DogSpeak raw dataset
│   ├── dogspeak_released/ # Audio files grouped by dog_id
│   ├── metadata.csv       # Metadata (77,202 records)
│   └── README.md           # Official documentation
├── emotion/               # Emotion classification dataset
│   ├── angry/             # Angry (1,200 samples)
│   ├── anxious/            # Anxious (1,012 samples)
│   ├── happy/             # Happy (775 samples)
│   ├── lonely/            # Lonely (603 samples)
│   └── sad/               # Sad (639 samples)
└── DATA_DESCRIPTION.md    # This file
```

---

## 1. DogSpeak Dataset

### Source
Real dog bark recordings collected from social media videos (77,202 Barkseqs)

### Metadata Fields
| Field | Description | Example |
|-------|-------------|---------|
| `filename` | Audio filename | `0_chihuahua_M_dog_1.wav` |
| `breed` | Dog breed | chihuahua, german shepherd, husky, pitbull, shiba inu |
| `sex` | Gender | male, female |
| `dog_id` | Unique dog ID | dog_1, dog_2, ... |

### Statistics
| Property | Value |
|----------|--------|
| Total audio files | 77,202 |
| Number of dogs | 156 |
| Breed distribution | Husky(35,968), Shiba Inu(19,418), Chihuahua(8,155), German Shepherd(7,551), Pitbull(6,110) |
| Gender distribution | Male: 41,999, Female: 35,203 |
| Sample rate | Uniform 16000 Hz |
| Average duration | ~1.33s |

---

## 2. Emotion Dataset

### Source
Emotion-labeled subset from DogSpeak

### Class Distribution
| Emotion | English | Samples |
|---------|---------|---------|
| Angry | angry | 1,200 |
| Anxious | anxious | 1,012 |
| Happy | happy | 775 |
| Lonely | lonely | 603 |
| Sad | sad | 639 |
| **Total** | | **4,229** |

### Audio Characteristics
| Emotion | Avg Duration | Duration Range | Channels | Sample Rate |
|---------|--------------|----------------|----------|-------------|
| angry | 3.72s | 0.61-11.56s | Mixed (2ch/1ch) | 22050, 44100 Hz |
| anxious | 4.46s | 0.75-9.15s | Mono | 22050, 44100, 48000 Hz |
| happy | 3.15s | 0.36-9.48s | Mixed (2ch/1ch) | 11025, 22050, 44100 Hz |
| lonely | 3.69s | 0.80-10.33s | Mixed (2ch/1ch) | 16000, 44100, 48000 Hz |
| sad | 3.86s | 0.63-11.72s | Mono | 22050, 44100, 48000 Hz |

---

## 3. Data Preprocessing

### 3.1 Problem Analysis

1. **Inconsistent sample rates**: 11025, 16000, 22050, 44100, 48000 Hz mixed
2. **Mixed audio formats**: PCM (fmt=1) and IEEE Float (fmt=3) mixed
3. **Unified channels**: Some stereo, some mono
4. **Large duration variance**: 0.36s - 11.72s

### 3.2 Preprocessing Steps

#### Step 1: Install Dependencies

```bash
pip install librosa soundfile numpy pandas
```

#### Step 2: Unified Sample Rate + Mono + Standardized Duration

```python
import os
import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm

TARGET_SAMPLE_RATE = 22050
TARGET_DURATION = 4.0  # Standardize to 4 seconds
TARGET_samples = int(TARGET_SAMPLE_RATE * TARGET_DURATION)

def preprocess_audio(filepath, output_dir):
    """Preprocess a single audio file."""
    try:
        # Load audio, resample to 22050Hz, convert to mono
        y, sr = librosa.load(filepath, sr=TARGET_SAMPLE_RATE, mono=True)

        # Standardize duration to TARGET_DURATION seconds
        if len(y) > TARGET_samples:
            # Truncate
            y = y[:TARGET_samples]
        elif len(y) < TARGET_samples:
            # Pad with zeros
            y = np.pad(y, (0, TARGET_samples - len(y)), mode='constant')

        return y
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return None

def preprocess_dataset(input_dir, output_dir, extensions=['.wav']):
    """Batch preprocess dataset."""
    os.makedirs(output_dir, exist_ok=True)

    for root, dirs, files in os.walk(input_dir):
        # Preserve directory structure
        rel_path = os.path.relpath(root, input_dir)
        target_subdir = os.path.join(output_dir, rel_path)
        os.makedirs(target_subdir, exist_ok=True)

        for file in tqdm(files):
            if any(file.endswith(ext) for ext in extensions):
                input_path = os.path.join(root, file)
                output_path = os.path.join(target_subdir, file)

                # Skip if already processed
                if os.path.exists(output_path):
                    continue

                y = preprocess_audio(input_path, output_dir)
                if y is not None:
                    sf.write(output_path, y, TARGET_SAMPLE_RATE)

# Usage example
preprocess_dataset('emotion', 'emotion_preprocessed')
```

#### Step 3: Generate Augmented Version (Optional)

```python
import librosa
import numpy as np

def augment_audio(y, sr):
    """Data augmentation: time shift, pitch shift, add noise."""
    augmentations = []

    # 1. Time Shifting
    shift = np.random.randint(-int(sr * 0.2), int(sr * 0.2))
    y_shift = np.roll(y, shift)
    augmentations.append(('shift', y_shift))

    # 2. Pitch Shifting
    n_steps = np.random.uniform(-2, 2)
    y_pitch = librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)
    augmentations.append(('pitch', y_pitch))

    # 3. Adding Noise
    noise = np.random.normal(0, 0.005, y.shape)
    y_noise = y + noise
    augmentations.append(('noise', y_noise))

    # 4. Time Stretching
    rate = np.random.uniform(0.8, 1.2)
    y_speed = librosa.effects.time_stretch(y, rate=rate)
    # Ensure consistent length
    if len(y_speed) > TARGET_samples:
        y_speed = y_speed[:TARGET_samples]
    else:
        y_speed = np.pad(y_speed, (0, TARGET_samples - len(y_speed)), mode='constant')
    augmentations.append(('speed', y_speed))

    return augmentations
```

#### Step 4: Generate Training Data CSV

```python
import pandas as pd
import os

def generate_metadata(input_dir, output_csv):
    """Generate training data metadata CSV."""
    data = []

    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.wav'):
                filepath = os.path.join(root, file)

                # Get label from directory structure (assumes emotion/label/*.wav)
                rel_path = os.path.relpath(root, input_dir)
                label = rel_path.split(os.sep)[-1] if rel_path != '.' else 'unknown'

                data.append({
                    'filepath': filepath,
                    'filename': file,
                    'label': label,
                    'label_id': {'angry': 0, 'anxious': 1, 'happy': 2, 'lonely': 3, 'sad': 4}.get(label, -1)
                })

    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    return df

# Usage example
df = generate_metadata('emotion_preprocessed', 'emotion_train.csv')
print(df.head())
print(f"\nLabel distribution:\n{df['label'].value_counts()}")
```

#### Step 5: Extract Mel Spectrogram Features

```python
import librosa
import numpy as np

def extract_mel_spectrogram(y, sr=22050, n_mels=128, n_fft=2048, hop_length=512):
    """Extract mel spectrogram features."""
    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length
    )
    # Convert to decibel units
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db

def extract_features_for_dataset(input_dir, output_dir, extensions=['.wav']):
    """Batch extract features."""
    os.makedirs(output_dir, exist_ok=True)

    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if any(file.endswith(ext) for ext in extensions):
                input_path = os.path.join(root, file)
                rel_path = os.path.relpath(root, input_dir)

                try:
                    y, sr = librosa.load(input_path, sr=TARGET_SAMPLE_RATE)
                    mel_spec = extract_mel_spectrogram(y, sr)

                    output_path = os.path.join(output_dir, rel_path.replace('.wav', '.npy'))
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    np.save(output_path, mel_spec)
                except Exception as e:
                    print(f"Error extracting features from {input_path}: {e}")
```

---

## 4. Preprocessed Data Structure

```
data/
├── emotion_preprocessed/     # Preprocessed audio
│   ├── angry/*.wav
│   ├── anxious/*.wav
│   ├── happy/*.wav
│   ├── lonely/*.wav
│   └── sad/*.wav
├── emotion_features/         # Mel spectrogram features
│   ├── angry/*.npy
│   ├── anxious/*.npy
│   ├── happy/*.npy
│   ├── lonely/*.npy
│   └── sad/*.npy
├── emotion_train.csv         # Training data metadata
└── emotion_augmented/        # Augmented audio
```

---

## 5. Quick Start Script

```python
#!/usr/bin/env python3
"""
data_preprocessing.py
One-click preprocessing for PetEmotion dataset
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from tqdm import tqdm

# Default config
DEFAULT_SR = 22050
DEFAULT_DURATION = 4.0

def preprocess_audio(filepath, target_sr=DEFAULT_SR, target_samples=None):
    """Preprocess a single audio file."""
    try:
        y, sr = librosa.load(filepath, sr=target_sr, mono=True)

        if target_samples:
            if len(y) > target_samples:
                y = y[:target_samples]
            elif len(y) < target_samples:
                y = np.pad(y, (0, target_samples - len(y)), mode='constant')

        return y, sr
    except Exception as e:
        print(f"Error: {filepath} - {e}")
        return None, None

def main():
    parser = argparse.ArgumentParser(description='PetEmotion Dataset Preprocessing')
    parser.add_argument('--input', '-i', default='emotion', help='Input directory')
    parser.add_argument('--output', '-o', default='emotion_preprocessed', help='Output directory')
    parser.add_argument('--sr', type=int, default=DEFAULT_SR, help='Target sample rate')
    parser.add_argument('--duration', type=float, default=DEFAULT_DURATION, help='Target duration (seconds)')
    args = parser.parse_args()

    target_samples = int(args.sr * args.duration)
    os.makedirs(args.output, exist_ok=True)

    for emotion in ['angry', 'anxious', 'happy', 'lonely', 'sad']:
        input_dir = os.path.join(args.input, emotion)
        output_dir = os.path.join(args.output, emotion)

        if not os.path.exists(input_dir):
            print(f"Warning: {input_dir} does not exist, skipping")
            continue

        os.makedirs(output_dir, exist_ok=True)
        files = [f for f in os.listdir(input_dir) if f.endswith('.wav')]

        print(f"\nProcessing {emotion} ({len(files)} files)...")

        for file in tqdm(files):
            input_path = os.path.join(input_dir, file)
            output_path = os.path.join(output_dir, file)

            if os.path.exists(output_path):
                continue

            y, sr = preprocess_audio(input_path, args.sr, target_samples)
            if y is not None:
                sf.write(output_path, y, args.sr)

    # Generate CSV
    print("\nGenerating metadata CSV...")
    data = []
    for root, dirs, files in os.walk(args.output):
        for file in files:
            if file.endswith('.wav'):
                filepath = os.path.join(root, file)
                rel_path = os.path.relpath(root, args.output)
                label = rel_path.split(os.sep)[-1]

                data.append({
                    'filepath': filepath,
                    'filename': file,
                    'label': label
                })

    df = pd.DataFrame(data)
    csv_path = os.path.join(os.path.dirname(args.output), 'train_metadata.csv')
    df.to_csv(csv_path, index=False)

    print(f"\nDone! Metadata saved to {csv_path}")
    print(f"Label distribution:\n{df['label'].value_counts()}")

if __name__ == '__main__':
    main()
```

**Usage:**
```bash
python data_preprocessing.py --input emotion --output emotion_preprocessed --sr 22050 --duration 4.0
```

---

## 6. Notes

1. **Some files in happy directory are corrupted**: Some wav files use Float format that cannot be read by standard libraries. Use `librosa.load()` to bypass this.
2. **Class imbalance**: happy(775) vs angry(1200). Consider using weighted sampling or data augmentation.
3. **Data source**: Emotion dataset may be a subset of DogSpeak. File correspondence needs further verification.