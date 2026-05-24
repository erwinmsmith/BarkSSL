# BarkSSL

**Metadata-Free Canine Vocal Representation Learning for Dog Bark Emotion Recognition**

A self-supervised learning framework for dog bark emotion recognition, inspired by HuBERT and WavLM. The framework uses DogSpeak (77,202 unlabeled bark sequences) for self-supervised pretraining, then adapts to the DogEmotionSound dataset (4,229 labeled samples) for downstream emotion classification.

## Overview

This project implements a complete pipeline for canine vocal representation learning:

1. **Stage 0**: Data preparation
2. **Stage 1**: Canine Acoustic Unit Discovery (k-means clustering on MFCC/log-mel features)
3. **Stage 2**: Masked Unit Prediction pretraining on DogSpeak
4. **Stage 3**: Emotion-space adaptation with CE + Supervised Contrastive Loss on DogEmotionSound

## Project Structure

```
petemotion/
├── configs/
│   └── config.yaml              # Configuration file
├── src/
│   ├── main.py                  # Main entry point
│   ├── data/
│   │   ├── dataset.py           # DogSpeakDataset, EmotionDataset
│   │   ├── transforms.py        # Audio augmentation
│   │   └── dataloader.py        # DataLoader utilities
│   ├── models/
│   │   ├── components.py        # CNNEncoder, TransformerEncoder, Pooling
│   │   ├── canine_encoder.py     # Canine-HuBERT encoder
│   │   ├── classifier.py        # Emotion classifier
│   │   └── base.py              # Base model classes
│   ├── pretraining/
│   │   ├── acoustic_unit.py     # Stage 1: k-means acoustic unit discovery
│   │   └── masked_pretrain.py    # Stage 2: masked unit prediction
│   ├── training/
│   │   ├── losses.py            # CE + Supervised Contrastive Loss
│   │   ├── trainer.py           # Training loop
│   │   └── evaluator.py          # Evaluation metrics
│   ├── utils/
│   │   ├── config.py            # Configuration loader
│   │   ├── device.py            # GPU/CUDA auto-detection
│   │   └── logger.py            # Logging utilities
│   └── inference/
│       └── predictor.py         # Inference utilities
├── data/                        # Data directory
├── checkpoints/                 # Model checkpoints
├── logs/                        # Training logs
└── main.py                      # Main entry point
```

## Installation

```bash
pip install -r requirements.txt
```

### Requirements

- Python >= 3.8
- PyTorch >= 2.0
- torchaudio >= 2.0
- librosa
- soundfile
- scikit-learn
- pyyaml
- tqdm
- tensorboard

## Usage

### 1. Test Data Loading and Model Initialization

```bash
python main.py test --test all
```

### 2. Self-Supervised Pretraining (Stage 1 + 2)

Pretrain the canine encoder on DogSpeak dataset:

```bash
# Single GPU / CPU
python scripts/pretrain.py \
    --scale small \
    --batch-size 32 \
    --epochs 100

# Multi-GPU (recommended for faster training)
torchrun --nproc_per_node=4 scripts/pretrain.py \
    --scale small \
    --batch-size 64 \
    --epochs 100
```

**Note**: `--batch-size` is per-GPU. With 4 GPUs and batch-size=64, effective batch size = 256.

Key parameters:
- `--scale`: Model scale (tiny/small/base/large)
- `--kmeans-k`: Number of acoustic units (default: 100)
- `--mask-prob`: Mask probability (default: 0.075)
- `--lr`: Learning rate (default: 0.0001)
- `--num-workers`: Data loading workers per GPU
- `--max-length`: Max audio length in seconds (default: 10s, prevents OOM)

### 3. Downstream Emotion Classification Training (Stage 3)

Fine-tune the pretrained encoder on DogEmotionSound:

```bash
python main.py train \
    --config configs/config.yaml \
    --checkpoint checkpoints/canine_encoder.pt \
    --epochs 30
```

### 4. Evaluate Model

Evaluate on the test set:

```bash
python main.py eval \
    --checkpoint checkpoints/best_model.pt \
    --data data/emotion
```

### 5. Run Inference

Single file prediction:

```bash
python main.py infer \
    --checkpoint checkpoints/best_model.pt \
    --input audio.wav
```

Batch prediction (directory):

```bash
python main.py infer \
    --checkpoint checkpoints/best_model.pt \
    --input data/emotion \
    --batch
```

## Configuration

All parameters are managed via `configs/config.yaml`:

```yaml
# Model configuration
model:
  scale: small              # small | medium
  small:
    hidden_dim: 384
    num_layers: 6
    num_heads: 6
  medium:
    hidden_dim: 768
    num_layers: 12
    num_heads: 12
  kmeans_k: 100            # Number of acoustic units
  mask_prob: 0.075         # Mask probability

# Training configuration
training:
  pretrain_epochs: 50
  finetune_epochs: 30
  learning_rate: 0.0001
  supcon_lambda: 0.1        # Supervised Contrastive loss weight
  supcon_temperature: 0.07
```

## Model Scales

| Scale   | Parameters | Hidden Dim | Layers | Heads |
|---------|------------|------------|--------|-------|
| Small   | ~12M       | 384        | 6      | 6     |
| Medium  | ~95M       | 768        | 12     | 12    |

## Dataset

### DogSpeak
- 77,202 bark sequences from 156 dogs
- 5 breeds: Husky, Shiba Inu, Chihuahua, German Shepherd, Pitbull
- Used for self-supervised pretraining (metadata not used as supervision)

### DogEmotionSound
- 4,229 samples across 5 emotion categories
- Categories: angry (1,200), anxious (1,012), happy (775), lonely (603), sad (639)

## Loss Functions

The framework supports multiple loss functions:

1. **Cross-Entropy**: Standard classification loss
2. **Supervised Contrastive Loss**: Pulls same-class embeddings together, pushes different classes apart
3. **Combined Loss**: CE + λ × SupCon (recommended)

## Audio Preprocessing

Default settings:
- Sampling rate: 16,000 Hz
- Duration: 4 seconds (padded/truncated)
- Channels: Mono
- Features: Log-mel spectrogram (80 bins)

## Reference Code

This implementation is inspired by:

- [Microsoft WavLM](https://github.com/microsoft/unilm/tree/master/wavlm) - CNN encoder, masked prediction
- [fairseq HuBERT](https://github.com/facebookresearch/fairseq/tree/main/examples/hubert) - k-means pseudo-label generation
- [S3PRL](https://github.com/s3prl/s3prl) - Downstream evaluation

## Citation

If you use this code in your research, please cite:

```bibtex
@software{barkssl,
  title = {TBD},
  author = {Zhenke Duan},
  year = {2025}
}
```

## License

This project is licensed under the MIT License.