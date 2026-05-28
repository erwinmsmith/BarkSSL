# BarkSSL

**Metadata-Free Canine Vocal Representation Learning for Dog Bark Emotion Recognition**

A self-supervised learning framework for dog bark emotion recognition, inspired by HuBERT and WavLM. The framework uses DogSpeak (77,202 unlabeled bark sequences) for self-supervised pretraining, then adapts to the DogEmotionSound dataset (4,229 labeled samples) for downstream emotion classification.

## Complete Pipeline

### Stage 0: Data Preprocessing

```bash
# Preprocess emotion and dogspeak datasets
python3 scripts/preprocess.py \
    --input-emotion data/emotion \
    --input-dogspeak data/dogspeak/dogspeak_released \
    --output-emotion data/emotion_preprocessed \
    --output-dogspeak data/dogspeak_preprocessed \
    --sr 16000 \
    --duration 4.0
```

### Stage 1: Self-Supervised Pretraining

```bash
# Single GPU
python3 scripts/pretrain.py \
    --scale small \
    --hidden-dim 384 \
    --num-layers 6 \
    --num-heads 6 \
    --kmeans-k 100 \
    --batch-size 32 \
    --epochs 100 \
    --data-dir data/dogspeak_preprocessed \
    --output-dir outputs/pretrain

# Multi-GPU (recommended)
torchrun --nproc_per_node=4 scripts/pretrain.py \
    --scale small \
    --hidden-dim 384 \
    --num-layers 6 \
    --num-heads 6 \
    --kmeans-k 100 \
    --batch-size 64 \
    --epochs 100 \
    --max-length 10 \
    --data-dir data/dogspeak_preprocessed \
    --output-dir outputs/pretrain
```

### Different Model Scales

#### Tiny Model (~1M parameters)
```bash
torchrun --nproc_per_node=4 scripts/pretrain.py \
    --scale tiny \
    --hidden-dim 128 \
    --num-layers 2 \
    --num-heads 2 \
    --kmeans-k 50 \
    --batch-size 64 \
    --epochs 100 \
    --max-length 10 \
    --output-dir outputs/pretrain_tiny
```

#### Small Model (~12M parameters)
```bash
torchrun --nproc_per_node=4 scripts/pretrain.py \
    --scale small \
    --hidden-dim 384 \
    --num-layers 6 \
    --num-heads 6 \
    --kmeans-k 100 \
    --batch-size 64 \
    --epochs 100 \
    --max-length 10 \
    --output-dir outputs/pretrain_small
```

#### Base Model (~47M parameters)
```bash
torchrun --nproc_per_node=4 scripts/pretrain.py \
    --scale base \
    --hidden-dim 768 \
    --num-layers 12 \
    --num-heads 12 \
    --kmeans-k 100 \
    --batch-size 32 \
    --epochs 100 \
    --max-length 10 \
    --output-dir outputs/pretrain_base
```

#### Large Model (~95M parameters)
```bash
torchrun --nproc_per_node=4 scripts/pretrain.py \
    --scale large \
    --hidden-dim 1024 \
    --num-layers 24 \
    --num-heads 16 \
    --kmeans-k 100 \
    --batch-size 16 \
    --epochs 100 \
    --max-length 10 \
    --output-dir outputs/pretrain_large
```

### Extended Pretraining (Stronger Representations)

For better downstream performance, increase epochs:

```bash
# 300 epochs pretraining
torchrun --nproc_per_node=4 scripts/pretrain.py \
    --scale small \
    --epochs 300 \
    --kmeans-k 100 \
    --batch-size 64 \
    --output-dir outputs/pretrain_300ep
```

**Key Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--scale` | small | Model scale (tiny/small/base/large) |
| `--hidden-dim` | 384 | Hidden dimension |
| `--num-layers` | 6 | Number of transformer layers |
| `--num-heads` | 6 | Number of attention heads |
| `--kmeans-k` | 100 | Number of acoustic units |
| `--mask-prob` | 0.075 | Mask probability |
| `--max-length` | 10.0 | Max audio length in seconds (prevents OOM) |

### Stage 2: Emotion Classification Fine-tuning

```bash
python3 scripts/finetune.py \
    --encoder outputs/pretrain/checkpoints/final_encoder.pt \
    --epochs 30 \
    --batch-size 32 \
    --lr 1e-4 \
    --pooling attentive \
    --data-dir data/emotion_preprocessed \
    --output-dir outputs/finetune
```

**Key Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--encoder` | required | Path to pretrained encoder |
| `--epochs` | 30 | Number of training epochs |
| `--batch-size` | 16 | Batch size |
| `--lr` | 1e-4 | Learning rate |
| `--pooling` | attentive | Pooling type (mean/max/attentive) |

### Stage 3: Model Evaluation

```bash
python3 scripts/evaluate.py \
    --model outputs/finetune/checkpoints/best_model.pt \
    --data-dir data/emotion_preprocessed \
    --batch-size 16
```

Output: Accuracy, F1, Precision, Recall, Confusion Matrix

### Stage 4: Inference

```bash
# Single file
python3 scripts/inference.py \
    --model outputs/finetune/checkpoints/best_model.pt \
    --input audio.wav

# Batch inference
python3 scripts/inference.py \
    --model outputs/finetune/checkpoints/best_model.pt \
    --input data/emotion/angry \
    --batch \
    --output predictions.json
```

## One-Command Pipeline

Run the entire pipeline at once:

```bash
python3 scripts/run_pipeline.py \
    --stages preprocess,pretrain,finetune,evaluate \
    --pretrain-epochs 100 \
    --finetune-epochs 30 \
    --num-gpus 4 \
    --output-dir outputs/bark_ssl
```

Or run specific stages:

```bash
# Only pretrain and finetune (skip if already done)
python3 scripts/run_pipeline.py \
    --stages pretrain,finetune \
    --encoder outputs/pretrain/checkpoints/final_encoder.pt
```

## TensorBoard Logging

View training metrics in real-time:

```bash
tensorboard --logdir outputs/bark_ssl/pretrain/tensorboard --port 6006
tensorboard --logdir outputs/bark_ssl/finetune/tensorboard --port 6007
```

## Model Scales

| Scale | Parameters | Hidden Dim | Layers | Heads |
|-------|------------|------------|--------|-------|
| Tiny  | ~1M        | 128        | 2      | 2    |
| Small | ~12M       | 384        | 6      | 6    |
| Base  | ~47M       | 768        | 12     | 12   |
| Large | ~95M       | 1024       | 24     | 16   |

## Project Structure

```
BarkSSL/
├── scripts/
│   ├── preprocess.py      # Data preprocessing
│   ├── pretrain.py        # Self-supervised pretraining
│   ├── finetune.py        # Emotion classification fine-tuning
│   ├── evaluate.py        # Model evaluation
│   ├── inference.py       # Inference on new audio
│   └── run_pipeline.py    # Run complete pipeline
├── src/
│   ├── models/
│   │   ├── canine_encoder.py   # Canine encoder architecture
│   │   └── classifier.py       # Emotion classifier
│   ├── data/
│   │   ├── dataset.py          # Dataset classes
│   │   └── preprocessed_dataset.py
│   ├── pretraining/
│   │   ├── acoustic_unit.py   # k-means clustering
│   │   └── masked_pretrain.py  # Masked prediction
│   └── training/
│       ├── trainer.py         # Training loop
│       └── evaluator.py       # Evaluation metrics
└── data/
    ├── emotion/              # Raw emotion data
    ├── emotion_preprocessed/ # Processed emotion data
    ├── dogspeak/            # Raw dogspeak data
    └── dogspeak_preprocessed/ # Processed dogspeak data
```

## Dataset

### DogSpeak
- 77,202 bark sequences from 156 dogs
- 5 breeds: Husky, Shiba Inu, Chihuahua, German Shepherd, Pitbull
- Used for self-supervised pretraining

### DogEmotionSound
- 4,229 samples across 5 emotion categories
- Categories: angry (1,200), anxious (1,012), happy (775), lonely (603), sad (639)

## Requirements

```bash
pip install torch torchaudio librosa soundfile scikit-learn tqdm tensorboard
```

## Installation

```bash
pip install -r requirements.txt
```

## Citation

If you use this code in your research, please cite:

```bibtex
@software{barkssl,
  title = {BarkSSL: Metadata-Free Canine Vocal Representation Learning},
  author = {Zhenke Duan},
  year = {2025}
}
```

## License

MIT License