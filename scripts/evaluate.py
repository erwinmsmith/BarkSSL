#!/usr/bin/env python3
"""
BarkSSL Evaluation Script
Evaluate model on test set with comprehensive metrics.
"""

import argparse
import torch
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.canine_encoder import CanineEncoder
from src.models.classifier import EmotionClassifier
from src.data.preprocessed_dataset import PreprocessedEmotionDataset
from src.training.evaluator import Evaluator
from sklearn.metrics import classification_report, confusion_matrix


def parse_args():
    parser = argparse.ArgumentParser(description='BarkSSL Evaluation')

    parser.add_argument('--model', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--data-dir', type=str,
                       default='data/emotion_preprocessed',
                       help='Emotion preprocessed data directory')
    parser.add_argument('--split', type=str, default='all',
                       choices=['all', 'train', 'val', 'test'],
                       help='Which split to evaluate')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                       help='Training set ratio (for split)')
    parser.add_argument('--val-ratio', type=float, default=0.1,
                       help='Validation set ratio (for split)')
    parser.add_argument('--num-classes', type=int, default=5,
                       help='Number of classes')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of workers')

    return parser.parse_args()


def collate_fixed_length(batch):
    """Collate function for fixed-length emotion samples."""
    import numpy as np
    waveforms = torch.FloatTensor(np.array([b['waveform'] for b in batch]))
    labels = torch.LongTensor([b['label_id'] for b in batch])

    return {
        'waveforms': waveforms,
        'labels': labels,
    }


def main():
    args = parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load checkpoint
    print(f"Loading model from: {args.model}")
    checkpoint = torch.load(args.model, map_location=device)

    # Try to extract encoder config from checkpoint
    encoder_config = checkpoint.get('config', None)

    # If config exists but doesn't have correct values, try to infer from state_dict
    model_state = checkpoint.get('model_state_dict', {})

    if isinstance(model_state, dict):
        actual_encoder = model_state.get('encoder', model_state)

        # Infer from first encoder layer
        if 'cnn_encoder.conv_layers.0.weight' in actual_encoder:
            hidden_dim = actual_encoder['cnn_encoder.conv_layers.0.weight'].shape[0]

            # Count transformer layers
            num_layers = 0
            while f'transformer_encoder.layers.{num_layers}.self_attn.in_proj_weight' in actual_encoder:
                num_layers += 1

            num_heads = hidden_dim // 64  # 2 for 128, 6 for 384

            # Infer kmeans_k from prediction head
            kmeans_k = 100
            for k in actual_encoder.keys():
                if 'prediction_head.2.weight' in k:
                    kmeans_k = actual_encoder[k].shape[0]
                    break

            print(f"Inferred from weights: hidden_dim={hidden_dim}, num_layers={num_layers}, num_heads={num_heads}, kmeans_k={kmeans_k}")

            # Use inferred config if it differs from stored config
            if encoder_config is None or encoder_config.get('hidden_dim', 384) != hidden_dim:
                encoder_config = {
                    'scale': 'tiny' if hidden_dim == 128 else 'small',
                    'hidden_dim': hidden_dim,
                    'num_layers': num_layers,
                    'num_heads': num_heads,
                    'kmeans_k': kmeans_k,
                }
                print(f"Using inferred config: {encoder_config}")

    if encoder_config is None:
        encoder_config = {'scale': 'small', 'hidden_dim': 384, 'num_layers': 6, 'num_heads': 6, 'kmeans_k': 100}

    print(f"Final encoder config: {encoder_config}")

    encoder = CanineEncoder(
        scale=encoder_config.get('scale', 'small'),
        hidden_dim=encoder_config.get('hidden_dim', 384),
        num_layers=encoder_config.get('num_layers', 6),
        num_heads=encoder_config.get('num_heads', 6),
        kmeans_k=encoder_config.get('kmeans_k', 100),
    )

    # Load encoder weights
    encoder_state = None

    if isinstance(model_state, dict):
        encoder_state = model_state.get('encoder', model_state)
    else:
        encoder_state = model_state

    if encoder_state:
        encoder.load_state_dict(encoder_state)
        print("Encoder loaded successfully")

    encoder.to(device)

    # Create classifier
    classifier = EmotionClassifier(
        encoder=encoder,
        num_classes=args.num_classes,
        pooling_type='attentive',
    )

    # Load classifier weights if available
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        if 'classifier' in state_dict:
            classifier.load_state_dict(state_dict['classifier'])

    classifier.to(device)

    # Combine into model wrapper
    class ModelWrapper(torch.nn.Module):
        def __init__(self, encoder, classifier):
            super().__init__()
            self.encoder = encoder
            self.classifier = classifier

        def forward(self, batch):
            output = self.classifier(batch['waveforms'])
            return {
                'logits': output['logits'],
                'labels': batch['labels'],
            }

        def eval_step(self, batch):
            return self.forward(batch)

    model = ModelWrapper(encoder, classifier)
    model.to(device)

    # Load dataset
    print(f"Loading dataset from: {args.data_dir}")
    dataset = PreprocessedEmotionDataset(
        root_dir=args.data_dir,
        target_sr=16000,
        target_duration=4.0,
    )

    # Split dataset if not evaluating all
    if args.split != 'all':
        total = len(dataset)
        train_size = int(total * args.train_ratio)
        val_size = int(total * args.val_ratio)
        test_size = total - train_size - val_size

        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            dataset,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42),
        )

        if args.split == 'train':
            eval_dataset = train_dataset
        elif args.split == 'val':
            eval_dataset = val_dataset
        else:
            eval_dataset = test_dataset

        print(f"Using {args.split} split: {len(eval_dataset)} samples")
    else:
        eval_dataset = dataset
        print(f"Using all data: {len(dataset)} samples")

    # Create dataloader
    test_loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fixed_length,
        pin_memory=True,
    )

    print(f"Evaluating {len(eval_dataset)} samples")

    # Create evaluator
    evaluator = Evaluator(
        model=model,
        device=device,
        num_classes=args.num_classes,
        class_names=['angry', 'anxious', 'happy', 'lonely', 'sad'],
    )

    # Evaluate
    print("\nEvaluating...")
    metrics = evaluator.evaluate(test_loader)

    # Print results
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)

    print(f"\nOverall Metrics:")
    print(f"  Accuracy:    {metrics['accuracy']:.4f}")
    print(f"  Macro F1:    {metrics['macro_f1']:.4f}")
    print(f"  Weighted F1: {metrics['weighted_f1']:.4f}")

    print(f"\nPer-Class Metrics:")
    print(f"{'Class':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}")
    print("-" * 48)

    class_names = ['angry', 'anxious', 'happy', 'lonely', 'sad']
    for name in class_names:
        print(f"{name:<12} "
              f"{metrics.get(f'{name}_precision', 0):.4f}       "
              f"{metrics.get(f'{name}_recall', 0):.4f}       "
              f"{metrics.get(f'{name}_f1', 0):.4f}")

    # Confusion matrix
    print("\n" + "=" * 60)
    print("Confusion Matrix")
    print("=" * 60)

    # Get predictions and labels
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            output = model(batch)
            preds = output['logits'].argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch['labels'].cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    print("\nPredicted ->")
    print("Actual    ", end="")
    for name in class_names:
        print(f" {name[:3]:>6}", end="")
    print()

    for i, name in enumerate(class_names):
        print(f"{name[:3]:>8}", end="")
        for j in range(len(class_names)):
            print(f" {cm[i][j]:>6}", end="")
        print()

    print("\n" + "=" * 60)


if __name__ == '__main__':
    main()