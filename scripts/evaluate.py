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

    # Determine config
    if 'config' in checkpoint:
        config = checkpoint['config']
    else:
        config = {
            'scale': 'small',
            'hidden_dim': 384,
            'num_layers': 6,
            'num_heads': 6,
            'kmeans_k': 100,
        }

    # Create encoder
    encoder = CanineEncoder(
        scale=config.get('scale', 'small'),
        hidden_dim=config.get('hidden_dim', 384),
        num_layers=config.get('num_layers', 6),
        num_heads=config.get('num_heads', 6),
        kmeans_k=config.get('kmeans_k', 100),
    )

    # Load encoder weights
    if 'encoder_state_dict' in checkpoint:
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
    elif 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        if 'encoder' in state_dict:
            encoder.load_state_dict(state_dict['encoder'])
        else:
            encoder.load_state_dict(state_dict)

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

        def eval_step(self, batch):
            output = self.classifier(batch['waveforms'])
            return {
                'logits': output['logits'],
                'labels': batch['labels'],
            }

    model = ModelWrapper(encoder, classifier)
    model.to(device)

    # Load dataset
    print(f"Loading test dataset from: {args.data_dir}")
    dataset = PreprocessedEmotionDataset(
        root_dir=args.data_dir,
        target_sr=16000,
        target_duration=4.0,
    )

    # Create dataloader
    test_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fixed_length,
        pin_memory=True,
    )

    print(f"Test set size: {len(dataset)}")

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