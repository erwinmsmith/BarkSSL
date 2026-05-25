#!/usr/bin/env python3
"""
BarkSSL Inference Script
Run inference on test set or single audio file.
"""

import argparse
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.canine_encoder import CanineEncoder
from src.models.classifier import EmotionClassifier


def parse_args():
    parser = argparse.ArgumentParser(description='BarkSSL Inference')

    # Model
    parser.add_argument('--model', type=str, required=True,
                       help='Path to model checkpoint (encoder or full model)')
    parser.add_argument('--num-classes', type=int, default=5,
                       help='Number of classes')

    # Input
    parser.add_argument('--input', type=str, required=True,
                       help='Path to audio file or directory')
    parser.add_argument('--batch', action='store_true',
                       help='Process directory as batch')

    # Output
    parser.add_argument('--output', type=str, default=None,
                       help='Output file for predictions')

    return parser.parse_args()


class InferenceModel:
    """Wrapper for inference."""

    def __init__(self, encoder, classifier, device):
        self.encoder = encoder
        self.classifier = classifier
        self.device = device
        self.encoder.eval()
        self.classifier.eval()

    @torch.no_grad()
    def predict(self, waveform):
        output = self.classifier(waveform)
        return output['logits'].argmax(dim=-1).item()

    @torch.no_grad()
    def predict_batch(self, waveforms):
        output = self.classifier(waveforms)
        return output['logits'].argmax(dim=-1)


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
    if 'model_state_dict' in checkpoint and 'classifier' in checkpoint['model_state_dict']:
        classifier.load_state_dict(checkpoint['model_state_dict']['classifier'])

    classifier.to(device)

    # Create model wrapper
    model = InferenceModel(encoder, classifier, device)

    # Class names
    class_names = ['angry', 'anxious', 'happy', 'lonely', 'sad']

    # Process input
    input_path = Path(args.input)

    if input_path.is_file():
        # Single file
        import soundfile as sf
        import numpy as np

        waveform, sr = sf.read(str(input_path))
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)

        # Convert to tensor
        waveform = torch.FloatTensor(waveform).unsqueeze(0).to(device)

        pred = model.predict(waveform)
        pred_label = class_names[pred]

        print(f"File: {input_path}")
        print(f"Prediction: {pred_label} (class {pred})")
        print(f"Confidence: {torch.softmax(model.classifier(waveform)['logits'], dim=-1).max().item():.4f}")

    elif input_path.is_dir():
        # Batch directory
        from tqdm import tqdm
        import soundfile as sf

        results = []
        files = list(input_path.glob('**/*.wav'))

        print(f"Processing {len(files)} files...")

        for filepath in tqdm(files):
            try:
                waveform, sr = sf.read(str(filepath))
                if waveform.ndim > 1:
                    waveform = waveform.mean(axis=1)

                waveform_tensor = torch.FloatTensor(waveform).unsqueeze(0).to(device)
                pred = model.predict(waveform_tensor)
                pred_label = class_names[pred]

                results.append({
                    'file': str(filepath),
                    'prediction': pred_label,
                    'class_id': pred.item() if isinstance(pred, torch.Tensor) else pred,
                })
            except Exception as e:
                print(f"Error processing {filepath}: {e}")

        # Print results
        print("\n" + "=" * 50)
        print("Prediction Results")
        print("=" * 50)

        for r in results[:10]:
            print(f"{Path(r['file']).name}: {r['prediction']}")

        if len(results) > 10:
            print(f"... and {len(results) - 10} more")

        # Save to file if specified
        if args.output:
            import json
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to: {args.output}")

        # Print summary
        from collections import Counter
        counts = Counter([r['prediction'] for r in results])
        print("\nPrediction distribution:")
        for label, count in counts.most_common():
            print(f"  {label}: {count} ({count/len(results)*100:.1f}%)")

    else:
        print(f"Error: {args.input} is not a file or directory")


if __name__ == '__main__':
    main()