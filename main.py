#!/usr/bin/env python3
"""
BarkSSL Main Entry Point

A metadata-free canine vocal representation learning framework for
dog bark emotion recognition.

Usage:
    python main.py pretrain --config configs/config.yaml
    python main.py train --config configs/config.yaml
    python main.py infer --checkpoint checkpoints/best_model.pt --input audio.wav
    python main.py eval --checkpoint checkpoints/best_model.pt --data data/emotion
"""

import os
import sys
import argparse
import random
import numpy as np
import torch
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.utils.config import Config
from src.utils.device import get_device, get_device_info
from src.utils.logger import Logger, setup_logger
from src.models.canine_encoder import CanineEncoder
from src.models.classifier import EmotionClassifier, create_emotion_classifier
from src.data.dataset import DogSpeakDataset, EmotionDataset, collate_fn
from src.data.dataloader import create_dataloader, create_emotion_dataloader, create_split_dataloaders
from src.training.losses import CombinedLoss, create_loss_function
from src.training.trainer import Trainer, PretrainingTrainer, create_trainer
from src.training.evaluator import Evaluator, create_evaluator
from src.pretraining.acoustic_unit import AcousticUnitDiscovery
from src.pretraining.masked_pretrain import CanineHuBERTPretraining
from src.inference.predictor import EmotionPredictor, load_predictor_from_checkpoint


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='BarkSSL: Metadata-Free Canine Vocal Representation Learning',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Main command
    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Pretrain command
    pretrain_parser = subparsers.add_parser('pretrain', help='Self-supervised pretraining')
    pretrain_parser.add_argument('--config', '-c', required=True, help='Config file path')
    pretrain_parser.add_argument('--epochs', '-e', type=int, default=None, help='Number of epochs')
    pretrain_parser.add_argument('--batch-size', '-b', type=int, default=None, help='Batch size')
    pretrain_parser.add_argument('--lr', type=float, default=None, help='Learning rate')

    # Train command
    train_parser = subparsers.add_parser('train', help='Downstream emotion classification training')
    train_parser.add_argument('--config', '-c', required=True, help='Config file path')
    train_parser.add_argument('--checkpoint', help='Pretrained encoder checkpoint')
    train_parser.add_argument('--epochs', '-e', type=int, default=None, help='Number of epochs')
    train_parser.add_argument('--batch-size', '-b', type=int, default=None, help='Batch size')

    # Eval command
    eval_parser = subparsers.add_parser('eval', help='Evaluate model')
    eval_parser.add_argument('--checkpoint', '-ckpt', required=True, help='Checkpoint path')
    eval_parser.add_argument('--config', '-c', default='configs/config.yaml', help='Config file path')
    eval_parser.add_argument('--data', '-d', default='data/emotion', help='Data directory')

    # Infer command
    infer_parser = subparsers.add_parser('infer', help='Run inference')
    infer_parser.add_argument('--checkpoint', '-ckpt', required=True, help='Checkpoint path')
    infer_parser.add_argument('--input', '-i', required=True, help='Input audio file or directory')
    infer_parser.add_argument('--config', '-c', default='configs/config.yaml', help='Config file path')
    infer_parser.add_argument('--batch', action='store_true', help='Batch mode (input is directory)')

    # Test command
    test_parser = subparsers.add_parser('test', help='Test data loading and model init')
    test_parser.add_argument('--config', '-c', default='configs/config.yaml', help='Config file path')
    test_parser.add_argument('--test', choices=['data', 'model', 'all'], default='all', help='Test type')

    return parser.parse_args()


def cmd_pretrain(args):
    """Run self-supervised pretraining."""
    # Load config
    config = Config(args.config)

    # Override with CLI args
    if args.epochs:
        config.set('training.pretrain_epochs', args.epochs)
    if args.batch_size:
        config.set('training.pretrain_batch_size', args.batch_size)
    if args.lr:
        config.set('training.learning_rate', args.lr)

    # Setup logger
    log_dir = config.get('paths.log_dir', 'logs')
    logger = setup_logger('pretrain', log_dir)
    logger.info("=" * 60)
    logger.info("BarkSSL Self-Supervised Pretraining")
    logger.info("=" * 60)

    # Log device info
    device = get_device(config.get('device.cuda_if_available', True))
    logger.info(f"Device: {device}")
    logger.info(f"Config: {config}")

    # Set seed
    set_seed(config.get('seed', 42))

    # Create acoustic unit discovery
    logger.info("Initializing acoustic unit discovery...")
    acoustic_unit = AcousticUnitDiscovery(
        k=config.get('model.kmeans_k', 100),
        feature_type='log_mel',
        n_fft=config.get('data.n_fft', 400),
        hop_length=config.get('data.hop_length', 160),
        n_mels=config.get('data.n_mels', 80),
        sample_rate=config.get('data.target_sr', 16000),
    )

    # Create model
    logger.info("Creating Canine encoder...")
    model = CanineEncoder(
        scale=config.get('model.scale', 'small'),
        hidden_dim=config.get_model_config().get('hidden_dim', 384),
        num_layers=config.get_model_config().get('num_layers', 6),
        num_heads=config.get_model_config().get('num_heads', 6),
        intermediate_size=config.get_model_config().get('intermediate_size', 1536),
        dropout=config.get_model_config().get('dropout', 0.1),
        kmeans_k=config.get('model.kmeans_k', 100),
    )

    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {num_params:,}")

    # Create dataloader
    logger.info("Loading DogSpeak dataset...")
    dogspeak_root = config.get('data.dogspeak_root', 'data/dogspeak/dogspeak_released')

    dataset = DogSpeakDataset(
        root_dir=dogspeak_root,
        metadata_path=config.get('data.dogspeak_metadata'),
        target_sr=config.get('data.target_sr', 16000),
        target_duration=config.get('data.target_duration'),
    )

    dataloader = create_dataloader(
        dataset,
        batch_size=config.get('training.pretrain_batch_size', 32),
        shuffle=True,
        num_workers=config.get('device.num_workers', 4),
    )

    # Create pretraining
    pretraining = CanineHuBERTPretraining(
        encoder=model,
        acoustic_unit_discovery=acoustic_unit,
        mask_prob=config.get('model.mask_prob', 0.075),
        mask_span=config.get('model.mask_span', 10),
        use_denoising=config.get('training.use_denoising', False),
    )

    # Train
    trainer = PretrainingTrainer(
        model=pretraining,
        train_dataloader=dataloader,
        device=device,
        config=config.to_dict(),
        logger=logger,
    )

    num_epochs = config.get('training.pretrain_epochs', 50)
    logger.info(f"Starting pretraining for {num_epochs} epochs...")

    trainer.train(num_epochs)

    # Save final model
    final_ckpt = config.get('paths.pretrained_encoder', 'checkpoints/canine_encoder.pt')
    pretraining.save(final_ckpt)
    logger.info(f"Pretrained model saved to {final_ckpt}")


def cmd_train(args):
    """Run downstream emotion classification training."""
    # Load config
    config = Config(args.config)

    # Override with CLI args
    if args.epochs:
        config.set('training.finetune_epochs', args.epochs)
    if args.batch_size:
        config.set('training.finetune_batch_size', args.batch_size)

    # Setup logger
    log_dir = config.get('paths.log_dir', 'logs')
    logger = setup_logger('train', log_dir)
    logger.info("=" * 60)
    logger.info("BarkSSL Emotion Classification Training")
    logger.info("=" * 60)

    # Log device info
    device = get_device(config.get('device.cuda_if_available', True))
    logger.info(f"Device: {device}")

    # Set seed
    set_seed(config.get('seed', 42))

    # Load pretrained encoder
    logger.info("Loading pretrained encoder...")
    encoder = CanineEncoder(
        scale=config.get('model.scale', 'small'),
        hidden_dim=config.get_model_config().get('hidden_dim', 384),
        num_layers=config.get_model_config().get('num_layers', 6),
        num_heads=config.get_model_config().get('num_heads', 6),
        intermediate_size=config.get_model_config().get('intermediate_size', 1536),
        dropout=config.get_model_config().get('dropout', 0.1),
        kmeans_k=config.get('model.kmeans_k', 100),
    )

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        encoder.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
        logger.info(f"Loaded pretrained model from {args.checkpoint}")

    # Create classifier
    model = create_emotion_classifier(
        encoder=encoder,
        num_classes=config.get('data.num_classes', 5),
        pooling_type='attentive',
    )

    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total parameters: {num_params:,}")

    # Create dataloaders
    logger.info("Loading emotion dataset...")
    emotion_root = config.get('data.emotion_root', 'data/emotion')

    dataloaders = create_split_dataloaders(
        root_dir=emotion_root,
        batch_size=config.get('training.finetune_batch_size', 16),
        target_sr=config.get('data.target_sr', 16000),
        target_duration=config.get('data.target_duration', 4.0),
        num_workers=config.get('device.num_workers', 4),
    )

    logger.info(f"Train samples: {len(dataloaders['train'].dataset)}")
    logger.info(f"Val samples: {len(dataloaders['val'].dataset)}")

    # Create loss
    loss_fn = CombinedLoss(
        num_classes=config.get('data.num_classes', 5),
        supcon_temperature=config.get('training.supcon_temperature', 0.07),
        supcon_lambda=config.get('training.supcon_lambda', 0.1),
    )

    # Train
    trainer = create_trainer(
        model=model,
        train_dataloader=dataloaders['train'],
        val_dataloader=dataloaders['val'],
        config=config.to_dict(),
        loss_fn=loss_fn,
        logger=logger,
    )

    num_epochs = config.get('training.finetune_epochs', 30)
    logger.info(f"Starting training for {num_epochs} epochs...")

    trainer.train(num_epochs)

    # Save best model
    best_ckpt = config.get('paths.best_model', 'checkpoints/best_model.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config.to_dict(),
    }, best_ckpt)
    logger.info(f"Best model saved to {best_ckpt}")


def cmd_eval(args):
    """Evaluate model."""
    # Load config
    config = Config(args.config)

    # Setup logger
    logger = setup_logger('eval')
    logger.info("=" * 60)
    logger.info("BarkSSL Model Evaluation")
    logger.info("=" * 60)

    device = get_device(config.get('device.cuda_if_available', True))
    logger.info(f"Device: {device}")

    # Load model
    logger.info(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    model_config = config.get_model_config()
    encoder = CanineEncoder(
        scale=config.get('model.scale', 'small'),
        hidden_dim=model_config.get('hidden_dim', 384),
        num_layers=model_config.get('num_layers', 6),
        num_heads=model_config.get('num_heads', 6),
        intermediate_size=model_config.get('intermediate_size', 1536),
        dropout=model_config.get('dropout', 0.1),
        kmeans_k=config.get('model.kmeans_k', 100),
    )

    model = create_emotion_classifier(
        encoder=encoder,
        num_classes=config.get('data.num_classes', 5),
        pooling_type='attentive',
    )

    model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
    model.to(device)

    # Create dataloader
    dataloader = create_emotion_dataloader(
        root_dir=args.data,
        batch_size=16,
        target_sr=config.get('data.target_sr', 16000),
        target_duration=config.get('data.target_duration', 4.0),
        num_workers=2,
        shuffle=False,
    )

    # Evaluate
    evaluator = create_evaluator(model, config.to_dict())
    evaluator.device = device

    metrics = evaluator.evaluate(dataloader)

    logger.info("\n" + "=" * 40)
    logger.info("Evaluation Results")
    logger.info("=" * 40)
    for key, value in metrics.items():
        if isinstance(value, float):
            logger.info(f"{key}: {value:.4f}")
        else:
            logger.info(f"{key}: {value}")

    # Confusion matrix
    logger.info("\nClassification Report:")
    all_preds, all_labels = [], []
    for batch in dataloader:
        result = evaluator.predict_batch({k: v.to(device) for k, v in batch.items()})
        all_preds.extend(result['predictions'])
        all_labels.extend(result['labels'])

    logger.info("\n" + evaluator.print_report(np.array(all_preds), np.array(all_labels)))


def cmd_infer(args):
    """Run inference."""
    # Load config
    config = Config(args.config)

    # Setup logger
    logger = setup_logger('infer')

    device = get_device(config.get('device.cuda_if_available', True))

    # Load predictor
    logger.info(f"Loading checkpoint from {args.checkpoint}...")
    predictor = load_predictor_from_checkpoint(args.checkpoint, config, device)

    # Predict
    input_path = Path(args.input)

    if input_path.is_file():
        logger.info(f"Predicting for {input_path}...")
        result = predictor.predict_file(str(input_path), target_sr=config.get('data.target_sr', 16000))

        logger.info("\n" + "=" * 40)
        logger.info("Prediction Result")
        logger.info("=" * 40)
        logger.info(f"File: {result['filepath']}")
        logger.info(f"Emotion: {result['emotion']} {EmotionPredictor.EMOTION_EMOJI.get(result['emotion'], '')}")
        logger.info(f"Confidence: {result['confidence']:.4f}")

        if 'probabilities' in result:
            logger.info("\nClass Probabilities:")
            for emotion, prob in sorted(result['probabilities'].items(), key=lambda x: -x[1]):
                bar = '█' * int(prob * 20)
                logger.info(f"  {emotion:<8}: {prob:.4f} {bar}")

    elif input_path.is_dir():
        logger.info(f"Predicting for all files in {input_path}...")
        results = predictor.predict_from_directory(str(input_path), target_sr=config.get('data.target_sr', 16000))

        # Summary
        distribution = predictor.get_class_distribution(results)
        logger.info("\n" + "=" * 40)
        logger.info("Prediction Summary")
        logger.info("=" * 40)
        for emotion, count in distribution.items():
            emoji = EmotionPredictor.EMOTION_EMOJI.get(emotion, '')
            logger.info(f"{emotion:<8}: {count} {emoji}")

        # Save results
        import json
        output_file = input_path / 'predictions.json'
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"\nResults saved to {output_file}")
    else:
        logger.error(f"Input path not found: {input_path}")
        return 1

    return 0


def cmd_test(args):
    """Test data loading and model initialization."""
    config = Config(args.config)

    device = get_device(config.get('device.cuda_if_available', True))
    print(f"\nDevice: {device}")
    print(f"Config loaded: {len(config._config)} keys")

    if args.test in ['data', 'all']:
        print("\n--- Testing Data Loading ---")
        try:
            dataset = EmotionDataset(
                root_dir='data/emotion',
                target_sr=16000,
                target_duration=4.0,
            )
            print(f"Dataset loaded: {len(dataset)} samples")
            print(f"Class distribution: {dataset.get_class_distribution()}")

            sample = dataset[0]
            print(f"Sample shape: {sample['waveform'].shape}")
            print("Data loading: OK")
        except Exception as e:
            print(f"Data loading failed: {e}")

    if args.test in ['model', 'all']:
        print("\n--- Testing Model Initialization ---")
        try:
            encoder = CanineEncoder(
                scale='small',
                hidden_dim=384,
                num_layers=6,
                num_heads=6,
                kmeans_k=100,
            )
            print(f"CanineEncoder created: {encoder.get_num_parameters():,} parameters")

            classifier = create_emotion_classifier(encoder, num_classes=5)
            print(f"EmotionClassifier created: {classifier.get_num_parameters():,} parameters")

            # Test forward pass
            dummy_input = torch.randn(2, 16000)
            output = classifier(dummy_input)
            print(f"Forward pass output: logits shape {output['logits'].shape}")
            print("Model initialization: OK")
        except Exception as e:
            print(f"Model initialization failed: {e}")

    print("\nAll tests passed!" if args.test == 'all' else "")
    return 0


def main():
    """Main entry point."""
    args = parse_args()

    if args.command is None:
        # No command provided, show help
        print(__doc__)
        print("\nRun 'python main.py --help' to see available commands.")
        return 0

    commands = {
        'pretrain': cmd_pretrain,
        'train': cmd_train,
        'eval': cmd_eval,
        'infer': cmd_infer,
        'test': cmd_test,
    }

    return commands.get(args.command, lambda _: 1)(args)


if __name__ == '__main__':
    sys.exit(main() or 0)