#!/usr/bin/env python3
"""
BarkSSL Complete Pipeline Script
Run the entire pipeline: preprocess -> pretrain -> finetune -> evaluate

Usage:
    # Run everything
    python3 scripts/run_pipeline.py

    # Run specific stages
    python3 scripts/run_pipeline.py --stages preprocess,pretrain,finetune,evaluate

    # Custom config
    python3 scripts/run_pipeline.py --pretrain-epochs 100 --finetune-epochs 30
"""

import argparse
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args():
    parser = argparse.ArgumentParser(description='BarkSSL Complete Pipeline')

    # Stages to run
    parser.add_argument('--stages', type=str, default='all',
                       help='Comma-separated stages: preprocess,pretrain,finetune,evaluate,inference')

    # Data paths
    parser.add_argument('--raw-emotion-dir', type=str, default='data/emotion',
                       help='Raw emotion data directory')
    parser.add_argument('--raw-dogspeak-dir', type=str, default='data/dogspeak/dogspeak_released',
                       help='Raw dogspeak data directory')
    parser.add_argument('--processed-emotion-dir', type=str, default='data/emotion_preprocessed',
                       help='Processed emotion output directory')
    parser.add_argument('--processed-dogspeak-dir', type=str, default='data/dogspeak_preprocessed',
                       help='Processed dogspeak output directory')

    # Preprocessing
    parser.add_argument('--sr', type=int, default=16000, help='Target sample rate')
    parser.add_argument('--duration', type=float, default=4.0, help='Emotion duration (seconds)')

    # Pretrain config
    parser.add_argument('--pretrain-scale', type=str, default='small',
                       choices=['tiny', 'small', 'base', 'large'],
                       help='Pretrain model scale')
    parser.add_argument('--pretrain-hidden-dim', type=int, default=384, help='Hidden dimension')
    parser.add_argument('--pretrain-layers', type=int, default=6, help='Number of layers')
    parser.add_argument('--pretrain-heads', type=int, default=6, help='Number of heads')
    parser.add_argument('--kmeans-k', type=int, default=100, help='Number of acoustic units')
    parser.add_argument('--pretrain-epochs', type=int, default=100, help='Pretrain epochs')
    parser.add_argument('--pretrain-batch-size', type=int, default=32, help='Batch size per GPU')
    parser.add_argument('--pretrain-lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--pretrain-max-length', type=float, default=10.0, help='Max audio length (seconds)')
    parser.add_argument('--pretrain-max-samples', type=int, default=None,
                       help='Max samples for pretrain (None for all)')

    # Finetune config
    parser.add_argument('--finetune-epochs', type=int, default=30, help='Finetune epochs')
    parser.add_argument('--finetune-batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--finetune-lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--pooling', type=str, default='attentive',
                       choices=['mean', 'max', 'attentive'], help='Pooling type')

    # Multi-GPU
    parser.add_argument('--num-gpus', type=int, default=1, help='Number of GPUs for pretrain')
    parser.add_argument('--num-workers', type=int, default=4, help='Data loading workers')

    # Output
    parser.add_argument('--output-dir', type=str, default='outputs/bark_ssl',
                       help='Output directory')
    parser.add_argument('--encoder-checkpoint', type=str, default=None,
                       help='Use existing encoder checkpoint')
    parser.add_argument('--finetune-checkpoint', type=str, default=None,
                       help='Use existing finetune checkpoint')

    # Inference
    parser.add_argument('--infer-input', type=str, default=None,
                       help='Input file/directory for inference')
    parser.add_argument('--infer-output', type=str, default=None,
                       help='Output file for inference predictions')

    # Misc
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--yes', '-y', action='store_true', help='Skip confirmation prompts')

    return parser.parse_args()


def run_preprocess(args):
    """Run data preprocessing."""
    print("\n" + "=" * 60)
    print("STAGE 1: Data Preprocessing")
    print("=" * 60)

    cmd = f"""
python3 scripts/preprocess.py \
    --input-emotion {args.raw_emotion_dir} \
    --input-dogspeak {args.raw_dogspeak_dir} \
    --output-emotion {args.processed_emotion_dir} \
    --output-dogspeak {args.processed_dogspeak_dir} \
    --sr {args.sr} \
    --duration {args.duration}
"""
    print(f"Running: {cmd}")
    os.system(cmd)
    return True


def run_pretrain(args):
    """Run self-supervised pretraining."""
    print("\n" + "=" * 60)
    print("STAGE 2: Self-Supervised Pretraining")
    print("=" * 60)

    if args.encoder_checkpoint and Path(args.encoder_checkpoint).exists():
        print(f"Using existing encoder: {args.encoder_checkpoint}")
        return args.encoder_checkpoint

    pretrain_output = f"{args.output_dir}/pretrain"
    os.makedirs(pretrain_output, exist_ok=True)

    if args.num_gpus > 1:
        cmd = f"""
torchrun --nproc_per_node={args.num_gpus} scripts/pretrain.py \
    --scale {args.pretrain_scale} \
    --hidden-dim {args.pretrain_hidden_dim} \
    --num-layers {args.pretrain_layers} \
    --num-heads {args.pretrain_heads} \
    --kmeans-k {args.kmeans_k} \
    --batch-size {args.pretrain_batch_size} \
    --epochs {args.pretrain_epochs} \
    --lr {args.pretrain_lr} \
    --max-length {args.pretrain_max_length} \
    --data-dir {args.processed_dogspeak_dir} \
    --output-dir {pretrain_output} \
    --num-workers {args.num_workers}
"""
    else:
        cmd = f"""
python3 scripts/pretrain.py \
    --scale {args.pretrain_scale} \
    --hidden-dim {args.pretrain_hidden_dim} \
    --num-layers {args.pretrain_layers} \
    --num-heads {args.pretrain_heads} \
    --kmeans-k {args.kmeans_k} \
    --batch-size {args.pretrain_batch_size} \
    --epochs {args.pretrain_epochs} \
    --lr {args.pretrain_lr} \
    --max-length {args.pretrain_max_length} \
    --data-dir {args.processed_dogspeak_dir} \
    --output-dir {pretrain_output} \
    --num-workers {args.num_workers}
"""

    if args.pretrain_max_samples:
        cmd += f" --max-samples {args.pretrain_max_samples}"

    print(f"Running: {cmd}")
    os.system(cmd)

    encoder_path = f"{pretrain_output}/checkpoints/final_encoder.pt"
    if Path(encoder_path).exists():
        return encoder_path
    else:
        # Try to find any encoder checkpoint
        checkpoints = list(Path(pretrain_output).glob("checkpoints/*.pt"))
        if checkpoints:
            return str(checkpoints[-1])
    return None


def run_finetune(args, encoder_path=None):
    """Run emotion classification fine-tuning."""
    print("\n" + "=" * 60)
    print("STAGE 3: Emotion Classification Fine-tuning")
    print("=" * 60)

    if args.finetune_checkpoint and Path(args.finetune_checkpoint).exists():
        print(f"Using existing finetune model: {args.finetune_checkpoint}")
        return args.finetune_checkpoint

    if encoder_path is None:
        print("Error: No encoder checkpoint provided for fine-tuning")
        return None

    finetune_output = f"{args.output_dir}/finetune"
    os.makedirs(finetune_output, exist_ok=True)

    cmd = f"""
python3 scripts/finetune.py \
    --encoder {encoder_path} \
    --epochs {args.finetune_epochs} \
    --batch-size {args.finetune_batch_size} \
    --lr {args.finetune_lr} \
    --pooling {args.pooling} \
    --data-dir {args.processed_emotion_dir} \
    --output-dir {finetune_output} \
    --num-workers {args.num_workers}
"""
    print(f"Running: {cmd}")
    os.system(cmd)

    best_model = f"{finetune_output}/checkpoints/best_model.pt"
    if Path(best_model).exists():
        return best_model
    else:
        checkpoints = list(Path(finetune_output).glob("checkpoints/*.pt"))
        if checkpoints:
            return str(checkpoints[-1])
    return None


def run_evaluate(args, model_path=None):
    """Run model evaluation."""
    print("\n" + "=" * 60)
    print("STAGE 4: Model Evaluation")
    print("=" * 60)

    if model_path is None:
        model_path = f"{args.output_dir}/finetune/checkpoints/best_model.pt"

    if not Path(model_path).exists():
        print(f"Error: Model checkpoint not found: {model_path}")
        return None

    cmd = f"""
python3 scripts/evaluate.py \
    --model {model_path} \
    --data-dir {args.processed_emotion_dir} \
    --batch-size {args.finetune_batch_size}
"""
    print(f"Running: {cmd}")
    os.system(cmd)
    return model_path


def run_inference(args, model_path=None):
    """Run inference."""
    print("\n" + "=" * 60)
    print("STAGE 5: Inference")
    print("=" * 60)

    if model_path is None:
        model_path = f"{args.output_dir}/finetune/checkpoints/best_model.pt"

    if not Path(model_path).exists():
        print(f"Error: Model checkpoint not found: {model_path}")
        return None

    if args.infer_input is None:
        print("Error: --infer-input required for inference")
        return None

    cmd = f"""
python3 scripts/inference.py \
    --model {model_path} \
    --input {args.infer_input}
"""
    if args.infer_output:
        cmd += f" --output {args.infer_output}"

    print(f"Running: {cmd}")
    os.system(cmd)
    return model_path


def main():
    args = parse_args()

    print("\n" + "=" * 60)
    print("BarkSSL Complete Pipeline")
    print("=" * 60)

    # Parse stages
    if args.stages == 'all':
        stages = ['preprocess', 'pretrain', 'finetune', 'evaluate']
    else:
        stages = [s.strip() for s in args.stages.split(',')]

    print(f"\nStages to run: {stages}")
    print(f"Output directory: {args.output_dir}")

    # Confirm
    if not args.yes:
        confirm = input("\nContinue? [y/N]: ").strip().lower()
        if confirm != 'y':
            print("Aborted.")
            return

    # Track results
    encoder_path = args.encoder_checkpoint
    finetune_path = args.finetune_checkpoint

    # Run stages
    try:
        if 'preprocess' in stages:
            run_preprocess(args)

        if 'pretrain' in stages:
            encoder_path = run_pretrain(args)
            if encoder_path is None:
                print("Pretraining failed!")
                return

        if 'finetune' in stages:
            finetune_path = run_finetune(args, encoder_path)
            if finetune_path is None:
                print("Fine-tuning failed!")
                return

        if 'evaluate' in stages:
            run_evaluate(args, finetune_path)

        if 'inference' in stages:
            run_inference(args, finetune_path)

        print("\n" + "=" * 60)
        print("Pipeline Complete!")
        print("=" * 60)
        print(f"\nResults saved to: {args.output_dir}")
        print(f"Encoder: {encoder_path}")
        print(f"Finetune model: {finetune_path}")

    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user.")
        return


if __name__ == '__main__':
    main()