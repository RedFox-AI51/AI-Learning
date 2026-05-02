#!/usr/bin/env python3
"""
train_locally.py — Train the student LLM on local NESA syllabuses

This wrapper script loads the AI_Learning_Kaggle notebook components and trains
the model on your local syllabuses directory. Useful for running the full training
pipeline without needing Kaggle.

Usage:
    python train_locally.py                           # train on all subjects
    python train_locally.py --subject Biology         # train on one subject
    python train_locally.py --checkpoint checkpoints/model.pt  # resume from checkpoint
"""

import argparse
import sys
from pathlib import Path


def train_all_subjects(data_dir="/kaggle/input/datasets/ninjanick/nesa-core-subjects", checkpoint_path=None):
    """Train on all available subjects in the curriculum order."""
    # Import here to allow --help without torch installed
    from tokenizer import ByteTokenizer
    from model import StudentTransformer, StudentConfig
    from trainer import Trainer, TrainerConfig
    from corpus_manager import CorpusManager
    
    # Initialize
    tok = ByteTokenizer()
    cfg = StudentConfig()
    
    # Load or create model
    if checkpoint_path and Path(checkpoint_path).exists():
        print(f"Loading checkpoint: {checkpoint_path}")
        trainer = Trainer.load(checkpoint_path, tok)
    else:
        model = StudentTransformer(cfg)
        trainer = Trainer(model, tok)
    
    # Load corpus manager
    cm = CorpusManager(trainer, data_dir=data_dir)
    
    # Ingest all subjects
    print("\n" + "="*70)
    print("TRAINING ON ALL SUBJECTS")
    print("="*70 + "\n")
    cm.ingest_all()
    
    # Save checkpoint
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_file = checkpoint_dir / "student.pt"
    trainer.save(str(checkpoint_file))
    
    # Print summary
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    trainer.print_history()


def train_single_subject(subject, data_dir="/kaggle/input/datasets/ninjanick/nesa-core-subjects", checkpoint_path=None):
    """Train on a single subject."""
    # Import here to allow --help without torch installed
    from tokenizer import ByteTokenizer
    from model import StudentTransformer, StudentConfig
    from trainer import Trainer, TrainerConfig
    from corpus_manager import CorpusManager
    
    # Initialize
    tok = ByteTokenizer()
    cfg = StudentConfig()
    
    # Load or create model
    if checkpoint_path and Path(checkpoint_path).exists():
        print(f"Loading checkpoint: {checkpoint_path}")
        trainer = Trainer.load(checkpoint_path, tok)
    else:
        model = StudentTransformer(cfg)
        trainer = Trainer(model, tok)
    
    # Load corpus manager
    cm = CorpusManager(trainer, data_dir=data_dir)
    
    # Ingest single subject
    print("\n" + "="*70)
    print(f"TRAINING ON: {subject}")
    print("="*70 + "\n")
    cm.ingest_subject(subject, section_by_section=True)
    
    # Save checkpoint
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_file = checkpoint_dir / "student.pt"
    trainer.save(str(checkpoint_file))
    
    # Print summary
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    trainer.print_history()


def main():
    parser = argparse.ArgumentParser(
        description="Train the student LLM on local NESA syllabuses",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_locally.py                              # train on all subjects
  python train_locally.py --subject Biology            # train on Biology only
  python train_locally.py --subject Chemistry --data /kaggle/input/datasets/ninjanick/nesa-core-subjects
  python train_locally.py --checkpoint checkpoints/backup.pt  # resume training
        """,
    )
    
    parser.add_argument(
        "--subject", "-s",
        default=None,
        help="Subject to train on. If not specified, trains on all available subjects."
    )
    parser.add_argument(
        "--data", "-d",
        default="/kaggle/input/datasets/ninjanick/nesa-core-subjects",
        dest="data_dir",
        help="Path to directory containing .txt syllabus files (default: /kaggle/input/datasets/ninjanick/nesa-core-subjects)"
    )
    parser.add_argument(
        "--checkpoint", "-c",
        default=None,
        dest="checkpoint_path",
        help="Path to checkpoint to resume from (optional)"
    )
    
    args = parser.parse_args()
    
    # Verify data directory exists
    if not Path(args.data_dir).exists():
        print(f"Error: Data directory not found: {args.data_dir}")
        sys.exit(1)
    
    try:
        if args.subject:
            train_single_subject(args.subject, args.data_dir, args.checkpoint_path)
        else:
            train_all_subjects(args.data_dir, args.checkpoint_path)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"Error during training: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
