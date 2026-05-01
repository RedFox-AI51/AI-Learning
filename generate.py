#!/usr/bin/env python3
"""
generate.py — Generate responses from a trained student LLM

Useful for testing the model after training. Loads a checkpoint and generates
responses to prompts interactively or from command line.

Usage:
    python generate.py                                # interactive mode
    python generate.py "What is photosynthesis?"     # single query
    python generate.py --checkpoint checkpoints/model.pt "Your question here"
"""

import argparse
import sys
from pathlib import Path


def interactive_mode(checkpoint_path="checkpoints/student.pt"):
    """Run interactive generation mode."""
    # Import here to allow --help without torch installed
    from tokenizer import ByteTokenizer
    from trainer import Trainer
    
    # Load checkpoint
    if not Path(checkpoint_path).exists():
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        print("Have you trained the model yet? Run train_locally.py first.")
        sys.exit(1)
    
    tok = ByteTokenizer()
    trainer = Trainer.load(checkpoint_path, tok)
    
    print("\n" + "="*70)
    print("STUDENT LLM — INTERACTIVE GENERATION")
    print("="*70)
    print(f"Model checkpoint: {checkpoint_path}")
    print(f"Trained steps: {trainer.step}")
    print("\nType 'quit' to exit.\n")
    
    while True:
        try:
            prompt = input("Prompt: ").strip()
            if prompt.lower() in ("quit", "exit", "q"):
                print("Goodbye!")
                break
            if not prompt:
                print("(empty prompt, skipping)")
                continue
            
            response = trainer.generate_response(prompt)
            print(f"Response: {response}\n")
            
        except KeyboardInterrupt:
            print("\nInterrupted.")
            break
        except Exception as e:
            print(f"Error: {e}")


def single_query_mode(prompt, checkpoint_path="checkpoints/student.pt", max_tokens=200):
    """Generate response for a single prompt."""
    # Import here to allow --help without torch installed
    from tokenizer import ByteTokenizer
    from trainer import Trainer
    
    # Load checkpoint
    if not Path(checkpoint_path).exists():
        print(f"Error: Checkpoint not found: {checkpoint_path}", file=sys.stderr)
        sys.exit(1)
    
    tok = ByteTokenizer()
    trainer = Trainer.load(checkpoint_path, tok)
    
    # Generate
    response = trainer.generate_response(prompt, max_new_tokens=max_tokens)
    print(response)


def main():
    parser = argparse.ArgumentParser(
        description="Generate responses from a trained student LLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate.py                                  # interactive mode
  python generate.py "What is photosynthesis?"       # single query
  python generate.py --checkpoint model.pt "Your question"
  python generate.py --max-tokens 300 "Explain..."   # longer generation
        """,
    )
    
    parser.add_argument(
        "prompt",
        nargs="?",
        default=None,
        help="Prompt for generation. If not provided, enters interactive mode."
    )
    parser.add_argument(
        "--checkpoint", "-c",
        default="checkpoints/student.pt",
        help="Path to model checkpoint (default: checkpoints/student.pt)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=200,
        help="Maximum number of new tokens to generate (default: 200)"
    )
    
    args = parser.parse_args()
    
    try:
        if args.prompt:
            single_query_mode(args.prompt, args.checkpoint, args.max_tokens)
        else:
            interactive_mode(args.checkpoint)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
