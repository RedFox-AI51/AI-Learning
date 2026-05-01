#!/usr/bin/env python3
"""
corpus_cli.py — Manage and inspect NESA syllabus corpus

CLI wrapper around CorpusManager for inspecting available syllabuses,
checking ingestion status, and previewing content.

Usage:
    python corpus_cli.py list                    # list all subjects
    python corpus_cli.py preview Biology         # preview Biology content
    python corpus_cli.py sections Physics        # list sections in Physics
"""

import sys

from corpus_manager import CorpusManager


def main():
    if len(sys.argv) < 2:
        print_help()
        return
    
    cmd = sys.argv[1]
    cm = CorpusManager(data_dir="nesa_syllabuses")
    
    if cmd == "list":
        cm.status()
    
    elif cmd == "preview":
        if len(sys.argv) < 3:
            print("Usage: python corpus_cli.py preview <subject>")
            sys.exit(1)
        subject = sys.argv[2]
        n_sections = int(sys.argv[3]) if len(sys.argv) > 3 else 1
        try:
            cm.preview(subject, n_sections=n_sections)
        except KeyError as e:
            print(f"Error: {e}")
            sys.exit(1)
    
    elif cmd == "sections":
        if len(sys.argv) < 3:
            print("Usage: python corpus_cli.py sections <subject>")
            sys.exit(1)
        subject = sys.argv[2]
        try:
            cm.list_sections(subject)
        except KeyError as e:
            print(f"Error: {e}")
            sys.exit(1)
    
    elif cmd in ("help", "-h", "--help"):
        print_help()
    
    else:
        print(f"Unknown command: {cmd}")
        print_help()
        sys.exit(1)


def print_help():
    help_text = """
NESA Syllabus Corpus Manager
=============================

Commands:
  list                        List all available subjects and ingestion status
  preview <subject> [N]       Preview first N sections of a subject (default: 1)
  sections <subject>          List all sections in a subject file
  help                        Show this help message

Examples:
  python corpus_cli.py list
  python corpus_cli.py preview Biology
  python corpus_cli.py preview Physics 3
  python corpus_cli.py sections Chemistry

Available subjects:
  biology, chemistry, physics, maths_advanced, english_advanced,
  modern_history, ancient_history, legal_studies, economics
    """
    print(help_text)


if __name__ == "__main__":
    main()
