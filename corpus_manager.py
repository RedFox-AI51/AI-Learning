"""
corpus_manager.py — NESA Stage 6 Corpus Manager (txt-based)
-------------------------------------------------------------
Reads pre-extracted .txt syllabus files and feeds them to the student LLM
trainer in curriculum order.

Expected file layout
--------------------
    nesa_syllabuses/
        nesa_biology.txt
        nesa_chemistry.txt
        nesa_physics.txt
        nesa_maths_advanced.txt
        nesa_english_advanced.txt
        nesa_modern_history.txt
        nesa_ancient_history.txt
        nesa_legal_studies.txt
        nesa_economics.txt

Each .txt file is the clean extracted syllabus document (one per subject).
The manager reads the file, splits it into sections on the "--- Module"
delimiters written by the extraction pipeline, and trains on each section
separately so progress is tracked per-module.

Usage
-----
    from corpus_manager import CorpusManager
    from trainer import Trainer
    from model import StudentTransformer, StudentConfig
    from tokenizer import ByteTokenizer

    tok     = ByteTokenizer()
    model   = StudentTransformer(StudentConfig())
    trainer = Trainer(model, tok)

    cm = CorpusManager(trainer, data_dir="nesa_syllabuses")
    cm.ingest_subject("Biology")   # train on Biology
    cm.ingest_all()                # train on everything
    cm.status()                    # show what has been learned

CLI
---
    python corpus_manager.py list                   # list subjects + status
    python corpus_manager.py sections Biology       # list sections in a file
    python corpus_manager.py preview Biology        # print first section
    python corpus_manager.py ingest Biology         # train on Biology
    python corpus_manager.py ingest --all           # train on everything
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Syllabus catalogue
# ---------------------------------------------------------------------------
SYLLABUSES: Dict[str, dict] = {
    # Science
    "biology": {
        "name": "Biology",
        "year": "Stage 6 (2017)",
        "kla":  "Science",
        "txt_file": "nesa_biology.txt",
    },
    "chemistry": {
        "name": "Chemistry",
        "year": "Stage 6 (2017)",
        "kla":  "Science",
        "txt_file": "nesa_chemistry.txt",
    },
    "physics": {
        "name": "Physics",
        "year": "Stage 6 (2017)",
        "kla":  "Science",
        "txt_file": "nesa_physics.txt",
    },
    # Mathematics
    "maths_advanced": {
        "name": "Mathematics Advanced",
        "year": "Stage 6 (2017)",
        "kla":  "Mathematics",
        "txt_file": "nesa_maths_advanced.txt",
    },
    # English
    "english_advanced": {
        "name": "English Advanced",
        "year": "Stage 6 (2017)",
        "kla":  "English",
        "txt_file": "nesa_english_advanced.txt",
    },
    # HSIE
    "modern_history": {
        "name": "Modern History",
        "year": "Stage 6 (2017)",
        "kla":  "HSIE",
        "txt_file": "nesa_modern_history.txt",
    },
    "ancient_history": {
        "name": "Ancient History",
        "year": "Stage 6 (2017)",
        "kla":  "HSIE",
        "txt_file": "nesa_ancient_history.txt",
    },
    "legal_studies": {
        "name": "Legal Studies",
        "year": "Stage 6 (2003)",
        "kla":  "HSIE",
        "txt_file": "nesa_legal_studies.txt",
    },
    "economics": {
        "name": "Economics",
        "year": "Stage 6 (2009)",
        "kla":  "HSIE",
        "txt_file": "nesa_economics.txt",
    },
}

KLA_ORDER = ["English", "Mathematics", "Science", "HSIE",
             "TAS", "Creative Arts", "PDHPE"]

# Matches dividers like:  --- Module 1: Cells as the Basis of Life ---
_SECTION_RE = re.compile(r"^---\s+(.+?)\s+---\s*$", re.MULTILINE)


# ---------------------------------------------------------------------------
# CorpusManager
# ---------------------------------------------------------------------------
class CorpusManager:
    """Reads .txt syllabus files and feeds them to the Trainer."""

    INDEX_FILE = "corpus_index.json"

    def __init__(self, trainer=None, data_dir: str = "nesa_syllabuses"):
        self.trainer  = trainer
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.index: Dict[str, dict] = self._load_index()

    # ------------------------------------------------------------------
    # Index
    # ------------------------------------------------------------------
    def _index_path(self) -> Path:
        return self.data_dir / self.INDEX_FILE

    def _load_index(self) -> Dict[str, dict]:
        p = self._index_path()
        if p.exists():
            with open(p) as f:
                return json.load(f)
        return {}

    def _save_index(self):
        with open(self._index_path(), "w") as f:
            json.dump(self.index, f, indent=2)

    # ------------------------------------------------------------------
    # File helpers
    # ------------------------------------------------------------------
    def _txt_path(self, key: str) -> Path:
        return self.data_dir / SYLLABUSES[key]["txt_file"]

    def _file_exists(self, key: str) -> bool:
        return self._txt_path(key).exists()

    # ------------------------------------------------------------------
    # Read + split into sections
    # ------------------------------------------------------------------
    def _read_sections(self, key: str) -> List[Tuple[str, str]]:
        """Read a .txt file and return (label, text) pairs per section.

        Sections are split on lines like:  --- Module N: Title ---
        Everything before the first delimiter becomes a 'header' section.
        If no delimiters exist, the whole file is one section.
        """
        txt_path = self._txt_path(key)
        if not txt_path.exists():
            raise FileNotFoundError(
                f"Syllabus file not found: {txt_path}\n"
                f"Place the .txt file for '{SYLLABUSES[key]['name']}' "
                f"at that path and try again."
            )

        raw     = txt_path.read_text(encoding="utf-8")
        matches = list(_SECTION_RE.finditer(raw))

        if not matches:
            return [(f"{key}/full", raw.strip())]

        sections: List[Tuple[str, str]] = []

        # Header (rationale, outcomes, etc.)
        header_text = raw[: matches[0].start()].strip()
        if header_text:
            sections.append((f"{key}/header", header_text))

        # Module sections
        for i, match in enumerate(matches):
            title = match.group(1).strip()
            start = match.end()
            end   = matches[i + 1].start() if i + 1 < len(matches) else len(raw)
            body  = raw[start:end].strip()
            if body:
                sections.append((f"{key}/{title}", f"{title}\n\n{body}"))

        return sections

    # ------------------------------------------------------------------
    # Ingest
    # ------------------------------------------------------------------
    def ingest_subject(self, subject: str, section_by_section: bool = True):
        """Read a .txt file and train the model on it.

        Parameters
        ----------
        subject:
            Subject name or key, e.g. "Biology" or "maths_advanced".
        section_by_section:
            If True (default), train one section at a time and save
            progress after each. Allows resuming if interrupted.
            If False, train on the whole file as one document.
        """
        if self.trainer is None:
            raise RuntimeError("No trainer attached. Pass trainer= to CorpusManager.")

        key      = self._resolve_key(subject)
        info     = SYLLABUSES[key]
        sections = self._read_sections(key)

        print(f"\n[ingest] {info['name']}  |  {len(sections)} section(s)\n")

        already_done = set(self.index.get(key, {}).get("sections_done", []))

        if section_by_section:
            for label, text in sections:
                if label in already_done:
                    print(f"  [skip] {label}")
                    continue
                self.trainer.learn_document(text, label=label)
                self._mark_section_done(key, label)
        else:
            full_text = "\n\n".join(text for _, text in sections)
            self.trainer.learn_document(full_text, label=info["name"])

        self.index[key] = {
            **self.index.get(key, {}),
            "name":        info["name"],
            "kla":         info["kla"],
            "ingested":    True,
            "ingested_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "sections":    len(sections),
        }
        self._save_index()
        print(f"\n[ingest] {info['name']} complete.\n")

    def ingest_all(self):
        """Ingest all subjects in KLA order, skipping already-ingested ones."""
        keys_by_kla: Dict[str, List[str]] = {}
        for key, info in SYLLABUSES.items():
            keys_by_kla.setdefault(info["kla"], []).append(key)

        for kla in KLA_ORDER:
            for key in keys_by_kla.get(kla, []):
                if self.index.get(key, {}).get("ingested"):
                    print(f"  [skip] {SYLLABUSES[key]['name']} already ingested.")
                    continue
                if not self._file_exists(key):
                    print(f"  [missing] {SYLLABUSES[key]['name']} — "
                          f"{self._txt_path(key)} not found, skipping.")
                    continue
                try:
                    self.ingest_subject(key)
                except Exception as e:
                    print(f"  [error] {key}: {e}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _resolve_key(self, subject: str) -> str:
        s = subject.lower().replace(" ", "_")
        if s in SYLLABUSES:
            return s
        for key, info in SYLLABUSES.items():
            if info["name"].lower() == subject.lower():
                return key
        raise KeyError(
            f"Unknown subject: {subject!r}. "
            f"Valid keys: {list(SYLLABUSES.keys())}"
        )

    def _mark_section_done(self, key: str, label: str):
        entry = self.index.setdefault(key, {
            "name": SYLLABUSES[key]["name"],
            "ingested": False,
            "sections_done": [],
        })
        entry.setdefault("sections_done", []).append(label)
        self._save_index()

    # ------------------------------------------------------------------
    # Status + preview
    # ------------------------------------------------------------------
    def status(self):
        """Print a summary table of all subjects and ingestion status."""
        print(f"\n{'─'*70}")
        print(f"{'Subject':<22} {'KLA':<14} {'Ingested':<10} {'Sections':<10} {'File'}")
        print(f"{'─'*70}")
        for key, info in SYLLABUSES.items():
            entry    = self.index.get(key, {})
            done     = "✓" if entry.get("ingested") else "–"
            sections = str(entry.get("sections", "?"))
            file_ok  = "txt✓" if self._file_exists(key) else "no file"
            print(f"{info['name']:<22} {info['kla']:<14} {done:<10} "
                  f"{sections:<10} {file_ok}")
        print(f"{'─'*70}\n")

    def preview(self, subject: str, n_sections: int = 1):
        """Print the first N sections of a subject file."""
        key      = self._resolve_key(subject)
        sections = self._read_sections(key)
        for label, text in sections[:n_sections]:
            print(f"=== {label} ===\n")
            print(text[:2000])
            if len(text) > 2000:
                print(f"\n... [{len(text) - 2000} more characters]")
            print()

    def list_sections(self, subject: str):
        """List all section labels and word counts for a subject."""
        key      = self._resolve_key(subject)
        sections = self._read_sections(key)
        print(f"\n{SYLLABUSES[key]['name']} — {len(sections)} section(s):")
        for i, (label, text) in enumerate(sections, 1):
            print(f"  {i:2d}. {label}  ({len(text.split())} words)")
        print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _cli():
    import argparse

    parser = argparse.ArgumentParser(
        description="NESA Stage 6 Corpus Manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python corpus_manager.py list                    # list all subjects + status
  python corpus_manager.py sections Biology        # list sections in Biology
  python corpus_manager.py preview Biology         # print first section
  python corpus_manager.py preview Biology --sections 3
  python corpus_manager.py ingest Biology          # train on Biology
  python corpus_manager.py ingest Biology --whole  # train as one document
  python corpus_manager.py ingest --all            # train on everything
        """,
    )
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("list", help="List subjects and ingestion status")

    p_sec = sub.add_parser("sections", help="List sections in a subject file")
    p_sec.add_argument("subject")

    p_prev = sub.add_parser("preview", help="Preview a subject's content")
    p_prev.add_argument("subject")
    p_prev.add_argument("--sections", type=int, default=1,
                        dest="n_sections", metavar="N")

    p_ing = sub.add_parser("ingest", help="Train on a subject's syllabus")
    p_ing.add_argument("subject", nargs="?")
    p_ing.add_argument("--all",   action="store_true",
                       help="Ingest all available subjects")
    p_ing.add_argument("--whole", action="store_true",
                       help="Train on whole file as one document")

    args = parser.parse_args()

    def _make_trainer():
        from tokenizer import ByteTokenizer
        from model     import StudentTransformer, StudentConfig
        from trainer   import Trainer
        tok   = ByteTokenizer()
        model = StudentTransformer(StudentConfig())
        ckpt  = Path("checkpoints/student.pt")
        if ckpt.exists():
            return Trainer.load(str(ckpt), tok)
        return Trainer(model, tok)

    if args.cmd == "list":
        CorpusManager(data_dir="nesa_syllabuses").status()

    elif args.cmd == "sections":
        CorpusManager(data_dir="nesa_syllabuses").list_sections(args.subject)

    elif args.cmd == "preview":
        CorpusManager(data_dir="nesa_syllabuses").preview(
            args.subject, n_sections=args.n_sections
        )

    elif args.cmd == "ingest":
        trainer = _make_trainer()
        cm      = CorpusManager(trainer, data_dir="nesa_syllabuses")
        section_by_section = not args.whole
        if args.all:
            cm.ingest_all()
        elif args.subject:
            cm.ingest_subject(args.subject, section_by_section=section_by_section)
        else:
            print("Specify a subject name or --all")
            sys.exit(1)
        trainer.save("checkpoints/student.pt")

    else:
        parser.print_help()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        _cli()
    else:
        cm = CorpusManager(data_dir="nesa_syllabuses")
        cm.status()
        try:
            cm.preview("biology", n_sections=1)
        except FileNotFoundError as e:
            print(e)