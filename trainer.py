"""
trainer.py — Student LLM training loop
----------------------------------------
Two learning modes:

  1. learn_document(text)
     Standard next-token prediction over a sliding window.
     Call this whenever you feed the model a new story, article, etc.

  2. learn_from_assessment(prompt, correct_answer, score, max_score, feedback)
     Supervised fine-tuning on the correct answer, with loss scaled by
     how wrong the model was.  A score of 2/10 applies full gradient
     pressure; 9/10 applies a gentle nudge.

  3. generate_response(prompt) -> str
     Ask the model a question and get its current best answer back.

Checkpointing
-------------
  trainer.save(path)   — saves model weights + optimiser state + step count
  Trainer.load(path)   — restores everything so training can resume

Usage
-----
    from trainer import Trainer, TrainerConfig
    from model   import StudentTransformer, StudentConfig
    from tokenizer import ByteTokenizer

    tok     = ByteTokenizer()
    model   = StudentTransformer(StudentConfig())
    trainer = Trainer(model, tok)

    trainer.learn_document("The mitochondria is the powerhouse of the cell...")
    response = trainer.generate_response("What does the mitochondria do?")
    trainer.learn_from_assessment(
        prompt         = "What does the mitochondria do?",
        correct_answer = "The mitochondria produces ATP through cellular respiration.",
        score          = 3,
        max_score      = 10,
        feedback       = "You missed ATP synthesis and cellular respiration.",
    )
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from model import StudentConfig, StudentTransformer
from tokenizer import ByteTokenizer


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class TrainerConfig:
    # Optimiser
    lr:            float = 3e-4
    weight_decay:  float = 0.1
    betas:         tuple = (0.9, 0.95)
    grad_clip:     float = 1.0

    # Document learning
    doc_batch_size:  int   = 4       # windows per gradient step
    doc_stride:      int   = 256     # sliding window stride (tokens)
    doc_epochs:      int   = 1       # passes over the document
    doc_lr_scale:    float = 1.0     # lr multiplier for document mode

    # Assessment learning
    assess_epochs:   int   = 3       # fine-tuning passes per assessment
    assess_lr_scale: float = 0.5     # lower lr — surgical update only
    min_loss_scale:  float = 0.05    # floor so a perfect score still trains a tiny bit

    # Generation defaults
    max_new_tokens:  int   = 200
    temperature:     float = 0.8
    top_k:           int   = 40
    top_p:           float = 0.9

    # Logging
    log_every:  int = 10     # print loss every N steps
    device:     str = "auto"


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
class Trainer:
    """Wraps StudentTransformer with document and assessment learning."""

    def __init__(
        self,
        model:  StudentTransformer,
        tok:    ByteTokenizer,
        cfg:    TrainerConfig = None,
    ):
        self.model = model
        self.tok   = tok
        self.cfg   = cfg or TrainerConfig()
        self.step  = 0
        self.history: List[dict] = []   # log of all training events

        # Device
        if self.cfg.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.cfg.device)
        self.model.to(self.device)

        # Optimiser — separate weight decay for embeddings vs rest
        decay_params    = [p for n, p in model.named_parameters()
                           if p.requires_grad and p.dim() >= 2]
        no_decay_params = [p for n, p in model.named_parameters()
                           if p.requires_grad and p.dim() < 2]
        self.optimizer = AdamW(
            [{"params": decay_params,    "weight_decay": self.cfg.weight_decay},
             {"params": no_decay_params, "weight_decay": 0.0}],
            lr=self.cfg.lr, betas=self.cfg.betas,
        )

        print(f"Trainer ready  |  device={self.device}  |  "
              f"params={model.num_params():,}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _to_device(self, t: torch.Tensor) -> torch.Tensor:
        return t.to(self.device)

    def _set_lr(self, scale: float):
        for g in self.optimizer.param_groups:
            g["lr"] = self.cfg.lr * scale

    def _gradient_step(self, input_ids: torch.Tensor, targets: torch.Tensor,
                        loss_scale: float = 1.0) -> float:
        """Run one forward + backward + optimiser step.

        Returns the raw loss value (float).
        """
        self.model.train()
        input_ids = self._to_device(input_ids)
        targets   = self._to_device(targets)

        _, loss = self.model(input_ids, targets=targets)
        scaled_loss = loss * loss_scale
        scaled_loss.backward()

        nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

        self.step += 1
        return loss.item()

    def _windows(self, ids: List[int]) -> List[torch.Tensor]:
        """Slice a token list into overlapping context windows."""
        max_len = self.model.cfg.max_seq_len
        stride  = self.cfg.doc_stride
        windows = []
        start   = 0
        while start < len(ids) - 1:
            end = min(start + max_len, len(ids))
            windows.append(torch.tensor(ids[start:end], dtype=torch.long))
            if end == len(ids):
                break
            start += stride
        return windows

    def _make_targets(self, ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Split a 1D token tensor into (input, target) pairs for LM training."""
        return ids[:-1].unsqueeze(0), ids[1:].unsqueeze(0)

    # ------------------------------------------------------------------
    # Mode 1: document learning
    # ------------------------------------------------------------------
    def learn_document(self, text: str, label: str = "document") -> dict:
        """Train on a text document using next-token prediction.

        Parameters
        ----------
        text:
            The full document string.
        label:
            Optional tag for the training log (e.g. 'biology_ch1').

        Returns
        -------
        dict with keys: steps, mean_loss, duration_s
        """
        ids     = self.tok.encode(text, add_special=True)
        windows = self._windows(ids)

        if not windows:
            print(f"[learn_document] Text too short to train on (got {len(ids)} tokens).")
            return {}

        self._set_lr(self.cfg.doc_lr_scale)
        losses = []
        t0     = time.time()

        print(f"[learn_document] '{label}'  |  "
              f"{len(ids)} tokens  |  {len(windows)} windows  |  "
              f"{self.cfg.doc_epochs} epoch(s)")

        for epoch in range(self.cfg.doc_epochs):
            epoch_losses = []
            # Shuffle windows each epoch so adjacent windows aren't always seen together
            idx = torch.randperm(len(windows)).tolist()
            for i in range(0, len(idx), self.cfg.doc_batch_size):
                batch_idx = idx[i : i + self.cfg.doc_batch_size]
                for wi in batch_idx:
                    inp, tgt = self._make_targets(windows[wi])
                    loss_val = self._gradient_step(inp, tgt)
                    epoch_losses.append(loss_val)
                    if self.step % self.cfg.log_every == 0:
                        print(f"  step {self.step:5d}  loss={loss_val:.4f}")

            mean_epoch = sum(epoch_losses) / len(epoch_losses)
            print(f"  epoch {epoch+1}/{self.cfg.doc_epochs}  "
                  f"mean_loss={mean_epoch:.4f}")
            losses.extend(epoch_losses)

        duration = time.time() - t0
        mean_loss = sum(losses) / len(losses)
        record = dict(event="document", label=label,
                      steps=len(losses), mean_loss=mean_loss, duration_s=duration)
        self.history.append(record)
        print(f"[learn_document] Done  |  steps={len(losses)}  "
              f"mean_loss={mean_loss:.4f}  ({duration:.1f}s)\n")
        return record

    # ------------------------------------------------------------------
    # Mode 2: assessment feedback
    # ------------------------------------------------------------------
    def learn_from_assessment(
        self,
        prompt:         str,
        correct_answer: str,
        score:          float,
        max_score:      float = 10.0,
        feedback:       str   = "",
    ) -> dict:
        """Fine-tune on the correct answer, scaled by how wrong the model was.

        The training text is structured as:
            <bos> [prompt] [correct_answer + feedback] <eos>

        Loss scale = 1 - (score / max_score), floored at min_loss_scale.
        A score of 0/10 → scale=1.0 (full gradient).
        A score of 10/10 → scale=min_loss_scale (almost nothing).

        Parameters
        ----------
        prompt:
            The question or task given to the model.
        correct_answer:
            The ideal response the model should learn.
        score:
            The grade you gave the model's response.
        max_score:
            The maximum possible score.
        feedback:
            Optional extra explanation to include in the training text.
        """
        # Build the supervised training text
        feedback_str = f"\n[Feedback: {feedback}]" if feedback else ""
        train_text   = f"{prompt}\n{correct_answer}{feedback_str}"

        ids = self.tok.encode(train_text, add_special=True)
        if len(ids) < 2:
            print("[learn_from_assessment] Training text too short, skipping.")
            return {}

        # Loss scale: how wrong was the model?
        loss_scale = max(
            self.cfg.min_loss_scale,
            1.0 - (score / max_score),
        )

        self._set_lr(self.cfg.assess_lr_scale)
        losses = []
        t0     = time.time()
        windows = self._windows(ids)

        print(f"[learn_from_assessment]  score={score}/{max_score}  "
              f"loss_scale={loss_scale:.2f}  |  "
              f"{len(ids)} tokens  |  {self.cfg.assess_epochs} epoch(s)")

        for epoch in range(self.cfg.assess_epochs):
            epoch_losses = []
            for window in windows:
                inp, tgt = self._make_targets(window)
                loss_val = self._gradient_step(inp, tgt, loss_scale=loss_scale)
                epoch_losses.append(loss_val)
                if self.step % self.cfg.log_every == 0:
                    print(f"  step {self.step:5d}  loss={loss_val:.4f}  "
                          f"(scaled={loss_val*loss_scale:.4f})")
            mean_epoch = sum(epoch_losses) / len(epoch_losses)
            print(f"  epoch {epoch+1}/{self.cfg.assess_epochs}  "
                  f"mean_loss={mean_epoch:.4f}")
            losses.extend(epoch_losses)

        duration  = time.time() - t0
        mean_loss = sum(losses) / len(losses)
        record = dict(
            event="assessment", score=score, max_score=max_score,
            loss_scale=loss_scale, steps=len(losses),
            mean_loss=mean_loss, duration_s=duration,
        )
        self.history.append(record)
        print(f"[learn_from_assessment] Done  |  steps={len(losses)}  "
              f"mean_loss={mean_loss:.4f}  ({duration:.1f}s)\n")
        return record

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate_response(
        self,
        prompt:         str,
        max_new_tokens: int   = None,
        temperature:    float = None,
        top_k:          int   = None,
        top_p:          float = None,
    ) -> str:
        """Generate a response string from the model given a prompt."""
        self.model.eval()
        cfg = self.cfg

        prompt_ids = self.tok.encode(prompt, add_special=True)
        input_tensor = self._to_device(
            torch.tensor(prompt_ids, dtype=torch.long).unsqueeze(0)
        )

        out_ids = self.model.generate(
            input_tensor,
            max_new_tokens = max_new_tokens or cfg.max_new_tokens,
            temperature    = temperature    or cfg.temperature,
            top_k          = top_k          or cfg.top_k,
            top_p          = top_p          or cfg.top_p,
        )

        return self.tok.decode(out_ids[0].tolist())

    # ------------------------------------------------------------------
    # Checkpoint save / load
    # ------------------------------------------------------------------
    def save(self, path: str):
        """Save full trainer state to a .pt file."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save({
            "model_state":     self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "model_cfg":       self.model.cfg,
            "trainer_cfg":     self.cfg,
            "step":            self.step,
            "history":         self.history,
        }, path)
        print(f"[save] Checkpoint written → {path}  (step {self.step})")

    @classmethod
    def load(cls, path: str, tok: ByteTokenizer = None) -> "Trainer":
        """Restore a Trainer from a checkpoint file."""
        ckpt       = torch.load(path, map_location="cpu", weights_only=False)
        model_cfg  = ckpt["model_cfg"]
        model      = StudentTransformer(model_cfg)
        model.load_state_dict(ckpt["model_state"])

        trainer          = cls(model, tok or ByteTokenizer(), ckpt["trainer_cfg"])
        trainer.optimizer.load_state_dict(ckpt["optimizer_state"])
        trainer.step     = ckpt.get("step", 0)
        trainer.history  = ckpt.get("history", [])
        print(f"[load] Checkpoint restored ← {path}  (step {trainer.step})")
        return trainer

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------
    def print_history(self):
        """Print a compact summary of all training events."""
        if not self.history:
            print("No training history yet.")
            return
        print(f"\n{'─'*60}")
        print(f"{'#':>4}  {'event':<14}  {'steps':>5}  {'mean_loss':>9}  {'info'}")
        print(f"{'─'*60}")
        for i, r in enumerate(self.history):
            info = (f"score={r['score']}/{r['max_score']}  "
                    f"scale={r['loss_scale']:.2f}"
                    if r["event"] == "assessment"
                    else r.get("label", ""))
            print(f"{i+1:>4}  {r['event']:<14}  {r['steps']:>5}  "
                  f"{r['mean_loss']:>9.4f}  {info}")
        print(f"{'─'*60}\n")


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
def _demo():
    tok     = ByteTokenizer()
    model   = StudentTransformer(StudentConfig())
    trainer = Trainer(model, tok)

    # --- Feed a short document ---
    doc = """
    The mitochondria is the powerhouse of the cell.
    It produces ATP through a process called cellular respiration.
    ATP stands for adenosine triphosphate and is the main energy currency of the cell.
    The process occurs in the inner mitochondrial membrane via the electron transport chain.
    Glucose is broken down in glycolysis, then pyruvate enters the Krebs cycle,
    and finally the electron transport chain produces the majority of ATP.
    """
    trainer.learn_document(doc.strip(), label="mitochondria_intro")

    # --- Ask a question before assessment ---
    q = "What does the mitochondria produce?"
    print(f"Q: {q}")
    print(f"A (before assessment): {trainer.generate_response(q)!r}\n")

    # --- Simulate assessment with feedback ---
    trainer.learn_from_assessment(
        prompt         = q,
        correct_answer = "The mitochondria produces ATP through cellular respiration.",
        score          = 2,
        max_score      = 10,
        feedback       = "Mention ATP and cellular respiration explicitly.",
    )

    # --- Ask again after assessment ---
    print(f"Q: {q}")
    print(f"A (after assessment):  {trainer.generate_response(q)!r}\n")

    # --- History ---
    trainer.print_history()

    # --- Save / load round-trip ---
    trainer.save("checkpoints/demo.pt")
    trainer2 = Trainer.load("checkpoints/demo.pt", tok)
    assert trainer2.step == trainer.step
    print("Save/load round-trip: OK")


if __name__ == "__main__":
    _demo()