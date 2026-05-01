"""
model.py — Generative Transformer (student LLM)
-------------------------------------------------
A small decoder-only transformer designed for real-time incremental training.

Architecture
------------
  - Byte-level vocab  : 260 tokens  (from tokenizer.py)
  - Embedding         : token + learned positional
  - Decoder blocks    : N × (causal self-attention + FFN)
  - Tied weights      : embedding matrix reused for output projection
  - Layer norm        : pre-norm (more stable for continual training)

Default config (StudentConfig) is tuned to train on CPU or a modest GPU
while still being expressive enough to learn from documents and assessments.

Usage
-----
    from model import StudentTransformer, StudentConfig
    cfg = StudentConfig()          # tweak as needed
    model = StudentTransformer(cfg)
    logits = model(input_ids)      # (B, T) -> (B, T, vocab_size)
    loss   = model(input_ids, targets=input_ids[:, 1:])
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class StudentConfig:
    # Vocabulary (must match tokenizer.VOCAB_SIZE)
    vocab_size: int = 260

    # Sequence
    max_seq_len: int = 512          # context window in tokens

    # Architecture
    n_layers: int    = 6            # transformer blocks
    n_heads: int     = 8            # attention heads
    d_model: int     = 256          # embedding / hidden dimension
    d_ff: int        = 1024         # feed-forward inner dimension
    dropout: float   = 0.1

    # Special token IDs (must match tokenizer.py)
    pad_id: int = 0
    bos_id: int = 2
    eos_id: int = 3

    def __post_init__(self):
        assert self.d_model % self.n_heads == 0, (
            f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
        )

    @property
    def d_head(self) -> int:
        return self.d_model // self.n_heads

    def n_params(self, model: nn.Module) -> int:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Causal self-attention
# ---------------------------------------------------------------------------
class CausalSelfAttention(nn.Module):
    """Multi-head causal (masked) self-attention."""

    def __init__(self, cfg: StudentConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.d_head  = cfg.d_head
        self.d_model = cfg.d_model

        # Fused QKV projection
        self.qkv_proj = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.out_proj  = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.dropout   = nn.Dropout(cfg.dropout)

        # Causal mask — registered as a buffer so it moves with .to(device)
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(cfg.max_seq_len, cfg.max_seq_len, dtype=torch.bool))
            .unsqueeze(0).unsqueeze(0),   # (1, 1, T, T)
        )

    def forward(self, x: torch.Tensor, pad_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, C = x.shape

        # Project and split into Q, K, V
        qkv = self.qkv_proj(x)                        # (B, T, 3C)
        q, k, v = qkv.split(self.d_model, dim=-1)     # each (B, T, C)

        # Reshape to (B, n_heads, T, d_head)
        def split_heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)

        # Scaled dot-product attention
        scale  = math.sqrt(self.d_head)
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale  # (B, H, T, T)

        # Apply causal mask
        scores = scores.masked_fill(~self.causal_mask[:, :, :T, :T], float("-inf"))

        # Apply padding mask if provided — shape (B, 1, 1, T)
        if pad_mask is not None:
            scores = scores.masked_fill(pad_mask, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Weighted sum and merge heads
        out = torch.matmul(attn, v)                            # (B, H, T, d_head)
        out = out.transpose(1, 2).contiguous().view(B, T, C)   # (B, T, C)
        return self.out_proj(out)


# ---------------------------------------------------------------------------
# Feed-forward block
# ---------------------------------------------------------------------------
class FeedForward(nn.Module):
    """Position-wise FFN: Linear -> GELU -> Linear."""

    def __init__(self, cfg: StudentConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff, bias=False),
            nn.GELU(),
            nn.Linear(cfg.d_ff, cfg.d_model, bias=False),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Transformer block (pre-norm)
# ---------------------------------------------------------------------------
class TransformerBlock(nn.Module):
    """Pre-LayerNorm decoder block: LN -> Attn -> residual, LN -> FFN -> residual."""

    def __init__(self, cfg: StudentConfig):
        super().__init__()
        self.ln1  = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2  = nn.LayerNorm(cfg.d_model)
        self.ffn  = FeedForward(cfg)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, pad_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.drop(self.attn(self.ln1(x), pad_mask=pad_mask))
        x = x + self.drop(self.ffn(self.ln2(x)))
        return x


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------
class StudentTransformer(nn.Module):
    """Decoder-only transformer for the student LLM.

    Parameters
    ----------
    cfg: StudentConfig
        Hyperparameters and special token IDs.
    """

    def __init__(self, cfg: StudentConfig):
        super().__init__()
        self.cfg = cfg

        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model,
                                      padding_idx=cfg.pad_id)
        self.pos_emb   = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.emb_drop  = nn.Dropout(cfg.dropout)

        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_out = nn.LayerNorm(cfg.d_model)

        # Output projection — tied to token embedding weights
        # This reduces parameters and improves generalisation
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight   # weight tying

        self._init_weights()

    def _init_weights(self):
        """GPT-2 style initialisation."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()

    def _pad_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Build additive attention mask for PAD tokens.

        Returns shape (B, 1, 1, T) — True where we want to mask.
        """
        return (input_ids == self.cfg.pad_id).unsqueeze(1).unsqueeze(2)

    def forward(
        self,
        input_ids: torch.Tensor,                  # (B, T)
        targets:   Optional[torch.Tensor] = None, # (B, T) shifted labels
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass.

        Returns
        -------
        logits: (B, T, vocab_size)
        loss:   scalar cross-entropy if targets provided, else None
        """
        B, T = input_ids.shape
        assert T <= self.cfg.max_seq_len, (
            f"Sequence length {T} exceeds max_seq_len {self.cfg.max_seq_len}"
        )
        device = input_ids.device

        # Embeddings
        positions = torch.arange(T, device=device).unsqueeze(0)  # (1, T)
        x = self.emb_drop(self.token_emb(input_ids) + self.pos_emb(positions))

        # Padding mask
        pad_mask = self._pad_mask(input_ids)

        # Transformer blocks
        for block in self.blocks:
            x = block(x, pad_mask=pad_mask)

        x = self.ln_out(x)
        logits = self.lm_head(x)   # (B, T, vocab_size)

        # Loss
        loss = None
        if targets is not None:
            # targets should be input_ids shifted left by 1
            # Flatten for cross-entropy: ignore PAD positions
            loss = F.cross_entropy(
                logits.view(-1, self.cfg.vocab_size),
                targets.reshape(-1),
                ignore_index=self.cfg.pad_id,
            )

        return logits, loss

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,        # (1, T) — already on correct device
        max_new_tokens: int = 200,
        temperature: float = 0.8,
        top_k: int = 40,
        top_p: float = 0.9,
        eos_id: Optional[int] = None,
    ) -> torch.Tensor:
        """Autoregressive generation with top-k + top-p (nucleus) sampling.

        Parameters
        ----------
        temperature:
            > 1 = more random, < 1 = more focused. 0 = greedy.
        top_k:
            Keep only the top-k logits before sampling.
        top_p:
            Nucleus sampling threshold (0–1). Use 1.0 to disable.
        eos_id:
            Stop early when this token is produced.
        """
        eos = eos_id if eos_id is not None else self.cfg.eos_id
        ids = prompt_ids.clone()

        for _ in range(max_new_tokens):
            # Crop to context window
            ctx = ids[:, -self.cfg.max_seq_len:]
            logits, _ = self.forward(ctx)
            logits = logits[:, -1, :]   # last position: (1, vocab_size)

            if temperature == 0.0:
                next_id = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature

                # Top-k
                if top_k > 0:
                    top_k_clamped = min(top_k, logits.size(-1))
                    kth_val = logits.topk(top_k_clamped, dim=-1).values[:, -1, None]
                    logits = logits.masked_fill(logits < kth_val, float("-inf"))

                # Top-p (nucleus)
                if 0.0 < top_p < 1.0:
                    sorted_logits, sorted_idx = logits.sort(dim=-1, descending=True)
                    cumprobs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
                    # Remove tokens where cumulative prob exceeds top_p
                    remove = cumprobs - sorted_logits.softmax(dim=-1) > top_p
                    sorted_logits[remove] = float("-inf")
                    # Scatter back to original ordering
                    logits = torch.zeros_like(logits).scatter_(
                        1, sorted_idx, sorted_logits
                    )

                probs   = F.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)

            ids = torch.cat([ids, next_id], dim=1)

            if next_id.item() == eos:
                break

        return ids[:, prompt_ids.shape[1]:]   # return only new tokens

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        cfg = self.cfg
        return (
            f"StudentTransformer("
            f"layers={cfg.n_layers}, heads={cfg.n_heads}, "
            f"d_model={cfg.d_model}, d_ff={cfg.d_ff}, "
            f"vocab={cfg.vocab_size}, ctx={cfg.max_seq_len}, "
            f"params={self.num_params():,})"
        )


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
def _demo():
    import time

    cfg   = StudentConfig()
    model = StudentTransformer(cfg)
    print(model)
    print(f"\nParameter breakdown:")
    print(f"  token_emb : {model.token_emb.weight.numel():>10,}")
    print(f"  pos_emb   : {model.pos_emb.weight.numel():>10,}")
    total_block = sum(p.numel() for b in model.blocks for p in b.parameters())
    print(f"  blocks×{cfg.n_layers}  : {total_block:>10,}")
    print(f"  ln_out    : {sum(p.numel() for p in model.ln_out.parameters()):>10,}")
    print(f"  lm_head   : tied (0 extra params)")
    print(f"  TOTAL     : {model.num_params():>10,}")

    # Forward pass sanity check
    B, T = 2, 64
    ids  = torch.randint(4, 260, (B, T))   # random byte tokens (skip specials)
    tgt  = torch.randint(4, 260, (B, T))

    t0 = time.time()
    logits, loss = model(ids, targets=tgt)
    print(f"\nForward pass:  logits={tuple(logits.shape)}  loss={loss.item():.4f}  "
          f"({(time.time()-t0)*1000:.1f}ms)")

    # Generation sanity check
    from tokenizer import ByteTokenizer
    tok    = ByteTokenizer()
    prompt = "The mitochondria"
    p_ids  = torch.tensor([tok.encode(prompt, add_special=True)])

    t0  = time.time()
    out = model.generate(p_ids, max_new_tokens=40, temperature=0.8)
    txt = tok.decode(out[0].tolist())
    print(f"Generation:    prompt={prompt!r}")
    print(f"               output={txt!r}  ({(time.time()-t0)*1000:.1f}ms)")
    print("\n(random output expected — model is untrained)")


if __name__ == "__main__":
    _demo()