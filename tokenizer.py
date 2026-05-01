"""
tokenizer.py — Byte-level frozen tokenizer
-------------------------------------------
Vocab is permanently fixed at 260 IDs:

  ID 0  <pad>   — padding / null
  ID 1  <unk>   — unknown (never actually produced, kept for compatibility)
  ID 2  <bos>   — beginning of sequence
  ID 3  <eos>   — end of sequence
  ID 4–259      — raw UTF-8 byte values 0x00–0xFF

No training needed. Instantiate once and reuse forever.
The embedding table in your transformer should have vocab_size=260.
"""

from __future__ import annotations

from typing import List


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PAD_ID  = 0
UNK_ID  = 1
BOS_ID  = 2
EOS_ID  = 3
BYTE_OFFSET = 4          # byte b maps to token ID  b + BYTE_OFFSET
VOCAB_SIZE  = 256 + BYTE_OFFSET   # 260


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------
class ByteTokenizer:
    """Fixed-vocabulary byte-level tokenizer.

    Encodes any UTF-8 string to a list of integer IDs in the range [0, 259].
    Decodes back losslessly. No state, no training, no files.

    Special tokens
    --------------
    <pad>  0   — use for padding batches to equal length
    <unk>  1   — reserved; never emitted by encode()
    <bos>  2   — prepended when add_special=True
    <eos>  3   — appended  when add_special=True

    Byte tokens
    -----------
    byte b  ->  ID  b + 4   (e.g. ord('A') = 65  ->  ID 69)
    """

    # Class-level constants so they're accessible without an instance
    PAD_ID      = PAD_ID
    UNK_ID      = UNK_ID
    BOS_ID      = BOS_ID
    EOS_ID      = EOS_ID
    VOCAB_SIZE  = VOCAB_SIZE

    # Human-readable names for the four special IDs
    SPECIAL_TOKENS = {PAD_ID: "<pad>", UNK_ID: "<unk>",
                      BOS_ID: "<bos>", EOS_ID: "<eos>"}

    # ------------------------------------------------------------------
    # Encode
    # ------------------------------------------------------------------
    def encode(
        self,
        text: str,
        add_special: bool = False,
        max_length: int | None = None,
    ) -> List[int]:
        """Encode *text* to a list of token IDs.

        Parameters
        ----------
        text:
            Any Python string (unicode fully supported).
        add_special:
            If True, wrap the output with BOS_ID … EOS_ID.
        max_length:
            If set, truncate to this many tokens *after* adding specials.
            Truncation removes from the right, preserving BOS if present.
        """
        ids = [b + BYTE_OFFSET for b in text.encode("utf-8")]

        if add_special:
            ids = [BOS_ID] + ids + [EOS_ID]

        if max_length is not None and len(ids) > max_length:
            # Keep BOS, truncate body, always end with EOS when add_special
            if add_special:
                ids = ids[:max_length - 1] + [EOS_ID]
            else:
                ids = ids[:max_length]

        return ids

    def encode_batch(
        self,
        texts: List[str],
        add_special: bool = False,
        pad: bool = True,
        max_length: int | None = None,
    ) -> List[List[int]]:
        """Encode a list of strings, optionally padding to equal length.

        Parameters
        ----------
        pad:
            If True, right-pad every sequence with PAD_ID so all rows are
            the same length (useful for batching into tensors).
        """
        encoded = [self.encode(t, add_special=add_special,
                               max_length=max_length) for t in texts]
        if pad and encoded:
            max_len = max(len(s) for s in encoded)
            encoded = [s + [PAD_ID] * (max_len - len(s)) for s in encoded]
        return encoded

    # ------------------------------------------------------------------
    # Decode
    # ------------------------------------------------------------------
    def decode(
        self,
        ids: List[int],
        skip_special: bool = True,
    ) -> str:
        """Decode a list of token IDs back to a string.

        Parameters
        ----------
        skip_special:
            If True (default), strip <pad>, <bos>, <eos>, <unk> before
            decoding. Set to False if you want to inspect the raw output.
        """
        if skip_special:
            ids = [i for i in ids if i not in self.SPECIAL_TOKENS]

        raw_bytes = bytes(
            i - BYTE_OFFSET
            for i in ids
            if BYTE_OFFSET <= i < BYTE_OFFSET + 256
        )
        return raw_bytes.decode("utf-8", errors="replace")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def id_to_token(self, id: int) -> str:
        """Human-readable label for a single ID (useful for debugging)."""
        if id in self.SPECIAL_TOKENS:
            return self.SPECIAL_TOKENS[id]
        if BYTE_OFFSET <= id < BYTE_OFFSET + 256:
            b = id - BYTE_OFFSET
            ch = bytes([b]).decode("utf-8", errors="replace")
            return f"0x{b:02X}({ch!r})"
        return f"<invalid:{id}>"

    def token_ids_for(self, text: str) -> dict:
        """Show the byte-to-ID mapping for every character in *text*.

        Handy during development to inspect what the model actually sees.
        """
        result = {}
        for ch in text:
            bs = ch.encode("utf-8")
            ids = [b + BYTE_OFFSET for b in bs]
            result[ch] = ids
        return result

    @property
    def vocab_size(self) -> int:
        return VOCAB_SIZE

    def __repr__(self) -> str:
        return f"ByteTokenizer(vocab_size={VOCAB_SIZE}, fixed=True)"


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
def _demo():
    tok = ByteTokenizer()
    print(tok)
    print(f"PAD={tok.PAD_ID}  UNK={tok.UNK_ID}  BOS={tok.BOS_ID}  EOS={tok.EOS_ID}\n")

    # --- Basic encode / decode ---
    sample = "Hello, world! 🌍"
    ids = tok.encode(sample, add_special=True)
    recovered = tok.decode(ids)
    print(f"Original : {sample!r}")
    print(f"IDs      : {ids}")
    print(f"Decoded  : {recovered!r}")
    assert recovered == sample, "Round-trip failed!"
    print("Round-trip: OK\n")

    # --- Truncation ---
    long_text = "A" * 20
    truncated = tok.encode(long_text, add_special=True, max_length=10)
    print(f"Truncated (max_length=10): {truncated}")
    assert truncated[0] == BOS_ID and truncated[-1] == EOS_ID
    assert len(truncated) == 10
    print("Truncation: OK\n")

    # --- Batch encode with padding ---
    texts = ["hi", "hello", "hey there"]
    batch = tok.encode_batch(texts, add_special=True, pad=True)
    lengths = [len(row) for row in batch]
    print(f"Batch (padded to len={lengths[0]}):")
    for t, row in zip(texts, batch):
        print(f"  {t!r:12s} -> {row}")
    assert len(set(lengths)) == 1, "Padding failed — rows differ in length"
    print("Batch padding: OK\n")

    # --- ID inspection ---
    print("Byte mapping for 'Hi!':")
    for ch, ids in tok.token_ids_for("Hi!").items():
        labels = [tok.id_to_token(i) for i in ids]
        print(f"  {ch!r} -> IDs {ids} -> {labels}")


if __name__ == "__main__":
    _demo()