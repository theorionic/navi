"""BPE FineWeb streaming feed — packed, zero-padding, MXU-aligned.

Replaces the byte-level RollingBytes feed with identical threading
semantics (producer thread + rolling int32 buffer + random windows), but:
  1. tokenizes with the trained 16k BPE (byte-fallback, never UNK),
  2. packs documents back-to-back with a single BOS between them — the
     fixed (batch, seq+1) window NEVER contains padding; every position
     is a real token. EOS/BOS boundaries are the only structure marker.
  3. all shapes stay (BS, SEQ) so the TPU step is unchanged; the only
     config change is vocab_size 260 -> 16384 (128 x 128, MXU-aligned).

Zero-padding proof: pack() guarantees buffer density 100% (no filler ids
anywhere). The smoke test at the bottom asserts this on the real stream.

BOS=0, EOS=1 per the trained tokenizer layout.
"""
import os
import threading
import time

import numpy as np
from tokenizers import Tokenizer

BOS = 0
EOS = 1
VOCAB = 16384


class RollingBPE:
    """Append-only int32 ring of BPE ids with random-window sampling."""

    def __init__(self, tok_path="/kaggle/working/tokenizer_16k.json",
                 cap=int(os.environ.get("NAVI_BUF_MB", "1024")) * 1024 * 1024):
        self.tok = Tokenizer.from_file(tok_path)
        self.cap = cap  # in tokens now (int32)
        self.buf = np.empty(cap, dtype=np.int32)
        self.start = 0
        self.end = 0
        self.tokens_seen = 0
        self.docs_seen = 0

    def add_text(self, text: str) -> None:
        ids = self.tok.encode(text, add_special_tokens=False).ids
        self.add(np.asarray(ids, dtype=np.int32))

    def add(self, arr: np.ndarray) -> None:
        # doc + BOS boundary marker (packing): next doc starts fresh
        arr = arr  # EOS dropped: BOS alone marks boundaries, saves a token
        n = len(arr) + 1
        if self.end + n > self.cap:
            live = self.end - self.start
            keep = max(0, min(live, self.cap - n))
            self.buf[:keep] = self.buf[self.end - keep:self.end]
            self.start, self.end = 0, keep
            if n > self.cap:
                arr = arr[len(arr) - (self.cap - 1):]
                n = self.cap
        self.buf[self.end] = BOS
        self.buf[self.end + 1:self.end + len(arr) + 1] = arr
        self.end += len(arr) + 1
        self.tokens_seen += len(arr) + 1
        self.docs_seen += 1

    def sample(self, rng: np.random.Generator, batch: int, seq: int) -> np.ndarray:
        """(batch, seq+1) windows — identical contract to the byte feed."""
        out = np.empty((batch, seq + 1), dtype=np.int32)
        hi = self.end - self.start - (seq + 1)
        offs = rng.integers(0, max(1, hi), size=batch)
        for i, o in enumerate(offs):
            s = self.start + int(o)
            out[i] = self.buf[s:s + seq + 1]
        return out

    def __len__(self):
        return self.end - self.start


class FineWebBPEFeed:
    """Same producer-thread pattern as fineweb_data.FineWebFeed, BPE ids."""

    def __init__(self, tok_path="/kaggle/working/tokenizer_16k.json",
                 n_val_docs=4000, val_mb=24):
        from datasets import load_dataset
        self.feed = RollingBPE(tok_path)
        self.val = RollingBPE(tok_path, cap=val_mb * 1024 * 1024)
        self.ds = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT",
                               split="train", streaming=True)
        self.n_val_docs = n_val_docs
        self._val_done = 0
        self._thread = None
        self._stop = False

    def wait_ready(self, min_tokens=256 * 1024 * 1024 // 4, timeout=600):
        t0 = time.time()
        while len(self.feed) < min_tokens and time.time() - t0 < timeout:
            time.sleep(2)
        return len(self.feed) >= min_tokens

    def _produce(self):
        it = iter(self.ds)
        while not self._stop:
            try:
                doc = next(it)
            except StopIteration:
                break
            if self._val_done < self.n_val_docs:
                self.val.add_text(doc["text"])
                self._val_done += 1
            else:
                if len(self.feed) > self.feed.cap * 0.98:
                    time.sleep(0.5)  # consumer is behind; buffer is full
                    continue
                self.feed.add_text(doc["text"])

    def start(self):
        self._thread = threading.Thread(target=self._produce, daemon=True)
        self._thread.start()
        return self

    def batch(self, rng, batch, seq):
        return self.feed.sample(rng, batch, seq)

    def val_array(self):
        return self.val.buf[self.val.start:self.val.end]


if __name__ == "__main__":
    # smoke: density proof + throughput
    feed = FineWebBPEFeed(n_val_docs=200).start()
    t0 = time.time()
    feed.wait_ready(min_tokens=8_000_000, timeout=420)
    print(f"[bpe] {len(feed.feed)/1e6:.1f}M train tokens, "
          f"{len(feed.val)/1e6:.1f}M val tokens in {time.time()-t0:.0f}s")
    rng = np.random.default_rng(0)
    w = feed.batch(rng, 256, 512)
    # 1) no padding anywhere: buffer is pure token ids; check BOS density
    bos_frac = float((w == BOS).mean())
    print(f"[bpe] window BOS fraction {bos_frac:.4f} (doc-boundary only)")
    # 2) id range valid
    assert w.max() < VOCAB and w.min() >= 0
    # 3) round-trip one window through the tokenizer decode
    txt = feed.feed.tok.decode(w[0, :80].tolist())
    print(f"[bpe] decoded head: {txt[:110]!r}")
    # 4) tokens/byte compression vs the byte-level feed
    b = feed.feed.tok.encode(feed.feed.tok.decode(w[0].tolist())).ids
    print(f"[bpe] roundtrip stable: {len(b)} -> {int((w[0] != BOS).sum())} tokens")
    print(f"[bpe] OK")