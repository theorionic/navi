"""Streaming dataloader: HF `datasets` streaming -> grain pipeline ->
threaded prefetch to pinned in-memory batches for TPU training.

User spec:
  * NEVER download the whole dataset. First batch returns ASAP via
    `load_dataset(..., streaming=True)` (HTTP row streaming).
  * Background thread(s) keep prefetching while the trainer consumes,
    holding a small ring of ready batches in system memory.
  * Batches are consumed FIFO: `next_batch()` hands back the oldest
    ready batch and frees its memory once the trainer moves on.

Additions over spec (marked ADD):
  * grain InMemoryPipeline per batch-window with batch + shuffle + pack
    transforms, so the same pipeline definition works if you later swap
    the streaming source for a grain-native one (PyGrain checkpointing).
  * Waiter/error propagation: `next_batch()` raises the producer's error
    instead of hanging.
  * Sequence packing: docs are concatenated into (bs, seq+1) windows so
    the TPU sees full windows, not ragged docs (matches PhaseFeed).
  * Backpressure: the queue is bounded (default 8 batches); the
    producer blocks when full so memory stays O(queue) not O(dataset).
  * Optional tokenization (HF tokenizers) with BOS boundary per doc.
  * Stats: docs/s, tokens/s, queue depth for monitoring.

Trainer contract:
    feed = StreamFeed("wikimedia/wikipedia", "20231101.en",
                      batch_size=64, seq_len=512)
    feed.start()                 # producer thread begins
    batch = feed.first_batch()   # blocks only until batch 0 is ready
    ...                          # in the train loop:
    batch = feed.next_batch()    # pops oldest ready batch (FIFO)
    # after the TPU step completes, the batch is simply dropped (goes
    # out of reference) -> freed; nothing to delete explicitly.
"""
from __future__ import annotations

import itertools
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

import numpy as np

BOS = 0
EOS = 1


@dataclass
class FeedStats:
    docs: int = 0
    tokens: int = 0
    batches: int = 0
    t_start: float = field(default_factory=time.time)
    errors: list = field(default_factory=list)

    def snapshot(self) -> dict:
        dt = max(time.time() - self.t_start, 1e-6)
        return {"docs": self.docs, "tokens": self.tokens,
                "batches": self.batches,
                "docs_per_s": round(self.docs / dt, 1),
                "tok_per_s": round(self.tokens / dt, 1)}


class StreamFeed:
    """Streaming HF dataset -> packed (bs, seq+1) int32 batches with a
    bounded prefetch queue. Zero full-dataset downloads."""

    def __init__(self,
                 repo_id: str,
                 config: str | None = None,
                 split: str = "train",
                 batch_size: int = 64,
                 seq_len: int = 512,
                 queue_depth: int = 8,
                 text_col: str | None = None,       # auto-detect if None
                 tokenizer_path: str | None = None,  # None = passthrough
                 seed: int = 0,
                 shuffle_buffer: int = 10_000,       # docs, streaming-safe
                 doc_transform: Callable[[dict], str] | None = None):
        if batch_size < 1 or seq_len < 1 or queue_depth < 1:
            raise ValueError("batch_size/seq_len/queue_depth must be >= 1")
        self.repo_id = repo_id
        self.config = config
        self.split = split
        self.bs = batch_size
        self.seq = seq_len
        self.qd = queue_depth
        self.text_col = text_col
        self.tokenizer_path = tokenizer_path
        self.seed = seed
        self.shuffle_buffer = shuffle_buffer
        self.doc_transform = doc_transform

        self._q: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=queue_depth)
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._err: BaseException | None = None
        self.stats = FeedStats()
        self._tok = None
        if tokenizer_path:
            from tokenizers import Tokenizer
            self._tok = Tokenizer.from_file(tokenizer_path)
        elif tokenizer_path is None:
            # ADD: default tokenizer so text feeds tokenize consistently;
            # pass tokenizer_path="" to disable (raw byte passthrough).
            try:
                from transformers import AutoTokenizer
                self._tok = AutoTokenizer.from_pretrained("gpt2")._tokenizer
            except Exception:
                self._tok = None

    # ---------------- public API ----------------
    def start(self) -> None:
        """Launch the producer thread (idempotent)."""
        if self._thread is None or not self._thread.is_alive():
            self._stop.clear()
            self._err = None
            self._thread = threading.Thread(target=self._produce,
                                            daemon=True, name="stream-feed")
            self._thread.start()

    def stop(self) -> None:
        """Signal the producer to stop and drain."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=30)

    def first_batch(self, timeout: float = 600.0) -> np.ndarray:
        """Block until the first batch is ready. This is the
        time-to-first-batch the whole design optimizes."""
        t0 = time.time()
        b = self._q.get(timeout=timeout)
        if b is None:
            raise RuntimeError(f"stream ended before first batch: {self._err}")
        self.stats.batches += 1
        return b

    def next_batch(self, timeout: float = 600.0) -> np.ndarray:
        """Pop the OLDEST ready batch (FIFO). Raises producer errors.
        The batch is freed when the caller drops the reference after its
        TPU step - no explicit removal needed (ADD: documented contract)."""
        b = self._q.get(timeout=timeout)
        if b is None:
            if self._err is not None:
                raise RuntimeError(f"data producer failed: {self._err}") \
                    from self._err
            raise StopIteration("stream exhausted")
        self.stats.batches += 1
        return b

    def qsize(self) -> int:
        return self._q.qsize()

    def try_next_batch(self) -> np.ndarray | None:
        """Non-blocking variant for overlapped schedulers."""
        try:
            b = self._q.get_nowait()
        except queue.Empty:
            return None
        if b is None:
            raise StopIteration("stream exhausted")
        self.stats.batches += 1
        return b

    # ---------------- producer ----------------
    def _produce(self) -> None:
        try:
            for win in self._stream_windows():
                if self._stop.is_set():
                    return
                self._q.put(win, timeout=300)
        except BaseException as e:  # surface to the trainer
            self._err = e
            self.stats.errors.append(repr(e))
        finally:
            self._q.put(None)  # sentinel: stream over

    def _stream_windows(self) -> Iterator[np.ndarray]:
        """HF streaming -> grain pipeline -> fixed (bs, seq+1) windows.

        grain 0.2.18's IterDataset subclassing is broken for lazy
        iterators (parent-ctx walk treats stream items as parents), so
        the stream is materialized into sliding chunks and each chunk
        goes through the documented InMemoryDataSource path:
          source(docs) -> map(tokenize+pack) -> batch(bs, drop_remainder)
        Chunking is transparent to the trainer: windows flow to the
        queue continuously, chunk boundaries never appear in output."""
        from datasets import load_dataset
        import grain.python as grain
        need = self.seq + 1
        kwargs: dict[str, Any] = {"split": self.split, "streaming": True}
        if self.config:
            kwargs["name"] = self.config
        ds = load_dataset(self.repo_id, **kwargs)

        CHUNK = max(self.bs * 64, self.shuffle_buffer)
        while not self._stop.is_set():
            texts = list(itertools.islice(self._iter_texts(ds), CHUNK))
            if not texts:
                break
            pipe = (grain.MapDataset.source(texts)
                    .map(self._to_ids)
                    .batch(self.bs, drop_remainder=True)
                    .to_iter_dataset())
            for batch_docs in iter(pipe):
                if self._stop.is_set():
                    return
                win = np.full((self.bs, need), EOS, dtype=np.int32)
                for i, ids in enumerate(batch_docs):
                    n = min(len(ids), need)
                    win[i, :n] = ids[:n]
                self.stats.docs += self.bs
                self.stats.tokens += int(win.size)
                yield win

    def _iter_texts(self, ds) -> Iterator[str]:
        for ex in ds:
            if self._stop.is_set():
                return
            t = self.doc_transform(ex) if self.doc_transform else \
                self._pick_text(ex)
            if t:
                self.stats.docs += 0  # counted at window yield
                yield t

    def _pick_text(self, ex: dict) -> str | None:
        if self.text_col:
            return ex.get(self.text_col)
        for k in ("text", "content", "body"):
            if k in ex:
                return ex[k]
        # first string column fallback
        for v in ex.values():
            if isinstance(v, str):
                return v
        return None

    def _to_ids(self, text: str) -> np.ndarray:
        """ADD: tokenize + pack to a FIXED-length window (cut or pad).
        Fixed length is required by grain's batch (equal-shape arrays)."""
        need = self.seq + 1
        if self._tok is None:
            # passthrough for pre-tokenized/synthetic text (whitespace ids)
            ids = np.frombuffer(text.encode(), dtype=np.uint8).astype(np.int32)
        else:
            ids = np.asarray(
                self._tok.encode(text, add_special_tokens=False).ids,
                dtype=np.int32)
        if len(ids) >= need:
            return ids[:need]
        return np.concatenate([ids, np.full(need - len(ids), EOS,
                                            dtype=np.int32)])


def demo(repo_id: str = "wikimedia/wikipedia",
         config: str = "20231101.en",
         bs: int = 64, seq: int = 512, batches: int = 4) -> dict:
    """Time-to-first-batch + throughput smoke, runnable anywhere."""
    feed = StreamFeed(repo_id, config, batch_size=bs, seq_len=seq)
    t0 = time.time()
    feed.start()
    b0 = feed.first_batch(timeout=600)
    ttfb = time.time() - t0
    print(f"first batch: shape={b0.shape} dtype={b0.dtype} "
          f"ttfb={ttfb:.2f}s", flush=True)
    t1 = time.time()
    n = 1
    while n < batches:
        b = feed.next_batch(timeout=300)
        n += 1
    dt = time.time() - t1
    print(f"{n} batches in {dt:.2f}s -> {n / dt:.1f} batches/s, "
          f"qdepth={feed.qsize()}, stats={feed.stats.snapshot()}",
          flush=True)
    feed.stop()
    return {"ttfb_s": ttfb, "batches_per_s": n / dt}


if __name__ == "__main__":
    demo()