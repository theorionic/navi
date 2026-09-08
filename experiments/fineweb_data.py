"""HuggingFace FineWeb (sample-10BT) streaming loader for byte-level LM.

Producer thread pulls documents over the HF streaming API, tokenizes to
UTF-8 bytes (BOS + bytes + EOS per doc) and tops up a rolling int32
buffer (~1GB of text). The consumer only ever samples random windows
from that buffer, so network jitter never touches the training loop --
the buffer IS the prefetch. On buffer-full, the older half is dropped
(fine-web is stationary enough at this scale that a 500MB mixing window
is plenty). The first VAL_DOCS documents are diverted to a held-out
val buffer before the training stream starts.
"""
import os
import threading
import time

import numpy as np

BOS = 256
EOS = 257


class RollingBytes:
    """Append-only byte ring with random-window sampling."""

    def __init__(self, cap=int(os.environ.get("NAVI_BUF_MB", "1024")) * 1024 * 1024):
        self.cap = cap
        self.buf = np.empty(cap, dtype=np.int32)
        self.start = 0   # logical start (data may have been compacted)
        self.end = 0     # logical end
        self.bytes_seen = 0

    def __len__(self):
        return self.end - self.start

    def add(self, arr: np.ndarray) -> None:
        n = len(arr)
        if self.end + n > self.cap:
            # slide: keep the newest (cap - n) bytes of live data so the
            # doc always fits. keep==cap (buffer frozen at start==0) was a
            # livelock: free=0 -> every later doc silently dropped.
            live = self.end - self.start
            keep = max(0, min(live, self.cap - n))
            self.buf[:keep] = self.buf[self.end - keep:self.end]
            self.start, self.end = 0, keep
            if n > self.cap:  # pathological: doc bigger than the buffer
                arr = arr[len(arr) - self.cap:]
                n = len(arr)
        self.buf[self.end:self.end + n] = arr
        self.end += n
        self.bytes_seen += n

    def sample(self, rng: np.random.Generator, batch: int, seq: int) -> np.ndarray:
        """(batch, seq+1) windows, BOS-padded like sweep_real.batch_iter."""
        if len(self) < seq + 2:
            raise RuntimeError(f"buffer starved: {len(self)} bytes")
        max_off = len(self) - seq - 2
        offs = self.start + rng.integers(0, max_off, size=batch)
        idx = offs[:, None] + np.arange(seq + 1)[None, :]
        win = self.buf[idx]
        win[:, 0] = BOS
        return win


def token_stream(doc_iter, n_val_docs=4000, val_cap=24 * 1024 * 1024):
    """Yields ('val', arr) for the first n_val_docs, then ('train', arr).

    val docs are capped at val_cap bytes total; train docs stream forever.
    """
    for i, doc in enumerate(doc_iter):
        text = doc["text"]
        if not text:
            continue
        ids = np.frombuffer(text.encode("utf-8", errors="ignore"), dtype=np.uint8)
        ids = ids[ids < 256].astype(np.int32)
        ids = np.concatenate([[BOS], ids, [EOS]]).astype(np.int32)
        yield ("val" if i < n_val_docs else "train", ids)


class FineWebFeed:
    """Background-prefetched FineWeb byte stream + val buffer."""

    def __init__(self, n_val_docs=4000):
        import datasets  # deferred: import cost paid on the kernel only
        self.stream = datasets.load_dataset(
            "HuggingFaceFW/fineweb", "sample-10BT",
            split="train", streaming=True)
        self.val = RollingBufferVal()
        self.train_buf = RollingBytes()
        self.docs = 0
        self.err = None
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, args=(n_val_docs,), daemon=True)
        self._thread.start()

    def _alive(self) -> bool:
        return self._thread.is_alive()

    def wait_ready(self, min_bytes=64 * 1024 * 1024, timeout_s=600) -> None:
        """Block until the train buffer holds min_bytes (or producer died)."""
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            if not self._alive():
                raise RuntimeError(f"feed producer died: {self.err}")
            with self._lock:
                if len(self.train_buf) >= min_bytes:
                    return
            time.sleep(2)
        raise RuntimeError(f"feed not ready after {timeout_s}s: {self.err}")

    def _run(self, n_val_docs):
        # HF streaming drops connections occasionally; retry the whole
        # stream forever -- buffer decouples us from outages up to
        # minutes long. (Streaming restarts from doc 0; with sample-10BT
        # that means re-reading the same prefix, acceptable at this scale
        # and simpler than resuming at an offset.)
        backoff = 5
        while True:
            try:
                for kind, ids in token_stream(iter(self.stream), n_val_docs=n_val_docs):
                    with self._lock:
                        self.docs += 1
                        if kind == "val" and not self.val.full:
                            self.val.add(ids)
                        else:
                            self.train_buf.add(ids)
                backoff = 5  # clean end of stream (shouldn't happen; loop)
            except Exception as e:  # surface + retry
                self.err = repr(e)
            time.sleep(backoff)
            backoff = min(backoff * 2, 120)

    def batch(self, rng: np.random.Generator, batch: int, seq: int) -> np.ndarray:
        with self._lock:
            return self.train_buf.sample(rng, batch, seq)


class RollingBufferVal:
    """Small capped store of val bytes (list of docs, sampled jointly)."""

    def __init__(self, cap=24 * 1024 * 1024):
        self.cap = cap
        self.chunks = []
        self.total = 0
        self.full = False

    def add(self, ids: np.ndarray) -> None:
        if self.full:
            return
        self.chunks.append(ids)
        self.total += len(ids)
        if self.total >= self.cap:
            self.full = True

    def array(self) -> np.ndarray:
        return np.concatenate(self.chunks)