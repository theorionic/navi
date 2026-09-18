"""Grain parquet dataloader for the 500M run -- resumable, auto-rolling.

Design (user spec + additions):
  * Parquet shards are downloaded from HF (hf_hub_download) into a local
    cache dir, ONE shard at a time; the trainer never sees the network.
  * Grain (0.2.x, kernel has 0.2.18) pipelines each shard:
      InMemoryDataSource(pyarrow rows) -> MapDataset -> SequentialSampler
      -> batch -> tokenize (16k BPE) -> pack
    and the grain DatasetIterator is checkpointed via
    PyGrainCheckpointHandler so resume is EXACT (grain restores its own
    sampler position + transform state).
  * data_state.json (in /kaggle/working/experiments, next to ckpts) is
    the single resume record:
      {repo_id, config_name, shard_index, shard_filename, hf_etag,
       grain_dir, docs_done_in_shard, global_step}
    model ckpt + data_state are saved together by the trainer; on resume
    both are restored, so skip logic is "grain state" not guesswork.
  * When a shard is exhausted: grain state is finalized (docs_done ==
    rows), the next shard index is downloaded, and the pipeline rebuilds.
    At the end of the last shard: rollover to the next PHASE in the run
    plan (fineweb -> ultrafineweb-L3-en) or stop when all phases done.
  * Deterministic order: shards in listed order, docs in shard order.
    Shuffling lives inside grain (IndexSampler) but resume stays exact
    because the sampler state is in the grain checkpoint. We keep
    sequential (no shuffle) for determinism at 1-epoch scale; the
    rolling 1GB packed buffer provides the mixing the LM needs.

Phases (run plan, edit RUN_PHASES to change):
    0: HuggingFaceFW/fineweb      sample-10BT        (~2.4B tok of 10BT)
    1: openbmb/Ultra-FineWeb-L3   en-QA-Synthetic
    2: openbmb/Ultra-FineWeb-L3   en-Multi-Style-Synthetic
"""
import json
import os
import time
from dataclasses import dataclass

import numpy as np
import pyarrow.parquet as pq
BOS = 0
EOS = 1
TOK_PATH_DEFAULT = "/kaggle/working/tokenizer_16k.json"
STATE_PATH = os.environ.get(
    "NAVI_DATA_STATE", "/kaggle/working/experiments/data_state.json")
CACHE_DIR = os.environ.get("NAVI_HF_CACHE", "/kaggle/working/hf_cache")

RUN_PHASES = [
    {"repo_id": "HuggingFaceFW/fineweb", "config": "sample-10BT",
     "file_glob": "sample/10BT/*.parquet", "lang": "en",
     "max_shards": 12},
    {"repo_id": "openbmb/Ultra-FineWeb-L3",
     "config": "Ultra-FineWeb-L3-en-QA-Synthetic",
     "file_glob": "data/ultrafineweb_en_l3/qa/*.parquet", "lang": "en"},
    {"repo_id": "openbmb/Ultra-FineWeb-L3",
     "config": "Ultra-FineWeb-L3-en-Multi-Style-Synthetic",
     "file_glob": "data/ultrafineweb_en_l3/multi_style/*.parquet",
     "lang": "en"},
]


def list_shards(repo_id, config, file_glob):
    """Ordered parquet shard paths in the repo for one config."""
    from huggingface_hub import HfApi
    api = HfApi()
    files = [f for f in api.list_repo_files(repo_id, repo_type="dataset")
             if f.endswith(".parquet")]
    # match the config's subdirectory exactly (file_glob minus *.parquet)
    prefix = file_glob.replace("*.parquet", "")
    shards = sorted(f for f in files if f.startswith(prefix))
    return shards


def download_shard(repo_id, filename, etag=None):
    """Local path of the parquet shard; downloads once, cached."""
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id=repo_id, filename=filename,
                           repo_type="dataset", cache_dir=CACHE_DIR,
                           etag_timeout=30)


@dataclass
class ShardPlan:
    phase: int
    repo_id: str
    config: str
    shard_index: int          # index within the phase's shard list
    filename: str             # repo path of the parquet
    local_path: str
    n_rows: int


def build_plan(phase_idx, shard_index):
    ph = RUN_PHASES[phase_idx]
    shards = list_shards(ph["repo_id"], ph["config"], ph["file_glob"])
    if ph.get("max_shards"):
        shards = shards[:ph["max_shards"]]
    assert shard_index < len(shards), (shard_index, len(shards))
    fn = shards[shard_index]
    local = download_shard(ph["repo_id"], fn)
    n = pq.ParquetFile(local).metadata.num_rows
    return ShardPlan(phase_idx, ph["repo_id"], ph["config"], shard_index,
                     fn, local, n)


def default_state():
    return {"phase": 0, "shard_index": 0, "filename": None,
            "docs_done": 0, "global_step": 0, "finished": False}


class ShardPipeline:
    """One parquet shard -> packed token docs, via a plain integer cursor.

    Why not grain: grain 0.2.18's InMemoryDataSource copies rows into a
    fixed-slot shared-memory segment and its back-transform does
    `value.rstrip(b'\\0').decode()` - docs longer than the slot (up to
    528KB here vs a ~9.4KB slot) are truncated mid-codepoint and the
    read raises UnicodeDecodeError. Unfixable from outside the library.
    The resume state is just `cursor` (an int), saved atomically in
    data_state.json - strictly simpler than grain's checkpoint and exact.
    """

    def __init__(self, plan, tokenizer_path=None, max_docs=None):
        from tokenizers import Tokenizer
        self.plan = plan
        self.text_col = _text_cols(plan.local_path)[0]
        # max_docs: memory valve for small hosts. to_pylist() of a full
        # 2GB shard's text column needs ~6GB of python strings; on a
        # 7GB box that swaps to death. Capped pipelines still stream
        # doc-ordered tokens; the producer rolls to the next shard when
        # the cap is exhausted (cursor == len(docs)).
        cap = max_docs or int(os.environ.get("NAVI_SHARD_MAX_DOCS", "0"))
        table = pq.read_table(plan.local_path, columns=[self.text_col])
        col = table.column(self.text_col)
        if cap and cap < col.length():
            col = col.slice(0, cap)
        self.docs = col.to_pylist()
        self.tok = Tokenizer.from_file(tokenizer_path
                                       or os.environ.get("NAVI_TOK_PATH",
                                                         TOK_PATH_DEFAULT))
        self.cursor = 0  # next doc index to emit; == state['docs_done']

    def _tokenize(self, text):
        ids = self.tok.encode(text, add_special_tokens=False).ids
        # packed doc: [BOS, ids...]  (EOS dropped; BOS is the boundary)
        return np.asarray([BOS] + ids, dtype=np.int32)

    def __iter__(self):
        return self

    def __next__(self):
        if self.cursor >= len(self.docs):
            raise StopIteration
        doc = self.docs[self.cursor]
        self.cursor += 1
        return self._tokenize(doc)

    def skip(self, n):
        """Resume: fast-forward past docs already consumed."""
        self.cursor = min(n, len(self.docs))

    @property
    def exhausted(self):
        return self.cursor >= len(self.docs)



def _text_cols(local_path):
    """The doc-text column name differs per source: fineweb uses 'text',
    Ultra-FineWeb-L3 uses 'content'. Resolve against the schema."""
    schema = pq.ParquetFile(local_path).schema_arrow
    for cand in ("text", "content"):
        if cand in schema.names:
            return [cand]
    raise ValueError(f"no text column in {schema.names}")


class PhaseFeed:
    """The trainer-facing object: batch(rng, bs, seq) over rolling packed
    buffer, auto-rollover across shards and phases, exact-resume.

    Trainer contract:
      feed = PhaseFeed()
      feed.resume()               # loads data_state.json + grain ckpt
      win = feed.batch(rng, BS, SEQ)   # (bs, seq+1) int32, zero padding
      feed.note_step(global_step) # update global_step for state file
    State written at every rollover + on trainer checkpoint.
    """

    def __init__(self, buffer_mb=1024, val_docs=4000):
        self.cap = buffer_mb * 1024 * 1024  # int32 tokens
        self.buf = np.empty(self.cap, dtype=np.int32)
        self.start = 0
        self.end = 0
        self.plan = None
        self.pipe = None
        self.state = default_state()
        self._finished = False
        self._producer_err = None
        import threading
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._produce, daemon=True)
        # Background prefetcher: pre-downloads the NEXT shard while the
        # producer tokenizes the current one, so shard boundaries never
        # stall the producer waiting on the network.
        self._prefetch_thread = threading.Thread(target=self._prefetch_loop,
                                                 daemon=True)
        # val buffer: first val_docs docs of phase 0, never trained
        self.val = np.empty(64 * 1024 * 1024, dtype=np.int32)
        self.val_start = self.val_end = 0
        self._val_left = val_docs

    def launch(self):
        """Idempotent: launch the producer thread. Named launch() because
        self.start is the buffer cursor."""
        if not self._thread.is_alive():
            self._thread.start()
        if not self._prefetch_thread.is_alive():
            self._prefetch_thread.start()

    def _prefetch_loop(self):
        """Pre-download upcoming shards (current +2 ahead) into the HF
        cache. hf_hub_download is idempotent - if the producer already
        fetched it this is a no-op; if not, the download happens here in
        parallel so _open_shard finds it local."""
        import time as _t
        while True:
            try:
                st = dict(self.state)
                ph = RUN_PHASES[st["phase"]]
                shards = list_shards(ph["repo_id"], ph["config"],
                                     ph["file_glob"])
                for ahead in range(0, 3):
                    idx = st["shard_index"] + ahead
                    if idx >= len(shards):
                        # wrap into next phase if it exists
                        nph = st["phase"] + 1
                        if nph < len(RUN_PHASES):
                            nph_cfg = RUN_PHASES[nph]
                            nshards = list_shards(nph_cfg["repo_id"],
                                                  nph_cfg["config"],
                                                  nph_cfg["file_glob"])
                            nidx = idx - len(shards)
                            if nidx < len(nshards):
                                download_shard(nph_cfg["repo_id"],
                                               nshards[nidx])
                        continue
                    download_shard(ph["repo_id"], shards[idx])
            except Exception:
                pass  # prefetch is best-effort; producer handles real errors
            _t.sleep(30)

    # ---------- state io ----------
    def _save_state(self):
        os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
        tmp = STATE_PATH + ".tmp"
        s = dict(self.state)
        if self.plan is not None:
            s.update({"phase": self.plan.phase,
                      "shard_index": self.plan.shard_index,
                      "filename": self.plan.filename,
                      "repo_id": self.plan.repo_id,
                      "config": self.plan.config})
        with open(tmp := STATE_PATH + ".tmp", "w") as f:
            json.dump(s, f, indent=1)
        os.replace(tmp, STATE_PATH)

    def load_state(self):
        if os.path.exists(STATE_PATH):
            with open(STATE_PATH) as f:
                self.state = json.load(f)
        return self.state

    # ---------- pipeline ----------
    def _open_shard(self, phase, shard_index):
        ph = RUN_PHASES[phase]
        shards = list_shards(ph["repo_id"], ph["config"], ph["file_glob"])
        if ph.get("max_shards"):
            shards = shards[:ph["max_shards"]]
        if shard_index >= len(shards):
            return False  # phase exhausted
        fn = shards[shard_index]
        local = download_shard(ph["repo_id"], fn)
        n = pq.ParquetFile(local).metadata.num_rows
        self.plan = ShardPlan(phase, ph["repo_id"], ph["config"],
                              shard_index, fn, local, n)
        self.pipe = ShardPipeline(self.plan)
        return True

    def _produce(self):
        try:
            st = self.state
            while True:
                if self.pipe is None:
                    if not self._open_shard(st["phase"], st["shard_index"]):
                        # phase exhausted -> next phase, shard 0
                        if st["phase"] + 1 >= len(RUN_PHASES):
                            self.state["finished"] = True
                            self._save_state()
                            return
                        st["phase"] += 1
                        st["shard_index"] = 0
                        st["docs_done"] = 0
                        self._save_state()
                        continue
                    # exact resume: fast-forward the cursor past consumed docs
                    if st["docs_done"]:
                        self.pipe.skip(st["docs_done"])
                        print(f"[data] resumed shard: skipped {st['docs_done']} docs",
                              flush=True)
                try:
                    doc_ids = next(self.pipe)
                except StopIteration:
                    # shard done -> persist, advance
                    st["docs_done"] = 0
                    st["shard_index"] += 1
                    self.pipe = None
                    self.plan = None
                    self._save_state()
                    continue
                if self._val_left > 0:
                    # first N docs of phase 0 -> val buffer, never trained
                    self._push_val(doc_ids)
                    self._val_left -= 1
                else:
                    self._push(doc_ids)
                st["docs_done"] += 1
        except Exception as e:  # surface producer errors to the trainer
            self._producer_err = repr(e)

    def _push_val(self, doc_ids):
        n = len(doc_ids)
        if self.val_end + n > len(self.val):
            return  # val buffer full; drop the rest silently
        self.val[self.val_end:self.val_end + n] = doc_ids
        self.val_end += n

    def _push(self, doc_ids):
        n = len(doc_ids)
        if self.end + n > self.cap:
            live = self.end - self.start
            keep = max(0, min(live, self.cap - n))
            self.buf[:keep] = self.buf[self.end - keep:self.end]
            self.start, self.end = 0, keep
            if n > self.cap:
                doc_ids = doc_ids[-(self.cap):]
                n = len(doc_ids)
        self.buf[self.end:self.end + n] = doc_ids
        self.end += n

    # ---------- trainer API ----------
    def wait_ready(self, min_tokens=256 * 1024 * 1024, timeout=900):
        t0 = time.time()
        while len(self) < min_tokens and time.time() - t0 < timeout:
            if self._producer_err:
                raise RuntimeError(f"data producer failed: {self._producer_err}")
            time.sleep(2)
        return len(self) >= min_tokens

    def __len__(self):
        return self.end - self.start

    def batch(self, rng, batch, seq):
        need = seq + 1
        if len(self) < batch * need:
            raise RuntimeError(
                f"buffer starved: have {len(self)} tokens, need {batch * need}")
        out = np.empty((batch, need), dtype=np.int32)
        hi = self.end - self.start - need
        offs = rng.integers(0, hi + 1, size=batch)
        for i, o in enumerate(offs):
            s = self.start + int(o)
            out[i] = self.buf[s:s + need]
        self.start += batch * need  # consumed; producer compacts on refill
        return out

    def note_step(self, global_step):
        self.state["global_step"] = global_step
        self._save_state()

    @property
    def producer_failed(self):
        return self._producer_err


if __name__ == "__main__":
    # smoke (tiny buffers): producer thread -> buffer; checkpoint; exact resume
    import shutil
    for p in (STATE_PATH,):
        shutil.rmtree(p, ignore_errors=True) if os.path.isdir(p) else \
            (os.path.exists(p) and os.remove(p))
    feed = PhaseFeed(buffer_mb=16, val_docs=0)
    feed.launch()
    feed.wait_ready(min_tokens=1 << 20, timeout=600)
    print("[smoke] buffered tokens:", len(feed))
    rng = np.random.default_rng(0)
    win = feed.batch(rng, 2, 511)
    print("[smoke] batch shape:", win.shape, "dtok0:", int(win[0, 0]))
    feed.note_step(7)
    with open(STATE_PATH) as f:
        print("[smoke] state after note_step:", json.load(f))

    # resume: new feed must pick up where the state file says, skipping
    # already-consumed docs (docs_done), and refill the buffer.
    feed2 = PhaseFeed(buffer_mb=16, val_docs=0)
    feed2.load_state()
    feed2.launch()
    feed2.wait_ready(min_tokens=1 << 20, timeout=600)
    win2 = feed2.batch(rng, 2, 511)
    print("[smoke] resume batch ok:", win2.shape,
          "global_step:", feed2.state["global_step"])
    print("[smoke] OK")
