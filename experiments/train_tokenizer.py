"""Train the production BPE tokenizer for the 500M run — on the TPU kernel.

Vocab sizing rule (MXU fit): the model's per-class value dim is
d_model / n_classes = 512/4 = 128, and the MXU-native tile on v5e is
128x128. We want vocab_size ≡ 0 (mod 128) so embed/head matrices and any
padding-blocked loads are exactly tiled:

    16384 = 128 x 128  -> perfectly MXU-aligned, zero padding.

    0         <|bos|>
    1         <|eos|>
    2..257    the 256 byte-fallback tokens (ByteLevel alphabet)
    258..16383  learned BPE merges

    (HF tokenizers assigns special ids FIRST, then the byte alphabet;
    verified 2026-09-14. BOS=0, EOS=1 for the model config.)

Corpus: FineWeb sample-10BT first ~2.5GB of text (matches training data
distribution exactly). Byte-fallback BPE so nothing is ever UNK.

Writes tokenizer_16k.json to /kaggle/working/ (synced to the repo via the
FUSE mount by the caller).
"""
import json
import os
import time

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

VOCAB = 16384          # 128-aligned: MXU-friendly
SPECIAL = 258          # bytes + BOS + EOS
CORPUS_GB = 2.5
OUT = "/kaggle/working/tokenizer_16k.json"
SAMPLE = "/kaggle/working/tokenizer_corpus.txt"

from datasets import load_dataset

def build_corpus():
    if os.path.exists(SAMPLE) and os.path.getsize(SAMPLE) > CORPUS_GB * 1e9 * 0.95:
        print(f"[tok] corpus exists ({os.path.getsize(SAMPLE)/1e9:.2f} GB), skip download")
        return
    t0 = time.time()
    ds = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT",
                      split="train", streaming=True)
    n_bytes = 0
    with open(SAMPLE, "wb") as f:
        for doc in ds:
            b = doc["text"].encode("utf-8")
            f.write(b)
            f.write(b"\n\n")
            n_bytes += len(b) + 2
            if n_bytes >= CORPUS_GB * 1e9:
                break
    print(f"[tok] corpus {n_bytes/1e9:.2f} GB in {time.time()-t0:.0f}s")

def train():
    t0 = time.time()
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB,
        special_tokens=["<|bos|>", "<|eos|>"],   # ids 256, 257 after 256 byte tokens
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tok.train([SAMPLE], trainer)
    tok.save(OUT)
    print(f"[tok] trained in {time.time()-t0:.0f}s -> {OUT}")

    # verify: byte-roundtrip + id layout
    e = tok.encode("Hello, world — the quick brown fox jumps.")
    assert tok.decode(e.ids) == "Hello, world — the quick brown fox jumps."
    b = tok.encode(bytearray(range(256)).decode("latin1"))
    assert tok.decode(b.ids), "byte fallback broken"
    v = tok.get_vocab()
    bos = v["<|bos|>"]; eos = v["<|eos|>"]
    print(f"[tok] vocab {len(v)} | bos {bos} eos {eos} (expect 0/1)")
    assert bos == 0 and eos == 1
    print(f"[tok] sample ids: {e.ids[:16]}")
    print(f"[tok] compression: {len('Hello, world — the quick brown fox jumps.')*1.0/len(e.ids):.2f} bytes/token")

from tokenizers import decoders  # noqa: E402

build_corpus()
train()
print("[tok] OK")