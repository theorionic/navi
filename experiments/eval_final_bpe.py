"""Final eval for bpe500m run22f: GEN samples, tight COV, slot traffic.
Run on the TPU kernel after training completes. Read-only on the ckpt."""
import sys
sys.path.insert(0, "/kaggle/working/code")
sys.path.insert(0, "/kaggle/working/code/tok")
import numpy as np
import optax
from navi.config import MemoryConfig, ModelConfig
from navi.model import Navi
from navi.train import init_params
from grain_parquet_data import BOS, EOS, PhaseFeed
from tokenizers import Tokenizer

TAG = "bpeeval"
CKPT = "/kaggle/working/experiments/ckpt_bpe500m_step019999.pkl"
TOK_PATH = "/kaggle/working/tokenizer_16k.json"
SEQ = 512
CAND_K = 8

mem_cfg = MemoryConfig(c1=512, c2=512, cand_k=CAND_K, side_top=64, n_classes=4,
                       score_temp=4.0)
cfg_m = ModelConfig(d_model=512, n_layers=8, n_heads=8, memory_every=2,
                    vocab_size=16384)
model = Navi(cfg_m, mem_cfg)
model_ra = Navi(cfg_m, mem_cfg, return_aux=True)

print(f"[{TAG}] loading ckpt...", flush=True)
with open(CKPT, "rb") as f:
    st = pickle.load(f)
p = st["params"]
print(f"[{TAG}] ckpt step {st['step']}", flush=True)

# val tokens: same recipe as trainer (PhaseFeed val tail)
feed = PhaseFeed(buffer_mb=256, val_docs=4000)
feed.launch()
feed.load_state()
feed.wait_ready(min_tokens=1, timeout=600)
val = feed.val[:feed.val_end].copy()
print(f"[{TAG}] val buffer {len(val)/1024/1024:.1f}MB", flush=True)


# exec the trainer's generate helpers verbatim (lines 171-210)
src = open("/kaggle/working/code/experiments/train500m_bpe.py").read()
lines = src.splitlines()
helpers = "\n".join(lines[170:211])   # PROMPTS .. generate() incl _gen_scan
ns = dict(globals())
exec(helpers, ns)

def do_gen(model, params, tok, step_i):
    texts = ns["generate"](model, params, tok, ns["PROMPTS"], 128, temp=0.8,
                           seed=step_i + 42)
    for pr, txt in zip(ns["PROMPTS"], texts):
        one = txt.replace("\r", " ").replace("\n", " ")
        print(f"[{TAG}] GEN [{pr!r}] -> {one[:200]}", flush=True)

do_gen(model, p, Tokenizer.from_file(TOK_PATH), st["step"])

# tight COV: 8 batches x 16 windows
def pool_cov(model_ra, params, val_tokens, n_batches=8):
    n_slots = 512 * 512 * 4
    rng = np.random.default_rng(11)
    hi = len(val_tokens) - SEQ - 2
    per_block = {}
    for _ in range(n_batches):
        offs = rng.integers(0, hi, size=16)
        idx = offs[:, None] + np.arange(SEQ + 1)[None, :]
        win = val_tokens[idx]
        ids = jax.device_put(win[:, :-1])
        _, aux, _ = model_ra.apply(params, ids, train=False)
        for name, s in aux.items():
            arr = np.asarray(s)
            sets = per_block.setdefault(name, set())
            for cls in range(arr.shape[-1] // CAND_K):
                sl = arr[..., cls*CAND_K:(cls+1)*CAND_K].reshape(-1)
                sets.update((cls, int(v)) for v in sl)
    names = sorted(per_block)
    return [len(per_block[n]) / n_slots for n in names]

cov = pool_cov(model_ra, p, val)
print(f"[{TAG}] COV-final " + " ".join(f"b{bi}={c*100:.2f}%" for bi, c in enumerate(cov)), flush=True)

# slot-traffic concentration: how peaked is routing per token (top-100 share, Gini)
def slot_traffic(model_ra, params, val_tokens, n_batches=4):
    rng = np.random.default_rng(13)
    hi = len(val_tokens) - SEQ - 2
    counts = {}
    for _ in range(n_batches):
        offs = rng.integers(0, hi, size=16)
        idx = offs[:, None] + np.arange(SEQ + 1)[None, :]
        win = val_tokens[idx]
        ids = jax.device_put(win[:, :-1])
        _, aux, _ = model_ra.apply(params, ids, train=False)
        for name, s in aux.items():
            arr = np.asarray(s).reshape(-1)
            c = counts.setdefault(name, {})
            import collections
            bc = np.bincount(arr, minlength=512*512)
            cur = c.get("bc")
            c["bc"] = bc if cur is None else cur + bc
    out = {}
    for name, c in counts.items():
        bc = c["bc"].astype(np.float64)
        tot = bc.sum()
        top100 = np.sort(bc)[::-1][:100].sum() / tot
        p_ = bc / tot
        gini = float((np.abs(p_[:, None] - p_[None, :]).sum() / (2 * p_.size * p_.sum())))
        alive = int((bc > 0).sum())
        out[name] = (top100, gini, alive)
    return out

tr = slot_traffic(model_ra, p, val)
for name, (t100, g, alive) in sorted(tr.items()):
    print(f"[{TAG}] TRAFFIC {name}: top100share={t100:.4f} alive_slots={alive} "
          f"({alive/(512*512*4)*100:.1f}% of space)", flush=True)
print(f"[{TAG}] DONE", flush=True)
