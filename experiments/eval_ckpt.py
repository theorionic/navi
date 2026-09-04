"""Early eval on the current MREAL checkpoint: banks the science before
the kernel stops. Runs alongside training (small eval batches fit TPU).

Usage: python3 eval_ckpt.py
- loads ckpt_MREAL.pkl (latest atomic state)
- evaluates held-out bpc on: intact / zeroed values / shuffled values
- prints slot-usage stats
- appends a RESULT line to sweep_real_mem.log
"""
import sys, os, pickle, time
sys.path.insert(0, "/kaggle/working")
os.environ.setdefault("NAVI_STEPS", "8000")
import jax, jax.numpy as jnp
import numpy as np
import sweep_real as S
from navi.model import Navi

CKPT = "/kaggle/working/experiments/ckpt_MREAL.pkl"

train_split, val_split = S.load_enwik8()
mem_cfg = S.MemoryConfig(c1=512, c2=512, cand_k=8, side_top=64, n_classes=4,
                         score_temp=4.0)
cfg_m = S.ModelConfig(d_model=256, n_layers=8, n_heads=4, memory_every=2,
                      vocab_size=S.VOCAB)
model = Navi(cfg_m, mem_cfg)
with open(CKPT, "rb") as f:
    state = pickle.load(f)
params = state["params"]
step = state["step"]
print(f"== early eval on ckpt at step {step} ==", flush=True)

bpc = S.evaluate(model, params, "MREAL-EARLY", "none", val_split)
bpc_zero = S.evaluate(model, S.zero_pool(params), "MREAL-EARLY", "zero", val_split)
bpc_shuf = S.evaluate(model, S.shuffle_pool(params, jax.random.PRNGKey(4242)),
                      "MREAL-EARLY", "shuffle", val_split)
S.slot_stats(cfg_m, mem_cfg, params, val_split)

lines = [
    f"[MREAL-EARLY] RESULT early@{step}: bpc={bpc:.4f} zero={bpc_zero:.4f} "
    f"shuffle={bpc_shuf:.4f}",
    f"[MREAL-EARLY] delta: pool-contrib={bpc_zero-bpc:+.4f} (zero-intact) "
    f"shuffle-contrib={bpc_shuf-bpc:+.4f}",
]
with open("/kaggle/working/experiments/sweep_real_mem.log", "a") as f:
    for ln in lines:
        f.write(ln + "\n")
for ln in lines:
    print(ln, flush=True)