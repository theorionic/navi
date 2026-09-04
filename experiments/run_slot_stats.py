"""Run the missing slot-stats against the saved final MREAL checkpoint."""
import sys, os, pickle
sys.path.insert(0, "/kaggle/working")
import sweep_real as S

train_split, val_split = S.load_enwik8()
mem_cfg = S.MemoryConfig(c1=512, c2=512, cand_k=8, side_top=64, n_classes=4,
                         score_temp=4.0)
cfg_m = S.ModelConfig(d_model=256, n_layers=8, n_heads=4, memory_every=2,
                      vocab_size=S.VOCAB)
with open("/kaggle/working/experiments/ckpt_MREAL.pkl", "rb") as f:
    state = pickle.load(f)
params, step = state["params"], state["step"]
print(f"== slot stats on final ckpt (step {step}) ==", flush=True)
S.slot_stats(cfg_m, mem_cfg, params, val_split)