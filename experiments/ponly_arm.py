import sys
sys.path.insert(0, "/content")
import attribution as A
from navi.config import ModelConfig

r = {}
r["ponly"] = A.run_arm(
    "pool-only", ModelConfig(d_model=256, n_layers=8, n_heads=4, memory_every=1),
    "ponly", sabotages=("none", "zero"),
)
print("== POOL-ONLY SUMMARY ==", flush=True)
for arm, accs in r.items():
    print(f"{arm:12s}: " + " ".join(f"{k}={v:.4f}" for k, v in accs.items()), flush=True)
