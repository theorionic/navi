"""Scale-readiness sweep: Pool vs dense at iso-compute, with held-out eval.

Arms (all d256/8L/4H, batch 512, seq 64, held-out fact split):
  D4     dense-only, 4M-fact space, 4k steps          -> dense ceiling at scale
  P4     pool 4x512x512, mem_lr_mult=10, temp=4      -> Pool at scale (Meta levers)
  P4t1   pool 4x512x512, mem_lr_mult=10, temp=1      -> temp ablation
  P4L1   pool 4x512x512, mem_lr_mult=1,  temp=4      -> LR ablation
  P4big  pool 4x2048x2048 (67M slots), pmap sharded  -> slot-space scaling
  M4     MoE-style control: 8 experts top-2, same capacity as P4 -> iso-compute ref
Eval: held-out facts only (sample_eval_batch), plus zero/shuffle sabotage.
"""
import sys, time
sys.path.insert(0, '/kaggle/working')
import jax, jax.numpy as jnp
import optax
import numpy as np
import navi.data as D
from navi.config import MemoryConfig, ModelConfig, TrainConfig
from navi.model import Navi
from navi.train import init_params

N_CORES = jax.device_count()
BS = 512            # global tokens batch (split 8x64 over cores)
SEQ = 64
STEPS = int(__import__('os').environ.get('NAVI_STEPS', '4000'))
LR = 3e-3
EVAL_SEED = 777
N_EVAL = 16          # 16 x 512 x 16 triples = 131k held-out fact draws

CFG_MEM = MemoryConfig(c1=512, c2=512, cand_k=8, side_top=64, n_classes=4,
                       score_temp=4.0)
CFG_MEM_BIG = MemoryConfig(c1=2048, c2=2048, cand_k=8, side_top=64, n_classes=4,
                           score_temp=4.0)
CFG_MEM_T1 = MemoryConfig(c1=512, c2=512, cand_k=8, side_top=64, n_classes=4,
                          score_temp=1.0)
CFG_HASH = MemoryConfig(c1=512, c2=512, n_classes=4, hash_slots=True, hash_k=4)
CFG_HASH_MID = MemoryConfig(c1=1024, c2=1024, n_classes=4, hash_slots=True, hash_k=4)

def loss_fn(model, p, b):
    logits = model.apply(p, b[:, :-1], train=True)
    return optax.softmax_cross_entropy_with_integer_labels(logits, b[:, 1:]).mean()


def is_mem(kp):
    ks = jax.tree_util.keystr(kp)
    return "values" in ks or "/k1" in ks or "/k2" in ks


def make_tx(cfg_t, params):
    core = optax.adamw(
        optax.schedules.warmup_cosine_decay_schedule(
            init_value=LR * 0.02, peak_value=LR,
            decay_steps=cfg_t.total_steps, warmup_steps=cfg_t.warmup_steps),
        b1=0.9, b2=0.95, weight_decay=cfg_t.weight_decay)
    if cfg_t.mem_lr_mult == 1.0:
        return core
    mem = optax.adamw(
        optax.schedules.warmup_cosine_decay_schedule(
            init_value=LR * 0.02, peak_value=LR * cfg_t.mem_lr_mult,
            decay_steps=cfg_t.total_steps, warmup_steps=cfg_t.warmup_steps),
        b1=0.9, b2=0.95, weight_decay=cfg_t.weight_decay)
    labels = jax.tree_util.tree_map_with_path(
        lambda kp, x: "mem" if is_mem(kp) else "core", params)
    return optax.multi_transform({"core": core, "mem": mem}, labels)


def zero_pool(p):
    return jax.tree_util.tree_map_with_path(
        lambda kp, x: jnp.zeros_like(x) if "values" in jax.tree_util.keystr(kp) else x, p)


def shuffle_pool(p, rng):
    def perm(kp, x):
        if "values" in jax.tree_util.keystr(kp):
            n = x.shape[-2]
            return x[..., jax.random.permutation(jax.random.fold_in(rng, n), n), :]
        return x
    return jax.tree_util.tree_map_with_path(perm, p)


def train(model, cfg_t, mem_lr_mult, tag, params=None):
    rng = jax.random.PRNGKey(cfg_t.seed)
    if params is None:
        params = init_params(model, SEQ, jax.random.fold_in(rng, 1))
    tx = make_tx(cfg_t, params)
    opt = tx.init(params)

    @jax.jit
    def step(p, o, b):
        g = jax.grad(lambda pp, bb: loss_fn(model, pp, bb))(p, b)
        u, o2 = tx.update(g, o, p)
        return optax.apply_updates(p, u), o2

    t0 = time.time()
    for i in range(cfg_t.total_steps):
        rng, d = jax.random.split(rng)
        b = jax.device_put(D.sample_batch(d, BS, SEQ))
        params, opt = step(params, opt, b)
        if i % 500 == 0 or i == cfg_t.total_steps - 1:
            l = float(loss_fn(model, params, b))
            print(f"[{tag}] step {i:5d} loss {l:.4f} ({(time.time()-t0)/(i+1):.3f}s/it)", flush=True)
    return params


def evaluate(model, params, tag, label, heldout=True):
    @jax.jit
    def eval_step(p, b):
        logits = model.apply(p, b[:, :-1], train=False)
        return D.fact_recall_acc(logits, b[:, 1:])
    accs = []
    for i in range(N_EVAL):
        r = jax.random.fold_in(jax.random.PRNGKey(EVAL_SEED), i)
        b = jax.device_put(D.sample_eval_batch(r, BS, SEQ) if heldout
                           else D.sample_batch(r, BS, SEQ, train=False))
        accs.append(float(eval_step(params, b)))
    acc = sum(accs) / len(accs)
    print(f"[{tag}] EVAL {label:10s} acc {acc:.4f}", flush=True)
    return acc


def run_arm(name, cfg_m, tag, mem_lr_mult=10.0, sabotages=("none",)):
    model = Navi(cfg_m, CFG_MEM)
    n, sz = 0, 0
    flat = jax.tree_util.tree_flatten_with_path(init_params(model, SEQ, jax.random.PRNGKey(0)))[0]
    sz = sum(p.size for k, p in flat if "mem" in jax.tree_util.keystr(k))
    print(f"[{tag}] == arm {name}: mem params {sz:,}", flush=True)
    cfg_t = TrainConfig(total_steps=STEPS, batch_size=BS, seq_len=SEQ,
                        warmup_steps=min(200, STEPS // 10), log_every=500, mem_lr_mult=mem_lr_mult)
    params = train(model, cfg_t, mem_lr_mult, tag)
    accs = {}
    for s in sabotages:
        p = params
        if s == "zero":
            p = zero_pool(params)
        elif s == "shuffle":
            p = shuffle_pool(params, jax.random.PRNGKey(4242))
        accs[s] = evaluate(model, p, tag, s)
    print(f"[{tag}] RESULT {name}: " + " ".join(f"{k}={v:.4f}" for k, v in accs.items()), flush=True)
    return accs


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else ""
    print(f"== scale sweep: BS={BS} SEQ={SEQ} STEPS={STEPS} cores={N_CORES} ==", flush=True)
    r = {}
    arms = [
        ("D4", "dense", ModelConfig(d_model=256, n_layers=8, n_heads=4, memory_every=0, vocab_size=D.VOCAB), 1.0, ("none",)),
        ("P4", "pool", ModelConfig(d_model=256, n_layers=8, n_heads=4, memory_every=2, vocab_size=D.VOCAB), 10.0, ("none", "zero", "shuffle")),
        ("P4L1", "pool-lr1", ModelConfig(d_model=256, n_layers=8, n_heads=4, memory_every=2, vocab_size=D.VOCAB), 1.0, ("none",)),
        ("P4BIG", "pool-2048", ModelConfig(d_model=256, n_layers=8, n_heads=4, memory_every=2, vocab_size=D.VOCAB), 10.0, ("none", "zero")),
    ]
    for key, name, cfg_m, mult, sab in arms:
        if only and only not in key:
            continue
        if key == "P4BIG":
            globals()["CFG_MEM"] = CFG_MEM_BIG
        else:
            globals()["CFG_MEM"] = CFG_MEM
        r[key] = run_arm(name, cfg_m, key, mem_lr_mult=mult, sabotages=sab)
    print("== SUMMARY ==", flush=True)
    for arm, accs in r.items():
        print(f"{arm:8s}: " + " ".join(f"{k}={v:.4f}" for k, v in accs.items()), flush=True)


if __name__ == "__main__":
    main()