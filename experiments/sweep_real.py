"""Byte-level language modeling on a real corpus: the Pool vs dense on
real data, at the same scale where the synthetic fact-recall results hold.

Dataset: enwik8 (100MB of Wikipedia XML, ~536M characters... no, 100M bytes).
- vocab: raw bytes (256), BOS token prepended to each window
- split: first 90M bytes train, last 10M bytes held-out test
- task: next-byte prediction (bpc = bits-per-byte = CE(log2))
- arms: MEM (Pool every other FFN) vs DENSE (all FFN), identical backbone,
        plus Pool sabotage (zero / shuffle) on the trained MEM model to
        attribute what the Pool contributes on real text
- also report slot-usage entropy / distinct-slot stats on held-out text:
  routing collapse detection on real data
"""

import sys
sys.path.insert(0, "/kaggle/working")
import os, time, pickle
import jax, jax.numpy as jnp
import numpy as np
import optax
from navi.config import MemoryConfig, ModelConfig, TrainConfig
from navi.model import Navi
from navi.train import init_params

# ---------------------------------------------------------------- data ----
VOCAB = 260          # 0-255 bytes + BOS(256) + EOS(257)? keep 258 free: 260
BOS = 256
SEQ = 64
BS = 512
STEPS = int(os.environ.get("NAVI_STEPS", "4000"))
LR = 3e-3
N_CORES = jax.device_count()
EVAL_BS = 256

# ---------------------------------------------------------------- data ----

def load_enwik8(path=os.environ.get("NAVI_DATA", "/kaggle/working/experiments/enwik8")):
    """Download on demand; returns train/val byte arrays + train-set stats."""
    import zipfile, urllib.request
    raw = path + ".raw"
    if os.path.exists(raw):
        data = open(raw, "rb").read()
    elif os.path.exists(path) and os.path.getsize(path) == 100_000_000:
        data = open(path, "rb").read()  # already the extracted corpus
    elif os.path.exists(path):
        with zipfile.ZipFile(path) as z:
            data = z.read("enwik8")
        with open(raw, "wb") as f:
            f.write(data)
    else:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        url = "https://mattmahoney.net/dc/enwik8.zip"
        print(f"downloading {url} ...", flush=True)
        urllib.request.urlretrieve(url, path)
        with zipfile.ZipFile(path) as z:
            data = z.read("enwik8")
        with open(raw, "wb") as f:
            f.write(data)
    print(f"loaded {len(data):,} bytes", flush=True)
    arr = np.frombuffer(data, dtype=np.uint8).astype(np.int32)
    n_train = 90_000_000
    return arr[:n_train], arr[n_train:]  # 90M train / 10M test bytes


def batch_iter(split, rng_key, batch, seq, offset=0):
    """Random windows from a byte array; returns (batch, seq+1) int32 with
    a BOS prepended at position 0 of each window."""
    data = split
    max_off = len(data) - seq - 1
    starts = np.asarray(jax.random.randint(rng_key, (batch,), 0, max_off))
    idx = starts[:, None] + np.arange(seq + 1)[None, :]  # (batch, seq+1)
    win = data[idx]                                       # gather bytes
    win = np.concatenate([np.full((batch, 1), BOS, np.int32), win[:, :-1]], axis=1)
    return win  # inputs; targets are the original window


def loss_fn(model, p, ids, tg):
    logits = model.apply(p, ids, train=True)
    return optax.softmax_cross_entropy_with_integer_labels(logits, tg).mean()


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


def train(model, cfg_t, mem_lr_mult, tag, train_split, ckpt_path=None):
    rng = jax.random.PRNGKey(cfg_t.seed)
    start = 0
    params = init_params(model, SEQ, jax.random.fold_in(rng, 1))
    tx = make_tx(cfg_t, params)
    opt = tx.init(params)
    if ckpt_path and os.path.exists(ckpt_path):
        with open(ckpt_path, "rb") as f:
            state = pickle.load(f)
        params, opt, rng, start = state["params"], state["opt"], state["rng"], state["step"] + 1
        print(f"[{tag}] RESUMED from {ckpt_path} at step {start}", flush=True)

    @jax.jit
    def step(p, o, ids, tg):
        g = jax.grad(lambda pp, a, t: loss_fn(model, pp, a, t))(p, ids, tg)
        u, o2 = tx.update(g, o, p)
        return optax.apply_updates(p, u), o2

    t0 = time.time()
    losses = []
    for i in range(start, cfg_t.total_steps):
        rng, d = jax.random.split(rng)
        win = batch_iter(train_split, d, BS, SEQ)
        ids = jax.device_put(win[:, :-1])
        tg = jax.device_put(win[:, 1:])
        params, opt = step(params, opt, ids, tg)
        if i % 250 == 0 or i == cfg_t.total_steps - 1:
            l = float(loss_fn(model, params, ids, tg))
            losses.append(l)
            print(f"[{tag}] step {i:5d} loss {l:.4f} bpc {l/np.log(2):.4f} "
                  f"({(time.time()-t0)/(i+1):.3f}s/it)", flush=True)
        if ckpt_path and (i % 500 == 499 or i == cfg_t.total_steps - 1):
            with open(ckpt_path + ".tmp", "wb") as f:
                pickle.dump({"params": params, "opt": opt, "rng": rng,
                             "step": i, "losses": losses}, f)
            os.replace(ckpt_path + ".tmp", ckpt_path)  # atomic
            print(f"[{tag}] CKPT saved at step {i}", flush=True)
            if os.path.exists(ckpt_path + ".evalflag"):
                _early_eval(model, cfg_m, mem_cfg, params, i, val_split, tag)
                os.remove(ckpt_path + ".evalflag")
    return params


def _early_eval(model, cfg_m, mem_cfg, params, step_i, val_split, tag):
    """In-process eval triggered by ckpt path + '.evalflag' sentinel."""
    print(f"[{tag}-EARLY] eval on params at step {step_i}", flush=True)
    bpc = evaluate(model, params, f"{tag}-EARLY", "none", val_split)
    b0 = evaluate(model, zero_pool(params), f"{tag}-EARLY", "zero", val_split)
    b1 = evaluate(model, shuffle_pool(params, jax.random.PRNGKey(4242)),
                  f"{tag}-EARLY", "shuffle", val_split)
    slot_stats(cfg_m, mem_cfg, params, val_split)
    with open("/kaggle/working/experiments/early_eval.txt", "w") as f:
        f.write(f"step={step_i} bpc={bpc:.4f} zero={b0:.4f} shuffle={b1:.4f}\n")
    print(f"[{tag}-EARLY] RESULT step={step_i} bpc={bpc:.4f} "
          f"zero={b0:.4f} shuffle={b1:.4f}", flush=True)


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


def evaluate(model, params, tag, label, val_split, n_batches=40):
    """Held-out next-byte CE in bits-per-byte on contiguous val windows."""
    @jax.jit
    def eval_step(p, ids, tg):
        logits = model.apply(p, ids, train=False)
        return optax.softmax_cross_entropy_with_integer_labels(logits, tg).mean()

    ces = []
    for i in range(n_batches):
        r = jax.random.fold_in(jax.random.PRNGKey(2026), i)
        win = batch_iter(val_split, r, EVAL_BS, SEQ)
        ces.append(float(eval_step(params,
                                   jax.device_put(win[:, :-1]),
                                   jax.device_put(win[:, 1:]))))
    bpc = float(np.mean(ces)) / np.log(2)
    print(f"[{tag}] EVAL {label:10s} bpc {bpc:.4f}", flush=True)
    return bpc


def slot_stats(model_cfg, mem_cfg, params, val_split):
    """Routing health on real text: distinct slots touched, traffic Gini,
    top-1% share. High Gini / tiny distinct count = routing collapse."""
    model = Navi(model_cfg, mem_cfg, return_aux=True)
    @jax.jit
    def slots_of(p, ids):
        _, aux, _lb = model.apply(p, ids, train=False)
        return aux
    per_layer = {}
    aux_first = None
    for i in range(20):
        r = jax.random.fold_in(jax.random.PRNGKey(99), i)
        win = batch_iter(val_split, r, EVAL_BS, SEQ)
        aux = slots_of(params, jax.device_put(win[:, :-1]))
        if aux_first is None:
            aux_first = aux
        for k in sorted(aux):  # mem_0, mem_2, ...
            a = np.asarray(aux[k])
            s = a.reshape(-1, a.shape[-1])  # (b*l, cand_k)
            counts = np.bincount(s.reshape(-1), minlength=mem_cfg.n_classes * mem_cfg.c1 * mem_cfg.c2)
            gini = (2 * (np.arange(1, len(counts)+1) * np.sort(counts.astype(np.float64))).sum()
                    / (len(counts) * counts.sum()) - (len(counts) + 1) / len(counts))
            top1 = float(np.sort(counts.astype(np.float64))[-max(1, len(counts)//100):].sum() / counts.sum())
            per_layer[k] = (int((counts > 0).sum()), float(gini), top1)
    print("[slots] layer: distinct/total, gini, top1%share", flush=True)
    for k, (d, g, t) in per_layer.items():
        print(f"[slots] {k}: {d}/{mem_cfg.n_classes * mem_cfg.c1 * mem_cfg.c2} "
              f"gini={g:.3f} top1%={t:.3f}", flush=True)
    return per_layer


def run_arm(name, cfg_m, mem_cfg, tag, mem_lr_mult, sabotages, train_split, val_split):
    global BS
    if "MREAL" in tag:
        BS = 256  # Pool grads+moments for 270M mem params need the headroom
    model = Navi(cfg_m, mem_cfg)
    flat = jax.tree_util.tree_flatten_with_path(init_params(model, SEQ, jax.random.PRNGKey(0)))[0]
    sz = sum(p.size for _, p in flat)
    msz = sum(p.size for k, p in flat if "mem" in jax.tree_util.keystr(k))
    print(f"[{tag}] == arm {name}: params {sz:,} (mem {msz:,})", flush=True)
    cfg_t = TrainConfig(total_steps=STEPS, batch_size=BS, seq_len=SEQ,
                        warmup_steps=min(200, STEPS // 10), log_every=250,
                        mem_lr_mult=mem_lr_mult)
    ckpt = f"/kaggle/working/experiments/ckpt_{tag}.pkl"
    params = train(model, cfg_t, mem_lr_mult, tag, train_split, ckpt_path=ckpt)
    results = {"bpc": evaluate(model, params, tag, "none", val_split)}
    for s in sabotages:
        p = params
        if s == "zero":
            p = zero_pool(params)
        elif s == "shuffle":
            p = shuffle_pool(params, jax.random.PRNGKey(4242))
        results[s] = evaluate(model, p, tag, s, val_split)
    if mem_cfg is not None and "MEM" in tag:
        results["slots"] = slot_stats(cfg_m, mem_cfg, params, val_split)
    print(f"[{tag}] RESULT {name}: " +
          " ".join(f"{k}={v:.4f}" for k, v in results.items() if k != "slots"), flush=True)
    return params, results


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else ""
    train_split, val_split = load_enwik8()
    print(f"== real-data sweep: enwik8 train {len(train_split):,} bytes, "
          f"val {len(val_split):,} bytes, BS={BS} SEQ={SEQ} STEPS={STEPS} "
          f"cores={N_CORES} ==", flush=True)
    mem_cfg = MemoryConfig(c1=512, c2=512, cand_k=8, side_top=64, n_classes=4,
                           score_temp=4.0)
    arms = [
        ("DREAL", "dense", ModelConfig(d_model=256, n_layers=8, n_heads=4,
                                       memory_every=0, vocab_size=VOCAB),
         mem_cfg, 1.0, ()),
        ("MREAL", "pool", ModelConfig(d_model=256, n_layers=8, n_heads=4,
                                      memory_every=2, vocab_size=VOCAB),
         mem_cfg, 10.0, ("zero", "shuffle")),
    ]
    out = {}
    for key, name, cfg_m, mc, mult, sab in arms:
        if only and only not in key:
            continue
        out[key] = run_arm(name, cfg_m, mc, key, mult, sab, train_split, val_split)
    print("== SUMMARY (bpc, lower=better) ==", flush=True)
    for arm, (params, res) in out.items():
        print(f"{arm:8s}: " + " ".join(f"{k}={v:.4f}" for k, v in res.items()
                                        if k != "slots"), flush=True)


if __name__ == "__main__":
    main()