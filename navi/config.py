"""Model + memory hyperparameters in one place; sized to fit 7 GB RAM / CPU."""

from flax import struct


@struct.dataclass
class ModelConfig:
    vocab_size: int = 339  # matches navi.data.VOCAB (16 keys + 64 nonces + 256 values)
    d_model: int = 128
    n_layers: int = 6
    n_heads: int = 4
    dropout: float = 0.0
    # memory-layer positions: replace every other FFN, Meta-style; 0 = FFN baseline
    memory_from_layer: int = 0
    memory_every: int = 2

    @property
    def head_dim(self) -> int:
        if self.d_model % self.n_heads:
            raise ValueError(f"d_model={self.d_model} not divisible by n_heads={self.n_heads}")
        return self.d_model // self.n_heads


@struct.dataclass
class MemoryConfig:
    # The slot space is c1 x c2 per class. CPU: 64 x 64 x 4 classes = 16k slots.
    # Scale path: c1=c2=2048 (16.8M slots/class x 4), cand_k 8-32, side_top 64-256.
    cand_k: int = 8  # slots (subkey pairs) retrieved per token per class
    side_top: int = 16  # per-side candidates before the product top-k (16 keeps grid small)
    n_classes: int = 4  # independent key/value planes
    c1: int = 64
    c2: int = 64
    # Meta's value-noise regularizer for huge memories; requires rngs at apply.
    value_noise: float = 0.0
    # Sharper readout: softmax(scores * score_temp) over cand_k gathers.
    score_temp: float = 1.0
    # Hash placement: slot candidates = hash(context token IDs) mod n_slots
    # per position, bypassing learned key routing for WRITES/reads alike.
    # Deterministic first-touch gradients: no key sharpening needed at scale.
    hash_slots: bool = False
    hash_k: int = 4  # hash-derived slots gathered per position per class
    # Router-balance aux: lb_weight * mean negative entropy of the
    # side-score softmaxes (s1/s2). Spreads candidate mass off hot
    # subkeys; gradients reach all side keys. Scale O(1).
    lb_weight: float = 0.0
    # Split the read: w = softmax(temp * scores) * (1 - lb_eps) + lb_eps/n
    # -- floor on unselected-slot value mass; 0 = off.
    lb_eps: float = 0.0
    # E2: 3-factor pool. c3 >= 2 enables a third subkey table and a
    # conditional 3-stage top-k (exact for additive grids). Slot index =
    # (i1*c2 + i2)*c3 + i3. Values table becomes (C, c1*c2*c3, D).
    c3: int = 0
    cand2: int = 8   # stage-2 candidates per stage-1 survivor
    cand3: int = 4   # stage-3 candidates per (i1,i2) pair
    # E3: hybrid read -- mix hash-derived rows into the candidate set.
    hybrid_hash: bool = False
    hybrid_eps: float = 0.15   # weight floor for hash rows in the softmax
    # E7: usage-weighted hash -- bias the hash slot choice toward cold
    # slots via a host-maintained visit-count table (passed as visit_off
    # third element). No learned params, no new loss.
    usage_hash: bool = False
    # E6: learned injection head -- small Dense on x producing scores
    # over the flattened slot grid; top inject_k become appended
    # candidates (like hash rows) with the same hybrid_eps floor.
    learned_inject: bool = False
    inject_k: int = 2          # injected candidates per token per class
    inject_hidden: int = 64    # head hidden width
    # E8: drift-gated exploration -- hybrid_eps scales with token
    # distribution drift. eps_t = hybrid_eps * (1 + drift_gain *
    # drift_t), where drift_t is the host-computed KL(batch unigram vs
    # EMA unigram), normalized 0..1. Zero cost on steady data; boosts
    # injection exactly when new topics arrive.
    drift_gate: bool = False
    drift_boost: float = 2.0   # eps multiplier at full drift
    # E4: exploration offset on side scores (host-updated visit EMA).
    explore_beta: float = 0.0
    # E9: on-device usage-balance. A per-slot touch-fraction EMA (maintained
    # inside the training step, returned to the host as data) biases the
    # SELECTION scores before top-k: s_sel = s - balance_beta * rel(usage),
    # rel = clip(usage/mean - 1, -1, 1). Hot subkeys lose the selection
    # race, cold ones re-enter; the readout softmax keeps RAW scores. As
    # usage equalizes the offsets vanish and learned routing dominates:
    # a self-erasing negative feedback, not a permanent distortion.
    balance_beta: float = 0.0
    balance_ema: float = 0.999  # visit-table decay (1/EMA = smoothing window)
    # DENSE read path: softmax over the FULL c1*c2 grid per token per class.
    # Every key/value row gets gradient each step (no top-k selection =>
    # no zero-gradient death). Value rows are combined in fixed-size token
    # chunks (dense_chunk) so transient memory is O(chunk * n_slots),
    # independent of pool AND backbone size -- any pool size on any host.
    # The forward readout is the same functional form as the sparse path
    # (softmax(temp*scores) weighted value sum), so dense-trained params
    # transfer to the sparse two-sided read with no export step.
    dense_warmup_steps: int = 0  # >0: dense read for first N steps (train.py owns the flag flip)
    dense_chunk: int = 32  # tokens per dense chunk; transient ~ chunk * c1*c2 * D fp32

@struct.dataclass
class TrainConfig:
    lr: float = 3e-3
    # Multiplicative LR boost for Pool params (keys + values) vs backbone.
    # Meta's memory-layer recipe: memory trains much faster than the backbone.
    mem_lr_mult: float = 1.0
    warmup_steps: int = 100
    total_steps: int = 1500
    batch_size: int = 64
    seq_len: int = 64
    weight_decay: float = 0.01
    log_every: int = 200
    seed: int = 0