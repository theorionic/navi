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