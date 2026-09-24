"""Autopilot: an intelligent monitor that sits beside the training loop.

Every log interval the trainer feeds it the debug signals it already
computes (loss, grad norms, per-block coverage, step, temp) plus
optionally raw batch stats. The autopilot validates the data (NaN,
stale, out-of-range), runs a rule engine over the recent history, and
returns AT MOST ONE hyperparameter change per evaluation - the single
change predicted to most reduce loss / improve healthy coverage.

Design constraints:
  * Conservative by default: one change at a time, min-cooldown between
    changes, every change logged with the rule + evidence that fired.
  * Changes are APPLIED IN PROCESS without stopping training:
      - mem lb_weight / lb_eps: read from MemoryConfig by pkm.py each
        step (loss_fn captures mem_cfg object) -> mutating the
        dataclass field takes effect on next jitted call. Safe: float
        fields only, no shape change.
      - router score_temp: same mechanism.
      - grad-scale for memory params (10x in trainer): not live-tunable
        without rejit -> exposed as a recommendation only.
      - pool-shape changes (c1/c2/side_top): shape-changing -> the
        autopilot flags them as REBUILD recommendations, never applies.
  * Bounded history; rules are pure functions of history windows.

Signals consumed (all optional; rules skip missing ones):
  step, loss (nats), gn_core, gn_mem, coverage per block (dict b0..),
  lr, temp, queue depth, docs_per_s (feed health), nan flags.

Rules (priority order; first fire wins):
  R1 NaN/Inf guard      -> recommend immediate rollback to last ckpt
  R2 loss plateau       -> if loss slope ~0 over W and grad norm low,
                           raise LR x1.5 (bounded)
  R3 loss spike         -> if loss > mu + 4*sigma and transient, ignore;
                           if persistent 2 evals, halve LR
  R4 router collapse    -> coverage falling AND min(block cov) < 8% and
                           lb_weight == 0 -> set lb_weight 0.05
  R5 router still flat  -> lb_weight active but min block cov still
                           falling after cooldown -> lb_weight x2 (cap 0.2)
  R6 dead deep block    -> deep block cov < 10% while block0 > 20% for
                           3 evals -> REBUILD recommendation (move/shrink)
  R7 feed starvation    -> queue depth 0 or docs/s < 10 for 3 evals ->
                           REBUILD recommendation (parallel producers)
  R8 gn_mem explosion   -> gn_mem / (gn_core+eps) > 10 -> recommend
                           lowering mem grad scale (trainer-side knob)
"""
from __future__ import annotations

import dataclasses
import math
import time
from collections import deque
from dataclasses import dataclass, field


@dataclass
class Action:
    """A single autopilot decision."""
    step: int
    rule: str
    param: str          # dotted path into the live config
    value: object       # new value (None = recommendation only)
    rebuild: bool       # True -> cannot be applied live; trainer must
                        # decide (pool shape, producer topology)
    reason: str
    applied: bool = False

    def value_str(self) -> str:
        return "None" if self.value is None else str(self.value)

    def __str__(self):
        tag = "APPLY" if self.applied else "RECOMMEND"
        return (f"[autopilot] step {self.step} {self.rule}: "
                f"{self.param} -> {self.value_str()} [{tag}] "
                f"({self.reason})")

    def __repr__(self):
        return self.__str__()


@dataclass
class TrainState:
    """Snapshot the trainer hands the autopilot each evaluation."""
    step: int = 0
    loss: float | None = None
    gn_core: float | None = None
    gn_mem: float | None = None
    coverage: dict | None = None     # {"b0": 0.24, ...}
    temp: float | None = None
    lr: float | None = None
    queue_depth: int | None = None
    docs_per_s: float | None = None
    wall: float = field(default_factory=time.time)


def _slope(xs: list[float], ys: list[float]) -> float:
    """Least-squares slope; None-safe."""
    n = len(xs)
    if n < 2:
        return 0.0
    mx, my = sum(xs) / n, sum(ys) / n
    den = sum((x - mx) ** 2 for x in xs)
    if den == 0:
        return 0.0
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den


class Autopilot:
    """Monitor -> validate -> rule engine -> one action per evaluation.

    Trainer contract:
        ap = Autopilot(mem_cfg=mem_cfg, eval_every=100, log=print)
        # in the loop, every eval_every steps:
        act = ap.observe(TrainState(step=i, loss=l, gn_core=..., ...))
        if act and act.param == "mem_cfg.lb_weight":  # applied live
            pass  # mem_cfg already mutated; next jit call uses it
        elif act and act.rebuild:
            ...  # surface to the operator / restart logic

    Live-mutable: mem_cfg.lb_weight, mem_cfg.lb_eps, mem_cfg.score_temp.
    Everything else is a logged recommendation.
    """

    LIVE_PARAMS = {"mem_cfg.lb_weight", "mem_cfg.lb_eps",
                   "mem_cfg.score_temp", "lr"}

    def __init__(self, mem_cfg=None, lr_scheduler=None,
                 eval_every: int = 100,
                 min_cooldown_steps: int = 500,
                 history: int = 200,
                 log=print,
                 # rule thresholds
                 plateau_window: int = 8,
                 plateau_rel_slope: float = 1e-4,
                 spike_sigma: float = 4.0,
                 collapse_cov_fall: float = 0.005,
                 dead_block_cov: float = 0.10,
                 dead_block_ref: float = 0.20,
                 gn_mem_ratio: float = 10.0,
                 starve_qdepth: int = 1,
                 starve_dps: float = 10.0):
        self._cfg_box = [mem_cfg]         # single-element box: the
        # trainer reads autopilot.mem_cfg each step (frozen dataclass
        # -> replaced not mutated; see _apply)
        self.mem_cfg = mem_cfg
        self.lr_scheduler = lr_scheduler  # callable(step) -> lr, or None
        self.eval_every = eval_every
        self.cooldown = min_cooldown_steps
        self.log = log
        self.hist: deque[TrainState] = deque(maxlen=history)
        self.last_change_step = -10**9
        self.actions: list[Action] = []
        self.cov_floor_seen: dict[str, int] = {}
        self.starve_count = 0
        self.spike_count = 0
        self.plateau_window = plateau_window
        self.plateau_rel_slope = plateau_rel_slope
        self.spike_sigma = spike_sigma
        self.collapse_cov_fall = collapse_cov_fall
        self.dead_block_cov = dead_block_cov
        self.dead_block_ref = dead_block_ref
        self.gn_mem_ratio = gn_mem_ratio
        self.starve_qdepth = starve_qdepth
        self.starve_dps = starve_dps
        self.lr_mult = 1.0
        self._lr_base = None
        self._fired_once: set = set()  # rule+key: fire once per session

    # ---------------- main entry ----------------
    def evaluate(self, s: TrainState) -> Action | None:
        """Validate + analyze one snapshot; return <=1 Action."""
        if not self._validate(s):
            return None
        self.hist.append(s)
        if s.step - self.last_change_step < self.cooldown:
            return None
        for rule in (self._r_nan, self._r_spike, self._r_starve,
                     self._r_collapse, self._r_plateau,
                     self._r_dead_block, self._r_gn_mem):
            act = rule()
            if act is not None:
                act.step = s.step
                self._apply(act)
                self.actions.append(act)
                self.last_change_step = s.step
                self.log(str(act))
                return act
        return None

    def evaluate_loss_only(self, step: int, loss: float, **kw) -> Action | None:
        """Convenience wrapper for trainers that only have loss."""
        return self.evaluate(TrainState(step=step, loss=loss, **kw))

    # ---------------- validation ----------------
    def _validate(self, s: TrainState) -> bool:
        if s.loss is not None and (math.isnan(s.loss)
                                   or math.isinf(s.loss)):
            # R1 handled inside rules with history context; still record
            self.hist.append(s)
            return True
        if s.coverage is not None:
            for k, v in s.coverage.items():
                if not (0.0 <= v <= 1.0):
                    self.log(f"[autopilot] invalid coverage {k}={v}; "
                             f"dropping sample")
                    return False
        return True

    # ---------------- actions ----------------
    def _apply(self, act: Action) -> None:
        if act.value is None or act.rebuild:
            return
        # MemoryConfig is a frozen flax struct.dataclass: mutate via
        # dataclasses.replace. The trainer must re-read
        # autopilot.mem_cfg each step (or hold autopilot._cfg_box).
        if act.param == "lr":
            cur = self._current_lr()
            if cur:
                self.lr_mult *= act.value / cur
                act.applied = True
            return
        if self.mem_cfg is None:
            return
        field_name = act.param.split(".", 1)[-1]
        if not hasattr(self.mem_cfg, field_name):
            return
        try:
            new_cfg = dataclasses.replace(self.mem_cfg,
                                          **{field_name: float(act.value)})
            self._cfg_box[0] = new_cfg
            self.mem_cfg = new_cfg
            act.applied = True
        except Exception as e:
            self.log(f"[autopilot] apply failed for {act.param}: {e!r}")

    # ---------------- rules (priority order) ----------------
    def _r_spike(self) -> Action | None:
        ls = [h.loss for h in self.hist if h.loss is not None
              and not math.isnan(h.loss)]
        if len(ls) < 8:
            return None
        recent, base = ls[-1], ls[:-1]
        mu = sum(base) / len(base)
        var = sum((x - mu) ** 2 for x in base) / len(base)
        sd = math.sqrt(var) + 1e-9
        if recent > mu + self.spike_sigma * sd:
            self.spike_count += 1
            if self.spike_count >= 2:
                self.spike_count = 0
                lr = self._current_lr()
                if lr is not None and lr > 1e-5:
                    return Action(0, "R3-loss-spike", "lr", lr * 0.5,
                                  False, f"loss {recent:.3f} > mu+4sd "
                                  f"({mu:.3f}) for 2 evals; halving LR")
            return Action(0, "R3-loss-spike", "lr", None, True,
                          f"loss spike {recent:.3f} vs mu {mu:.3f}; "
                          f"watching (transient if next eval normal)")
        self.spike_count = 0
        return None

    def _r_collapse(self) -> Action | None:
        covs = [h.coverage for h in self.hist if h.coverage]
        if len(covs) < 4:
            return None
        keys = covs[0].keys()
        for k in keys:
            vals = [c.get(k) for c in covs if c.get(k) is not None]
            if len(vals) < 4:
                continue
            falling = _slope(list(range(len(vals))), vals)
            min_cov = min(vals)
            cur = self.mem_cfg.lb_weight if self.mem_cfg else 0.0
            if falling < -self.collapse_cov_fall and min_cov < 0.15:
                if cur == 0.0:
                    return Action(0, "R4-router-collapse",
                                  "mem_cfg.lb_weight", 0.05, False,
                                  f"{k} falling {falling:.5f}/eval, "
                                  f"min {min_cov:.3f}; enabling lb aux")
                elif cur < 0.2 and len(covs) >= 6 and \
                        _slope(list(range(len(vals[-3:]))),
                               vals[-3:]) < -self.collapse_cov_fall:
                    return Action(0, "R5-still-flat",
                                  "mem_cfg.lb_weight", min(cur * 2, 0.2),
                                  False, f"{k} still falling on lb="
                                  f"{cur}; doubling (cap 0.2)")
        return None

    def _r_dead_block(self) -> Action | None:
        covs = [h.coverage for h in self.hist if h.coverage]
        if len(covs) < 3:
            return None
        keys = sorted(covs[-1].keys())
        if len(keys) < 2:
            return None
        ref_key, ref_val = max(covs[-1].items(), key=lambda kv: kv[1])
        for k in keys:
            if k == ref_key:
                continue
            if all(c.get(k, 1.0) < self.dead_block_cov for c in covs[-3:]) \
                    and ref_val > self.dead_block_ref:
                n = self.cov_floor_seen.get(k, 0) + 1
                self.cov_floor_seen[k] = n
                if n >= 2 and ("R6", k) not in self._fired_once:
                    self._fired_once.add(("R6", k))
                    return Action(0, "R6-dead-block", f"pool.{k}", None,
                                  True, f"{k} < {self.dead_block_cov:.0%} "
                                  f"for 3+ evals while {ref_key}="
                                  f"{ref_val:.0%}; rebuild: shrink or "
                                  f"relocate this block's pool")
            else:
                self.cov_floor_seen[k] = 0
        return None

    def _r_starve(self) -> Action | None:
        h = self.hist[-1]
        starved = False
        if h.queue_depth is not None:
            starved |= h.queue_depth <= self.starve_qdepth
        if h.docs_per_s is not None:
            starved |= h.docs_per_s < self.starve_dps
        if starved:
            self.starve_count += 1
            if self.starve_count >= 3 and ("R7",) not in self._fired_once:
                self.starve_count = 0
                self._fired_once.add(("R7",))
                dps = "n/a" if h.docs_per_s is None else \
                    f"{h.docs_per_s:.0f}"
                return Action(0, "R7-feed-starvation", "data.producers",
                              None, True, f"qdepth {h.queue_depth}, "
                              f"docs/s {dps} for 3 evals; "
                              f"rebuild: parallel producers")
        else:
            self.starve_count = 0
        return None

    def _r_gn_mem(self) -> Action | None:
        h = self.hist[-1]
        if h.gn_core is None or h.gn_mem is None or h.gn_core <= 0:
            return None
        r = h.gn_mem / (h.gn_core + 1e-9)
        if r > self.gn_mem_ratio:
            return Action(0, "R8-gn-mem-explosion", "trainer.mem_grad_scale",
                          None, True, f"gn_mem/gn_core = {r:.1f} > "
                          f"{self.gn_mem_ratio}; lower memory grad scale "
                          f"(trainer-side, requires code)")
        return None

    def _r_plateau(self) -> Action | None:
        pts = [(h.step, h.loss) for h in self.hist
               if h.loss is not None and not math.isnan(h.loss)]
        if len(pts) < self.plateau_window:
            return None
        pts = pts[-self.plateau_window:]
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        sl = _slope(xs, ys)
        mean = sum(ys) / len(ys)
        rel = sl / (abs(mean) + 1e-9)
        gn = self.hist[-1].gn_core
        if abs(rel) < self.plateau_rel_slope and (gn is None or gn < 1.0):
            lr = self._current_lr()
            if lr is not None and lr < 1e-2:
                gn_s = "n/a" if gn is None else f"{gn:.3f}"
                return Action(0, "R2-plateau", "lr", lr * 1.5, False,
                              f"loss slope {rel:.2e} flat over "
                              f"{len(pts)} evals, gn_core {gn_s}; "
                              f"raising LR x1.5")
        return None

    def _r_nan(self) -> Action | None:
        h = self.hist[-1]
        if h.loss is not None and (math.isnan(h.loss) or math.isinf(h.loss)):
            return Action(0, "R1-nan-guard", "trainer.rollback", None, True,
                          "loss NaN/Inf; rollback to last checkpoint "
                          "and halve LR before resuming")
        return None

    def _current_lr(self) -> float | None:
        base = self._lr_base if self._lr_base is not None else (
            float(self.lr_scheduler(self.hist[-1].step))
            if self.lr_scheduler is not None else None)
        if base is None:
            return None
        return base * self.lr_mult

    @property
    def current_lr(self) -> float | None:
        """Trainer contract: effective LR = schedule(step) * lr_mult.
        R2/R3 adjust lr_mult; the trainer reads this property."""
        return self.lr_mult


if __name__ == "__main__":
    # self-test: synthetic pathological trajectories -> expected actions
    import dataclasses
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    from navi.config import MemoryConfig

    mc = MemoryConfig(c1=48, c2=48, cand_k=8, side_top=8, n_classes=4,
                      score_temp=4.0, lb_weight=0.0, lb_eps=0.0)
    ap = Autopilot(mem_cfg=mc, eval_every=100, min_cooldown_steps=0,
                   lr_scheduler=lambda s: 1e-3, log=print)

    # R4: falling coverage
    fired = []
    for i, cov in enumerate([(0.24, 0.06), (0.23, 0.05), (0.21, 0.04),
                             (0.19, 0.03), (0.17, 0.02), (0.15, 0.02)]):
        a = ap.evaluate(TrainState(step=(i + 1) * 100, loss=7.0 - 0.01 * i,
                                   gn_core=0.5,
                                   coverage={"b0": cov[0], "b1": cov[1]}))
        if a:
            fired.append(a)
            break
    assert fired and fired[0].param == "mem_cfg.lb_weight", fired
    assert ap.mem_cfg.lb_weight == 0.05
    print("R4 collapse -> lb_weight applied:", ap.mem_cfg.lb_weight)

    # R2: plateau with low grad norm (after cooldown)
    for i in range(10):
        a = ap.evaluate(TrainState(step=1000 + i * 100, loss=6.5,
                                   gn_core=0.2))
        if a:
            print("R2 fired:", a)
            break
    assert a is not None and a.param == "lr", a
    print("R2 plateau -> lr override:", a.value, "mult:", ap.lr_mult)

    # R1: nan
    a = ap.evaluate(TrainState(step=3000, loss=float("nan")))
    assert a is not None and a.rule == "R1-nan-guard", a
    print("R1 nan ->", a.reason)

    # R7: starvation
    for i in range(3):
        a = ap.evaluate(TrainState(step=4000 + i * 100, loss=6.0,
                                   queue_depth=0, docs_per_s=2.0))
    assert a is not None and a.rule == "R7-feed-starvation", a
    print("R7 starve ->", a.reason)
    print("AUTOPILOT-SELFTEST-OK")