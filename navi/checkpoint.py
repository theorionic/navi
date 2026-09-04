"""Orbax checkpointing: save/restore full parameter trees + train state."""

from pathlib import Path

import jax
import orbax.checkpoint as ocp


def save_params(path: str, params) -> None:
    """Save a parameter pytree to an orbax checkpoint directory."""
    ckpt = ocp.PyTreeCheckpointer()
    ckpt.save(Path(path), jax.tree_util.tree_map(lambda x: x, params), force=True)
    ckpt.close()


def restore_params(path: str, template):
    """Restore a parameter pytree; template supplies structure/dtypes."""
    ckpt = ocp.PyTreeCheckpointer()
    restored = ckpt.restore(Path(path), item=template)
    ckpt.close()
    return restored


def save_meta(path: str, meta: dict) -> None:
    ckpt = ocp.JsonCheckpointHandler()
    ckpt.save(Path(path), meta, force=True)
    ckpt.close()


def load_meta(path: str) -> dict:
    ckpt = ocp.JsonCheckpointHandler()
    meta = ckpt.restore(Path(path))
    ckpt.close()
    return meta