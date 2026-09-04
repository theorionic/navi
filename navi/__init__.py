"""Navi: a reasoner + trainable product-key memory pool (Pool) architecture.

The Reasoner is a dense transformer backbone. The Pool is a trainable
key -> value memory queried per token via product-key lookup (Meta,
"Memory Layers at Scale", arXiv:2412.09764): knowledge capacity is
decoupled from reasoning FLOPs, which is what lets the Pool live
outside VRAM in the shipping design while the Reasoner stays resident.
"""

from navi.model import ModelConfig, Navi
from navi.pkm import MemoryConfig, ProductKeyMemory

__all__ = [
    "ModelConfig",
    "MemoryConfig",
    "Navi",
    "ProductKeyMemory",
]