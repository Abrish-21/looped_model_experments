"""Utilities for reproducing looped latent-reasoning experiments."""

from .latent_reasoning import (
    OptimizerSpec,
    LoopedLatentReasoner,
    run_optimizer_comparison,
)

__all__ = [
    "OptimizerSpec",
    "LoopedLatentReasoner",
    "run_optimizer_comparison",
]
