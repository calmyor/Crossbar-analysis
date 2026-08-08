"""Behavioral compute-SNR analysis for voltage-driven resistive crossbars."""

from .model import (
    DEVICE_PRESETS,
    CrossbarConfig,
    Device,
    SimulationResult,
    find_optimum,
    recommended_rs_grid,
    simulate_sweep,
)

__all__ = [
    "DEVICE_PRESETS",
    "CrossbarConfig",
    "Device",
    "SimulationResult",
    "find_optimum",
    "recommended_rs_grid",
    "simulate_sweep",
]
