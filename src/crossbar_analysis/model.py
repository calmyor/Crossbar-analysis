"""Monte Carlo model corresponding to the ISCAS 2022 crossbar analysis.

The public API uses the paper's logical dot-product dimension ``N``. Each
logical weight is differentially encoded by two conductances, so the physical
crossbar contains ``2N`` columns. This differs from several archived scripts,
which called the physical-column count ``N``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class Device:
    """Binary resistive-device profile used by the paper model."""

    name: str
    ron_ohm: float
    roff_ohm: float

    def __post_init__(self) -> None:
        if self.ron_ohm <= 0 or self.roff_ohm <= 0:
            raise ValueError("Device resistances must be positive.")
        if self.roff_ohm < self.ron_ohm:
            raise ValueError("roff_ohm must be greater than or equal to ron_ohm.")

    @property
    def gon_siemens(self) -> float:
        return 1.0 / self.ron_ohm

    @property
    def goff_siemens(self) -> float:
        return 1.0 / self.roff_ohm

    @property
    def contrast(self) -> float:
        return self.roff_ohm / self.ron_ohm


DEVICE_PRESETS = {
    "mram": Device("MRAM", ron_ohm=3e3, roff_ohm=6e3),
    "reram": Device("ReRAM", ron_ohm=25e3, roff_ohm=300e3),
    "fefet": Device("FeFET", ron_ohm=1e6, roff_ohm=1e9),
}


@dataclass(frozen=True)
class CrossbarConfig:
    """Simulation conditions for one logical crossbar dot product."""

    dot_product_dimension: int = 512
    input_bits: int = 5
    adc_bits: int = 6
    v_lsb_volt: float = 3e-3
    adc_clip_ampere: float = 2e-6
    dac_mismatch_ratio: float = 0.04
    conductance_variation_ratio: float = 0.04
    samples: int = 10_000
    seed: int = 2021

    def __post_init__(self) -> None:
        if self.dot_product_dimension < 1:
            raise ValueError("dot_product_dimension must be at least 1.")
        if self.input_bits < 2 or self.adc_bits < 1:
            raise ValueError("input_bits must be >= 2 and adc_bits must be >= 1.")
        if self.v_lsb_volt <= 0 or self.adc_clip_ampere <= 0:
            raise ValueError("Voltage and clipping-current parameters must be positive.")
        if self.dac_mismatch_ratio < 0 or self.conductance_variation_ratio < 0:
            raise ValueError("Variation ratios cannot be negative.")
        if self.samples < 2:
            raise ValueError("samples must be at least 2.")

    @property
    def physical_columns(self) -> int:
        return 2 * self.dot_product_dimension


@dataclass(frozen=True)
class SimulationResult:
    """Signal and error statistics at one sensing resistance."""

    sensing_resistance_ohm: float
    snr_db: float
    signal_power_a2: float
    total_noise_power_a2: float
    dac_noise_power_a2: float
    bitcell_noise_power_a2: float
    quantization_noise_power_a2: float
    clipping_noise_power_a2: float
    clipping_probability: float
    current_scaling_factor: float

    def to_record(self) -> dict[str, float]:
        return {
            "sensing_resistance_ohm": self.sensing_resistance_ohm,
            "snr_db": self.snr_db,
            "signal_power_a2": self.signal_power_a2,
            "total_noise_power_a2": self.total_noise_power_a2,
            "dac_noise_power_a2": self.dac_noise_power_a2,
            "bitcell_noise_power_a2": self.bitcell_noise_power_a2,
            "quantization_noise_power_a2": self.quantization_noise_power_a2,
            "clipping_noise_power_a2": self.clipping_noise_power_a2,
            "clipping_probability": self.clipping_probability,
            "current_scaling_factor": self.current_scaling_factor,
        }


@dataclass(frozen=True)
class _Trials:
    signal_unscaled_a: np.ndarray
    dac_unscaled_a: np.ndarray
    bitcell_unscaled_a: np.ndarray
    quantization_a: np.ndarray
    array_resistance_ohm: float


def _make_trials(device: Device, config: CrossbarConfig) -> _Trials:
    # RandomState intentionally preserves the archived NumPy sampling stream.
    # This makes the refactored model numerically comparable with the 2021
    # exploratory scripts, which used np.random.seed followed by module-level
    # random functions.
    rng = np.random.RandomState(config.seed)
    sample_count = config.samples
    gon = device.gon_siemens
    goff = device.goff_siemens
    nominal_delta_g = gon - goff
    sigma_delta_g = config.conductance_variation_ratio * np.hypot(gon, goff)
    input_low = -(2 ** (config.input_bits - 1))
    input_high = 2 ** (config.input_bits - 1)

    dimension = config.dot_product_dimension
    weights = 2.0 * (rng.uniform(size=dimension) > 0.5) - 1.0
    differential_conductance = nominal_delta_g * weights

    # Preserve the archived draw order: bitcell variation, inputs, DAC
    # mismatch, then ADC quantization. Temporary arrays are released between
    # stages to keep the 10,000-sample paper configuration practical.
    delta_conductance = rng.normal(size=(dimension, sample_count)) * sigma_delta_g
    inputs = rng.randint(input_low, input_high, size=(dimension, sample_count))
    voltage = inputs.astype(np.float64) * config.v_lsb_volt
    signal = np.einsum("ij,i->j", voltage, differential_conductance)
    bitcell_noise = np.sum(voltage * delta_conductance, axis=0)
    del delta_conductance

    dac_sigma = (
        np.sqrt(2.0 * np.abs(inputs.astype(np.float64)))
        * config.dac_mismatch_ratio
        * config.v_lsb_volt
    )
    delta_voltage = rng.normal(size=voltage.shape) * dac_sigma
    dac_noise = np.einsum("ij,i->j", delta_voltage, differential_conductance)

    quantization_limit = config.adc_clip_ampere / (2**config.adc_bits)
    quantization = rng.uniform(-quantization_limit, quantization_limit, sample_count)
    total_array_conductance = config.dot_product_dimension * (gon + goff)

    return _Trials(
        signal_unscaled_a=signal,
        dac_unscaled_a=dac_noise,
        bitcell_unscaled_a=bitcell_noise,
        quantization_a=quantization,
        array_resistance_ohm=1.0 / total_array_conductance,
    )


def _evaluate_trials(
    trials: _Trials,
    config: CrossbarConfig,
    sensing_resistance_ohm: float,
) -> SimulationResult:
    if sensing_resistance_ohm <= 0:
        raise ValueError("Sensing resistance must be positive.")

    scale = trials.array_resistance_ohm / (
        trials.array_resistance_ohm + sensing_resistance_ohm
    )
    signal = scale * trials.signal_unscaled_a
    dac_noise = scale * trials.dac_unscaled_a
    bitcell_noise = scale * trials.bitcell_unscaled_a
    preclip = signal + dac_noise + bitcell_noise + trials.quantization_a
    output = np.clip(preclip, -config.adc_clip_ampere, config.adc_clip_ampere)
    clipping_noise = output - preclip
    total_noise = output - signal

    signal_power = float(np.mean(np.square(signal)))
    total_noise_power = float(np.mean(np.square(total_noise)))
    snr_db = float(10.0 * np.log10(signal_power / total_noise_power))

    return SimulationResult(
        sensing_resistance_ohm=float(sensing_resistance_ohm),
        snr_db=snr_db,
        signal_power_a2=signal_power,
        total_noise_power_a2=total_noise_power,
        dac_noise_power_a2=float(np.mean(np.square(dac_noise))),
        bitcell_noise_power_a2=float(np.mean(np.square(bitcell_noise))),
        quantization_noise_power_a2=float(np.mean(np.square(trials.quantization_a))),
        clipping_noise_power_a2=float(np.mean(np.square(clipping_noise))),
        clipping_probability=float(np.mean(np.abs(preclip) > config.adc_clip_ampere)),
        current_scaling_factor=float(scale),
    )


def simulate_sweep(
    device: Device,
    config: CrossbarConfig,
    sensing_resistances_ohm: Iterable[float],
) -> list[SimulationResult]:
    """Evaluate one shared Monte Carlo sample set across a sensing sweep."""

    resistances = [float(value) for value in sensing_resistances_ohm]
    if not resistances:
        raise ValueError("At least one sensing resistance is required.")
    trials = _make_trials(device, config)
    return [_evaluate_trials(trials, config, value) for value in resistances]


def find_optimum(results: Sequence[SimulationResult]) -> SimulationResult:
    """Return the evaluated sensing point with the maximum compute SNR."""

    if not results:
        raise ValueError("Cannot find an optimum in an empty result sequence.")
    return max(results, key=lambda result: result.snr_db)


def recommended_rs_grid(device_key: str, points: int = 41) -> np.ndarray:
    """Return a device-appropriate logarithmic sensing-resistance grid."""

    if points < 2:
        raise ValueError("points must be at least 2.")
    bounds = {
        "mram": (1e2, 1e4),
        "reram": (1e2, 1e5),
        "fefet": (1e3, 1e6),
    }
    try:
        start, stop = bounds[device_key.lower()]
    except KeyError as exc:
        raise ValueError(f"Unknown device preset: {device_key}") from exc
    return np.geomspace(start, stop, points)
