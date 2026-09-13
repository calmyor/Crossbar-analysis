"""Parasitic-aware Monte Carlo model for voltage-driven resistive crossbars.

The ISCAS 2022 model in :mod:`crossbar_analysis.model` treats each source line
as an ideal conductance sum. Measured crossbar chips show that wire and driver
resistance, together with DAC and device nonlinearity, dominate once the array
grows. This module solves the full resistive network of an ``M x 2N`` crossbar:

* bit-line and source-line wire resistance between neighboring cells,
* the output resistance of the input drivers and an optional access transistor,
* current-mode (virtual ground through ``R_s``) or voltage-mode sensing,
* static DAC nonlinearity (random per-code INL and full-scale compression) and
  input-dependent DAC mismatch,
* cubic device I-V nonlinearity evaluated at the IR-dropped cell voltages,
* bitcell conductance variation, programming error, ADC clipping, and ADC
  quantization.

For a fixed set of conductances the wire network is linear, so wire and driver
resistance alone act as a deterministic distortion of the effective weights.
DAC nonlinearity, device I-V nonlinearity, and variation make the error input
dependent. The simulation reports compute SNR at three compensation levels:

``raw``
    Output compared with the ideal, parasitic-free output.
``calibrated``
    After a per-output gain and offset correction, as applied by chips with
    per-column affine calibration.
``ir_aware``
    After removing the deterministic distortion predicted from the nominal
    design (wire and driver IR drop), as a mapping or training flow that knows
    the array layout would. Random programming error remains.
``chip_in_loop``
    After fitting the best linear map from inputs to each measured output, which
    models chip-in-the-loop fine-tuning that also absorbs the random static
    weight error of one die. The residual contains only input-dependent error.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .model import Device

# Per-cell wire resistance extracted from the 22 nm MRAM IMC layout used in
# Roy and Shanbhag, IEEE JxCDC 2024 (183 ohm and 84 ohm across 512 cells).
LAYOUT_22NM_R_BL_OHM = 183.0 / 512.0
LAYOUT_22NM_R_SL_OHM = 84.0 / 512.0

COMPENSATION_LEVELS = ("raw", "calibrated", "ir_aware", "chip_in_loop")
_MIN_RESISTANCE_OHM = 1e-4


@dataclass(frozen=True)
class ParasiticConfig:
    """Circuit, input, and sampling conditions for one crossbar bank."""

    dot_product_dimension: int = 64
    active_rows: int = 1
    input_bits: int = 5
    adc_bits: int = 6
    v_lsb_volt: float = 3e-3
    weight_distribution: str = "binary"
    conductance_variation_ratio: float = 0.04
    programming_sigma_ratio: float = 0.0
    input_encoding: str = "voltage"
    dac_mismatch_ratio: float = 0.04
    dac_inl_lsb: float = 0.0
    dac_compression: float = 0.0
    iv_nonlinearity: float = 0.0
    r_bl_ohm: float = LAYOUT_22NM_R_BL_OHM
    r_sl_ohm: float = LAYOUT_22NM_R_SL_OHM
    r_driver_ohm: float = 0.0
    r_access_ohm: float = 0.0
    r_sense_ohm: float = 1e3
    r_input_rail_ohm: float = 0.0
    parallel_cores: int = 1
    sensing: str = "current"
    weight_trials: int = 4
    input_samples: int = 0
    seed: int = 2022

    def __post_init__(self) -> None:
        if self.dot_product_dimension < 1 or self.active_rows < 1:
            raise ValueError("dot_product_dimension and active_rows must be positive.")
        if self.input_bits < 2 or self.adc_bits < 1:
            raise ValueError("input_bits must be >= 2 and adc_bits must be >= 1.")
        if self.weight_distribution not in ("binary", "uniform"):
            raise ValueError("weight_distribution must be 'binary' or 'uniform'.")
        if self.input_encoding not in ("voltage", "pwm"):
            raise ValueError("input_encoding must be 'voltage' or 'pwm'.")
        if self.sensing not in ("current", "voltage"):
            raise ValueError("sensing must be 'current' or 'voltage'.")
        resistances = (self.r_bl_ohm, self.r_sl_ohm, self.r_driver_ohm, self.r_access_ohm)
        if min(resistances) < 0 or self.r_sense_ohm <= 0 or self.v_lsb_volt <= 0:
            raise ValueError("Resistances must be non-negative; r_sense_ohm and v_lsb_volt positive.")
        if min(
            self.conductance_variation_ratio,
            self.programming_sigma_ratio,
            self.dac_mismatch_ratio,
            self.dac_inl_lsb,
            self.dac_compression,
            self.iv_nonlinearity,
        ) < 0:
            raise ValueError("Variation, mismatch, and nonlinearity parameters cannot be negative.")
        if self.weight_trials < 1:
            raise ValueError("weight_trials must be at least 1.")
        if self.r_input_rail_ohm < 0 or self.parallel_cores < 1:
            raise ValueError("r_input_rail_ohm must be non-negative and parallel_cores at least 1.")

    @property
    def physical_columns(self) -> int:
        return 2 * self.dot_product_dimension

    @property
    def full_scale_code(self) -> int:
        return 2 ** (self.input_bits - 1)

    @property
    def samples_per_trial(self) -> int:
        # The weight-aware fit needs comfortably more samples than unknowns.
        return self.input_samples or max(512, 3 * self.dot_product_dimension + 64)


@dataclass(frozen=True)
class CompensationResult:
    """Compute SNR before and after the ADC for one compensation level."""

    analog_snr_db: float
    total_snr_db: float
    clip_multiple: float


@dataclass(frozen=True)
class ParasiticResult:
    """Summary of a parasitic-aware crossbar simulation."""

    raw: CompensationResult
    calibrated: CompensationResult
    ir_aware: CompensationResult
    chip_in_loop: CompensationResult
    mean_gain: float
    config: ParasiticConfig

    def level(self, name: str) -> CompensationResult:
        if name not in COMPENSATION_LEVELS:
            raise ValueError(f"Unknown compensation level: {name}")
        return getattr(self, name)


def _cell_conductance(conductance: np.ndarray, r_access_ohm: float) -> np.ndarray:
    if r_access_ohm <= 0:
        return conductance
    return 1.0 / (1.0 / conductance + r_access_ohm)


class CrossbarNetwork:
    """Factorized nodal model of one crossbar with fixed conductances."""

    def __init__(self, conductance: np.ndarray, config: ParasiticConfig) -> None:
        rows, columns = conductance.shape
        cells = rows * columns
        self.rows, self.columns, self.config = rows, columns, config
        self.bitline = np.arange(cells).reshape(rows, columns)
        self.sourceline = cells + self.bitline
        self.size = 2 * cells
        self.g_cell = _cell_conductance(conductance, config.r_access_ohm)
        self.g_driver = 1.0 / max(config.r_driver_ohm, _MIN_RESISTANCE_OHM)
        self.drivers = self.bitline[0, :]
        self.outputs = self.sourceline[:, -1]

        g_bl = 1.0 / max(config.r_bl_ohm, _MIN_RESISTANCE_OHM)
        g_sl = 1.0 / max(config.r_sl_ohm, _MIN_RESISTANCE_OHM)
        first, second, values = [self.bitline.ravel()], [self.sourceline.ravel()], [self.g_cell.ravel()]
        if rows > 1:
            first.append(self.bitline[:-1, :].ravel())
            second.append(self.bitline[1:, :].ravel())
            values.append(np.full((rows - 1) * columns, g_bl))
        if columns > 1:
            first.append(self.sourceline[:, :-1].ravel())
            second.append(self.sourceline[:, 1:].ravel())
            values.append(np.full(rows * (columns - 1), g_sl))
        p, q, g = np.concatenate(first), np.concatenate(second), np.concatenate(values)

        diagonal = np.bincount(p, weights=g, minlength=self.size)
        diagonal += np.bincount(q, weights=g, minlength=self.size)
        diagonal[self.drivers] += self.g_driver
        if config.sensing == "current":
            diagonal[self.outputs] += 1.0 / config.r_sense_ohm
        index = np.arange(self.size)
        matrix = sp.coo_matrix(
            (np.concatenate([-g, -g, diagonal]), (np.concatenate([p, q, index]), np.concatenate([q, p, index]))),
            shape=(self.size, self.size),
        ).tocsc()
        self._solver = spla.splu(matrix)

        # Adjoint solves: one right-hand side per output. They give both the
        # driver-to-output transfer and the sensitivity of each output to a
        # current injected across any cell.
        self.transfer = np.empty((rows, columns))
        self._cell_sensitivity = np.empty((cells, rows))
        output_scale = 1.0 / config.r_sense_ohm if config.sensing == "current" else 1.0
        for start in range(0, rows, 32):
            stop = min(start + 32, rows)
            rhs = np.zeros((self.size, stop - start))
            rhs[self.outputs[start:stop], np.arange(stop - start)] = 1.0
            adjoint = self._solver.solve(rhs)
            self.transfer[start:stop, :] = adjoint[self.drivers, :].T * self.g_driver * output_scale
            self._cell_sensitivity[:, start:stop] = (
                adjoint[self.sourceline.ravel(), :] - adjoint[self.bitline.ravel(), :]
            ) * output_scale

    def outputs_for(self, driver_voltages: np.ndarray) -> np.ndarray:
        """Return outputs for ``samples x columns`` driver voltages."""

        linear = driver_voltages @ self.transfer.T
        beta = self.config.iv_nonlinearity
        if beta <= 0:
            return linear
        # One Picard step: cubic excess current evaluated at the linear cell
        # voltages, which already include IR drop.
        reference = self.config.full_scale_code * self.config.v_lsb_volt
        correction = np.empty_like(linear)
        g = self.g_cell.ravel()
        for start in range(0, driver_voltages.shape[0], 64):
            stop = min(start + 64, driver_voltages.shape[0])
            rhs = np.zeros((self.size, stop - start))
            rhs[self.drivers, :] = driver_voltages[start:stop].T * self.g_driver
            nodes = self._solver.solve(rhs)
            cell_voltage = nodes[self.bitline.ravel(), :] - nodes[self.sourceline.ravel(), :]
            excess = g[:, None] * beta * cell_voltage**3 / reference**2
            correction[start:stop] = (self._cell_sensitivity.T @ excess).T
        return linear + correction


def transfer_matrix(conductance: np.ndarray, config: ParasiticConfig) -> np.ndarray:
    """Return the ``rows x columns`` map from driver voltages to outputs.

    Outputs are source-line currents (A/V) for current-mode sensing and
    source-line voltages (V/V) for voltage-mode sensing.
    """

    return CrossbarNetwork(conductance, config).transfer


def ideal_transfer(conductance: np.ndarray, config: ParasiticConfig) -> np.ndarray:
    """Transfer matrix of the same conductances with ideal wires and drivers."""

    g_cell = _cell_conductance(conductance, config.r_access_ohm)
    total = g_cell.sum(axis=1, keepdims=True)
    if config.sensing == "current":
        return g_cell / (1.0 + config.r_sense_ohm * total)
    return g_cell / total


def weight_conductances(weights: np.ndarray, device: Device) -> np.ndarray:
    """Map logical weights in [-1, 1] onto differential conductance pairs."""

    span = device.gon_siemens - device.goff_siemens
    pairs = np.empty((weights.shape[0], 2 * weights.shape[1]))
    pairs[:, 0::2] = device.goff_siemens + span * np.clip(weights, 0.0, 1.0)
    pairs[:, 1::2] = device.goff_siemens + span * np.clip(-weights, 0.0, 1.0)
    return pairs


def _differential(inputs: np.ndarray) -> np.ndarray:
    physical = np.empty((inputs.shape[0], 2 * inputs.shape[1]))
    physical[:, 0::2] = inputs
    physical[:, 1::2] = -inputs
    return physical


def _dac_codes(
    inputs: np.ndarray, config: ParasiticConfig, inl_table: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Return effective input codes, in LSB, for each physical driver."""

    physical = _differential(inputs)
    if config.input_encoding == "pwm":
        return physical
    full_scale = config.full_scale_code
    effective = physical * (1.0 - config.dac_compression * np.square(physical / full_scale))
    if config.dac_inl_lsb > 0:
        codes = np.clip(physical + full_scale, 0, 2 * full_scale).astype(int)
        effective = effective + inl_table[np.arange(physical.shape[1]), codes]
    if config.dac_mismatch_ratio > 0:
        sigma = np.sqrt(2.0 * np.abs(physical)) * config.dac_mismatch_ratio
        effective = effective + rng.normal(size=physical.shape) * sigma
    return effective


def _input_rail_sag(
    driver_voltages: np.ndarray,
    conductance: np.ndarray,
    config: ParasiticConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Scale each input by the IR drop of the read-voltage rail that feeds it.

    All drivers of one polarity share a rail with resistance ``r_input_rail_ohm``.
    The rail current is the total current drawn by this bank plus, when several
    cores run in parallel, independent loads from the other cores. The sag
    changes from one input vector to the next, so per-column calibration cannot
    remove it.
    """

    if config.r_input_rail_ohm <= 0:
        return driver_voltages
    currents = driver_voltages * conductance.sum(axis=0)
    full_scale = config.full_scale_code * config.v_lsb_volt
    rails = []
    for polarity in (1.0, -1.0):
        rail = np.clip(polarity * currents, 0.0, None).sum(axis=1)
        others = config.parallel_cores - 1
        if others > 0:
            rail = rail + rng.normal(
                others * rail.mean(), np.sqrt(others) * rail.std(), size=rail.shape
            ).clip(min=0.0)
        rails.append(1.0 - config.r_input_rail_ohm * rail / full_scale)
    factor = np.where(driver_voltages >= 0, rails[0][:, None], rails[1][:, None])
    return driver_voltages * np.clip(factor, 0.0, 1.0)


def _quantize(values: np.ndarray, clip: float, bits: int) -> np.ndarray:
    step = 2.0 * clip / (2**bits)
    clipped = np.clip(values, -clip, clip - step)
    return (np.floor(clipped / step) + 0.5) * step


def _best_adc_snr(reference: np.ndarray, estimate: np.ndarray, bits: int) -> tuple[float, float]:
    power = float(np.mean(np.square(reference)))
    rms = np.sqrt(power)
    best = (-np.inf, 0.0)
    for multiple in np.linspace(0.5, 6.0, 23):
        noise = float(np.mean(np.square(_quantize(estimate, multiple * rms, bits) - reference)))
        snr = 10.0 * np.log10(power / noise)
        if snr > best[0]:
            best = (float(snr), float(multiple))
    return best


def _rms_ratio(target: np.ndarray, source: np.ndarray) -> float:
    source_power = float(np.mean(np.square(source)))
    if source_power <= 0.0:
        return 1.0
    return float(np.sqrt(np.mean(np.square(target)) / source_power))


def _affine_referred(
    reference: np.ndarray, output: np.ndarray, gains: list[float] | None = None
) -> np.ndarray:
    """Correct each output column with a least-squares gain and offset."""

    corrected = np.empty_like(output)
    ones = np.ones(output.shape[0])
    for row in range(output.shape[1]):
        design = np.column_stack([reference[:, row], ones])
        (gain, offset), *_ = np.linalg.lstsq(design, output[:, row], rcond=None)
        gain = gain if abs(gain) > 0 else 1.0
        if gains is not None:
            gains.append(gain)
        corrected[:, row] = (output[:, row] - offset) / gain
    return corrected


def _snr_db(signal: np.ndarray, noise: np.ndarray) -> float:
    return float(10.0 * np.log10(np.mean(np.square(signal)) / np.mean(np.square(noise))))


def simulate_parasitic(device: Device, config: ParasiticConfig) -> ParasiticResult:
    """Monte Carlo compute SNR of a crossbar bank with wire, DAC, and device non-idealities."""

    rng = np.random.default_rng(config.seed)
    rows, dimension = config.active_rows, config.dot_product_dimension
    samples, full_scale = config.samples_per_trial, config.full_scale_code
    if config.input_encoding == "pwm" and config.iv_nonlinearity > 0:
        # Pulse-width inputs read every cell at the full read amplitude, so the
        # cubic I-V term becomes a uniform gain change rather than distortion.
        config_for_network = replace(config, iv_nonlinearity=0.0)
    else:
        config_for_network = config

    collected = {name: ([], []) for name in COMPENSATION_LEVELS}
    gains = []
    for _ in range(config.weight_trials):
        if config.weight_distribution == "binary":
            weights = 2.0 * (rng.uniform(size=(rows, dimension)) > 0.5) - 1.0
        else:
            weights = rng.uniform(-1.0, 1.0, size=(rows, dimension))
        nominal = weight_conductances(weights, device)
        programmed = nominal * (1.0 + config.conductance_variation_ratio * rng.normal(size=nominal.shape))
        programmed += config.programming_sigma_ratio * device.gon_siemens * rng.normal(size=nominal.shape)
        programmed = np.maximum(programmed, 0.05 * device.goff_siemens)

        network = CrossbarNetwork(programmed, config_for_network)
        designed = CrossbarNetwork(nominal, replace(config_for_network, iv_nonlinearity=0.0))
        ideal = ideal_transfer(nominal, config)
        inl_table = config.dac_inl_lsb * rng.normal(size=(2 * dimension, 2 * full_scale + 1))

        inputs = rng.integers(-full_scale, full_scale, size=(samples, dimension))
        driver_voltages = _differential(inputs) * config.v_lsb_volt
        reference = driver_voltages @ ideal.T
        predicted = driver_voltages @ designed.transfer.T
        applied = _dac_codes(inputs, config, inl_table, rng) * config.v_lsb_volt
        output = network.outputs_for(_input_rail_sag(applied, programmed, config, rng))

        collected["raw"][0].append(reference)
        collected["raw"][1].append(output)
        collected["calibrated"][0].append(reference)
        collected["calibrated"][1].append(_affine_referred(reference, output, gains))
        collected["ir_aware"][0].append(reference)
        collected["ir_aware"][1].append(
            _affine_referred(predicted, output) * _rms_ratio(reference, predicted)
            + reference - predicted * _rms_ratio(reference, predicted)
        )

        basis = np.hstack([inputs * config.v_lsb_volt, np.ones((samples, 1))])
        coefficients, *_ = np.linalg.lstsq(basis, output, rcond=None)
        fitted = basis @ coefficients
        scale = _rms_ratio(reference, fitted)
        collected["chip_in_loop"][0].append(fitted * scale)
        collected["chip_in_loop"][1].append(output * scale)

    levels = {}
    for name, (references, estimates) in collected.items():
        reference = np.concatenate([r.ravel() for r in references])
        estimate = np.concatenate([e.ravel() for e in estimates])
        total, clip = _best_adc_snr(reference, estimate, config.adc_bits)
        levels[name] = CompensationResult(_snr_db(reference, estimate - reference), total, clip)

    return ParasiticResult(
        raw=levels["raw"],
        calibrated=levels["calibrated"],
        ir_aware=levels["ir_aware"],
        chip_in_loop=levels["chip_in_loop"],
        mean_gain=float(np.mean(gains)),
        config=config,
    )


def sweep_sensing(
    device: Device, config: ParasiticConfig, sensing_resistances_ohm: Sequence[float]
) -> list[tuple[float, ParasiticResult]]:
    """Evaluate the same bank across sensing resistances (current-mode only)."""

    return [
        (float(value), simulate_parasitic(device, replace(config, r_sense_ohm=float(value))))
        for value in sensing_resistances_ohm
    ]
