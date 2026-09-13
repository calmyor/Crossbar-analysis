#!/usr/bin/env python3
"""Fit the parasitic-aware crossbar model to measured chips and check held-out results.

Each chip keeps the circuit parameters stated in its paper. One unknown per chip,
the parameter its authors identify as the dominant unmodeled loss, is fitted so
that the predicted accuracy matches one reported result. A second reported result
that was not used in the fit then checks the prediction.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np

from crossbar_analysis.accuracy import NOISE_CURVES, predict_accuracy
from crossbar_analysis.model import Device
from crossbar_analysis.parasitics import ParasiticConfig, simulate_parasitic

ROOT = Path(__file__).resolve().parents[1]


def neurosim_wire_ohm_per_um(width_nm: float, aspect: float, rho: float, barrier_nm: float) -> float:
    """Unit wire resistance with the NeuroSim V1.4 size-effect and barrier model."""

    rho_eff = rho / (1 - ((2 * aspect * width_nm + width_nm) * barrier_nm / (aspect * width_nm**2)))
    return rho_eff / (width_nm * 1e-9 * width_nm * 1e-9 * aspect) * 1e-6


# Minimum-width metal per node (NeuroSim V1.4 Param.cpp), 1T1R cell pitch of 4 F.
WIRE_PER_CELL_OHM = {
    14: neurosim_wire_ohm_per_um(32, 2.0, 3.25e-8, 2.5) * 4 * 0.032,
    130: neurosim_wire_ohm_per_um(175, 1.6, 2.01e-8, 10.0) * 4 * 0.175,
}


@dataclass
class Observation:
    label: str
    level: str
    accuracy: float | None = None
    precision_bits: float | None = None
    overrides: dict = field(default_factory=dict)


@dataclass
class Chip:
    key: str
    label: str
    source: str
    device: Device
    config: ParasiticConfig
    curve: str | None
    baseline: float | None
    free_parameter: str | None
    bounds: tuple[float, float] | None
    fit: Observation | None
    checks: list[Observation]
    assumptions: list[str]


def chips(rows: int) -> list[Chip]:
    # NeuRRAM: weight noise injected in training is 10% of max |w|; with two
    # differential cells this corresponds to sigma = 0.10 * (gmax - gmin) / sqrt(2).
    neurram_device = Device("RRAM (NeuRRAM)", ron_ohm=1 / 40e-6, roff_ohm=1 / 1e-6)
    neurram_sigma = 0.10 * (40e-6 - 1e-6) / np.sqrt(2) / 40e-6
    # IBM: combined PCM read and write noise of about 3.8% of Gmax (Joshi et al.
    # 2020) for one device per polarity; two devices per polarity lower it by sqrt(2).
    ibm_device = Device("PCM (IBM HERMES)", ron_ohm=1 / 25e-6, roff_ohm=1 / 0.25e-6)
    return [
        Chip(
            key="neurram",
            label="NeuRRAM, 48-core RRAM CIM chip (130 nm)",
            source="W. Wan et al., Nature 608, 504-512 (2022)",
            device=neurram_device,
            config=ParasiticConfig(
                dot_product_dimension=128,
                active_rows=rows,
                input_bits=4,
                adc_bits=8,
                v_lsb_volt=0.5 / 8,
                weight_distribution="uniform",
                conductance_variation_ratio=0.0,
                programming_sigma_ratio=float(neurram_sigma),
                input_encoding="voltage",
                dac_mismatch_ratio=0.0,
                iv_nonlinearity=0.05,
                r_bl_ohm=WIRE_PER_CELL_OHM[130],
                r_sl_ohm=WIRE_PER_CELL_OHM[130],
                parallel_cores=4,
                sensing="voltage",
                weight_trials=1,
            ),
            curve="resnet20-cifar10-injection-20",
            baseline=None,
            free_parameter="r_input_rail_ohm",
            bounds=(0.0, 10.0),
            fit=Observation("ResNet-20 / CIFAR-10 measured before fine-tuning", "calibrated", accuracy=83.67),
            checks=[
                Observation(
                    "CIFAR-10 after chip-in-the-loop fine-tuning (+1.99 pt)", "chip_in_loop", accuracy=85.66
                ),
            ],
            assumptions=[
                "256 x 256 core modeled as N = 128 differential weights per output.",
                "Programming error set by the 10% weight noise used in training (3.87 uS worst-case relaxation at 40 uS).",
                "I-V nonlinearity of 5% at the full read voltage is assumed, not reported.",
                "Accuracy read directly from the chip team's ResNet-20 noise curve, without rescaling.",
                "Input IR drop during multi-core operation is lumped into a shared read-voltage rail with four "
                "cores in parallel; the rail resistance is the fitted parameter.",
                "Chip-in-the-loop compensation in the model absorbs all static error, so the check is an upper bound.",
            ],
        ),
        Chip(
            key="ibm-hermes",
            label="IBM HERMES 64-core PCM AIMC chip (14 nm)",
            source="M. Le Gallo et al., Nature Electronics 6, 680-693 (2023)",
            device=ibm_device,
            config=ParasiticConfig(
                dot_product_dimension=256,
                active_rows=rows,
                input_bits=8,
                adc_bits=12,
                v_lsb_volt=0.2 / 128,
                weight_distribution="uniform",
                conductance_variation_ratio=0.0,
                programming_sigma_ratio=0.038 / np.sqrt(2),
                input_encoding="pwm",
                dac_mismatch_ratio=0.0,
                iv_nonlinearity=0.05,
                r_sense_ohm=100.0,
                weight_trials=1,
            ),
            curve="resnet32-cifar10-matched-injection",
            baseline=93.67,
            free_parameter="wire_scale",
            bounds=(0.02, 2.0),
            fit=Observation("ResNet-9 / CIFAR-10 with two devices per polarity (TDP)", "calibrated", accuracy=92.81),
            checks=[
                Observation(
                    "ResNet-9 / CIFAR-10 with one device per polarity (ODP)",
                    "calibrated",
                    accuracy=92.23,
                    overrides={"programming_sigma_ratio": 0.038},
                ),
            ],
            assumptions=[
                "Unit cell of four PCM devices modeled as a differential pair on half-cell pitch.",
                "PCM programming plus read noise of 3.8% of Gmax (Joshi et al. 2020); TDP lowers it by sqrt(2).",
                "Pulse-width inputs remove amplitude DAC error; ADC gain and offset are calibrated per row.",
                "Wire resistance is the fitted parameter, as a multiple of 14 nm minimum-width metal.",
            ],
        ),
        Chip(
            key="hp-dpe",
            label="HP Labs dot-product engine, 128 x 64 1T1R (no fitted parameter)",
            source="M. Hu et al., Advanced Materials 30, 1705914 (2018); C. Li et al., Nature Electronics 1, 52-59 (2018)",
            device=Device("HfO2 1T1R (HP)", ron_ohm=1 / 900e-6, roff_ohm=1 / 100e-6),
            config=ParasiticConfig(
                dot_product_dimension=32,
                active_rows=64,
                input_bits=8,
                adc_bits=12,
                v_lsb_volt=0.2 / 128,
                weight_distribution="uniform",
                conductance_variation_ratio=0.0,
                programming_sigma_ratio=6e-6 / 900e-6,
                input_encoding="voltage",
                dac_mismatch_ratio=0.0,
                iv_nonlinearity=0.01,
                r_bl_ohm=1.0,
                r_sl_ohm=1.0,
                r_sense_ohm=10.0,
                weight_trials=1,
            ),
            curve=None,
            baseline=None,
            free_parameter=None,
            bounds=None,
            fit=None,
            checks=[
                Observation(
                    "Vector-matrix multiplication with the DPE conversion algorithm (design-aware)",
                    "ir_aware",
                    precision_bits=6.0,
                ),
                Observation(
                    "Same array with linear output correction only",
                    "calibrated",
                    precision_bits=6.0,
                ),
            ],
            assumptions=[
                "128 x 64 array modeled as 64 rows of 32 differential weights (128 physical columns).",
                "Conductance 100-900 uS and write error sigma 6 uS from the sister Li et al. (2018) paper on the same array.",
                "Wire segment resistance of 1 ohm, as calibrated against the array by Zhang and Hu (arXiv 1912.08716).",
                "Instrument-grade DAC and ADC; I-V nonlinearity of 1% at the 0.2 V read bias is assumed.",
                "Precision in bits is (SNR - 1.76) / 6.02; the paper reports an accuracy equivalent of 6 bits.",
                "No MNIST noise-tolerance curve exists for this network, so only precision is checked.",
            ],
        ),
    ]


def configure(chip: Chip, value: float | None, overrides: dict | None = None) -> ParasiticConfig:
    if chip.free_parameter is None:
        config = chip.config
    elif chip.free_parameter == "wire_scale":
        wire = WIRE_PER_CELL_OHM[14] / 2 * value
        config = replace(chip.config, r_bl_ohm=wire, r_sl_ohm=wire)
    else:
        config = replace(chip.config, **{chip.free_parameter: value})
    return replace(config, **(overrides or {}))


def evaluate(chip: Chip, value: float | None, observation: Observation) -> dict:
    result = simulate_parasitic(chip.device, configure(chip, value, observation.overrides))
    level = result.level(observation.level)
    row = {
        "observation": observation.label,
        "compensation": observation.level,
        "compute_snr_db": round(level.total_snr_db, 2),
        "analog_snr_db": round(level.analog_snr_db, 2),
        "raw_snr_db": round(result.raw.total_snr_db, 2),
        "mean_gain": round(result.mean_gain, 3),
    }
    if observation.precision_bits is not None:
        bits = (level.total_snr_db - 1.76) / 6.02
        row.update(
            reported_bits=observation.precision_bits,
            predicted_bits=round(bits, 2),
            error=round(bits - observation.precision_bits, 2),
        )
        return row
    estimate = predict_accuracy(level.total_snr_db, NOISE_CURVES[chip.curve], chip.baseline)
    row.update(
        reported_accuracy=observation.accuracy,
        predicted_accuracy=round(estimate.accuracy, 2),
        error=round(estimate.accuracy - observation.accuracy, 2),
        equivalent_weight_noise=round(estimate.weight_noise, 4),
        extrapolated_curve=estimate.extrapolated,
    )
    return row


def fit_chip(chip: Chip) -> dict:
    if chip.fit is None:
        return {
            "chip": chip.label,
            "source": chip.source,
            "noise_curve": None,
            "free_parameter": None,
            "fitted_value": None,
            "fit": None,
            "checks": [evaluate(chip, None, check) for check in chip.checks],
            "assumptions": chip.assumptions,
            "config": asdict(chip.config),
        }
    low, high = chip.bounds
    grid = np.linspace(low, high, 9)
    scores = [abs(evaluate(chip, value, chip.fit)["error"]) for value in grid]
    best = int(np.argmin(scores))
    lo, hi = grid[max(best - 1, 0)], grid[min(best + 1, len(grid) - 1)]
    for _ in range(10):
        a, b = lo + (hi - lo) / 3, hi - (hi - lo) / 3
        if abs(evaluate(chip, a, chip.fit)["error"]) <= abs(evaluate(chip, b, chip.fit)["error"]):
            hi = b
        else:
            lo = a
    value = float((lo + hi) / 2)
    return {
        "chip": chip.label,
        "source": chip.source,
        "noise_curve": NOISE_CURVES[chip.curve].label,
        "free_parameter": chip.free_parameter,
        "fitted_value": round(value, 4),
        "fit": evaluate(chip, value, chip.fit),
        "checks": [evaluate(chip, value, check) for check in chip.checks],
        "assumptions": chip.assumptions,
        "config": {k: v for k, v in asdict(configure(chip, value)).items()},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=64, help="Active rows simulated per bank")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "fits" / "crossbar_chip_fits.json")
    parser.add_argument("--only", nargs="*", help="Chip keys to fit")
    args = parser.parse_args()

    results = []
    for chip in chips(args.rows):
        if args.only and chip.key not in args.only:
            continue
        fitted = fit_chip(chip)
        results.append(fitted)
        print(f"\n{fitted['chip']}: {fitted['free_parameter']} = {fitted['fitted_value']}")
        rows = ([("fit  ", fitted["fit"])] if fitted["fit"] else []) + [("check", row) for row in fitted["checks"]]
        for tag, row in rows:
            if "predicted_bits" in row:
                detail = f"reported {row['reported_bits']:.1f} b, predicted {row['predicted_bits']:.2f} b ({row['error']:+.2f} b)"
            else:
                detail = (
                    f"reported {row['reported_accuracy']:.2f}%, predicted {row['predicted_accuracy']:.2f}% "
                    f"({row['error']:+.2f} pt), weight noise {100 * row['equivalent_weight_noise']:.1f}%"
                )
            print(
                f"  {tag} {row['observation']}: {detail}; SNR {row['compute_snr_db']:.1f} dB "
                f"(raw {row['raw_snr_db']:.1f} dB, gain {row['mean_gain']:.2f})"
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"rows_simulated": args.rows, "chips": results}, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
