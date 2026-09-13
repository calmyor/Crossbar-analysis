#!/usr/bin/env python3
"""Precompute parasitic-aware bank SNR and predicted accuracy for the website architect.

The grid covers the three ISCAS 2022 devices, logical bank dimensions from 16 to
512, two wire options, and two input schemes. Each point is simulated with the
full crossbar network model and mapped to ResNet-20 / CIFAR-10 accuracy with the
measured noise-tolerance curves. The chip fits are embedded so the page can show
how the model compares with measured crossbar chips.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from crossbar_analysis.accuracy import NOISE_CURVES, predict_accuracy
from crossbar_analysis.model import DEVICE_PRESETS
from crossbar_analysis.parasitics import (
    COMPENSATION_LEVELS,
    LAYOUT_22NM_R_BL_OHM,
    LAYOUT_22NM_R_SL_OHM,
    ParasiticConfig,
    simulate_parasitic,
)

ROOT = Path(__file__).resolve().parents[1]
DIMENSIONS = (16, 32, 64, 128, 256, 512)
INPUT_BITS = 5
V_LSB_VOLT = 3e-3
BASELINE_ACCURACY = 84.94  # ISCAS 2022 fixed-point ResNet-20 on CIFAR-10
# Characteristic I-V voltage consistent with the NeuRRAM (5% at 0.5 V) and HP
# (1% at 0.2 V) fits: excess current fraction = (V / 2.2 V)^2.
IV_CHARACTERISTIC_VOLT = 2.2
SENSE_OHM = {"mram": 300.0, "reram": 700.0, "fefet": 10_000.0}


def neurosim_wire_ohm_per_um(width_nm: float, aspect: float, rho: float, barrier_nm: float) -> float:
    rho_eff = rho / (1 - ((2 * aspect * width_nm + width_nm) * barrier_nm / (aspect * width_nm**2)))
    return rho_eff / (width_nm * 1e-9 * width_nm * 1e-9 * aspect) * 1e-6


MIN_WIDTH_22NM_OHM = neurosim_wire_ohm_per_um(40, 1.9, 2.97e-8, 2.5) * 4 * 0.040

WIRES = {
    "layout": {
        "label": "Wide metal, 22 nm layout",
        "detail": "0.36 ohm bit-line and 0.16 ohm source-line per cell, extracted from a 22 nm IMC layout (JxCDC 2024).",
        "r_bl_ohm": LAYOUT_22NM_R_BL_OHM,
        "r_sl_ohm": LAYOUT_22NM_R_SL_OHM,
    },
    "minimum": {
        "label": "Minimum-width 22 nm metal",
        "detail": f"{MIN_WIDTH_22NM_OHM:.2f} ohm per cell on a 4F 1T1R pitch (NeuroSim V1.4 interconnect model).",
        "r_bl_ohm": MIN_WIDTH_22NM_OHM,
        "r_sl_ohm": MIN_WIDTH_22NM_OHM,
    },
}

INPUTS = {
    "voltage": {
        "label": "Voltage DAC",
        "detail": "Amplitude-coded inputs with 4% finger mismatch and an assumed 0.5 LSB rms static INL.",
        "config": {"input_encoding": "voltage", "dac_inl_lsb": 0.5, "dac_mismatch_ratio": 0.04},
    },
    "pwm": {
        "label": "Pulse-width input",
        "detail": "Time-coded inputs at a fixed read amplitude, as in the IBM HERMES chip; no amplitude DAC error.",
        "config": {"input_encoding": "pwm", "dac_inl_lsb": 0.0, "dac_mismatch_ratio": 0.0},
    },
}

CURVES = {
    "trained": "resnet20-cifar10-injection-20",
    "untrained": "resnet20-cifar10-no-injection",
}

LEVEL_LABELS = {
    "raw": "No compensation",
    "calibrated": "Per-column calibration",
    "ir_aware": "IR-drop-aware mapping",
    "chip_in_loop": "Chip-in-the-loop tuning",
}


def accuracy_map(snr_db: float) -> tuple[dict, dict]:
    """Predicted accuracy per training curve, and whether each is an upper bound."""

    estimates = {
        name: predict_accuracy(snr_db, NOISE_CURVES[curve], BASELINE_ACCURACY)
        for name, curve in CURVES.items()
    }
    return (
        {name: round(estimate.accuracy, 2) for name, estimate in estimates.items()},
        {name: estimate.extrapolated for name, estimate in estimates.items()},
    )


def remap_points(points: list[dict]) -> list[dict]:
    for point in points:
        for level in point["levels"].values():
            level["accuracy"], level["bounded"] = accuracy_map(level["snr_db"])
    return points


def load_points(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    payload = json.loads(text[text.index("{") : text.rindex("}") + 1])
    return payload["points"]


def simulate_grid(rows: int, trials: int) -> list[dict]:
    full_scale_volt = 2 ** (INPUT_BITS - 1) * V_LSB_VOLT
    iv_nonlinearity = (full_scale_volt / IV_CHARACTERISTIC_VOLT) ** 2
    points = []
    for device_key, device in DEVICE_PRESETS.items():
        for wire_key, wire in WIRES.items():
            for input_key, scheme in INPUTS.items():
                for dimension in DIMENSIONS:
                    start = time.time()
                    config = ParasiticConfig(
                        dot_product_dimension=dimension,
                        active_rows=rows,
                        input_bits=INPUT_BITS,
                        adc_bits=6,
                        v_lsb_volt=V_LSB_VOLT,
                        conductance_variation_ratio=0.04,
                        iv_nonlinearity=iv_nonlinearity,
                        r_bl_ohm=wire["r_bl_ohm"],
                        r_sl_ohm=wire["r_sl_ohm"],
                        r_sense_ohm=SENSE_OHM[device_key],
                        weight_trials=trials,
                        **scheme["config"],
                    )
                    result = simulate_parasitic(device, config)
                    levels = {}
                    for level in COMPENSATION_LEVELS:
                        snr = result.level(level).total_snr_db
                        accuracy, bounded = accuracy_map(snr)
                        levels[level] = {"snr_db": round(snr, 2), "accuracy": accuracy, "bounded": bounded}
                    points.append(
                        {
                            "device": device_key,
                            "wire": wire_key,
                            "input": input_key,
                            "N": dimension,
                            "gain": round(result.mean_gain, 3),
                            "levels": levels,
                        }
                    )
                    print(
                        f"{device_key:5} {wire_key:7} {input_key:7} N={dimension:3d} "
                        f"raw {levels['raw']['snr_db']:6.1f} cal {levels['calibrated']['snr_db']:6.1f} "
                        f"ir {levels['ir_aware']['snr_db']:6.1f} loop {levels['chip_in_loop']['snr_db']:6.1f} dB "
                        f"[{time.time() - start:.0f}s]",
                        flush=True,
                    )
    return points


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=64)
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--fits", type=Path, default=ROOT / "data" / "fits" / "crossbar_chip_fits.json")
    parser.add_argument("--output", type=Path, default=ROOT / "docs" / "assets" / "data" / "crossbar-model.js")
    parser.add_argument(
        "--reuse",
        action="store_true",
        help="Keep the simulated SNRs in --output and only recompute accuracy and chip fits",
    )
    args = parser.parse_args()
    points = remap_points(load_points(args.output)) if args.reuse else simulate_grid(args.rows, args.trials)

    fits = json.loads(args.fits.read_text(encoding="utf-8")) if args.fits.exists() else {"chips": []}
    payload = {
        "generated_by": "scripts/build_architect_data.py",
        "rows_simulated": args.rows,
        "weight_trials": args.trials,
        "baseline_accuracy": BASELINE_ACCURACY,
        "dimensions": list(DIMENSIONS),
        "devices": {key: {"label": device.name, "contrast": device.contrast} for key, device in DEVICE_PRESETS.items()},
        "wires": {key: {k: v for k, v in wire.items() if k in ("label", "detail")} for key, wire in WIRES.items()},
        "inputs": {key: {k: v for k, v in scheme.items() if k in ("label", "detail")} for key, scheme in INPUTS.items()},
        "levels": LEVEL_LABELS,
        "curves": {name: {"label": NOISE_CURVES[key].label, "source": NOISE_CURVES[key].source} for name, key in CURVES.items()},
        "assumptions": [
            "Binary differential weights, signed 5-bit inputs with a 3 mV LSB, 4% bitcell variation, and a 6-bit ADC with optimized clipping, as in ISCAS 2022.",
            f"Device I-V nonlinearity follows (V / {IV_CHARACTERISTIC_VOLT} V)^2, consistent with the NeuRRAM and HP fits.",
            f"{args.rows} rows computed together per bank; current-mode sensing into a fixed sensing resistance per device.",
            "Accuracy maps compute SNR to an equivalent weight-noise level, then to ResNet-20 retention measured by NeuRRAM, scaled to the 84.94% ISCAS baseline.",
            "Beyond the largest measured weight noise, accuracy is shown as an upper bound, not extrapolated.",
        ],
        "points": points,
        "chip_fits": fits.get("chips", []),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "// Generated by scripts/build_architect_data.py. Do not edit by hand.\n"
        f"window.CROSSBAR_MODEL = {json.dumps(payload, separators=(',', ':'))};\n",
        encoding="utf-8",
    )
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
