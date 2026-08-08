#!/usr/bin/env python3
"""Reproduce the Figure 2 behavioral sweep and overlay archived SPICE points."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
import numpy as np

from crossbar_analysis import DEVICE_PRESETS, CrossbarConfig, find_optimum, simulate_sweep

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "figure2_reproduction.svg",
    )
    return parser.parse_args()


def load_spice_points() -> tuple[np.ndarray, np.ndarray]:
    path = ROOT / "data" / "spice" / "reram_n512_rs_sweep.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return (
        np.asarray([float(row["sensing_resistance_ohm"]) for row in rows]),
        np.asarray([float(row["snr_db"]) for row in rows]),
    )


def main() -> int:
    args = parse_args()
    config = CrossbarConfig(dot_product_dimension=512, adc_bits=6, samples=args.samples, seed=args.seed)
    rs_grid = np.geomspace(100.0, 10_000.0, 25)
    results = simulate_sweep(DEVICE_PRESETS["reram"], config, rs_grid)
    optimum = find_optimum(results)
    spice_rs, spice_snr = load_spice_points()

    fig, ax = plt.subplots(figsize=(8.0, 5.2), constrained_layout=True)
    ax.plot(
        [result.sensing_resistance_ohm for result in results],
        [result.snr_db for result in results],
        color="#3157c8",
        marker="o",
        markersize=4,
        linewidth=2,
        label=f"Behavioral model ({args.samples:,} samples)",
    )
    ax.scatter(spice_rs, spice_snr, color="#171717", marker="D", s=28, label="Archived SPICE points")
    ax.axvline(optimum.sensing_resistance_ohm, color="#cc2f2f", linestyle="--", linewidth=1.5)
    ax.annotate(
        f"best evaluated $R_s$ = {optimum.sensing_resistance_ohm:.0f} Ω",
        (optimum.sensing_resistance_ohm, optimum.snr_db),
        xytext=(8, -28),
        textcoords="offset points",
        fontsize=9,
    )
    ax.set_xscale("log")
    ax.set_xlabel("Sensing resistance $R_s$ (Ω)")
    ax.set_ylabel("Compute SNR (dB)")
    ax.set_title("ReRAM crossbar: logical N = 512, 6-bit ADC")
    ax.grid(True, which="both", linewidth=0.5, alpha=0.35)
    ax.legend(frameon=False)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, format=args.output.suffix.lstrip(".") or "svg", dpi=180)
    print(f"Wrote {args.output}")
    print(f"Best evaluated point: Rs={optimum.sensing_resistance_ohm:.6g} ohm, SNR={optimum.snr_db:.3f} dB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
