#!/usr/bin/env python3
"""Render the archived processed device trends used in paper Figure 3."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
COLORS = {"MRAM": "#202020", "ReRAM": "#3157c8", "FeFET": "#d43c35"}


def read_series(name: str) -> tuple[list[float], dict[str, list[float]]]:
    path = ROOT / "data" / "processed" / "figure3" / name
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    x_name = next(key for key in rows[0] if key not in COLORS)
    return (
        [float(row[x_name]) for row in rows],
        {device: [float(row[device]) for row in rows] for device in COLORS},
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "figure3_archived_summary.svg",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dimension, by_dimension = read_series("snr_vs_dimension.csv")
    bits, by_bits = read_series("snr_vs_adc_bits.csv")
    contrast, by_contrast = read_series("snr_vs_contrast.csv")

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2), constrained_layout=True)
    panels = [
        (axes[0], dimension, by_dimension, "Logical dot-product dimension N", True),
        (axes[1], bits, by_bits, "ADC precision (bits)", False),
        (axes[2], contrast, by_contrast, "Resistive contrast $R_{off}/R_{on}$", True),
    ]
    for axis, x_values, series, xlabel, log_scale in panels:
        for device, values in series.items():
            axis.plot(x_values, values, label=device, color=COLORS[device], linewidth=2, marker="o", markersize=3)
        if log_scale:
            axis.set_xscale("log")
        axis.set_xlabel(xlabel)
        axis.grid(True, which="both", linewidth=0.45, alpha=0.35)
    axes[0].set_ylabel("Maximum compute SNR (dB)")
    axes[1].legend(frameon=False, loc="lower right")
    fig.suptitle("Archived processed trends from ISCAS 2022 Figure 3", fontsize=13)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, format=args.output.suffix.lstrip(".") or "svg", dpi=180)
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
