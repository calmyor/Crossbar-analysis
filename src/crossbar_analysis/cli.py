"""Command-line entry point for reproducible sensing-resistance sweeps."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from .model import DEVICE_PRESETS, CrossbarConfig, find_optimum, simulate_sweep


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sweep sensing resistance for the ISCAS 2022 crossbar model."
    )
    parser.add_argument("--device", choices=sorted(DEVICE_PRESETS), default="reram")
    parser.add_argument("--dimension", type=int, default=512, help="Logical dot-product N")
    parser.add_argument("--input-bits", type=int, default=5)
    parser.add_argument("--adc-bits", type=int, default=6)
    parser.add_argument("--samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--dac-mismatch", type=float, default=0.04)
    parser.add_argument("--bitcell-variation", type=float, default=0.04)
    parser.add_argument("--rs-min", type=float, default=100.0)
    parser.add_argument("--rs-max", type=float, default=10_000.0)
    parser.add_argument("--rs-points", type=int, default=25)
    parser.add_argument("--output", type=Path, default=Path("outputs/sweep.csv"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.rs_min <= 0 or args.rs_max <= args.rs_min or args.rs_points < 2:
        raise SystemExit("Require 0 < --rs-min < --rs-max and --rs-points >= 2.")

    config = CrossbarConfig(
        dot_product_dimension=args.dimension,
        input_bits=args.input_bits,
        adc_bits=args.adc_bits,
        dac_mismatch_ratio=args.dac_mismatch,
        conductance_variation_ratio=args.bitcell_variation,
        samples=args.samples,
        seed=args.seed,
    )
    grid = np.geomspace(args.rs_min, args.rs_max, args.rs_points)
    results = simulate_sweep(DEVICE_PRESETS[args.device], config, grid)
    optimum = find_optimum(results)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0].to_record()))
        writer.writeheader()
        writer.writerows(result.to_record() for result in results)

    print(
        f"{DEVICE_PRESETS[args.device].name}: logical N={config.dot_product_dimension} "
        f"({config.physical_columns} physical columns), ADC={config.adc_bits} b"
    )
    print(
        f"Best evaluated point: Rs={optimum.sensing_resistance_ohm:.6g} ohm, "
        f"SNR={optimum.snr_db:.3f} dB"
    )
    print(f"Wrote {len(results)} points to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
