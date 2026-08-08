#!/usr/bin/env python3
"""Print the exact ResNet-20 accuracy comparison reported in the paper."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    path = ROOT / "data" / "network" / "paper_points.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    print("Device  exhaustive  SNR-selected  gap (percentage points)")
    for row in rows:
        exhaustive = float(row["exhaustive_accuracy_percent"])
        selected = float(row["snr_selected_accuracy_percent"])
        print(f"{row['device']:<7} {exhaustive:>10.2f} {selected:>13.2f} {exhaustive - selected:>8.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
