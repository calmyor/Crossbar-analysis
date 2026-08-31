<p align="center">
  <img src="docs/assets/crossbar-mark.svg" width="120" alt="Crossbar IMC project mark" />
</p>

# Crossbar IMC accuracy analysis

[Research website](https://calmyor.github.io/Crossbar-analysis/) · [IEEE paper](https://doi.org/10.1109/ISCAS48785.2022.9937336)

This repository is the curated companion to **“Fundamental Limits on the Computational Accuracy of Resistive Crossbar-based In-memory Architectures”** (IEEE ISCAS 2022). It reorganizes the original exploratory scripts into a reproducible behavioral model, retains the small processed results used in the paper, and serves the project website from `docs/`.

The central question is: for a voltage-driven resistive crossbar, which sensing resistance maximizes compute signal-to-noise ratio (SNR) once bitcell variation, input-DAC mismatch, ADC clipping, and ADC quantization are considered together?

## Repository map

```text
configs/                  Paper evaluation parameters
src/crossbar_analysis/    Reusable Monte Carlo model and command-line interface
scripts/                  Figure and result reproduction entry points
data/
  spice/                  Precomputed SPICE comparison points
  processed/figure3/      Archived aggregate SNR trends from the paper workflow
  network/                Curated paper-level ResNet-20 results
  legacy/network_grids/   Archived accuracy sweeps with provenance caveats
docs/                     Static research website (GitHub Pages)
paper/                    Publication links and scope notes
tests/                    Deterministic model and CLI checks
```

The original 1.1 GB working directory is intentionally excluded from Git. It remains available locally under `ISCAS-2022/`, `Python Model/`, and `Conference-LaTeX-template_10-17-19/`, but includes duplicate drafts, third-party papers, incomplete older experiments, and activation tensors that exceed GitHub’s file limit.

## Quick start

Python 3.9 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[plots]"

crossbar-sweep \
  --device reram \
  --dimension 512 \
  --adc-bits 6 \
  --samples 10000 \
  --rs-min 100 \
  --rs-max 10000 \
  --rs-points 25 \
  --output outputs/reram_n512.csv
```

`--dimension` is the paper’s logical dot-product length, (N). Differential weight encoding uses (2N) physical columns. Several archived scripts used `N` for the physical-column count, so an archived value of `N = 1024` corresponds to paper (N = 512).

## Reproduce the published evidence route

Generate a fresh behavioral sweep and compare it with the archived SPICE points used for the paper’s Figure 2:

```bash
python scripts/reproduce_fig2.py --samples 10000
```

Render the archived processed trends behind Figure 3:

```bash
python scripts/plot_archived_fig3.py
```

Summarize the exact paper-level ResNet-20 comparison points:

```bash
python scripts/summarize_network_results.py
```

Run the validation suite:

```bash
python -m unittest discover -s tests
```

## What is—and is not—reproduced

- **Open behavioral model:** the implementation follows the paper’s voltage-driven 1T1R assumptions, binary (R_\mathrm{on}/R_\mathrm{off}) states, differential encoding, and four reported compute-SNR error terms.
- **SPICE comparison:** `data/spice/` contains the published comparison points. The commercial 22 nm PDK and transistor-level netlists are not redistributed.
- **Network experiment:** `data/network/paper_points.csv` records the values reported in the paper. The archived grid files are retained separately because the historical files do not form one clean end-to-end regeneration path for every plotted annotation.
- **Multi-bank website explorer:** workload partitioning is an architectural illustration. The paper motivates multi-bank scaling but does not measure a complete multi-bank chip.
- **Model boundary:** wire parasitics are not included in the paper’s reported compute-SNR equation and are not silently folded into this implementation.

The reported “fundamental limits” are conditional on these assumptions and evaluation parameters; they are not universal limits for every device, circuit, or workload.

## Citation

If this code or website helps your work, please cite:

```bibtex
@inproceedings{roy2022crossbarlimits,
  author    = {Saion K. Roy and Ameya Patil and Naresh R. Shanbhag},
  title     = {Fundamental Limits on the Computational Accuracy of Resistive Crossbar-based In-memory Architectures},
  booktitle = {2022 IEEE International Symposium on Circuits and Systems (ISCAS)},
  pages     = {384--388},
  year      = {2022},
  doi       = {10.1109/ISCAS48785.2022.9937336}
}
```
