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
  parasitics.py           Parasitic-aware crossbar network model
  accuracy.py             Compute-SNR to network-accuracy mapping
scripts/                  Figure and result reproduction entry points
data/
  spice/                  Precomputed SPICE comparison points
  processed/figure3/      Archived aggregate SNR trends from the paper workflow
  network/                Curated paper-level ResNet-20 results
  legacy/network_grids/   Archived accuracy sweeps with provenance caveats
  fits/                   Parasitic model fits to measured crossbar chips
docs/                     Static research website (GitHub Pages)
paper/                    Publication links and scope notes
tests/                    Deterministic model and CLI checks
```

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

## Parasitic-aware crossbar model

The ISCAS 2022 equation treats each source line as an ideal conductance sum. Measured crossbar chips show that this is optimistic: wire IR drop, driver and input-rail resistance, and DAC and device nonlinearity dominate once the array grows. `crossbar_analysis.parasitics` solves the full resistive network of an `M x 2N` crossbar with these effects and reports compute SNR at four compensation levels:

| Level | What it removes |
|---|---|
| `raw` | Nothing |
| `calibrated` | Per-output gain and offset |
| `ir_aware` | The deterministic IR-drop distortion predicted from the nominal design |
| `chip_in_loop` | All static error of one die, leaving only input-dependent error |

`crossbar_analysis.accuracy` converts compute SNR into an equivalent weight-noise level and maps it to accuracy with noise-tolerance curves read from the NeuRRAM (Nature 2022) and IBM PCM (Nature Communications 2020) papers.

Fit the model to measured chips and check held-out results:

```bash
python scripts/fit_crossbar_chips.py --rows 64
```

Each chip keeps its published circuit parameters, and one unknown is fitted to one reported result:

| Chip | Fitted unknown | Held-out check |
|---|---|---|
| IBM HERMES, 14 nm PCM | Wire resistance, 0.33x minimum-width metal | One device per polarity: 92.59% predicted, 92.23% reported |
| NeuRRAM, 130 nm RRAM | Shared input-rail resistance, 2.5 ohm | After chip-in-the-loop tuning: 84.80% predicted, 85.66% reported |
| HP dot-product engine, 128 x 64 | None | 5.5 bits predicted, 6 bits reported |

Regenerate the data behind the website's multi-bank architect:

```bash
python scripts/build_architect_data.py --rows 64 --trials 2
```

The fits rest on stated assumptions, listed in `data/fits/crossbar_chip_fits.json`. The noise-tolerance curves are read from published figures, so differences below one percentage point are within reading error.

## What is—and is not—reproduced

- **Open behavioral model:** the implementation follows the paper’s voltage-driven 1T1R assumptions, binary (R_\mathrm{on}/R_\mathrm{off}) states, differential encoding, and four reported compute-SNR error terms.
- **SPICE comparison:** `data/spice/` contains the published comparison points. The commercial 22 nm PDK and transistor-level netlists are not redistributed.
- **Network experiment:** `data/network/paper_points.csv` records the values reported in the paper. The archived grid files are retained separately because the historical files do not form one clean end-to-end regeneration path for every plotted annotation.
- **Multi-bank website explorer:** bank SNR and accuracy come from the parasitic-aware model, not from measurements of a multi-bank chip. The paper motivates multi-bank scaling but does not measure one.
- **Model boundary:** wire parasitics are not part of the paper’s reported compute-SNR equation, and `model.py` does not fold them in. The separate `parasitics.py` extension adds them.

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
