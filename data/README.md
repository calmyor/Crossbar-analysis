# Data provenance

## `spice/`

`reram_n512_rs_sweep.csv` is the tabulated SPICE comparison embedded in the original Figure 2 plotting script. It represents the paper’s specific ReRAM validation configuration. The commercial 22 nm PDK and transistor-level netlists are not included.

## `processed/figure3/`

These CSV files are readable exports of the nine final `.npy` arrays consumed by the archived Figure 3 aggregation scripts. They are retained as processed historical outputs because the original generator scripts did not save every array required by the downstream plotters.

The dimension CSV reports the paper’s logical (N). The archived generators used a variable equal to the physical (2N) column count and the final plot divided that value by two.

## `network/`

`paper_points.csv` records the exact annotated ResNet-20/CIFAR-10 results reported in the paper. The fixed-point baseline was 84.94% accuracy; inputs were signed 5-bit values and weights were ternary.

## `legacy/network_grids/`

The six historical grid files are preserved for provenance only. The paper’s annotated Figure 4 points do not all come from a single consistently named “final” grid, and the archived surface plotter did not itself regenerate the red/blue annotations. Do not treat these grids as a complete end-to-end reproduction of Figure 4.

`SHA256SUMS` records the hashes of every curated data file in this repository.
