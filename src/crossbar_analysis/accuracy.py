"""Map crossbar compute SNR to network accuracy with measured noise-tolerance curves.

Compute SNR is first converted into an equivalent Gaussian weight-noise level.
For a dot product ``y = w . x`` with independent weight errors of standard
deviation ``sigma_w * max|w|``, the compute SNR is

    SNR = E[w^2] / (sigma_w^2 * max|w|^2),

so ``sigma_w = sqrt(rho / SNR)`` with ``rho = E[w^2] / max|w|^2``. Uniformly
distributed weights give ``rho = 1/3``, and weights trained with noise injection
approach that shape (Wan et al., Nature 2022, Extended Data Fig. 6d).

The weight-noise level is then mapped to accuracy using curves reported by
crossbar chip papers. The curves are read from published figures and carry
roughly one percentage point of reading error.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

UNIFORM_WEIGHT_POWER = 1.0 / 3.0


@dataclass(frozen=True)
class NoiseCurve:
    """Accuracy versus inference-time weight noise for one trained model."""

    key: str
    label: str
    source: str
    points: tuple[tuple[float, float], ...]
    chance: float = 10.0
    note: str = ""

    @property
    def clean_accuracy(self) -> float:
        return self.points[0][1]


@dataclass(frozen=True)
class AccuracyEstimate:
    """Accuracy predicted from a compute SNR.

    ``extrapolated`` is true when the equivalent weight noise lies beyond the
    measured curve. The accuracy is then the last measured value, an upper bound.
    """

    weight_noise: float
    accuracy: float
    retention: float
    extrapolated: bool


_NEURRAM_SOURCE = (
    "W. Wan et al., 'A compute-in-memory chip based on resistive random-access "
    "memory,' Nature 608, 504-512 (2022), Extended Data Fig. 6a (read from the figure)."
)
_JOSHI_SOURCE = (
    "V. Joshi et al., 'Accurate deep neural network inference using computational "
    "phase-change memory,' Nature Communications 11, 2473 (2020), Fig. 2 (read from the figure)."
)

NOISE_CURVES = {
    curve.key: curve
    for curve in (
        NoiseCurve(
            key="resnet20-cifar10-no-injection",
            label="ResNet-20 / CIFAR-10, trained without noise injection",
            source=_NEURRAM_SOURCE,
            points=((0.00, 86.5), (0.05, 66.5), (0.10, 45.0)),
            note="The 10% point lies below the plotted axis; 45% is an upper bound.",
        ),
        NoiseCurve(
            key="resnet20-cifar10-injection-5",
            label="ResNet-20 / CIFAR-10, trained with 5% weight noise",
            source=_NEURRAM_SOURCE,
            points=((0.00, 87.0), (0.05, 85.0), (0.10, 77.0), (0.15, 58.5)),
        ),
        NoiseCurve(
            key="resnet20-cifar10-injection-10",
            label="ResNet-20 / CIFAR-10, trained with 10% weight noise",
            source=_NEURRAM_SOURCE,
            points=((0.00, 86.0), (0.05, 85.5), (0.10, 83.0), (0.15, 74.5)),
        ),
        NoiseCurve(
            key="resnet20-cifar10-injection-20",
            label="ResNet-20 / CIFAR-10, trained with 20% weight noise (best model)",
            source=_NEURRAM_SOURCE,
            points=((0.00, 85.5), (0.05, 85.0), (0.10, 84.5), (0.15, 82.5)),
        ),
        NoiseCurve(
            key="resnet32-cifar10-matched-injection",
            label="ResNet-32 / CIFAR-10, training noise matched to inference noise",
            source=_JOSHI_SOURCE,
            points=(
                (0.000, 93.87),
                (0.026, 93.73),
                (0.033, 93.63),
                (0.038, 93.62),
                (0.043, 93.52),
                (0.0565, 93.28),
                (0.075, 92.78),
                (0.113, 91.45),
            ),
        ),
        NoiseCurve(
            key="resnet32-cifar10-injection-2.6",
            label="ResNet-32 / CIFAR-10, trained with 2.6% weight noise",
            source=_JOSHI_SOURCE,
            points=(
                (0.000, 93.90),
                (0.026, 93.78),
                (0.033, 93.68),
                (0.038, 93.55),
                (0.043, 93.40),
                (0.0565, 92.95),
                (0.075, 92.05),
                (0.113, 88.90),
            ),
        ),
    )
}


def equivalent_weight_noise(snr_db: float, weight_power: float = UNIFORM_WEIGHT_POWER) -> float:
    """Weight-noise level, as a fraction of the maximum weight, with the same SNR."""

    return float(np.sqrt(weight_power / 10.0 ** (snr_db / 10.0)))


def snr_for_weight_noise(weight_noise: float, weight_power: float = UNIFORM_WEIGHT_POWER) -> float:
    """Compute SNR, in dB, produced by a given relative weight-noise level."""

    if weight_noise <= 0:
        return float("inf")
    return float(10.0 * np.log10(weight_power / weight_noise**2))


def predict_accuracy(
    snr_db: float,
    curve: NoiseCurve,
    baseline: float | None = None,
    weight_power: float = UNIFORM_WEIGHT_POWER,
) -> AccuracyEstimate:
    """Predict accuracy from compute SNR with a measured noise-tolerance curve.

    When ``baseline`` is given, the curve's relative retention above chance is
    applied to that baseline, so a curve measured on one network can be used for
    a chip reported against a different software accuracy.
    """

    sigma = equivalent_weight_noise(snr_db, weight_power)
    noise = np.array([point[0] for point in curve.points])
    accuracy = np.array([point[1] for point in curve.points])
    extrapolated = sigma > noise[-1]
    if extrapolated:
        # No measurement exists this far out, so report the last measured
        # accuracy as an upper bound instead of extending the curve.
        value = float(accuracy[-1])
    else:
        value = float(np.interp(sigma, noise, accuracy))
    value = float(np.clip(value, curve.chance, curve.clean_accuracy))
    retention = (value - curve.chance) / (curve.clean_accuracy - curve.chance)
    if baseline is not None:
        value = curve.chance + (baseline - curve.chance) * retention
    return AccuracyEstimate(sigma, float(value), float(retention), bool(extrapolated))
