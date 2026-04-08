"""Pure-numpy benchmark metrics.

All functions take numpy arrays and return Python floats so the results
serialise cleanly to JSON. No torch dependency.

Conventions
-----------
- ``y``  : ground-truth values
- ``yh`` : predicted mean values
- ``s``  : predicted standard deviation (uncertainty)
- All array inputs may be 1-D or 2-D; metrics broadcast over the last axis.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Point-accuracy metrics
# ---------------------------------------------------------------------------


def rmse(y: NDArray, yh: NDArray) -> float:
    """Root mean squared error."""
    return float(np.sqrt(np.mean((y - yh) ** 2)))


def mae(y: NDArray, yh: NDArray) -> float:
    """Mean absolute error."""
    return float(np.mean(np.abs(y - yh)))


def mape(y: NDArray, yh: NDArray, eps: float = 1e-9) -> float:
    """Mean absolute percentage error (as a fraction, not percent)."""
    return float(np.mean(np.abs((y - yh) / (np.abs(y) + eps))))


def sam_deg(y: NDArray, yh: NDArray) -> float:
    """Spectral Angle Mapper (SAM) in degrees, averaged over samples.

    For each spectrum, computes the angle between predicted and true vectors.
    Inputs must be 2-D ``(N, W)`` where N is the number of spectra and W is
    the number of wavelengths. 1-D inputs are treated as a single spectrum.
    """
    y = np.atleast_2d(y).astype(np.float64)
    yh = np.atleast_2d(yh).astype(np.float64)
    dot = np.sum(y * yh, axis=-1)
    norm_y = np.linalg.norm(y, axis=-1)
    norm_yh = np.linalg.norm(yh, axis=-1)
    cos_sim = dot / np.maximum(norm_y * norm_yh, 1e-12)
    cos_sim = np.clip(cos_sim, -1.0, 1.0)
    angle_rad = np.arccos(cos_sim)
    return float(np.degrees(angle_rad).mean())


def r2_score(y: NDArray, yh: NDArray) -> float:
    """Coefficient of determination R²."""
    y = np.asarray(y, dtype=np.float64)
    yh = np.asarray(yh, dtype=np.float64)
    ss_res = float(np.sum((y - yh) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    if ss_tot == 0:
        return 0.0
    return 1.0 - ss_res / ss_tot


# ---------------------------------------------------------------------------
# Uncertainty calibration metrics
# ---------------------------------------------------------------------------


def coverage(y: NDArray, yh: NDArray, s: NDArray, k: float = 1.0) -> float:
    """Fraction of true values within ±k·σ of the prediction."""
    y = np.asarray(y, dtype=np.float64)
    yh = np.asarray(yh, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    return float(np.mean(np.abs(y - yh) <= k * s))


def picp(y: NDArray, yh: NDArray, s: NDArray, level: float = 0.95) -> float:
    """Prediction Interval Coverage Probability for a Gaussian PI at *level*."""
    from scipy.stats import norm

    half = norm.ppf(0.5 + level / 2.0)
    return coverage(y, yh, s, k=half)


def sharpness(s: NDArray) -> float:
    """Mean predicted standard deviation."""
    return float(np.mean(np.asarray(s)))


def gaussian_crps(y: NDArray, yh: NDArray, s: NDArray) -> float:
    """Closed-form Continuous Ranked Probability Score for a Normal forecast.

    CRPS(N(μ,σ²), y) = σ · [ z·(2·Φ(z) - 1) + 2·φ(z) - 1/√π ]
    where z = (y - μ) / σ.

    Lower is better. Combines accuracy and calibration into one number.
    """
    from scipy.stats import norm

    y = np.asarray(y, dtype=np.float64)
    yh = np.asarray(yh, dtype=np.float64)
    s = np.maximum(np.asarray(s, dtype=np.float64), 1e-9)
    z = (y - yh) / s
    crps = s * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1.0 / np.sqrt(np.pi))
    return float(np.mean(crps))


def gaussian_nll(y: NDArray, yh: NDArray, s: NDArray) -> float:
    """Mean negative log-likelihood under a Normal predictive distribution."""
    y = np.asarray(y, dtype=np.float64)
    yh = np.asarray(yh, dtype=np.float64)
    s = np.maximum(np.asarray(s, dtype=np.float64), 1e-9)
    nll = 0.5 * np.log(2 * np.pi * s ** 2) + 0.5 * ((y - yh) / s) ** 2
    return float(np.mean(nll))


# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------


def topk_accuracy(y_true: NDArray, probs: NDArray, k: int = 1) -> float:
    """Top-k classification accuracy.

    y_true : (N,) integer class labels
    probs  : (N, C) class probabilities
    """
    y_true = np.asarray(y_true)
    topk = np.argsort(probs, axis=-1)[:, -k:]
    hits = np.any(topk == y_true[:, None], axis=-1)
    return float(np.mean(hits))


def brier_multiclass(y_true: NDArray, probs: NDArray) -> float:
    """Multi-class Brier score (proper scoring rule, lower is better)."""
    y_true = np.asarray(y_true).astype(int)
    n, c = probs.shape
    one_hot = np.zeros((n, c))
    one_hot[np.arange(n), y_true] = 1.0
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=-1)))


def ece(y_true: NDArray, probs: NDArray, n_bins: int = 15) -> float:
    """Expected Calibration Error of the top-1 prediction.

    Bins predictions by confidence (max prob) and compares mean confidence
    to actual accuracy in each bin.
    """
    y_true = np.asarray(y_true).astype(int)
    confidences = probs.max(axis=-1)
    predictions = probs.argmax(axis=-1)
    accuracies = (predictions == y_true).astype(np.float64)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece_val = 0.0
    n = len(y_true)
    for i in range(n_bins):
        in_bin = (confidences > bin_edges[i]) & (confidences <= bin_edges[i + 1])
        if i == 0:
            in_bin = in_bin | (confidences == 0.0)
        n_in_bin = int(in_bin.sum())
        if n_in_bin == 0:
            continue
        acc_bin = float(accuracies[in_bin].mean())
        conf_bin = float(confidences[in_bin].mean())
        ece_val += (n_in_bin / n) * abs(acc_bin - conf_bin)
    return float(ece_val)


def per_class_prf1(
    y_true: NDArray, y_pred: NDArray, class_names: list[str]
) -> dict[str, dict[str, float]]:
    """Per-class precision/recall/F1 + support."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    out: dict[str, dict[str, float]] = {}
    for ci, name in enumerate(class_names):
        tp = int(((y_pred == ci) & (y_true == ci)).sum())
        fp = int(((y_pred == ci) & (y_true != ci)).sum())
        fn = int(((y_pred != ci) & (y_true == ci)).sum())
        support = int((y_true == ci).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        out[name] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": support,
        }
    return out


def macro_f1(per_class: dict[str, dict[str, float]]) -> float:
    """Unweighted mean of per-class F1 scores."""
    f1s = [v["f1"] for v in per_class.values() if v["support"] > 0]
    if not f1s:
        return 0.0
    return float(np.mean(f1s))


def confusion_matrix(
    y_true: NDArray, y_pred: NDArray, n_classes: int
) -> NDArray:
    """Standard confusion matrix; rows = true, cols = predicted."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < n_classes and 0 <= p < n_classes:
            cm[t, p] += 1
    return cm


def entropy_error_correlation(
    is_error: NDArray, entropies: NDArray
) -> float:
    """Pearson correlation between predicted entropy and 0/1 error.

    Higher = entropy is a useful indicator of mistakes.
    """
    is_error = np.asarray(is_error, dtype=np.float64)
    entropies = np.asarray(entropies, dtype=np.float64)
    if is_error.std() == 0 or entropies.std() == 0:
        return 0.0
    return float(np.corrcoef(is_error, entropies)[0, 1])
