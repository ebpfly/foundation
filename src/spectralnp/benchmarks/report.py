"""Report generation: JSON + summary.md + matplotlib plots."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def make_results_dir(base_dir: Path | str, model_name: str) -> Path:
    """Create timestamped output directory."""
    base = Path(base_dir)
    ts = dt.datetime.now().strftime("%Y-%m-%d_%H-%M")
    out = base / f"{model_name}_{ts}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def save_results_json(out_dir: Path, results: dict[str, Any]) -> Path:
    """Write the full results dict as JSON."""
    path = out_dir / "results.json"
    with path.open("w") as f:
        json.dump(results, f, indent=2, default=_json_default)
    return path


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    raise TypeError(f"Not JSON serialisable: {type(o)}")


# ---------------------------------------------------------------------------
# Summary markdown
# ---------------------------------------------------------------------------


def _continuous_table(by_sensor: dict[str, dict]) -> str:
    """Markdown table for use case 1 / 2 results."""
    lines = [
        "| Sensor | bands | RMSE | MAE | SAM° | R² | Cov 2σ | CRPS | Sharpness |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, m in by_sensor.items():
        lines.append(
            f"| {name} | {m['n_bands']} "
            f"| {m['rmse']:.4f} | {m['mae']:.4f} | {m['sam_deg']:.2f} "
            f"| {m['r2']:.3f} | {m['coverage_2sigma']:.3f} "
            f"| {m['crps']:.4f} | {m['sharpness']:.4f} |"
        )
    return "\n".join(lines)


def _scaling_table(scaling: dict[str, list]) -> str:
    if not scaling.get("n_bands"):
        return "_(no scaling data)_"
    lines = [
        "| n_bands | RMSE | Sharpness | CRPS |",
        "|---:|---:|---:|---:|",
    ]
    for nb, r, sh, c in zip(scaling["n_bands"], scaling["rmse"], scaling["sharpness"], scaling["crps"]):
        lines.append(f"| {nb} | {r:.4f} | {sh:.4f} | {c:.4f} |")
    return "\n".join(lines)


def _material_table(material: dict) -> str:
    lines = [
        f"- **Top-1 (category)**: {material.get('top1_category', 0):.3f}",
        f"- **Top-3 (category)**: {material.get('top3_category', 0):.3f}",
        f"- **Macro F1**: {material.get('macro_f1', 0):.3f}",
        f"- **ECE**: {material.get('ece_category', 0):.3f}",
        f"- **Brier**: {material.get('brier_category', 0):.3f}",
        f"- **Entropy↔error correlation**: {material.get('entropy_error_correlation', 0):.3f}",
        "",
        "### Per-category breakdown",
        "",
        "| Category | Precision | Recall | F1 | Support |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, m in material.get("per_class", {}).items():
        lines.append(
            f"| {name} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} | {m['support']} |"
        )
    return "\n".join(lines)


def write_summary_md(out_dir: Path, results: dict) -> Path:
    """Write a human-readable summary."""
    md = ["# SpectralNP Benchmark Results", ""]
    md.append(f"- **Model**: `{results.get('model_path', 'unknown')}`")
    md.append(f"- **Generated**: {results.get('timestamp', '')}")
    md.append(f"- **Test set size**: {results.get('n_test_spectra', '?')} held-out USGS spectra")
    md.append("")

    md.append("## 1. At-sensor radiance prediction")
    md.append("")
    md.append(_continuous_table(results["radiance"]["by_sensor"]))
    md.append("")
    md.append("### Scaling with band count")
    md.append("")
    md.append(_scaling_table(results["radiance"]["scaling"]))
    md.append("")

    md.append("## 2. Surface reflectance prediction")
    md.append("")
    md.append(_continuous_table(results["reflectance"]["by_sensor"]))
    md.append("")
    md.append("### Scaling with band count")
    md.append("")
    md.append(_scaling_table(results["reflectance"]["scaling"]))
    md.append("")

    md.append("## 3. Material classification")
    md.append("")
    md.append(_material_table(results["material"]))
    md.append("")

    path = out_dir / "summary.md"
    path.write_text("\n".join(md))
    return path


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_scaling(out_dir: Path, scaling: dict, title: str, filename: str) -> Path | None:
    if not scaling.get("n_bands"):
        return None
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    nb = scaling["n_bands"]
    axes[0].plot(nb, scaling["rmse"], "o-", color="#e74c3c")
    axes[0].set_title("RMSE")
    axes[0].set_xlabel("# input bands")
    axes[0].set_xscale("log")
    axes[1].plot(nb, scaling["sharpness"], "s-", color="#8e44ad")
    axes[1].set_title("Sharpness (mean σ̂)")
    axes[1].set_xlabel("# input bands")
    axes[1].set_xscale("log")
    axes[2].plot(nb, scaling["crps"], "^-", color="#27ae60")
    axes[2].set_title("CRPS")
    axes[2].set_xlabel("# input bands")
    axes[2].set_xscale("log")
    fig.suptitle(title, fontweight="bold")
    plt.tight_layout()
    path = out_dir / filename
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_calibration(out_dir: Path, by_sensor: dict, title: str, filename: str) -> Path:
    """Reliability diagram: empirical coverage vs nominal level for each sensor."""
    nominals = [0.68, 0.95, 0.997]
    keys = ["coverage_1sigma", "coverage_2sigma", "coverage_3sigma"]
    fig, ax = plt.subplots(figsize=(6, 6))
    for name, m in by_sensor.items():
        empirical = [m[k] for k in keys]
        ax.plot(nominals, empirical, "o-", label=name)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="ideal")
    ax.set_xlabel("Nominal coverage")
    ax.set_ylabel("Empirical coverage")
    ax.set_title(title, fontweight="bold")
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = out_dir / filename
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_confusion(out_dir: Path, material: dict, filename: str = "material_confusion.png") -> Path:
    cm = np.asarray(material["confusion_matrix"], dtype=np.float64)
    names = material["category_names"]
    cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticklabels(names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Material classification — row-normalised confusion", fontweight="bold")
    for i in range(len(names)):
        for j in range(len(names)):
            if cm[i, j] > 0:
                ax.text(j, i, f"{int(cm[i, j])}", ha="center", va="center",
                        color="white" if cm_norm[i, j] > 0.5 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    path = out_dir / filename
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def write_all(out_dir: Path, results: dict) -> dict[str, Path]:
    """Convenience: write JSON + markdown + all plots."""
    paths = {
        "json": save_results_json(out_dir, results),
        "summary": write_summary_md(out_dir, results),
    }
    p = plot_scaling(out_dir, results["radiance"]["scaling"],
                     "Radiance scaling with band count", "radiance_scaling.png")
    if p:
        paths["radiance_scaling"] = p
    paths["radiance_calibration"] = plot_calibration(
        out_dir, results["radiance"]["by_sensor"],
        "Radiance — coverage calibration", "radiance_calibration.png",
    )
    p = plot_scaling(out_dir, results["reflectance"]["scaling"],
                     "Reflectance scaling with band count", "reflectance_scaling.png")
    if p:
        paths["reflectance_scaling"] = p
    paths["reflectance_calibration"] = plot_calibration(
        out_dir, results["reflectance"]["by_sensor"],
        "Reflectance — coverage calibration", "reflectance_calibration.png",
    )
    paths["confusion"] = plot_confusion(out_dir, results["material"])
    return paths
