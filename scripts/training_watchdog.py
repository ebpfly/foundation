#!/usr/bin/env python3
"""Watchdog for SpectralNP training runs.

Parses the training log emitted by ``pretrain.py`` and reports a verdict:
  - ``ok``           — training is on track
  - ``warming``      — early epochs, can't judge yet
  - ``stalled``      — loss not decreasing for a while (after warmup)
  - ``diverging``    — loss going up after warmup
  - ``kl_collapse``  — KL term blew up (>1e6)
  - ``nan``          — any metric is NaN/Inf
  - ``crashed``      — training process gone or log file empty

Usage:
    python scripts/training_watchdog.py /tmp/training_v3.txt
    python scripts/training_watchdog.py /tmp/training_v3.txt --pid 12345

Exit codes:
    0  ok / warming
    1  stalled
    2  diverging
    3  kl_collapse
    4  nan
    5  crashed
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

EPOCH_RE = re.compile(
    r"Epoch (\d+)/(\d+)\s+"
    r"loss=(?P<loss>[-\d.eE+inf nan]+)\s+"
    r"spectral=(?P<spectral>[-\d.eE+inf nan]+)\s+"
    r"(?:reflectance=(?P<reflectance>[-\d.eE+inf nan]+)\s+)?"
    r"atmos=(?P<atmos>[-\d.eE+inf nan]+)\s+"
    r"kl=(?P<kl>[-\d.eE+inf nan]+)\s+"
    r"lr=(?P<lr>[-\d.eE+]+)"
)


@dataclass
class EpochMetrics:
    epoch: int
    total_epochs: int
    loss: float
    spectral: float
    reflectance: float
    atmos: float
    kl: float
    lr: float


def parse_log(log_path: Path) -> list[EpochMetrics]:
    metrics: list[EpochMetrics] = []
    if not log_path.exists():
        return metrics
    text = log_path.read_text()
    for line in text.splitlines():
        m = EPOCH_RE.search(line)
        if not m:
            continue
        try:
            metrics.append(EpochMetrics(
                epoch=int(m.group(1)),
                total_epochs=int(m.group(2)),
                loss=float(m.group("loss")),
                spectral=float(m.group("spectral")),
                reflectance=float(m.group("reflectance") or 0.0),
                atmos=float(m.group("atmos")),
                kl=float(m.group("kl")),
                lr=float(m.group("lr")),
            ))
        except (TypeError, ValueError):
            continue
    return metrics


def is_finite(x: float) -> bool:
    return math.isfinite(x)


def diagnose(
    metrics: list[EpochMetrics],
    pid: int | None = None,
    warmup_epochs: int = 5,
    stall_window: int = 10,
    kl_collapse_threshold: float = 1e6,
) -> tuple[str, str]:
    """Return (verdict, human-readable message)."""
    pid_alive = True
    if pid is not None:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            pid_alive = False
        except PermissionError:
            pass  # process exists but not ours

    if not metrics:
        # If the process is still running, just early — "warming".
        # If it's gone, then it crashed before producing any output.
        if pid_alive:
            return ("warming", "no epoch lines yet (cache populate / first epoch)")
        return ("crashed", "log file empty or no epoch lines parsed")
    if pid is not None and not pid_alive:
        return ("crashed", f"PID {pid} no longer running")

    last = metrics[-1]

    # NaN / Inf check
    for field in ("loss", "spectral", "reflectance", "atmos", "kl"):
        v = getattr(last, field)
        if not is_finite(v):
            return ("nan", f"epoch {last.epoch} {field}={v}")

    # KL collapse / explosion
    if abs(last.kl) > kl_collapse_threshold:
        return ("kl_collapse",
                f"epoch {last.epoch} kl={last.kl:.4g} > {kl_collapse_threshold:.0g}")

    # Still in warmup window — be patient
    if last.epoch <= warmup_epochs:
        return ("warming",
                f"epoch {last.epoch}/{last.total_epochs} (warmup)")

    # Look at the post-warmup trend on spectral loss + total loss.
    post_warmup = [m for m in metrics if m.epoch > warmup_epochs]
    if len(post_warmup) < 2:
        return ("warming", f"epoch {last.epoch}/{last.total_epochs} (just past warmup)")

    # Check NaN in post-warmup
    for m in post_warmup:
        if not all(is_finite(getattr(m, f)) for f in ("loss", "spectral", "kl")):
            return ("nan", f"epoch {m.epoch} contains NaN/Inf")

    # Diverging: spectral OR reflectance loss grew by ≥3× from post-warmup minimum.
    spectral_history = [m.spectral for m in post_warmup]
    min_spectral = min(spectral_history)
    if min_spectral > 0 and last.spectral > min_spectral * 3.0:
        return ("diverging",
                f"epoch {last.epoch} spectral={last.spectral:.4g} >> "
                f"min={min_spectral:.4g} (×{last.spectral/min_spectral:.1f})")

    # Reflectance NLL can be negative (well-calibrated Gaussian), so use
    # the absolute swing relative to min: if it grew by 5+ from its min
    # AND is now positive, that's a divergence.
    refl_history = [m.reflectance for m in post_warmup]
    min_refl = min(refl_history)
    swing = last.reflectance - min_refl
    if swing > 5.0 and last.reflectance > 0.5:
        return ("diverging",
                f"epoch {last.epoch} reflectance={last.reflectance:.4g} "
                f"(min was {min_refl:.4g}, swing +{swing:.2f})")

    # Stalled: spectral loss hasn't improved in the last `stall_window` epochs
    if len(post_warmup) >= stall_window:
        recent = post_warmup[-stall_window:]
        recent_min = min(m.spectral for m in recent)
        baseline = post_warmup[-stall_window].spectral
        improvement = (baseline - recent_min) / max(abs(baseline), 1e-12)
        if improvement < 0.005:  # less than 0.5% improvement over the window
            return ("stalled",
                    f"epoch {last.epoch} spectral stuck near {recent_min:.4g} "
                    f"(no improvement over last {stall_window} epochs)")

    return ("ok",
            f"epoch {last.epoch}/{last.total_epochs} loss={last.loss:.4f} "
            f"spectral={last.spectral:.4f} kl={last.kl:.4g} lr={last.lr:.2e}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("log_path", type=Path)
    p.add_argument("--pid", type=int, default=None,
                   help="Training PID to check liveness")
    p.add_argument("--warmup-epochs", type=int, default=5)
    p.add_argument("--stall-window", type=int, default=10)
    args = p.parse_args()

    metrics = parse_log(args.log_path)
    verdict, msg = diagnose(
        metrics, pid=args.pid,
        warmup_epochs=args.warmup_epochs,
        stall_window=args.stall_window,
    )
    print(f"{verdict}: {msg}")
    code = {
        "ok": 0, "warming": 0,
        "stalled": 1, "diverging": 2,
        "kl_collapse": 3, "nan": 4, "crashed": 5,
    }.get(verdict, 99)
    sys.exit(code)


if __name__ == "__main__":
    main()
