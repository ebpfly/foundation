"""Training script for the disentangled spectral foundation model.

The model uses a multi-pixel architecture internally — today it trains
with K=1 (one pixel per scene) by wrapping every dataset sample in a
K=1 dimension, but the architecture is already set up to handle
multi-pixel scenes without any model changes.

Usage:
    # Quick test with synthetic spectra (no external data):
    python -m spectralnp.training.train_foundation --epochs 10

    # Full training with USGS data:
    python -m spectralnp.training.train_foundation \
        --usgs-data /path/to/ASCIIdata_splib07a \
        --epochs 100 --samples-per-epoch 5000
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from spectralnp.data.dataset import SpectralNPDataset, collate_spectral_batch
from spectralnp.model.foundation import FoundationConfig, SpectralFoundation


def collect_pca_spectra(
    dataset: SpectralNPDataset,
    n_samples: int = 5000,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate samples from the dataset to fit PCA bases.

    Returns (radiance, reflectance) arrays of shape (n_samples, n_wavelengths).
    """
    loader = DataLoader(
        dataset,
        batch_size=min(128, n_samples),
        collate_fn=collate_spectral_batch,
        num_workers=0,
    )
    rad_list, refl_list = [], []
    n = 0
    for batch in loader:
        rad_list.append(batch["target_radiance"].numpy())
        refl_list.append(batch["target_reflectance"].numpy())
        n += batch["target_radiance"].shape[0]
        if n >= n_samples:
            break
    return (
        np.concatenate(rad_list)[:n_samples],
        np.concatenate(refl_list)[:n_samples],
    )


def train(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Spectral library ----
    if args.usgs_data:
        from spectralnp.data.usgs_speclib import load_from_directory
        speclib = load_from_directory(args.usgs_data)
        print(f"Loaded {len(speclib)} USGS spectra")
    else:
        from spectralnp.data.synthetic_speclib import generate_synthetic_library
        wl_grid = np.arange(380.0, 2501.0, 5.0)
        speclib = generate_synthetic_library(
            n_per_class=60, wavelength_nm=wl_grid, seed=args.seed,
        )
        print(f"Generated {len(speclib)} synthetic spectra")

    # ---- Dataset ----
    dense_wl = np.arange(380.0, 2501.0, 5.0)
    dataset = SpectralNPDataset(
        spectral_library=speclib,
        dense_wavelength_nm=dense_wl,
        samples_per_epoch=args.samples_per_epoch,
        n_bands_range=(3, 200),
        seed=args.seed,
    )
    n_material_classes = dataset.n_material_classes

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=collate_spectral_batch,
        num_workers=0,
        drop_last=True,
    )

    # ---- Fit PCA ----
    print(f"Collecting {args.n_pca_samples} spectra for PCA fitting...")
    rad_spectra, refl_spectra = collect_pca_spectra(dataset, args.n_pca_samples)

    config = FoundationConfig(
        n_pca_radiance=args.n_pca,
        n_pca_reflectance=args.n_pca,
        z_atm_dim=args.z_atm_dim,
        z_surf_dim=args.z_surf_dim,
        feature_dim=args.feature_dim,
        n_material_classes=n_material_classes,
        dropout=args.dropout,
        w_refl=args.w_refl,
        w_atmos=args.w_atmos,
        w_temp=args.w_temp,
        w_material=args.w_material,
        w_physics=args.w_physics,
        beta_atm=args.beta_atm,
        beta_surf=args.beta_surf,
    )

    model = SpectralFoundation(config).to(device)
    diag = model.fit_pca(rad_spectra, refl_spectra, dense_wl)
    ev_rad = float(diag["radiance_explained_var"])
    ev_refl = float(diag["reflectance_explained_var"])
    print(
        f"Radiance    PCA: {ev_rad * 100:.4f}% variance explained "
        f"({model.config.n_pca_radiance} components)"
    )
    print(
        f"Reflectance PCA: {ev_refl * 100:.4f}% variance explained "
        f"({model.config.n_pca_reflectance} components)"
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")
    print(f"Material classes: {n_material_classes}")

    # ---- Optimiser ----
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)

    if args.lr_constant:
        scheduler = None
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimiser, T_max=args.epochs, eta_min=args.lr * 0.01,
        )

    # ---- W&B ----
    if args.wandb:
        import wandb
        wandb.init(project="spectral-foundation", config=vars(args))
        wandb.watch(model, log_freq=100)

    # ---- Training loop ----
    best_loss = float("inf")
    log_path = output_dir / "training.log"

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses = {
            k: 0.0 for k in
            ["total", "reflectance", "physics", "atmosphere",
             "temperature", "material", "kl_atm", "kl_surf"]
        }
        n_batches = 0
        t0 = time.time()

        # KL annealing: linearly ramp beta_atm and beta_surf
        if args.beta_warmup > 0 and epoch <= args.beta_warmup:
            warm = epoch / args.beta_warmup
            model.config.beta_atm = args.beta_atm * warm
            model.config.beta_surf = args.beta_surf * warm
        else:
            model.config.beta_atm = args.beta_atm
            model.config.beta_surf = args.beta_surf

        for batch in loader:
            wl = batch["wavelength"].to(device)
            fw = batch["fwhm"].to(device)
            rad = batch["radiance"].to(device)
            mask = batch["pad_mask"].to(device)
            target_refl = batch["target_reflectance"].to(device)
            target_rad = batch["target_radiance"].to(device)
            target_atmos = batch["atmos_params"].to(device)
            target_temp_k = batch["surface_temperature_k"].to(device)
            target_material = batch["material_idx"].to(device)

            # Normalise temperature: (T - 290) / 40 → roughly [-1, 1]
            target_temp_norm = ((target_temp_k - 290.0) / 40.0).unsqueeze(-1)

            output = model(wl, fw, rad, mask)
            losses = model.loss(
                output,
                target_reflectance=target_refl,
                target_radiance=target_rad,
                target_atmos=target_atmos,
                target_temperature=target_temp_norm,
                target_material=target_material,
            )

            optimiser.zero_grad()
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()

            for k in epoch_losses:
                epoch_losses[k] += losses[k].item()
            n_batches += 1

        if scheduler is not None:
            scheduler.step()

        # ---- Logging ----
        dt = time.time() - t0
        avg = {k: v / max(n_batches, 1) for k, v in epoch_losses.items()}
        lr = optimiser.param_groups[0]["lr"]

        line = (
            f"[{epoch:4d}/{args.epochs}]  "
            f"tot={avg['total']:7.3f}  "
            f"refl={avg['reflectance']:.4f}  "
            f"phys={avg['physics']:7.3f}  "
            f"atm={avg['atmosphere']:.3f}  "
            f"temp={avg['temperature']:.3f}  "
            f"mat={avg['material']:.3f}  "
            f"klA={avg['kl_atm']:.2f}  "
            f"klS={avg['kl_surf']:.2f}  "
            f"β_a={model.config.beta_atm:.3f}  "
            f"lr={lr:.2e}  "
            f"{dt:.1f}s"
        )
        print(line)
        with open(log_path, "a") as f:
            f.write(line + "\n")

        if args.wandb:
            import wandb
            wandb.log({f"train/{k}": v for k, v in avg.items()}, step=epoch)
            wandb.log({"train/lr": lr, "train/beta_atm": model.config.beta_atm},
                       step=epoch)

        # ---- Checkpointing ----
        if avg["total"] < best_loss:
            best_loss = avg["total"]
            _save_checkpoint(model, config, dense_wl, optimiser, epoch,
                             output_dir / "best.pt")

        if epoch % max(args.epochs // 5, 1) == 0 or epoch == args.epochs:
            _save_checkpoint(
                model, config, dense_wl, optimiser, epoch,
                output_dir / f"epoch_{epoch:04d}.pt",
            )

    _save_checkpoint(model, config, dense_wl, optimiser, args.epochs,
                     output_dir / "final.pt")
    print(f"\nTraining complete. Checkpoints saved to {output_dir}")


def _save_checkpoint(
    model: SpectralFoundation,
    config: FoundationConfig,
    wavelength_nm: np.ndarray,
    optimiser: torch.optim.Optimizer,
    epoch: int,
    path: Path,
) -> None:
    torch.save(
        {
            "config": config,
            "model_state_dict": model.state_dict(),
            "optimiser_state_dict": optimiser.state_dict(),
            "epoch": epoch,
            "wavelength_nm": wavelength_nm,
        },
        path,
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Train disentangled spectral foundation")

    # Data
    p.add_argument("--usgs-data", type=str, default=None)
    p.add_argument("--samples-per-epoch", type=int, default=5000)
    p.add_argument("--n-pca-samples", type=int, default=5000)

    # Model sizes
    p.add_argument("--n-pca", type=int, default=0,
                    help="PCA components (0 = full rank, recommended)")
    p.add_argument("--z-atm-dim", type=int, default=32)
    p.add_argument("--z-surf-dim", type=int, default=96)
    p.add_argument("--feature-dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.1)

    # Loss weights
    p.add_argument("--w-refl", type=float, default=1.0)
    p.add_argument("--w-atmos", type=float, default=1.0)
    p.add_argument("--w-temp", type=float, default=1.0)
    p.add_argument("--w-material", type=float, default=0.1)
    p.add_argument("--w-physics", type=float, default=0.5)
    p.add_argument("--beta-atm", type=float, default=0.01)
    p.add_argument("--beta-surf", type=float, default=0.01)
    p.add_argument("--beta-warmup", type=int, default=10)

    # Training
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--lr-constant", action="store_true")

    # Infrastructure
    p.add_argument("--output-dir", type=str, default="checkpoints/foundation")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--wandb", action="store_true")

    args = p.parse_args()

    if args.device == "cpu":
        if torch.cuda.is_available():
            args.device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            args.device = "mps"

    print(f"Device: {args.device}")
    print(
        f"Config: z_atm={args.z_atm_dim}, z_surf={args.z_surf_dim}, "
        f"feat={args.feature_dim}, β_atm={args.beta_atm}"
    )

    train(args)


if __name__ == "__main__":
    main()
