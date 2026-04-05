"""Train a spectral VAE on the USGS spectral library.

Usage:
    python -m spectralnp.training.train_vae \
        --usgs-data data/ASCIIdata_splib07a.zip \
        --epochs 200 --batch-size 128 --lr 1e-3

The trained model can generate novel, physically plausible reflectance
spectra by sampling from the learned latent space.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from spectralnp.data.usgs_speclib import SpectralLibrary, load_from_directory, load_from_zip
from spectralnp.model.spectral_vae import SpectralVAE, SpectralVAEConfig, vae_loss

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train Spectral VAE")
    p.add_argument("--usgs-data", type=str, required=True,
                    help="Path to USGS spectral library (directory or .zip)")
    p.add_argument("--output-dir", type=str, default="checkpoints/vae")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--z-dim", type=int, default=32)
    p.add_argument("--base-channels", type=int, default=64)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--beta", type=float, default=1.0,
                    help="KL weight (beta-VAE). <1 for sharper reconstructions, >1 for more disentangled latent.")
    p.add_argument("--beta-warmup", type=int, default=20,
                    help="Linearly anneal beta from 0 to target over this many epochs")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--spectrometer", type=str, default=None,
                    help="Filter to a specific spectrometer (ASD, BECK, NIC4, AVIRIS)")
    p.add_argument("--wavelength-lo", type=float, default=350.0,
                    help="Lower wavelength bound (nm)")
    p.add_argument("--wavelength-hi", type=float, default=2500.0,
                    help="Upper wavelength bound (nm)")
    p.add_argument("--wavelength-step", type=float, default=1.0,
                    help="Wavelength grid step (nm)")
    p.add_argument("--augment", action="store_true",
                    help="Enable spectral augmentation (scale, shift, noise)")
    return p


def get_device(arg: str) -> torch.device:
    if arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(arg)


def prepare_data(
    speclib: SpectralLibrary,
    wavelength_nm: np.ndarray,
    val_fraction: float = 0.1,
    seed: int = 42,
    augment: bool = False,
) -> tuple[TensorDataset, TensorDataset, np.ndarray]:
    """Build train/val datasets from the spectral library.

    Returns train_dataset, val_dataset, and the wavelength grid used.
    """
    mat = speclib.to_array(wavelength_nm)  # (N, W)

    # Replace NaN and clip to physical range.
    mat = np.nan_to_num(mat, nan=0.0)
    mat = np.clip(mat, 0.0, 1.5).astype(np.float32)

    # Drop spectra that are all-zero (out of range).
    valid_mask = mat.max(axis=1) > 0.01
    mat = mat[valid_mask]
    logger.info(f"Valid spectra after filtering: {mat.shape[0]}")

    # Shuffle and split.
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(mat))
    n_val = max(1, int(len(mat) * val_fraction))
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    train_data = mat[train_idx]
    val_data = mat[val_idx]

    if augment:
        # Simple augmentations: random scale + shift + noise
        n_aug = len(train_data)
        aug_scale = rng.uniform(0.8, 1.2, size=(n_aug, 1)).astype(np.float32)
        aug_shift = rng.uniform(-0.02, 0.02, size=(n_aug, 1)).astype(np.float32)
        aug_noise = rng.normal(0, 0.005, size=train_data.shape).astype(np.float32)
        augmented = np.clip(train_data * aug_scale + aug_shift + aug_noise, 0.0, 1.5)
        train_data = np.concatenate([train_data, augmented], axis=0)
        rng.shuffle(train_data)
        logger.info(f"Augmented training set: {len(train_data)} spectra")

    train_ds = TensorDataset(torch.from_numpy(train_data))
    val_ds = TensorDataset(torch.from_numpy(val_data))

    return train_ds, val_ds, wavelength_nm


def train_one_epoch(
    model: SpectralVAE,
    loader: DataLoader,
    optimiser: torch.optim.Optimizer,
    device: torch.device,
    beta: float,
) -> dict[str, float]:
    model.train()
    running = {}
    n = 0
    for (batch,) in loader:
        batch = batch.to(device)
        recon, mu, log_var = model(batch)
        losses = vae_loss(recon, batch, mu, log_var, beta=beta)
        optimiser.zero_grad()
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()
        for k, v in losses.items():
            running[k] = running.get(k, 0.0) + v.item()
        n += 1
    return {k: v / n for k, v in running.items()}


@torch.no_grad()
def validate(
    model: SpectralVAE,
    loader: DataLoader,
    device: torch.device,
    beta: float,
) -> dict[str, float]:
    model.eval()
    running = {}
    n = 0
    for (batch,) in loader:
        batch = batch.to(device)
        recon, mu, log_var = model(batch)
        losses = vae_loss(recon, batch, mu, log_var, beta=beta)
        for k, v in losses.items():
            running[k] = running.get(k, 0.0) + v.item()
        n += 1
    return {k: v / n for k, v in running.items()}


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = get_device(args.device)
    logger.info(f"Using device: {device}")

    # Load spectral library.
    usgs_path = Path(args.usgs_data)
    if usgs_path.suffix == ".zip":
        speclib = load_from_zip(usgs_path)
    else:
        speclib = load_from_directory(usgs_path)
    logger.info(f"Loaded {len(speclib)} spectra")

    if args.spectrometer:
        speclib = speclib.filter_spectrometer(args.spectrometer)
        logger.info(f"Filtered to {args.spectrometer}: {len(speclib)} spectra")

    # Filter to spectra with coverage across the wavelength range.
    speclib = speclib.filter_wavelength_range(args.wavelength_lo + 30, args.wavelength_hi - 100)
    logger.info(f"After wavelength filter: {len(speclib)} spectra")

    if len(speclib) == 0:
        raise RuntimeError("No spectra passed filtering. Check wavelength range or spectrometer.")

    # Build wavelength grid.
    wl_grid = np.arange(args.wavelength_lo, args.wavelength_hi + args.wavelength_step,
                        args.wavelength_step, dtype=np.float32)

    # Prepare data.
    train_ds, val_ds, wl_grid = prepare_data(
        speclib, wl_grid, args.val_fraction, args.seed, args.augment,
    )
    logger.info(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Wavelengths: {len(wl_grid)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    # Model.
    cfg = SpectralVAEConfig(
        n_wavelengths=len(wl_grid),
        z_dim=args.z_dim,
        base_channels=args.base_channels,
        n_layers=args.n_layers,
        beta=args.beta,
    )
    model = SpectralVAE(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {n_params:,}")

    # Optimiser.
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, args.epochs, eta_min=1e-6)

    # Output.
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Training loop.
    best_val = float("inf")
    for epoch in range(args.epochs):
        # Beta warmup.
        beta_eff = args.beta * min(1.0, epoch / max(args.beta_warmup, 1))

        train_metrics = train_one_epoch(model, train_loader, optimiser, device, beta_eff)
        val_metrics = validate(model, val_loader, device, beta_eff)
        scheduler.step()

        lr = scheduler.get_last_lr()[0]
        logger.info(
            f"Epoch {epoch+1:3d}/{args.epochs}  "
            f"train_loss={train_metrics['total']:.5f} "
            f"(recon={train_metrics['recon']:.5f} kl={train_metrics['kl']:.3f})  "
            f"val_loss={val_metrics['total']:.5f} "
            f"(recon={val_metrics['recon']:.5f} kl={val_metrics['kl']:.3f})  "
            f"beta={beta_eff:.3f} lr={lr:.2e}"
        )

        # Save best.
        if val_metrics["total"] < best_val:
            best_val = val_metrics["total"]
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimiser_state_dict": optimiser.state_dict(),
                "config": cfg,
                "wavelength_nm": wl_grid,
                "val_loss": best_val,
            }, out_dir / "best_vae.pt")

        if (epoch + 1) % 50 == 0:
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "config": cfg,
                "wavelength_nm": wl_grid,
            }, out_dir / f"vae_epoch_{epoch+1}.pt")

    # Final save with generation samples.
    model.eval()
    generated = model.generate(16, device=device).cpu()
    torch.save({
        "epoch": args.epochs,
        "model_state_dict": model.state_dict(),
        "config": cfg,
        "wavelength_nm": wl_grid,
        "val_loss": best_val,
        "generated_samples": generated,
    }, out_dir / "final_vae.pt")

    logger.info(f"Training complete. Best val loss: {best_val:.5f}")
    logger.info(f"Checkpoints saved to {out_dir}")


if __name__ == "__main__":
    main()
