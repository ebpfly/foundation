"""Train a PCA-latent VAE on the USGS spectral library.

Usage:
    python -m spectralnp.training.train_pca_vae \
        --usgs-data /path/to/ASCIIdata_splib07a \
        --epochs 300 --n-pca 64 --z-dim 16

For thermal IR (6-15 um):
    python -m spectralnp.training.train_pca_vae \
        --usgs-data /path/to/ASCIIdata_splib07a \
        --spectrometer NIC4 \
        --wavelength-lo 6000 --wavelength-hi 15000 --wavelength-step 10 \
        --epochs 300 --n-pca 48 --z-dim 16 \
        --output-dir checkpoints/pca_vae_tir
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from spectralnp.data.usgs_speclib import SpectralLibrary, load_from_directory, load_from_zip
from spectralnp.model.pca_vae import PCAVAE, PCAVAEConfig, pca_vae_loss

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train PCA-latent VAE")
    p.add_argument("--usgs-data", type=str, required=True)
    p.add_argument("--output-dir", type=str, default="checkpoints/pca_vae")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--n-pca", type=int, default=64)
    p.add_argument("--z-dim", type=int, default=16)
    p.add_argument("--hidden-dims", type=int, nargs="+", default=[256, 128])
    p.add_argument("--beta", type=float, default=0.01)
    p.add_argument("--beta-warmup", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--spectrometer", type=str, default=None)
    p.add_argument("--wavelength-lo", type=float, default=350.0)
    p.add_argument("--wavelength-hi", type=float, default=2500.0)
    p.add_argument("--wavelength-step", type=float, default=1.0)
    p.add_argument("--augment", action="store_true")
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (train_spectra, val_spectra, wavelength_nm) as numpy arrays."""
    mat = speclib.to_array(wavelength_nm)
    mat = np.nan_to_num(mat, nan=0.0)
    mat = np.clip(mat, 0.0, 1.5).astype(np.float32)

    valid_mask = mat.max(axis=1) > 0.01
    mat = mat[valid_mask]
    logger.info(f"Valid spectra after filtering: {mat.shape[0]}")

    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(mat))
    n_val = max(1, int(len(mat) * val_fraction))
    val_data = mat[indices[:n_val]]
    train_data = mat[indices[n_val:]]

    if augment:
        n_aug = len(train_data)
        aug_scale = rng.uniform(0.8, 1.2, size=(n_aug, 1)).astype(np.float32)
        aug_shift = rng.uniform(-0.02, 0.02, size=(n_aug, 1)).astype(np.float32)
        aug_noise = rng.normal(0, 0.005, size=train_data.shape).astype(np.float32)
        augmented = np.clip(train_data * aug_scale + aug_shift + aug_noise, 0.0, 1.5)
        train_data = np.concatenate([train_data, augmented], axis=0)
        rng.shuffle(train_data)
        logger.info(f"Augmented training set: {len(train_data)} spectra")

    return train_data, val_data, wavelength_nm


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

    speclib = speclib.filter_wavelength_range(args.wavelength_lo + 30, args.wavelength_hi - 100)
    logger.info(f"After wavelength filter: {len(speclib)} spectra")

    if len(speclib) == 0:
        raise RuntimeError("No spectra passed filtering.")

    wl_grid = np.arange(args.wavelength_lo, args.wavelength_hi + args.wavelength_step,
                        args.wavelength_step, dtype=np.float32)

    train_spectra, val_spectra, wl_grid = prepare_data(
        speclib, wl_grid, args.val_fraction, args.seed, args.augment,
    )
    logger.info(f"Train: {len(train_spectra)}, Val: {len(val_spectra)}, Wavelengths: {len(wl_grid)}")

    # Build model and fit PCA on training data.
    cfg = PCAVAEConfig(
        n_pca=args.n_pca,
        z_dim=args.z_dim,
        hidden_dims=tuple(args.hidden_dims),
        beta=args.beta,
        dropout=0.1,
    )
    model = PCAVAE(cfg)

    # Fit PCA on training data only (no augmented copies to avoid bias).
    pca_info = model.fit_pca(train_spectra)
    ev = pca_info["explained_variance_ratio"]
    logger.info(
        f"PCA: {cfg.n_pca} components explain "
        f"{ev[-1]*100:.1f}% variance "
        f"(first 10: {ev[9]*100:.1f}%, first 30: {ev[min(29, len(ev)-1)]*100:.1f}%)"
    )

    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {n_params:,} (vs ~18M for conv VAE)")

    # Data loaders.
    train_ds = TensorDataset(torch.from_numpy(train_spectra))
    val_ds = TensorDataset(torch.from_numpy(val_spectra))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    # Optimiser.
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, args.epochs, eta_min=1e-6)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Training loop.
    best_val = float("inf")
    for epoch in range(args.epochs):
        beta_eff = args.beta * min(1.0, epoch / max(args.beta_warmup, 1))

        # Train.
        model.train()
        train_running: dict[str, float] = {}
        n_train = 0
        for (batch,) in train_loader:
            batch = batch.to(device)
            recon, mu, log_var, pca_recon = model(batch)
            pca_target = model.to_pca(batch)
            losses = pca_vae_loss(recon, batch, mu, log_var, pca_recon, pca_target, beta=beta_eff)
            optimiser.zero_grad()
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            for k, v in losses.items():
                train_running[k] = train_running.get(k, 0.0) + v.item()
            n_train += 1
        train_metrics = {k: v / n_train for k, v in train_running.items()}

        # Validate.
        model.eval()
        val_running: dict[str, float] = {}
        n_val = 0
        with torch.no_grad():
            for (batch,) in val_loader:
                batch = batch.to(device)
                recon, mu, log_var, pca_recon = model(batch)
                pca_target = model.to_pca(batch)
                losses = pca_vae_loss(recon, batch, mu, log_var, pca_recon, pca_target, beta=beta_eff)
                for k, v in losses.items():
                    val_running[k] = val_running.get(k, 0.0) + v.item()
                n_val += 1
        val_metrics = {k: v / n_val for k, v in val_running.items()}

        scheduler.step()
        lr = scheduler.get_last_lr()[0]

        logger.info(
            f"Epoch {epoch+1:3d}/{args.epochs}  "
            f"train={train_metrics['total']:.6f} "
            f"(recon={train_metrics['recon']:.6f} pca={train_metrics['pca_recon']:.6f} "
            f"kl={train_metrics['kl']:.3f})  "
            f"val={val_metrics['total']:.6f} "
            f"(recon={val_metrics['recon']:.6f} pca={val_metrics['pca_recon']:.6f} "
            f"kl={val_metrics['kl']:.3f})  "
            f"beta={beta_eff:.4f} lr={lr:.2e}"
        )

        if val_metrics["total"] < best_val:
            best_val = val_metrics["total"]
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "config": cfg,
                "wavelength_nm": wl_grid,
                "val_loss": best_val,
            }, out_dir / "best.pt")

        if (epoch + 1) % 50 == 0:
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "config": cfg,
                "wavelength_nm": wl_grid,
            }, out_dir / f"epoch_{epoch+1}.pt")

    # Fit latent prior from training data for realistic generation.
    model.eval()
    model.cpu()
    model.fit_latent_prior(train_spectra)
    model = model.to(device)
    logger.info("Fitted latent prior from training data")

    # Final save with samples.
    generated = model.generate(16, device=device).cpu()
    torch.save({
        "epoch": args.epochs,
        "model_state_dict": model.state_dict(),
        "config": cfg,
        "wavelength_nm": wl_grid,
        "val_loss": best_val,
        "generated_samples": generated,
    }, out_dir / "final.pt")

    logger.info(f"Training complete. Best val loss: {best_val:.6f}")
    logger.info(f"Checkpoints saved to {out_dir}")


if __name__ == "__main__":
    main()
