"""Phase 1: Self-supervised spectral pretraining.

Trains SpectralNP on simulated at-sensor radiance from the USGS spectral
library + radiative transfer.  The model learns to reconstruct dense
spectra from sparse, randomly-sampled band subsets while simultaneously
estimating atmospheric parameters.

Usage:
    python -m spectralnp.training.pretrain \
        --usgs-data /path/to/usgs_splib07 \
        --epochs 100 \
        --batch-size 64 \
        --lr 3e-4
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from spectralnp.data.dataset import SpectralNPDataset, collate_spectral_batch
from spectralnp.data.usgs_speclib import SpectralLibrary, load_from_directory, load_from_zip
from spectralnp.model.spectralnp import SpectralNP, SpectralNPConfig
from spectralnp.training.losses import SpectralNPLoss

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Pretrain SpectralNP")
    p.add_argument("--usgs-data", type=str, required=True,
                    help="Path to USGS spectral library (directory or .zip)")
    p.add_argument("--output-dir", type=str, default="checkpoints",
                    help="Where to save checkpoints")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--samples-per-epoch", type=int, default=100_000)
    p.add_argument("--min-bands", type=int, default=3)
    p.add_argument("--max-bands", type=int, default=200)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--n-layers", type=int, default=6)
    p.add_argument("--z-dim", type=int, default=128)
    p.add_argument("--spectral-hidden", type=int, default=512)
    p.add_argument("--spectral-n-layers", type=int, default=4)
    p.add_argument("--n-frequencies", type=int, default=64,
                    help="Number of sinusoidal frequencies in the spectral "
                         "positional encoding. Default 64 resolves ~33nm. "
                         "256 resolves ~4nm (matches 5nm dense grid).")
    p.add_argument("--no-r-in-decoder", action="store_true",
                    help="Drop the deterministic representation r from the "
                         "spectral decoder input. Forces the decoder to depend "
                         "on the latent z, which can mitigate posterior collapse "
                         "where r dominates and z is ignored.")
    p.add_argument("--z-atm-dim", type=int, default=None,
                    help="Override z_atm_dim (default 32)")
    p.add_argument("--z-surf-dim", type=int, default=None,
                    help="Override z_surf_dim (default 96)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", type=str, default=None,
                    help="Path to checkpoint to resume from. Loads model weights "
                         "and optimizer state. Architecture flags must match.")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--wandb", action="store_true", help="Log to Weights & Biases")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--lut-path", type=str, default=None,
                    help="Path to pre-computed HDF5 layer-optical-depth LUT "
                         "(from build_lut.py). Enables fast LUT-based RTM.")
    p.add_argument("--abs-lookup", type=str, default=None,
                    help="Path to ARTS abs_lookup XML file (from atmgen). "
                         "Enables real ARTS line-by-line gas absorption with "
                         "per-atmosphere caching. Forces num-workers=0.")
    p.add_argument("--n-atmospheres-per-epoch", type=int, default=10,
                    help="When --abs-lookup is set: number of fresh random "
                         "atmospheres computed at the start of each epoch. "
                         "Each costs ~3 sec of ARTS work.")
    p.add_argument("--n-scenes-per-epoch", type=int, default=50,
                    help="When --abs-lookup is set: number of fresh "
                         "(atmosphere, geometry, aerosol) scenes computed "
                         "at the start of each epoch. Each is ~3 sec ARTS + "
                         "a fast path-integration. Per-sample work then "
                         "drops to ~1 ms (just the surface coupling).")
    # Loss weights (exposed for ablations and debugging KL collapse)
    p.add_argument("--w-spectral", type=float, default=1.0)
    p.add_argument("--w-reflectance", type=float, default=1.0)
    p.add_argument("--w-atmos", type=float, default=0.1)
    p.add_argument("--w-kl", type=float, default=0.001,
                    help="KL weight (default 0.001 — was 0.01 which caused "
                         "collapse on real ARTS data; lower is safer).")
    p.add_argument("--w-material", type=float, default=0.1)
    p.add_argument("--w-calibration", type=float, default=0.0,
                    help="Weight on the calibration regulariser that pulls "
                         "predicted variance toward empirical squared error. "
                         "Set >0 to fix overconfident uncertainty.")
    p.add_argument("--feature-weight", type=float, default=0.0,
                    help="Feature-weighted loss strength. When >0, wavelengths "
                         "with steep spectral gradients (absorption bands, "
                         "mineral features) get upweighted by up to 1+this.")
    p.add_argument("--pca-vae", type=str, default=None,
                    help="Path to trained PCA-VAE checkpoint (final.pt). "
                         "When set, 50%% of training samples use novel "
                         "generated spectra instead of USGS (anti-memorisation).")
    p.add_argument("--no-class-balance", action="store_true",
                    help="Disable class-balanced material loss "
                         "(default: enabled when --abs-lookup is used).")
    p.add_argument("--wl-min-nm", type=float, default=None,
                    help="Min wavelength of the dense reconstruction grid (nm). "
                         "Default: 380 nm (or 300 nm if --abs-lookup is set).")
    p.add_argument("--wl-max-nm", type=float, default=None,
                    help="Max wavelength of the dense reconstruction grid (nm). "
                         "Default: 2500 nm (or 16000 nm if --abs-lookup is set).")
    p.add_argument("--dense-n-points", type=int, default=None,
                    help="Use a UNIFORM dense grid with this many points "
                         "instead of the variable-resolution grid. Halves "
                         "model output cost vs the variable grid.")
    # KL annealing.
    p.add_argument("--kl-warmup-epochs", type=int, default=10,
                    help="Linearly anneal KL weight from 0 to target over this many epochs")
    p.add_argument("--lr-constant", action="store_true",
                    help="Use constant LR (no cosine decay) — useful for short iteration runs")
    return p


def get_device(arg: str) -> torch.device:
    if arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(arg)


def train_one_epoch(
    model: SpectralNP,
    loader: DataLoader,
    loss_fn: SpectralNPLoss,
    optimiser: torch.optim.Optimizer,
    device: torch.device,
    kl_weight: float = 1.0,
) -> dict[str, float]:
    """Train for one epoch, return mean losses."""
    model.train()
    running = {}
    n_batches = 0

    for batch in loader:
        # Move to device.
        wavelength = batch["wavelength"].to(device)
        fwhm = batch["fwhm"].to(device)
        radiance = batch["radiance"].to(device)
        pad_mask = batch["pad_mask"].to(device)
        target_wl = batch["target_wavelength"].to(device)
        target_rad = batch["target_radiance"].to(device)
        target_refl = batch["target_reflectance"].to(device)
        target_atmos = batch["atmos_params"].to(device)
        material_idx = batch["material_idx"].to(device)

        # Forward pass.
        output = model(
            wavelength=wavelength,
            fwhm=fwhm,
            radiance=radiance,
            pad_mask=pad_mask,
            query_wavelength=target_wl,
        )

        # Also compute "target" posterior (all bands) for NP KL.
        # In pretraining, we pass the dense spectrum through a second
        # encode call to get the full-information posterior.
        with torch.no_grad():
            # Approximate: use target radiance at dense wavelengths as "all bands".
            # For efficiency, subsample to ~50 query points.
            n_dense = target_wl.shape[1]
            subsample = torch.randperm(n_dense, device=device)[:50].sort().values
            dense_wl_sub = target_wl[:, subsample]
            dense_fwhm_sub = torch.ones_like(dense_wl_sub) * 5.0  # narrow bands
            dense_rad_sub = target_rad[:, subsample]
            _, _, _, _, _, prior_mu, prior_log_sigma = model.encode(
                dense_wl_sub, dense_fwhm_sub, dense_rad_sub
            )

        # Override KL weight for annealing.
        original_kl = loss_fn.w_kl
        loss_fn.w_kl = original_kl * kl_weight

        losses = loss_fn(
            output,
            target_radiance=target_rad,
            target_reflectance=target_refl,
            target_atmos=target_atmos,
            target_material=material_idx,
            prior_mu=prior_mu,
            prior_log_sigma=prior_log_sigma,
        )

        loss_fn.w_kl = original_kl

        # Backward.
        optimiser.zero_grad()
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()

        # Accumulate.
        for k, v in losses.items():
            running[k] = running.get(k, 0.0) + v.item()
        n_batches += 1

    return {k: v / n_batches for k, v in running.items()}


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
    # Filter to the ASD range with full coverage.
    speclib = speclib.filter_wavelength_range(380, 2400)
    logger.info(f"Loaded {len(speclib)} spectra from USGS library")

    if len(speclib) == 0:
        raise RuntimeError("No spectra with full 380-2400 nm coverage found. "
                           "Check your USGS data path.")

    # Optional ARTS abs_lookup simulator.
    arts_sim = None
    if args.abs_lookup is not None:
        from spectralnp.data.rtm_simulator import ARTSLookupSimulator
        logger.info(f"Loading ARTS abs_lookup from {args.abs_lookup}")
        arts_sim = ARTSLookupSimulator(args.abs_lookup)
        # Initial cache population happens at start of epoch 0 below.
        # pyarts is not fork-safe; force single-process data loading.
        if args.num_workers > 0:
            logger.info("Forcing --num-workers=0 because abs_lookup is in use")
            args.num_workers = 0

    # Dense reconstruction grid. Default to VNIR/SWIR; if abs_lookup is set,
    # default to the full 0.3–16 μm variable-resolution grid from lut.py.
    if args.abs_lookup is not None:
        from spectralnp.data.lut import make_lut_wavelength_grid
        wl_min = args.wl_min_nm if args.wl_min_nm is not None else 300.0
        wl_max = args.wl_max_nm if args.wl_max_nm is not None else 16000.0
        if args.dense_n_points is not None:
            dense_wl = np.linspace(wl_min, wl_max, args.dense_n_points)
            logger.info(
                f"Uniform dense grid: {wl_min:.0f}–{wl_max:.0f} nm, "
                f"{len(dense_wl)} points (~{(wl_max-wl_min)/args.dense_n_points:.1f} nm step)"
            )
        else:
            dense_wl = make_lut_wavelength_grid(wl_min=wl_min, wl_max=wl_max)
            logger.info(
                f"Full-spectrum dense grid: {wl_min:.0f}–{wl_max:.0f} nm, "
                f"{len(dense_wl)} points (variable resolution)"
            )
    else:
        wl_min = args.wl_min_nm if args.wl_min_nm is not None else 380.0
        wl_max = args.wl_max_nm if args.wl_max_nm is not None else 2500.0
        if args.dense_n_points is not None:
            dense_wl = np.linspace(wl_min, wl_max, args.dense_n_points)
        else:
            dense_wl = np.arange(wl_min, wl_max + 1.0, 5.0)
        logger.info(f"Dense grid: {wl_min:.0f}–{wl_max:.0f} nm, {len(dense_wl)} points")

    # Dataset.
    dataset = SpectralNPDataset(
        spectral_library=speclib,
        dense_wavelength_nm=dense_wl,
        samples_per_epoch=args.samples_per_epoch,
        n_bands_range=(args.min_bands, args.max_bands),
        lut_path=args.lut_path,
        arts_simulator=arts_sim,
        pca_vae_path=args.pca_vae,
        seed=args.seed,
    )
    if args.pca_vae:
        logger.info(f"PCA-VAE generator loaded from {args.pca_vae} "
                     "(50% of samples will be novel generated spectra)")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=collate_spectral_batch,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Model.
    cfg_kwargs = dict(
        d_model=args.d_model,
        n_layers=args.n_layers,
        z_dim=args.z_dim,
        n_frequencies=args.n_frequencies,
        spectral_hidden=args.spectral_hidden,
        spectral_n_layers=args.spectral_n_layers,
        spectral_decoder_use_r=not args.no_r_in_decoder,
        n_material_classes=dataset.n_material_classes,
    )
    if args.z_atm_dim is not None:
        cfg_kwargs["z_atm_dim"] = args.z_atm_dim
    if args.z_surf_dim is not None:
        cfg_kwargs["z_surf_dim"] = args.z_surf_dim
    cfg = SpectralNPConfig(**cfg_kwargs)
    model = SpectralNP(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {n_params:,}")

    # Optimiser.
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    if args.lr_constant:
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser, lr_lambda=lambda _: 1.0)
        logger.info("Using constant LR (no cosine decay).")
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, args.epochs)

    # Resume from checkpoint.
    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        opt_key = "optimiser_state_dict" if "optimiser_state_dict" in ckpt else "optimizer_state_dict"
        if opt_key in ckpt:
            optimiser.load_state_dict(ckpt[opt_key])
        if "epoch" in ckpt:
            start_epoch = ckpt["epoch"]
        logger.info(f"Resumed from {args.resume} (epoch {start_epoch})")

    # Loss.
    # Class-balanced material loss: weights inversely proportional to category size.
    class_weights = None
    if not args.no_class_balance:
        counts = np.bincount(
            dataset.category_id_by_spec, minlength=dataset.n_material_classes
        ).astype(np.float64)
        # Inverse frequency, normalised to mean=1.
        weights = (counts.sum() / np.maximum(counts, 1)) / dataset.n_material_classes
        class_weights = torch.tensor(weights, dtype=torch.float32, device=device)
        weights_str = ", ".join(
            f"{n}={w:.2f}" for n, w in zip(dataset.category_names, weights)
        )
        logger.info(f"Class weights: {weights_str}")

    loss_fn = SpectralNPLoss(
        w_spectral=args.w_spectral,
        w_reflectance=args.w_reflectance,
        w_atmos=args.w_atmos,
        w_kl=args.w_kl,
        w_material=args.w_material,
        w_calibration=args.w_calibration,
        feature_weight_strength=args.feature_weight,
        material_class_weights=class_weights,
    ).to(device)
    logger.info(
        f"Loss weights: spectral={args.w_spectral} reflectance={args.w_reflectance} "
        f"atmos={args.w_atmos} kl={args.w_kl} material={args.w_material} "
        f"calibration={args.w_calibration} feature_weight={args.feature_weight}"
    )

    # Optional W&B.
    if args.wandb:
        import wandb
        wandb.init(project="spectralnp", config=vars(args))
        wandb.watch(model, log_freq=100)

    # Output directory.
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Training loop.
    best_loss = float("inf")
    arts_rng = np.random.default_rng(args.seed + 10000) if arts_sim is not None else None
    for epoch in range(start_epoch, args.epochs):
        # Refresh ARTS scene cache once per epoch (scene-based fast path).
        if arts_sim is not None:
            import time as _t
            t0 = _t.time()
            scenes = arts_sim.populate_random_scenes(args.n_scenes_per_epoch, arts_rng)
            logger.info(
                f"Epoch {epoch+1}: refreshed {len(scenes)} scenes "
                f"in {_t.time()-t0:.0f}s "
                f"(WV {min(s['atmos'].water_vapour for s in scenes):.1f}-"
                f"{max(s['atmos'].water_vapour for s in scenes):.1f} g/cm², "
                f"AOD {min(s['atmos'].aod_550 for s in scenes):.2f}-"
                f"{max(s['atmos'].aod_550 for s in scenes):.2f}, "
                f"SZA {min(s['geometry'].solar_zenith_deg for s in scenes):.0f}-"
                f"{max(s['geometry'].solar_zenith_deg for s in scenes):.0f}°)"
            )

        # KL annealing.
        kl_weight = min(1.0, epoch / max(args.kl_warmup_epochs, 1))

        metrics = train_one_epoch(model, loader, loss_fn, optimiser, device, kl_weight)
        scheduler.step()

        lr = scheduler.get_last_lr()[0]
        logger.info(
            f"Epoch {epoch+1}/{args.epochs}  "
            f"loss={metrics['total']:.4f}  "
            f"spectral={metrics.get('spectral', 0):.4f}  "
            f"reflectance={metrics.get('reflectance', 0):.4f}  "
            f"atmos={metrics.get('atmos', 0):.4f}  "
            f"kl={metrics.get('kl', 0):.4f}  "
            f"lr={lr:.2e}"
        )

        if args.wandb:
            import wandb
            wandb.log({"epoch": epoch + 1, "lr": lr, **metrics})

        # Save checkpoint.
        if metrics["total"] < best_loss:
            best_loss = metrics["total"]
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimiser_state_dict": optimiser.state_dict(),
                "config": cfg,
                "loss": best_loss,
                "speclib_size": len(speclib),
                "category_names": dataset.category_names,
            }, out_dir / "best.pt")

        if (epoch + 1) % 10 == 0:
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimiser_state_dict": optimiser.state_dict(),
                "config": cfg,
                "speclib_size": len(speclib),
                "category_names": dataset.category_names,
            }, out_dir / f"epoch_{epoch+1}.pt")

    logger.info(f"Training complete. Best loss: {best_loss:.4f}")


if __name__ == "__main__":
    main()
