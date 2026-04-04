"""Train a small SpectralNP model for the demo notebook.

Uses synthetic spectra + simplified RTM so no external data or ARTS install needed.
Trains for enough epochs to show meaningful convergence.
"""

import logging
import numpy as np
import torch
from torch.utils.data import DataLoader

from spectralnp.data.synthetic_speclib import generate_synthetic_library
from spectralnp.data.dataset import SpectralNPDataset, collate_spectral_batch
from spectralnp.model.spectralnp import SpectralNP, SpectralNPConfig
from spectralnp.training.losses import SpectralNPLoss

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    logger.info(f"Device: {device}")

    # Generate synthetic spectral library.
    logger.info("Generating synthetic spectral library...")
    speclib = generate_synthetic_library(n_per_class=60, seed=42)
    logger.info(f"Generated {len(speclib)} spectra across 5 material classes")

    # Dataset: on-the-fly simulation + random sensor augmentation.
    dataset = SpectralNPDataset(
        spectral_library=speclib,
        dense_wavelength_nm=np.arange(380.0, 2501.0, 5.0),
        samples_per_epoch=8000,
        n_bands_range=(3, 100),
        seed=42,
    )
    loader = DataLoader(
        dataset,
        batch_size=32,
        collate_fn=collate_spectral_batch,
        num_workers=0,  # in-process for demo
    )

    # Small model for fast training.
    cfg = SpectralNPConfig(
        d_model=128,
        n_heads=4,
        n_layers=4,
        n_frequencies=32,
        n_latents=32,
        z_dim=64,
        spectral_hidden=256,
        spectral_n_layers=3,
        n_material_classes=len(speclib),
        n_atmos_params=4,
        dropout=0.1,
    )
    model = SpectralNP(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {n_params:,}")

    optimiser = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=40)
    loss_fn = SpectralNPLoss(
        w_spectral=1.0,
        w_atmos=0.1,
        w_kl=0.005,
        w_material=0.05,
        w_evidence_reg=0.01,
    )

    n_epochs = 40
    best_loss = float("inf")

    for epoch in range(n_epochs):
        model.train()
        running = {}
        n_batches = 0

        for batch in loader:
            wl = batch["wavelength"].to(device)
            fw = batch["fwhm"].to(device)
            rad = batch["radiance"].to(device)
            mask = batch["pad_mask"].to(device)
            t_wl = batch["target_wavelength"].to(device)
            t_rad = batch["target_radiance"].to(device)
            t_atmos = batch["atmos_params"].to(device)
            t_mat = batch["material_idx"].to(device)

            out = model(wl, fw, rad, pad_mask=mask, query_wavelength=t_wl)

            # KL annealing.
            kl_w_orig = loss_fn.w_kl
            loss_fn.w_kl = kl_w_orig * min(1.0, epoch / 10)

            losses = loss_fn(out, t_rad, t_atmos, t_mat)
            loss_fn.w_kl = kl_w_orig

            optimiser.zero_grad()
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()

            for k, v in losses.items():
                running[k] = running.get(k, 0.0) + v.item()
            n_batches += 1

        scheduler.step()
        avg = {k: v / n_batches for k, v in running.items()}

        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info(
                f"Epoch {epoch+1:3d}/{n_epochs}  "
                f"total={avg['total']:.4f}  "
                f"spectral={avg.get('spectral', 0):.4f}  "
                f"atmos={avg.get('atmos', 0):.4f}  "
                f"kl={avg.get('kl', 0):.4f}  "
                f"material={avg.get('material', 0):.4f}"
            )

        if avg["total"] < best_loss:
            best_loss = avg["total"]

    # Save checkpoint.
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": cfg,
        "loss": best_loss,
        "epoch": n_epochs,
        "speclib_size": len(speclib),
    }, "demo_model.pt")
    logger.info(f"Saved demo_model.pt (best loss: {best_loss:.4f})")


if __name__ == "__main__":
    main()
