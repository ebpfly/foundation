"""Inference pipeline for SpectralNP.

Provides a high-level API for predicting from any sensor's data
with calibrated uncertainty.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from spectralnp.model.spectralnp import SpectralNP, SpectralNPConfig


@dataclass
class SpectralPrediction:
    """Full prediction with uncertainty from SpectralNP."""

    # Spectral reconstruction (at-sensor radiance).
    spectral_mean: np.ndarray | None = None    # (Q,) or (B, Q)
    spectral_std: np.ndarray | None = None     # (Q,) or (B, Q)

    # Surface reflectance reconstruction.
    reflectance_mean: np.ndarray | None = None # (Q,) or (B, Q)
    reflectance_std: np.ndarray | None = None  # (Q,) or (B, Q)

    # Material probabilities.
    material_probs: np.ndarray | None = None   # (C,) or (B, C)
    material_entropy: np.ndarray | None = None # scalar or (B,)

    # Atmospheric parameters.
    atmos_mean: np.ndarray | None = None       # (P,) or (B, P)
    atmos_std: np.ndarray | None = None        # (P,) or (B, P)


class SpectralNPPredictor:
    """High-level inference wrapper.

    Example usage:

        predictor = SpectralNPPredictor.from_checkpoint("checkpoints/best.pt")

        # Predict from Sentinel-2 bands.
        pred = predictor.predict(
            wavelength_nm=[443, 490, 560, 665, 705, 740, 783, 842, 865],
            fwhm_nm=[20, 65, 35, 30, 15, 15, 20, 115, 20],
            radiance=[85.2, 92.1, 78.4, 45.3, 52.1, 60.8, 70.2, 68.5, 55.3],
            query_wavelength_nm=np.arange(400, 2501, 1),  # 1nm reconstruction
        )

        print(pred.spectral_mean)   # reconstructed spectrum
        print(pred.spectral_std)    # uncertainty (wider with fewer input bands)
        print(pred.material_probs)  # material classification
        print(pred.atmos_mean)      # estimated AOD, water vapour, etc.
    """

    def __init__(self, model: SpectralNP, device: torch.device | None = None) -> None:
        self.device = device or torch.device("cpu")
        self.model = model.to(self.device).eval()

    @classmethod
    def from_checkpoint(cls, path: str | Path, device: str = "cpu") -> SpectralNPPredictor:
        ckpt = torch.load(path, map_location=device, weights_only=False)
        cfg = ckpt.get("config", SpectralNPConfig())
        model = SpectralNP(cfg)
        model.load_state_dict(ckpt["model_state_dict"])
        return cls(model, torch.device(device))

    def predict(
        self,
        wavelength_nm: np.ndarray | list[float],
        fwhm_nm: np.ndarray | list[float],
        radiance: np.ndarray | list[float],
        query_wavelength_nm: np.ndarray | None = None,
        n_samples: int = 16,
    ) -> SpectralPrediction:
        """Run inference on a single observation.

        Parameters
        ----------
        wavelength_nm : band center wavelengths (nm)
        fwhm_nm : band widths (nm)
        radiance : at-sensor radiance values (W/m^2/sr/um)
        query_wavelength_nm : wavelengths for spectral reconstruction (nm)
        n_samples : number of z samples for uncertainty estimation

        Returns
        -------
        SpectralPrediction with means and uncertainties.
        """
        wl = torch.tensor(np.asarray(wavelength_nm, dtype=np.float32)).unsqueeze(0).to(self.device)
        fw = torch.tensor(np.asarray(fwhm_nm, dtype=np.float32)).unsqueeze(0).to(self.device)
        rad = torch.tensor(np.asarray(radiance, dtype=np.float32)).unsqueeze(0).to(self.device)

        q_wl = None
        if query_wavelength_nm is not None:
            q_wl = torch.tensor(
                np.asarray(query_wavelength_nm, dtype=np.float32)
            ).unsqueeze(0).to(self.device)

        results = self.model.predict_with_uncertainty(
            wl, fw, rad,
            query_wavelength=q_wl,
            n_samples=n_samples,
        )

        pred = SpectralPrediction()
        if "spectral_mean" in results:
            pred.spectral_mean = results["spectral_mean"][0].cpu().numpy()
            pred.spectral_std = results["spectral_std"][0].cpu().numpy()
        if "reflectance_mean" in results:
            pred.reflectance_mean = results["reflectance_mean"][0].cpu().numpy()
            pred.reflectance_std = results["reflectance_std"][0].cpu().numpy()
        if "material_probs" in results:
            pred.material_probs = results["material_probs"][0].cpu().numpy()
            pred.material_entropy = results["material_entropy"][0].cpu().numpy()
        if "atmos_mean" in results:
            pred.atmos_mean = results["atmos_mean"][0].cpu().numpy()
            pred.atmos_std = results["atmos_std"][0].cpu().numpy()

        return pred

    def predict_batch(
        self,
        wavelength_nm: list[np.ndarray],
        fwhm_nm: list[np.ndarray],
        radiance: list[np.ndarray],
        query_wavelength_nm: np.ndarray | None = None,
        n_samples: int = 16,
    ) -> SpectralPrediction:
        """Run inference on a batch of observations (potentially different sensors).

        Each element of the lists can have a different number of bands.
        """
        max_bands = max(len(w) for w in wavelength_nm)
        B = len(wavelength_nm)

        wl_pad = torch.zeros(B, max_bands)
        fw_pad = torch.zeros(B, max_bands)
        rad_pad = torch.zeros(B, max_bands)
        mask = torch.zeros(B, max_bands, dtype=torch.bool)

        for i in range(B):
            n = len(wavelength_nm[i])
            wl_pad[i, :n] = torch.from_numpy(np.asarray(wavelength_nm[i], dtype=np.float32))
            fw_pad[i, :n] = torch.from_numpy(np.asarray(fwhm_nm[i], dtype=np.float32))
            rad_pad[i, :n] = torch.from_numpy(np.asarray(radiance[i], dtype=np.float32))
            mask[i, :n] = True

        wl_pad = wl_pad.to(self.device)
        fw_pad = fw_pad.to(self.device)
        rad_pad = rad_pad.to(self.device)
        mask = mask.to(self.device)

        q_wl = None
        if query_wavelength_nm is not None:
            q_wl = torch.tensor(
                np.asarray(query_wavelength_nm, dtype=np.float32)
            ).unsqueeze(0).expand(B, -1).to(self.device)

        results = self.model.predict_with_uncertainty(
            wl_pad, fw_pad, rad_pad,
            pad_mask=mask,
            query_wavelength=q_wl,
            n_samples=n_samples,
        )

        pred = SpectralPrediction()
        if "spectral_mean" in results:
            pred.spectral_mean = results["spectral_mean"].cpu().numpy()
            pred.spectral_std = results["spectral_std"].cpu().numpy()
        if "reflectance_mean" in results:
            pred.reflectance_mean = results["reflectance_mean"].cpu().numpy()
            pred.reflectance_std = results["reflectance_std"].cpu().numpy()
        if "material_probs" in results:
            pred.material_probs = results["material_probs"].cpu().numpy()
            pred.material_entropy = results["material_entropy"].cpu().numpy()
        if "atmos_mean" in results:
            pred.atmos_mean = results["atmos_mean"].cpu().numpy()
            pred.atmos_std = results["atmos_std"].cpu().numpy()

        return pred
