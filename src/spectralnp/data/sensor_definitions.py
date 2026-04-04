"""Known sensor band definitions for validation and fine-tuning.

Each sensor is defined by its band center wavelengths (nm) and FWHM (nm).
These are used for:
1. Evaluating the model on realistic sensor configurations
2. Generating sensor-specific training samples during augmentation
3. Fine-tuning on real sensor data
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SensorDefinition:
    """Definition of a remote sensing instrument's spectral bands."""

    name: str
    center_wavelength_nm: np.ndarray
    fwhm_nm: np.ndarray
    description: str = ""

    @property
    def n_bands(self) -> int:
        return len(self.center_wavelength_nm)

    def spectral_response(self, wavelength_nm: np.ndarray) -> np.ndarray:
        """Compute pseudo-Voigt spectral response functions for all bands.

        Returns (n_bands, len(wavelength_nm)) array of response weights,
        each normalised to integrate to 1.
        """
        from spectralnp.data.srf import pseudo_voigt

        return pseudo_voigt(
            wavelength_nm[np.newaxis, :],
            self.center_wavelength_nm[:, np.newaxis],
            self.fwhm_nm[:, np.newaxis],
        )

    def convolve(self, wavelength_nm: np.ndarray, spectrum: np.ndarray) -> np.ndarray:
        """Convolve a high-resolution spectrum with this sensor's SRFs.

        Parameters
        ----------
        wavelength_nm : (W,) wavelength grid of the input spectrum
        spectrum : (..., W) spectral values

        Returns
        -------
        (..., n_bands) band-integrated values
        """
        srf = self.spectral_response(wavelength_nm)  # (n_bands, W)
        return spectrum @ srf.T


# ---- Specific sensor definitions ----

LANDSAT8_OLI = SensorDefinition(
    name="Landsat-8 OLI",
    center_wavelength_nm=np.array([443, 482, 561, 655, 865, 1609, 2201]),
    fwhm_nm=np.array([16, 60, 57, 37, 28, 85, 187]),
    description="Landsat-8 Operational Land Imager (coastal, blue, green, red, NIR, SWIR1, SWIR2)",
)

SENTINEL2_MSI = SensorDefinition(
    name="Sentinel-2 MSI",
    center_wavelength_nm=np.array([
        443, 490, 560, 665, 705, 740, 783, 842, 865, 945, 1375, 1610, 2190,
    ]),
    fwhm_nm=np.array([20, 65, 35, 30, 15, 15, 20, 115, 20, 20, 30, 90, 180]),
    description="Sentinel-2A MultiSpectral Instrument (13 bands, 443-2190 nm)",
)

AVIRIS_NG = SensorDefinition(
    name="AVIRIS-NG",
    center_wavelength_nm=np.arange(380, 2501, 5.0),
    fwhm_nm=np.full(len(np.arange(380, 2501, 5.0)), 5.0),
    description="AVIRIS Next Generation imaging spectrometer (5 nm sampling, 380-2500 nm)",
)

PRISMA = SensorDefinition(
    name="PRISMA",
    center_wavelength_nm=np.concatenate([
        np.arange(400, 1011, 9.0),   # VNIR (66 bands)
        np.arange(920, 2506, 9.0),   # SWIR (171 bands)
    ]),
    fwhm_nm=np.concatenate([
        np.full(len(np.arange(400, 1011, 9.0)), 9.0),
        np.full(len(np.arange(920, 2506, 9.0)), 9.0),
    ]),
    description="PRISMA hyperspectral (9 nm VNIR + 9 nm SWIR)",
)

ENMAP = SensorDefinition(
    name="EnMAP",
    center_wavelength_nm=np.concatenate([
        np.arange(420, 1001, 6.5),   # VNIR
        np.arange(900, 2451, 10.0),  # SWIR
    ]),
    fwhm_nm=np.concatenate([
        np.full(len(np.arange(420, 1001, 6.5)), 6.5),
        np.full(len(np.arange(900, 2451, 10.0)), 10.0),
    ]),
    description="EnMAP hyperspectral satellite",
)

# Registry of all known sensors.
SENSORS: dict[str, SensorDefinition] = {
    "landsat8": LANDSAT8_OLI,
    "sentinel2": SENTINEL2_MSI,
    "aviris_ng": AVIRIS_NG,
    "prisma": PRISMA,
    "enmap": ENMAP,
}
