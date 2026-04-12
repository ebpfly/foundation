"""Random virtual sensor augmentation.

The key data augmentation strategy that makes SpectralNP sensor-agnostic.
During training, each sample is observed through a randomly generated
"virtual sensor" with a random number of bands at random wavelength
positions with random spectral widths.  This forces the model to learn
general spectral reasoning rather than memorising specific sensor
configurations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class VirtualSensor:
    """A randomly generated sensor configuration."""

    center_wavelength_nm: np.ndarray  # (N_bands,)
    fwhm_nm: np.ndarray              # (N_bands,)

    @property
    def n_bands(self) -> int:
        return len(self.center_wavelength_nm)


def sample_virtual_sensor(
    rng: np.random.Generator,
    n_bands_range: tuple[int, int] = (3, 200),
    wavelength_range: tuple[float, float] = (380.0, 2500.0),
    fwhm_range: tuple[float, float] = (1.0, 100.0),
    strategy: str = "mixed",
) -> VirtualSensor:
    """Generate a random virtual sensor configuration.

    Parameters
    ----------
    rng : numpy random Generator.
    n_bands_range : (min, max) number of bands.
    wavelength_range : (min, max) wavelength in nm.
    fwhm_range : (min, max) band FWHM in nm.
    strategy : sampling strategy
        - "uniform": bands uniformly distributed across range
        - "clustered": bands clustered in 1-3 spectral regions
        - "regular": evenly spaced bands (like a hyperspectral sensor)
        - "mixed": randomly choose one of the above strategies

    Returns
    -------
    VirtualSensor with sorted band positions.
    """
    if strategy == "mixed":
        strategy = rng.choice([
            "uniform", "clustered", "regular",
            "edge_only", "gap_max",  # adversarial strategies
        ])

    n_bands = rng.integers(n_bands_range[0], n_bands_range[1] + 1)
    wl_lo, wl_hi = wavelength_range

    if strategy == "uniform":
        centers = rng.uniform(wl_lo, wl_hi, size=n_bands)
        fwhms = rng.uniform(fwhm_range[0], fwhm_range[1], size=n_bands)

    elif strategy == "clustered":
        # 1-3 spectral windows, simulating a multispectral sensor.
        n_clusters = rng.integers(1, 4)
        cluster_centers = rng.uniform(wl_lo + 100, wl_hi - 100, size=n_clusters)
        cluster_widths = rng.uniform(50, 400, size=n_clusters)

        # Distribute bands among clusters.
        bands_per_cluster = np.diff(
            np.sort(np.concatenate([[0], rng.integers(1, n_bands, size=n_clusters - 1), [n_bands]]))
        )
        bands_per_cluster = np.maximum(bands_per_cluster, 1)

        centers_list = []
        for cc, cw, nb in zip(cluster_centers, cluster_widths, bands_per_cluster):
            c = rng.normal(cc, cw / 3, size=int(nb))
            c = np.clip(c, wl_lo, wl_hi)
            centers_list.append(c)
        centers = np.concatenate(centers_list)[:n_bands]
        fwhms = rng.uniform(fwhm_range[0], fwhm_range[1], size=len(centers))

    elif strategy == "regular":
        # Evenly spaced, like a hyperspectral sensor.
        centers = np.linspace(wl_lo, wl_hi, n_bands)
        # Constant FWHM for this sensor.
        fwhm_val = rng.uniform(fwhm_range[0], min(fwhm_range[1], 20.0))
        fwhms = np.full(n_bands, fwhm_val)

    elif strategy == "edge_only":
        # Adversarial: bands only at the spectral edges, large gap in middle.
        n_lo = max(1, n_bands // 2)
        n_hi = n_bands - n_lo
        lo_bands = rng.uniform(wl_lo, wl_lo + (wl_hi - wl_lo) * 0.2, size=n_lo)
        hi_bands = rng.uniform(wl_hi - (wl_hi - wl_lo) * 0.2, wl_hi, size=n_hi)
        centers = np.concatenate([lo_bands, hi_bands])
        fwhms = rng.uniform(fwhm_range[0], fwhm_range[1], size=n_bands)

    elif strategy == "gap_max":
        # Adversarial: place bands to create large unobserved gaps.
        # Pick 2-4 narrow windows, leaving most of the range unobserved.
        n_windows = rng.integers(2, 5)
        window_width = (wl_hi - wl_lo) * 0.05  # each window is 5% of range
        window_centers = rng.uniform(wl_lo + window_width, wl_hi - window_width, size=n_windows)
        bands_per_window = max(1, n_bands // n_windows)
        centers_list = []
        for wc in window_centers:
            c = rng.uniform(wc - window_width / 2, wc + window_width / 2, size=bands_per_window)
            centers_list.append(c)
        centers = np.concatenate(centers_list)[:n_bands]
        centers = np.clip(centers, wl_lo, wl_hi)
        fwhms = rng.uniform(fwhm_range[0], fwhm_range[1], size=len(centers))

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # Sort by wavelength.
    order = np.argsort(centers)
    centers = centers[order]
    fwhms = fwhms[order] if len(fwhms) == len(centers) else fwhms[order[:len(fwhms)]]

    return VirtualSensor(
        center_wavelength_nm=centers.astype(np.float32),
        fwhm_nm=fwhms.astype(np.float32),
    )


def apply_sensor(
    sensor: VirtualSensor,
    wavelength_nm: np.ndarray,
    spectrum: np.ndarray,
) -> np.ndarray:
    """Convolve a high-resolution spectrum with a virtual sensor's SRFs.

    Parameters
    ----------
    sensor : VirtualSensor
    wavelength_nm : (W,) high-resolution wavelength grid
    spectrum : (W,) or (B, W) spectral values

    Returns
    -------
    (N_bands,) or (B, N_bands) band-integrated values.
    """
    from spectralnp.data.srf import pseudo_voigt

    srf = pseudo_voigt(
        wavelength_nm[np.newaxis, :],
        sensor.center_wavelength_nm[:, np.newaxis],
        sensor.fwhm_nm[:, np.newaxis],
    )

    if spectrum.ndim == 1:
        return srf @ spectrum
    return spectrum @ srf.T


def add_sensor_noise(
    radiance: np.ndarray,
    rng: np.random.Generator,
    snr_range: tuple[float, float] = (50.0, 500.0),
) -> np.ndarray:
    """Add realistic signal-dependent noise to radiance measurements.

    Models combined Poisson (signal-dependent) and Gaussian (read) noise.

    Parameters
    ----------
    radiance : (...,) radiance values.
    rng : numpy Generator.
    snr_range : (min, max) signal-to-noise ratio at reference radiance.

    Returns
    -------
    Noisy radiance with same shape.
    """
    snr = rng.uniform(snr_range[0], snr_range[1])
    # Reference radiance for SNR specification.
    ref_radiance = np.percentile(np.abs(radiance) + 1e-10, 75)
    # Noise standard deviation: sigma = L / SNR (signal-dependent).
    sigma = np.abs(radiance) / snr + ref_radiance / (snr * 10)  # + read noise floor
    noise = rng.normal(0, sigma)
    return radiance + noise
