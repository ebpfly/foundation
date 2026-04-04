"""USGS Spectral Library v7 loader.

Downloads and parses the USGS Spectral Library v7 ASCII data,
providing surface reflectance spectra for training data generation.

The library contains ~1300 spectra of minerals, soils, vegetation,
water, and man-made materials measured from 0.2-200 um.

We focus on the ASD range (0.35-2.5 um) which covers the common
remote sensing window.
"""

from __future__ import annotations

import io
import os
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Default cache location.
_CACHE_DIR = Path(os.environ.get("SPECTRALNP_DATA", Path.home() / ".spectralnp" / "data"))
_USGS_ZIP_URL = (
    "https://crustal.usgs.gov/speclab/QueryAll07a.php?quick_filter=702"
)

# We ship a download helper but the primary workflow is to point at
# an already-downloaded copy of the ASCII data.

MATERIAL_CATEGORIES = [
    "minerals",
    "soils",
    "vegetation",
    "water",
    "manmade",
    "mixtures",
    "coatings",
]


@dataclass
class Spectrum:
    """A single reflectance spectrum from the USGS library."""

    name: str
    category: str
    wavelength_um: np.ndarray   # wavelength in micrometres
    reflectance: np.ndarray     # dimensionless [0, 1] (may have negatives for noise)
    description: str = ""
    measurement_type: str = ""  # e.g., "ASD", "Beckman", "Nicolet"

    @property
    def wavelength_nm(self) -> np.ndarray:
        return self.wavelength_um * 1000.0

    def resample(self, target_wl_nm: np.ndarray) -> np.ndarray:
        """Linearly interpolate reflectance to target wavelengths (nm).

        Values outside the measured range are set to NaN.
        """
        return np.interp(
            target_wl_nm,
            self.wavelength_nm,
            self.reflectance,
            left=np.nan,
            right=np.nan,
        )


@dataclass
class SpectralLibrary:
    """Collection of USGS spectra ready for use in training."""

    spectra: list[Spectrum] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.spectra)

    def filter_category(self, category: str) -> SpectralLibrary:
        return SpectralLibrary([s for s in self.spectra if s.category == category])

    def filter_wavelength_range(self, lo_nm: float, hi_nm: float) -> SpectralLibrary:
        """Keep only spectra that have coverage in the given range."""
        out = []
        for s in self.spectra:
            wl = s.wavelength_nm
            if wl.min() <= lo_nm and wl.max() >= hi_nm:
                out.append(s)
        return SpectralLibrary(out)

    def to_array(self, wavelength_nm: np.ndarray) -> np.ndarray:
        """Resample all spectra onto a common wavelength grid.

        Returns (N_spectra, N_wavelengths) array.  NaN where data is missing.
        """
        return np.stack([s.resample(wavelength_nm) for s in self.spectra])


def _parse_asd_ascii(text: str, filename: str) -> Spectrum | None:
    """Parse a single USGS ASCII spectrum file.

    The USGS ASCII format has a header section followed by two-column data
    (wavelength in micrometres, reflectance).  Lines with -1.23e+34 are
    flagged as deleted/bad channels.
    """
    lines = text.strip().splitlines()
    if len(lines) < 10:
        return None

    # Extract name from first line.
    name = lines[0].strip()

    # Detect category from filename path.
    category = "unknown"
    fname_lower = filename.lower()
    for cat in MATERIAL_CATEGORIES:
        if cat in fname_lower:
            category = cat
            break

    # Find data start: first line that looks like two floats.
    data_start = None
    for i, line in enumerate(lines):
        parts = line.strip().split()
        if len(parts) >= 2:
            try:
                float(parts[0])
                float(parts[1])
                data_start = i
                break
            except ValueError:
                continue

    if data_start is None:
        return None

    wavelengths = []
    reflectances = []
    for line in lines[data_start:]:
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        try:
            wl = float(parts[0])
            refl = float(parts[1])
        except ValueError:
            continue
        # Skip deleted channels (flagged as -1.23e+34 in USGS format).
        if refl < -1e30:
            continue
        if wl <= 0:
            continue
        wavelengths.append(wl)
        reflectances.append(refl)

    if len(wavelengths) < 10:
        return None

    return Spectrum(
        name=name,
        category=category,
        wavelength_um=np.array(wavelengths, dtype=np.float64),
        reflectance=np.array(reflectances, dtype=np.float64),
    )


def load_from_directory(data_dir: str | Path) -> SpectralLibrary:
    """Load all ASCII spectra from a directory tree.

    Expects the USGS Spectral Library v7 ASCII data layout:
        data_dir/
            ASCIIdata/
                ...subdirectories with .txt files...

    Parameters
    ----------
    data_dir : path to extracted USGS spectral library data.

    Returns
    -------
    SpectralLibrary with all successfully parsed spectra.
    """
    data_dir = Path(data_dir)
    spectra: list[Spectrum] = []
    for txt_path in sorted(data_dir.rglob("*.txt")):
        try:
            text = txt_path.read_text(errors="replace")
        except Exception:
            continue
        spec = _parse_asd_ascii(text, str(txt_path))
        if spec is not None:
            spectra.append(spec)
    return SpectralLibrary(spectra)


def load_from_zip(zip_path: str | Path) -> SpectralLibrary:
    """Load spectra directly from the USGS zip archive without full extraction."""
    zip_path = Path(zip_path)
    spectra: list[Spectrum] = []
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            if not info.filename.endswith(".txt"):
                continue
            with zf.open(info) as f:
                text = io.TextIOWrapper(f, errors="replace").read()
            spec = _parse_asd_ascii(text, info.filename)
            if spec is not None:
                spectra.append(spec)
    return SpectralLibrary(spectra)
