"""USGS Spectral Library v7 loader.

Parses the USGS Spectral Library v7 (splib07a) ASCII data.
The library uses single-column reflectance files with separate
wavelength/bandpass files per spectrometer (ASD, Beckman, Nicolet, AVIRIS).

Sentinel value -1.23e+34 marks deleted/bad channels.
"""

from __future__ import annotations

import io
import os
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_CACHE_DIR = Path(os.environ.get("SPECTRALNP_DATA", Path.home() / ".spectralnp" / "data"))

MATERIAL_CATEGORIES = {
    "ChapterA": "artificial",
    "ChapterC": "coatings",
    "ChapterL": "liquids",
    "ChapterM": "minerals",
    "ChapterO": "organics",
    "ChapterS": "soils",
    "ChapterV": "vegetation",
}

# Map filename substrings to the spectrometer wavelength/bandpass files.
_SPECTROMETER_MAP = {
    "ASD": "ASD",
    "BECK": "BECK",
    "NIC4": "NIC4",
    "AVIRIS": "AVIRIS",
}

_BAD_VALUE_THRESHOLD = -1.2e34


@dataclass
class Spectrum:
    """A single reflectance spectrum from the USGS library."""

    name: str
    category: str
    wavelength_um: np.ndarray   # wavelength in micrometres
    reflectance: np.ndarray     # dimensionless (may have negatives for noise)
    description: str = ""
    spectrometer: str = ""      # e.g., "ASD", "BECK", "NIC4", "AVIRIS"

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

    def filter_spectrometer(self, spectrometer: str) -> SpectralLibrary:
        return SpectralLibrary([s for s in self.spectra if s.spectrometer == spectrometer])

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

    @property
    def categories(self) -> list[str]:
        return sorted({s.category for s in self.spectra})


def _read_single_column(text: str) -> np.ndarray:
    """Parse a single-column USGS ASCII file (1 header line, then floats)."""
    lines = text.strip().splitlines()
    if len(lines) < 2:
        return np.array([], dtype=np.float64)
    values = []
    for line in lines[1:]:  # skip header
        line = line.strip()
        if not line:
            continue
        try:
            values.append(float(line))
        except ValueError:
            continue
    return np.array(values, dtype=np.float64)


def _detect_spectrometer(filename: str) -> str | None:
    """Determine spectrometer type from filename."""
    fname = filename.upper()
    if "AVIRIS" in fname:
        return "AVIRIS"
    if "NIC4" in fname:
        return "NIC4"
    if "BECK" in fname:
        return "BECK"
    if "ASD" in fname:
        return "ASD"
    return None


def _detect_category(filepath: str) -> str:
    """Detect material category from directory path."""
    for key, cat in MATERIAL_CATEGORIES.items():
        if key in filepath:
            return cat
    return "unknown"


def _parse_spectrum(
    text: str,
    filepath: str,
    wavelength_um: np.ndarray,
) -> Spectrum | None:
    """Parse a single-column spectrum file paired with its wavelength grid."""
    values = _read_single_column(text)
    if len(values) == 0:
        return None

    # Wavelength and spectrum must have the same number of channels.
    if len(values) != len(wavelength_um):
        return None

    # Mask bad/deleted channels (sentinel -1.23e+34).
    good = values > _BAD_VALUE_THRESHOLD
    if good.sum() < 10:
        return None

    wl_good = wavelength_um[good]
    refl_good = values[good]

    # Extract name from first line.
    lines = text.strip().splitlines()
    name = lines[0].strip() if lines else filepath.split("/")[-1]
    # Clean up the record prefix.
    name = re.sub(r"^splib07a\s+Record=\d+:\s*", "", name).strip()

    spectrometer = _detect_spectrometer(filepath) or ""
    category = _detect_category(filepath)

    return Spectrum(
        name=name,
        category=category,
        wavelength_um=wl_good,
        reflectance=refl_good,
        spectrometer=spectrometer,
    )


def _find_wavelength_files(file_list: list[str]) -> dict[str, str]:
    """Map spectrometer code → wavelength filename."""
    mapping = {}
    for f in file_list:
        fname = f.split("/")[-1].upper()
        if "WAVELENGTH" not in fname:
            continue
        if "ASD" in fname:
            mapping["ASD"] = f
        elif "BECK" in fname:
            mapping["BECK"] = f
        elif "NIC4" in fname:
            mapping["NIC4"] = f
        elif "AVIRIS" in fname:
            mapping["AVIRIS"] = f
    return mapping


def _find_bandpass_files(file_list: list[str]) -> dict[str, str]:
    """Map spectrometer code → bandpass filename."""
    mapping = {}
    for f in file_list:
        fname = f.split("/")[-1].upper()
        if "BANDPASS" not in fname:
            continue
        if "ASDFR" in fname:
            mapping["ASDFR"] = f
        elif "ASDHR" in fname:
            mapping["ASDHR"] = f
        elif "ASDNG" in fname:
            mapping["ASDNG"] = f
        elif "AVIRIS" in fname:
            mapping["AVIRIS"] = f
        elif "BECK" in fname:
            mapping["BECK"] = f
        elif "NIC4" in fname:
            mapping["NIC4"] = f
    return mapping


def load_from_zip(zip_path: str | Path) -> SpectralLibrary:
    """Load spectra from the USGS splib07a zip without full extraction.

    Reads wavelength grids first, then pairs each spectrum file with
    its corresponding wavelength grid based on spectrometer type.
    """
    zip_path = Path(zip_path)
    spectra: list[Spectrum] = []

    with zipfile.ZipFile(zip_path) as zf:
        all_files = [info.filename for info in zf.infolist()]
        txt_files = [f for f in all_files if f.endswith(".txt")]

        # Load wavelength grids per spectrometer.
        wl_file_map = _find_wavelength_files(txt_files)
        wl_grids: dict[str, np.ndarray] = {}
        for spec_code, wl_file in wl_file_map.items():
            with zf.open(wl_file) as f:
                text = io.TextIOWrapper(f, errors="replace").read()
            wl_grids[spec_code] = _read_single_column(text)

        # Parse each spectrum file.
        spectrum_files = [
            f for f in txt_files
            if "Chapter" in f and f.endswith(".txt")
        ]

        for filepath in spectrum_files:
            spec_type = _detect_spectrometer(filepath)
            if spec_type is None or spec_type not in wl_grids:
                continue

            with zf.open(filepath) as f:
                text = io.TextIOWrapper(f, errors="replace").read()

            spec = _parse_spectrum(text, filepath, wl_grids[spec_type])
            if spec is not None:
                spectra.append(spec)

    return SpectralLibrary(spectra)


def load_from_directory(data_dir: str | Path) -> SpectralLibrary:
    """Load all ASCII spectra from an extracted splib07a directory.

    Expects:
        data_dir/
            ASCIIdata_splib07a/   (or files directly in data_dir)
                splib07a_Wavelengths_*.txt
                splib07a_Bandpass_*.txt
                ChapterM_Minerals/*.txt
                ChapterS_SoilsAndMixtures/*.txt
                ...
    """
    data_dir = Path(data_dir)
    txt_files = sorted(str(p) for p in data_dir.rglob("*.txt"))

    # Relativised names for the finder functions.
    rel_files = [str(p) for p in txt_files]

    # Load wavelength grids.
    wl_file_map = _find_wavelength_files(rel_files)
    wl_grids: dict[str, np.ndarray] = {}
    for spec_code, wl_file in wl_file_map.items():
        text = Path(wl_file).read_text(errors="replace")
        wl_grids[spec_code] = _read_single_column(text)

    # Parse spectra.
    spectra: list[Spectrum] = []
    for filepath in txt_files:
        if "Chapter" not in filepath:
            continue
        spec_type = _detect_spectrometer(filepath)
        if spec_type is None or spec_type not in wl_grids:
            continue
        try:
            text = Path(filepath).read_text(errors="replace")
        except Exception:
            continue
        spec = _parse_spectrum(text, filepath, wl_grids[spec_type])
        if spec is not None:
            spectra.append(spec)

    return SpectralLibrary(spectra)
