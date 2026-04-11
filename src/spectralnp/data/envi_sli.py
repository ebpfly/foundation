"""ENVI spectral library (.sli) reader and writer.

ENVI .sli format is a simple pair of files:
  * ``name.sli``  — raw binary, shape (n_spectra, n_bands), float32, row-major
  * ``name.hdr``  — ASCII header describing shape, wavelengths, names, metadata

This module provides round-tripping support for the subset of fields used
by the synthetic LWIR library generator.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------

def write_envi_sli(
    path: str | Path,
    wavelength_nm: np.ndarray,
    spectra: np.ndarray,
    spectra_names: list[str] | None = None,
    description: str = "",
    reflectance_scale: float = 1.0,
) -> None:
    """Write an ENVI spectral library.

    Parameters
    ----------
    path : base path (without or with extension; .sli and .hdr will both be written)
    wavelength_nm : (n_bands,) wavelengths in nm
    spectra : (n_spectra, n_bands) float array
    spectra_names : optional list of n_spectra names; defaults to auto-generated
    description : free-form description string for the header
    reflectance_scale : ENVI's "reflectance scale factor" field (1.0 for emissivity
        already in [0, 1])
    """
    path = Path(path)
    if path.suffix in (".sli", ".hdr"):
        path = path.with_suffix("")
    sli_path = path.with_suffix(".sli")
    hdr_path = path.with_suffix(".hdr")

    spectra = np.ascontiguousarray(spectra, dtype=np.float32)
    n_spectra, n_bands = spectra.shape

    if wavelength_nm.shape != (n_bands,):
        raise ValueError(
            f"wavelength_nm shape {wavelength_nm.shape} does not match "
            f"n_bands = {n_bands}"
        )

    if spectra_names is None:
        spectra_names = [f"spectrum_{i:06d}" for i in range(n_spectra)]
    if len(spectra_names) != n_spectra:
        raise ValueError(
            f"spectra_names has {len(spectra_names)} entries, "
            f"expected {n_spectra}"
        )

    # --- Binary data ---
    spectra.tofile(str(sli_path))

    # --- Header ---
    wl_um = wavelength_nm.astype(np.float64) / 1000.0  # ENVI prefers μm for thermal IR

    # ENVI separates list entries with ",\n " by convention
    spectra_names_block = "{\n" + ",\n".join(f" {n}" for n in spectra_names) + "}"
    wavelength_block = (
        "{\n" + ",\n".join(f" {w:.6f}" for w in wl_um) + "}"
    )

    hdr_lines = [
        "ENVI",
        f"description = {{\n {description}}}",
        f"samples = {n_bands}",
        f"lines = {n_spectra}",
        "bands = 1",
        "header offset = 0",
        "file type = ENVI Spectral Library",
        "data type = 4",
        "interleave = bsq",
        "sensor type = Unknown",
        "byte order = 0",
        "wavelength units = Micrometers",
        f"reflectance scale factor = {reflectance_scale:.6f}",
        "band names = { Spectral Library }",
        f"spectra names = {spectra_names_block}",
        f"wavelength = {wavelength_block}",
    ]
    hdr_path.write_text("\n".join(hdr_lines) + "\n")


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------

def read_envi_sli(
    path: str | Path,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Read an ENVI spectral library.

    Returns
    -------
    wavelength_nm : (n_bands,) float32 — always in nm (converted if header
        declares μm)
    spectra : (n_spectra, n_bands) float32
    names : list of n_spectra names
    """
    path = Path(path)
    if path.suffix in (".sli", ".hdr"):
        path = path.with_suffix("")
    hdr_path = path.with_suffix(".hdr")
    sli_path = path.with_suffix(".sli")

    hdr = _parse_envi_header(hdr_path)

    n_bands = int(hdr["samples"])
    n_spectra = int(hdr["lines"])
    data_type = int(hdr.get("data type", "4"))
    if data_type != 4:
        raise NotImplementedError(
            f"Only data type 4 (float32) supported, got {data_type}"
        )

    dtype = np.dtype(np.float32)
    byte_order = int(hdr.get("byte order", "0"))
    dtype = dtype.newbyteorder("<" if byte_order == 0 else ">")

    data = np.fromfile(str(sli_path), dtype=dtype).reshape(n_spectra, n_bands)
    data = data.astype(np.float32, copy=False)

    # Wavelengths
    wl_values = _parse_numeric_list(hdr.get("wavelength", ""))
    wl = np.array(wl_values, dtype=np.float64)
    units = hdr.get("wavelength units", "").lower()
    if "micro" in units or units.strip() == "um":
        wl = wl * 1000.0
    elif "nano" in units or units.strip() == "nm":
        pass
    # other unit strings fall through unchanged

    # Names
    names = _parse_string_list(hdr.get("spectra names", ""))
    if not names:
        names = [f"spectrum_{i:06d}" for i in range(n_spectra)]

    return wl.astype(np.float32), data, names


# ---------------------------------------------------------------------------
# Header parsing
# ---------------------------------------------------------------------------

def _parse_envi_header(path: Path) -> dict[str, str]:
    """Parse an ENVI header file into a {key: value} dict.

    Handles multi-line braced values (e.g. wavelength lists).
    """
    content = path.read_text()
    result: dict[str, str] = {}

    lines = content.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if not stripped or stripped == "ENVI" or "=" not in line:
            i += 1
            continue
        key, _, value = line.partition("=")
        key = key.strip().lower()
        value = value.strip()

        if value.startswith("{") and not value.endswith("}"):
            # Multi-line braced value — collect until matching brace
            parts = [value]
            i += 1
            while i < len(lines):
                parts.append(lines[i])
                if "}" in lines[i]:
                    break
                i += 1
            value = "\n".join(parts)
        result[key] = value
        i += 1
    return result


def _strip_braces(s: str) -> str:
    s = s.strip()
    if s.startswith("{"):
        s = s[1:]
    if s.endswith("}"):
        s = s[:-1]
    return s


def _parse_numeric_list(s: str) -> list[float]:
    s = _strip_braces(s)
    if not s.strip():
        return []
    return [float(part.strip()) for part in s.split(",") if part.strip()]


def _parse_string_list(s: str) -> list[str]:
    s = _strip_braces(s)
    if not s.strip():
        return []
    return [part.strip() for part in s.split(",") if part.strip()]
