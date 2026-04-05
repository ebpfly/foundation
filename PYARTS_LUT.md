# PyARTS LUT: Status & Notes

## Overview

The LUT caches ARTS line-by-line gas absorption (the expensive part) as per-layer optical depths. Runtime path integration (transmittance, emission, any geometry/altitude) is fast vectorised NumPy.

## 1. PyARTS API compatibility — DONE

Verified against **pyarts 2.6.18** (conda env `atmgen-pyarts`). The code now uses the ARTS 2.6 workspace-variable API:

- **Atmosphere**: `ws.p_grid`, `ws.t_field` (Tensor3), `ws.z_field` (Tensor3), `ws.vmr_field` (Tensor4) — *not* `AtmField`
- **Propmat at a point**: `ws.rtp_pressure/temperature/vmr/mag/los/nlte` → `propmat_clearskyInit()` + `propmat_clearskyAddFromLookup()`
- **Extract kappa**: `np.array(ws.propmat_clearsky.value.data)` → shape `(1, 1, n_freq, 1)` → `[0, 0, :, 0]`
- **Line catalogue**: Must call `abs_linesReadSpeciesSplitCatalog` + `abs_lines_per_speciesCreateFromLines` before `propmat_clearsky_agendaAuto()`
- **Species tags**: Updated to `CKDMT400` continua to match atmgen LUTs
- **Fast path**: Can load pre-built `abs_lookup` XML (from `~/.cache/atmgen/lut/`) via `ReadXML` + `abs_lookupAdapt` + `propmat_clearskyAddFromLookup`

## 2. Build a test LUT

Start with the coarse grid to verify end-to-end:

```bash
python scripts/build_lut.py -o lut/arts_test.h5 --quick --workers 1
```

This runs ~12 atmospheric states (vs ~2900 for the full grid). Should take minutes, not hours. Check the output:

```python
import h5py
with h5py.File("lut/arts_test.h5") as f:
    print(f"Wavelengths: {f['wavelength_nm'].shape}")
    print(f"tau_layers: {f['tau_layers'].shape}")
    print(f"File size: {f['tau_layers'].id.get_storage_size() / 1e6:.1f} MB compressed")
```

## 3. Validate the RT

Compare LUT-based RT against direct ARTS simulation:

```python
from spectralnp.data.lut import SpectralLUT
from spectralnp.data.rtm_simulator import ARTSSimulator, AtmosphericState, ViewGeometry
import numpy as np

lut = SpectralLUT("lut/arts_test.h5")

# LUT path integration
L_lut = lut.toa_radiance(
    surface_reflectance=0.3 * np.ones(len(lut.wavelength_nm)),
    water_vapour=2.0, ozone_du=300.0,
    surface_temperature_k=300.0,
    sensor_altitude_km=800.0,
    solar_zenith_deg=30.0, sensor_zenith_deg=0.0,
    aod_550=0.0,  # compare without aerosol first
)

# Direct ARTS simulation (same conditions)
sim = ARTSSimulator(wavelength_nm=lut.wavelength_nm)
result = sim.simulate(
    surface_reflectance=0.3 * np.ones(len(lut.wavelength_nm)),
    atmos=AtmosphericState(water_vapour=2.0, ozone_du=300.0,
                           surface_temperature_k=300.0, aod_550=0.0),
    geometry=ViewGeometry(solar_zenith_deg=30.0, sensor_zenith_deg=0.0),
)

# Compare
import matplotlib.pyplot as plt
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
ax1.plot(lut.wavelength_nm, L_lut, label="LUT")
ax1.plot(lut.wavelength_nm, result.toa_radiance, label="ARTS direct", alpha=0.7)
ax1.set_ylabel("Radiance [W/m²/sr/μm]")
ax1.legend()
ax2.plot(lut.wavelength_nm, (L_lut - result.toa_radiance) / (result.toa_radiance + 1e-10) * 100)
ax2.set_ylabel("Relative difference [%]")
ax2.set_xlabel("Wavelength [nm]")
plt.savefig("lut_validation.png", dpi=150)
```

Key things to check:
- Gas absorption features should align (H₂O, CO₂, O₃ bands)
- TIR thermal emission should match
- Differences should be < 5% (Rayleigh path radiance approximation is the main source)

## 4. Build the full LUT

Once validation looks good:

```bash
python scripts/build_lut.py -o lut/arts_lut.h5 --workers 8
```

The full grid is ~2900 atmospheric states. Each requires one ARTS `propmat_clearsky` evaluation per layer (50 layers × 2901 wavelengths). Expect:
- ~1–5 seconds per atmospheric state (depending on machine)
- ~1–4 hours total with 8 workers
- Output: ~200–500 MB compressed HDF5

## 5. Train with the LUT

```python
from spectralnp.data.dataset import SpectralNPDataset
from spectralnp.data.usgs_speclib import SpectralLibrary

speclib = SpectralLibrary.from_usgs("/path/to/usgs_splib07")
dataset = SpectralNPDataset(
    speclib,
    lut_path="lut/arts_lut.h5",
    dense_wavelength_nm=np.arange(380.0, 2501.0, 5.0),  # or extend for TIR
)
sample = dataset[0]
```

The dataset computes TOA radiance on the LUT grid, then resamples to your training grid. For TIR training, extend the dense grid:

```python
dense_wl = np.arange(300.0, 16001.0, 10.0)  # full range
dataset = SpectralNPDataset(speclib, lut_path="lut/arts_lut.h5", dense_wavelength_nm=dense_wl)
```

## 6. Likely adjustments needed

- **ARTS API**: The `propmat_clearsky` extraction pattern may need tweaking for your pyarts version. See step 1.
- **Rayleigh scattering**: Currently ARTS computes Rayleigh in `propmat_clearsky` (included in layer τ). The analytical Rayleigh path radiance in `path_integrate` may double-count slightly. If validation shows excess scattering, remove the `L_path_ray` term.
- **Solar spectrum**: The current code uses Planck @ 5778 K. ARTS has a proper solar spectrum (`sun/solar_spectrum_QUIET`). For better solar accuracy, you could store the ARTS solar spectrum in the LUT and use it in `path_integrate`.
- **Spherical albedo**: The current analytical approximation (`S = 0.05 * τ_ray`) is rough. For high-albedo surfaces this matters. A more accurate S could be computed from the layer τ using adding-doubling.
