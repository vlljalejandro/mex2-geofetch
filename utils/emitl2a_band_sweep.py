import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
from pathlib import Path
from osgeo import gdal


# =============================================================================
# Config
# =============================================================================
NC_PATH        = Path("data/04_cube_data/emitl2a_hyper60m.nc")
BASE_GRID_PATH = Path("data/00_base_grid/maaden_base_grid60m.tif")  # 0 = nodata expected, 1 = data expected

OUT_DIR_GRAYSCALE   = Path("figures/band_sweep/grayscale")
OUT_DIR_SLIDING_RGB = Path("figures/band_sweep/sliding_rgb")

VAR_NAME  = "refl"

GENERATE_GRAYSCALE   = False
GENERATE_SLIDING_RGB = True

STRIDE     = 20   # band offset between R/G/B channels in the sliding composite
FRAME_STEP = 1   # set >1 to skip bands (e.g. 2 = every other band)

WAVELENGTH_MIN = 380.0
WAVELENGTH_MAX = 2510.0
N_BANDS        = 285

STRETCH_LOW_PCT  = 2
STRETCH_HIGH_PCT = 98
STRETCH_SAMPLE_BANDS = 40   # bands sampled to compute the *global* stretch

DPI              = 150
FIGSIZE          = (10, 10)
COMPOSITE_ALPHA  = 1.0


# =============================================================================
# Wavelength / band-label helpers
# =============================================================================

def band_wavelengths(wl_min, wl_max, n_bands):
    return np.linspace(wl_min, wl_max, n_bands)


def band_label(idx):
    return f"b{idx + 1:03d}"


# =============================================================================
# Load full cube + base grid mask once
# =============================================================================

def load_cube(nc_path, var_name):
    ds = xr.open_dataset(nc_path)
    da = ds[var_name]
    band_dim = "band" if "band" in da.dims else f"{var_name}_band"
    data = da.values
    x = da.coords["x"].values
    y = da.coords["y"].values
    crs_wkt = ds["spatial_ref"].attrs["crs_wkt"]
    ds.close()
    return data, band_dim, da.dims, x, y, crs_wkt


def get_band_array(data, dims, band_dim, band_idx):
    axis = dims.index(band_dim)
    return np.take(data, band_idx, axis=axis)


def load_expected_data_mask(path):
    """1 = data expected, 0 = nodata expected, per the base grid convention."""
    ds = gdal.Open(str(path))
    if ds is None:
        raise FileNotFoundError(f"Could not open base grid: {path}")
    arr = ds.GetRasterBand(1).ReadAsArray()
    ds = None
    return arr == 1


# =============================================================================
# Per-band validity: a band is invalid if any NaN lands where data is expected
# =============================================================================

def compute_band_validity(data, dims, band_dim, n_bands, expected_mask):
    validity = np.ones(n_bands, dtype=bool)
    for idx in range(n_bands):
        arr = get_band_array(data, dims, band_dim, idx)
        if arr.shape != expected_mask.shape:
            raise ValueError(
                f"Band shape {arr.shape} != base grid shape {expected_mask.shape}. "
                "Confirm the base grid is on the same grid/alignment as the cube."
            )
        nan_on_expected_data = np.isnan(arr) & expected_mask
        validity[idx] = not np.any(nan_on_expected_data)
    return validity


# =============================================================================
# Global stretch (computed ONCE across a sample of bands, reused every frame)
# =============================================================================

def compute_global_stretch(data, dims, band_dim, n_bands, sample_bands, low_pct, high_pct):
    sample_idx = np.linspace(0, n_bands - 1, sample_bands).astype(int)
    samples = []
    for idx in sample_idx:
        arr = get_band_array(data, dims, band_dim, idx)
        valid = arr[np.isfinite(arr)]
        if valid.size:
            samples.append(valid)
    pooled = np.concatenate(samples)
    lo, hi = np.percentile(pooled, [low_pct, high_pct])
    return lo, hi


def stretch_to_uint8(arr, lo, hi):
    stretched = np.clip((arr - lo) / (hi - lo), 0, 1)
    stretched = np.nan_to_num(stretched, nan=0.0)
    return (stretched * 255).astype(np.uint8)


# =============================================================================
# Frame builders
# =============================================================================

def build_grayscale_frame(data, dims, band_dim, band_idx, lo, hi):
    arr = get_band_array(data, dims, band_dim, band_idx)
    gray = stretch_to_uint8(arr, lo, hi)
    rgba = np.zeros((*gray.shape, 4), dtype=np.uint8)
    rgba[..., 0] = gray
    rgba[..., 1] = gray
    rgba[..., 2] = gray
    rgba[..., 3] = np.where(np.isfinite(arr), 255, 0).astype(np.uint8)
    return rgba


def build_sliding_rgb_frame(data, dims, band_dim, idx_r, idx_g, idx_b, lo, hi):
    arr_r = get_band_array(data, dims, band_dim, idx_r)
    arr_g = get_band_array(data, dims, band_dim, idx_g)
    arr_b = get_band_array(data, dims, band_dim, idx_b)

    nodata = ~(np.isfinite(arr_r) & np.isfinite(arr_g) & np.isfinite(arr_b))

    rgba = np.zeros((*arr_r.shape, 4), dtype=np.uint8)
    rgba[..., 0] = stretch_to_uint8(arr_r, lo, hi)
    rgba[..., 1] = stretch_to_uint8(arr_g, lo, hi)
    rgba[..., 2] = stretch_to_uint8(arr_b, lo, hi)
    rgba[..., 3] = np.where(nodata, 0, 255).astype(np.uint8)
    return rgba


# =============================================================================
# Main sweep builder — figure built ONCE, reused for every frame
# in both the grayscale and sliding-RGB sets
# =============================================================================

def build_band_sweep():
    data, band_dim, dims, x, y, crs_wkt = load_cube(NC_PATH, VAR_NAME)
    wavelengths = band_wavelengths(WAVELENGTH_MIN, WAVELENGTH_MAX, N_BANDS)

    expected_mask = load_expected_data_mask(BASE_GRID_PATH)
    validity = compute_band_validity(data, dims, band_dim, N_BANDS, expected_mask)
    n_invalid = int((~validity).sum())
    print(f"[*] {n_invalid}/{N_BANDS} bands flagged invalid (NaN over expected-data pixels)")
    if n_invalid:
        invalid_labels = [band_label(i) for i in range(N_BANDS) if not validity[i]]
        print(f"    invalid: {invalid_labels}")

    lo, hi = compute_global_stretch(
        data, dims, band_dim, N_BANDS, STRETCH_SAMPLE_BANDS, STRETCH_LOW_PCT, STRETCH_HIGH_PCT
    )
    print(f"[*] Global stretch bounds: lo={lo:.4f}, hi={hi:.4f}")

    dx = abs(x[1] - x[0])
    dy = abs(y[1] - y[0])
    extent = [x.min() - dx / 2, x.max() + dx / 2, y.min() - dy / 2, y.max() + dy / 2]

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal")

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.add_artist(ScaleBar(1, location="lower left", box_alpha=0.7))
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99)

    placeholder_shape = get_band_array(data, dims, band_dim, 0).shape
    im = ax.imshow(
        np.zeros((*placeholder_shape, 4), dtype=np.uint8),
        extent=extent, alpha=COMPOSITE_ALPHA, zorder=3, origin="upper",
    )
    legend = ax.text(
        0.02, 0.04, "", transform=ax.transAxes, fontsize=8, va="bottom", ha="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8), zorder=10,
    )

    if GENERATE_GRAYSCALE:
        OUT_DIR_GRAYSCALE.mkdir(parents=True, exist_ok=True)
        n_saved = 0
        for idx in range(0, N_BANDS, FRAME_STEP):
            if not validity[idx]:
                continue
            rgba = build_grayscale_frame(data, dims, band_dim, idx, lo, hi)
            im.set_data(rgba)
            wl = wavelengths[idx]
            legend.set_text(f"{band_label(idx)} ({wl:.0f} nm)")
            out_path = OUT_DIR_GRAYSCALE / f"frame_{n_saved:03d}_{band_label(idx)}.png"
            fig.savefig(out_path, dpi=DPI)
            n_saved += 1
        print(f"[✓] Wrote {n_saved} grayscale frames to {OUT_DIR_GRAYSCALE}")

    if GENERATE_SLIDING_RGB:
        OUT_DIR_SLIDING_RGB.mkdir(parents=True, exist_ok=True)
        n_saved = 0
        for center_idx in range(0, N_BANDS, FRAME_STEP):
            idx_r = min(center_idx, N_BANDS - 1)
            idx_g = min(center_idx + STRIDE, N_BANDS - 1)
            idx_b = min(center_idx + 2 * STRIDE, N_BANDS - 1)

            if not (validity[idx_r] and validity[idx_g] and validity[idx_b]):
                continue

            rgba = build_sliding_rgb_frame(data, dims, band_dim, idx_r, idx_g, idx_b, lo, hi)
            im.set_data(rgba)
            legend.set_text(f"R={band_label(idx_r)} G={band_label(idx_g)} B={band_label(idx_b)}")
            out_path = OUT_DIR_SLIDING_RGB / f"frame_{n_saved:03d}_{band_label(center_idx)}.png"
            fig.savefig(out_path, dpi=DPI)
            n_saved += 1
        print(f"[✓] Wrote {n_saved} sliding-RGB frames to {OUT_DIR_SLIDING_RGB}")

    plt.close(fig)


if __name__ == "__main__":
    build_band_sweep()