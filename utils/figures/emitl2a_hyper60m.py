import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib_scalebar.scalebar import ScaleBar
from pathlib import Path
from pyproj import Transformer
import contextily.plotting as cx_plot


# =============================================================================
# Config
# =============================================================================
NC_PATH    = Path("04_cube_data/emitl2a_hyper60m.nc")
OUT_PATH   = Path("figures/emitl2a_hyper60m_composite_mineral.png")

VAR_NAME   = "refl"

# Mineral-ratio composite for EMIT hyperspectral data, built from band-depth
# style ratios around known diagnostic absorption features:
#
#   R = Clay/Al-OH ratio   = continuum(2120nm) / absorption(2200nm)
#       Highlights clay minerals (kaolinite, illite, smectite) via the
#       Al-OH absorption feature near 2200 nm.
#   G = Carbonate ratio    = continuum(2280nm) / absorption(2350nm)
#       Highlights carbonate minerals (calcite, dolomite) via the CO3
#       absorption feature near 2350 nm.
#   B = Iron oxide ratio   = reflectance(650nm) / reflectance(850nm)
#       Highlights iron-oxide / ferric-iron minerals (hematite, goethite)
#       via the characteristic reflectance drop-off shape in the
#       visible-NIR transition.
#
# NOTE: band-to-wavelength mapping below assumes EMIT's 285 bands are
# evenly spaced across 380-2510 nm (per cube metadata). This is an
# approximation — EMIT's true band centers follow the instrument's
# spectral response function and are not perfectly linear. Replace
# WAVELENGTH_MIN/MAX or supply an explicit band->wavelength table if you
# have EMIT's actual band-center file for more precise targeting.
WAVELENGTH_MIN = 380.0
WAVELENGTH_MAX = 2510.0
N_BANDS        = 285

RATIO_BANDS = {
    # channel: (numerator_wavelength_nm, denominator_wavelength_nm)
    "R": (2120, 2200),   # clay / Al-OH absorption
    "G": (2280, 2350),   # carbonate absorption
    "B": (650, 850),     # iron oxide
}

DPI            = 300
FIGSIZE        = (10, 10)
COMPOSITE_ALPHA = 0.85
BASEMAP_ALPHA   = 0.7

BASEMAP_SOURCE = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
GRID_INTERVAL_DEG = 0.5      # lat/lon tick spacing in decimal degrees


# =============================================================================
# Wavelength -> band label mapping
# =============================================================================

def wavelength_to_band_label(wavelength_nm, wl_min, wl_max, n_bands):
    """
    Map a target wavelength (nm) to the closest EMIT band label, assuming
    even spacing across [wl_min, wl_max] over n_bands. Band labels follow
    the 'b001'..'b285' convention used in this cube's band_labels attr.
    """
    wavelengths = np.linspace(wl_min, wl_max, n_bands)
    idx = int(np.argmin(np.abs(wavelengths - wavelength_nm)))
    actual_wl = wavelengths[idx]
    label = f"b{idx + 1:03d}"
    return label, actual_wl


# =============================================================================
# Load data
# =============================================================================

def load_bands(nc_path, var_name, band_names):
    ds = xr.open_dataset(nc_path)
    da = ds[var_name]

    band_dim = "band" if "band" in da.dims else f"{var_name}_band"
    labels   = da.attrs["band_labels"].split(", ")

    arrays = {}
    for name in band_names:
        if name not in labels:
            ds.close()
            raise ValueError(f"Band '{name}' not found in band_labels: {labels}")
        idx = labels.index(name)
        arrays[name] = da.isel({band_dim: idx}).values

    x = da.coords["x"].values
    y = da.coords["y"].values
    crs_wkt = ds["spatial_ref"].attrs["crs_wkt"]
    ds.close()
    return arrays, x, y, crs_wkt


# =============================================================================
# Ratio composite construction
# =============================================================================

def safe_ratio(numerator, denominator):
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = numerator / denominator
    ratio[~np.isfinite(ratio)] = np.nan
    return ratio


def stretch_to_uint8(arr, low_pct=2, high_pct=98):
    valid = arr[np.isfinite(arr)]
    if valid.size == 0:
        return np.zeros_like(arr, dtype=np.uint8)

    lo, hi = np.percentile(valid, [low_pct, high_pct])
    stretched = np.clip((arr - lo) / (hi - lo), 0, 1)
    stretched = np.nan_to_num(stretched, nan=0.0)
    return (stretched * 255).astype(np.uint8)


def build_rgb_composite(bands_dict, ratio_bands_by_label):
    """
    ratio_bands_by_label : dict {channel: (num_label, den_label)}
    """
    channels = {}
    nodata_mask = None

    for channel, (num_label, den_label) in ratio_bands_by_label.items():
        ratio = safe_ratio(bands_dict[num_label], bands_dict[den_label])
        channels[channel] = stretch_to_uint8(ratio)

        chan_nodata = ~np.isfinite(ratio)
        nodata_mask = chan_nodata if nodata_mask is None else (nodata_mask | chan_nodata)

    rgba = np.zeros((*channels["R"].shape, 4), dtype=np.uint8)
    rgba[..., 0] = channels["R"]
    rgba[..., 1] = channels["G"]
    rgba[..., 2] = channels["B"]
    rgba[..., 3] = np.where(nodata_mask, 0, 255).astype(np.uint8)

    return rgba


# =============================================================================
# Basemap helper
# =============================================================================

def add_basemap(ax, crs_wkt, source, alpha=0.7):
    """
    Add a satellite basemap to an axis already in the data's native CRS.
    Uses contextily's plotting submodule directly (bypasses the package's
    top-level __init__, which otherwise pulls in geopy/aiohttp).
    """
    cx_plot.add_basemap(ax, crs=crs_wkt, source=source, attribution_size=6, alpha=alpha)


# =============================================================================
# Graticule
# =============================================================================

def add_latlon_ticks(ax, crs_wkt, x_bounds, y_bounds, interval_deg=0.5):
    """
    Label the bottom and left axes with lat/lon coordinates instead of
    UTM easting/northing, without drawing any interior grid lines.
    """
    transformer_fwd = Transformer.from_crs("EPSG:4326", crs_wkt, always_xy=True)
    transformer_inv = Transformer.from_crs(crs_wkt, "EPSG:4326", always_xy=True)

    x_min, x_max = x_bounds
    y_min, y_max = y_bounds

    corners_x = [x_min, x_max, x_min, x_max]
    corners_y = [y_min, y_min, y_max, y_max]
    lons, lats = transformer_inv.transform(corners_x, corners_y)

    lon_min, lon_max = min(lons), max(lons)
    lat_min, lat_max = min(lats), max(lats)

    lon_ticks_deg = np.arange(
        np.ceil(lon_min / interval_deg) * interval_deg, lon_max, interval_deg,
    )
    lat_ticks_deg = np.arange(
        np.ceil(lat_min / interval_deg) * interval_deg, lat_max, interval_deg,
    )

    lon_tick_x, _ = transformer_fwd.transform(lon_ticks_deg, np.full_like(lon_ticks_deg, lat_min))
    _, lat_tick_y = transformer_fwd.transform(np.full_like(lat_ticks_deg, lon_min), lat_ticks_deg)

    def fmt_lon(v):
        return f"{abs(v):.1f}°{'E' if v >= 0 else 'W'}"

    def fmt_lat(v):
        return f"{abs(v):.1f}°{'N' if v >= 0 else 'S'}"

    ax.set_xticks(lon_tick_x)
    ax.set_xticklabels([fmt_lon(v) for v in lon_ticks_deg])
    ax.set_yticks(lat_tick_y)
    ax.set_yticklabels([fmt_lat(v) for v in lat_ticks_deg])

    return lon_ticks_deg, lat_ticks_deg


# =============================================================================
# North arrow
# =============================================================================

def add_north_arrow(ax, loc=(0.98, 0.98), size=0.06):
    x, y = loc
    ax.annotate(
        "N",
        xy=(x, y), xytext=(x, y - size),
        xycoords="axes fraction", textcoords="axes fraction",
        ha="center", va="center",
        fontsize=14, fontweight="bold", color="black",
        arrowprops=dict(arrowstyle="-|>", color="black", lw=2),
        zorder=10,
    )


# =============================================================================
# Main figure builder
# =============================================================================

def build_mineral_preview():
    # Resolve target wavelengths to actual band labels
    ratio_bands_by_label = {}
    resolved_info = {}  # for legend text
    for channel, (num_wl, den_wl) in RATIO_BANDS.items():
        num_label, num_actual = wavelength_to_band_label(num_wl, WAVELENGTH_MIN, WAVELENGTH_MAX, N_BANDS)
        den_label, den_actual = wavelength_to_band_label(den_wl, WAVELENGTH_MIN, WAVELENGTH_MAX, N_BANDS)
        ratio_bands_by_label[channel] = (num_label, den_label)
        resolved_info[channel] = (num_label, num_actual, den_label, den_actual)
        print(f"[*] Channel {channel}: {num_label} ({num_actual:.0f}nm) / "
              f"{den_label} ({den_actual:.0f}nm)")

    needed_bands = sorted(set(b for pair in ratio_bands_by_label.values() for b in pair))
    bands_dict, x, y, crs_wkt = load_bands(NC_PATH, VAR_NAME, needed_bands)

    rgba = build_rgb_composite(bands_dict, ratio_bands_by_label)

    dx = abs(x[1] - x[0])
    dy = abs(y[1] - y[0])
    extent = [x.min() - dx / 2, x.max() + dx / 2, y.min() - dy / 2, y.max() + dy / 2]

    fig, ax = plt.subplots(figsize=FIGSIZE)

    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal")
    add_basemap(ax, crs_wkt, BASEMAP_SOURCE, alpha=BASEMAP_ALPHA)

    ax.imshow(
        rgba,
        extent=extent,
        alpha=COMPOSITE_ALPHA,
        zorder=3,
        origin="upper",
    )

    add_latlon_ticks(ax, crs_wkt, (extent[0], extent[1]), (extent[2], extent[3]), GRID_INTERVAL_DEG)
    add_north_arrow(ax)
    ax.add_artist(ScaleBar(1, location="lower left", box_alpha=0.7))

    r_num, r_den = ratio_bands_by_label["R"]
    g_num, g_den = ratio_bands_by_label["G"]
    b_num, b_den = ratio_bands_by_label["B"]

    legend_text = (
        f"R = Clay/Al-OH ({r_num}/{r_den})\n"
        f"G = Carbonate ({g_num}/{g_den})\n"
        f"B = Iron oxide ({b_num}/{b_den})"
    )
    ax.text(
        0.02, 0.06, legend_text,
        transform=ax.transAxes,
        fontsize=8, va="bottom", ha="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        zorder=10,
    )

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("EMIT Mineral Ratio Composite", fontsize=13, fontweight="bold")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=0.06, right=0.995, bottom=0.09, top=0.96)
    fig.savefig(OUT_PATH, dpi=DPI)
    plt.close(fig)
    print(f"[✓] Saved → {OUT_PATH}")


if __name__ == "__main__":
    build_mineral_preview()