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
NC_PATH    = Path("04_cube_data/sentinel_refl10m.nc")
OUT_PATH   = Path("figures/sentinel_refl10m_composite_ctruecolor.png")

VAR_NAME   = "refl"

# True/natural color composite — the three visible bands rendered directly,
# no ratios or index math. Looks like a normal aerial/satellite photo.
#   R = B04 (Red)
#   G = B03 (Green)
#   B = B02 (Blue)
TRUE_COLOR_BANDS = {
    "R": "B04",
    "G": "B03",
    "B": "B02",
}

DPI            = 300
FIGSIZE        = (10, 10)
COMPOSITE_ALPHA = 0.85
BASEMAP_ALPHA   = 0.7

BASEMAP_SOURCE = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
GRID_INTERVAL_DEG = 0.5      # lat/lon tick spacing in decimal degrees


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
# Composite construction
# =============================================================================


def stretch_to_uint8(arr, low_pct=2, high_pct=98):
    """
    Percentile-stretch a single index band to 0-255 uint8 for RGB display.
    Normalized-difference indices are bounded [-1, 1] in theory, but a
    percentile stretch still gives better visual contrast than a fixed
    [-1, 1] -> [0, 255] linear stretch, since most pixels cluster near 0.

    Kept entirely in float32 and computed in-place to avoid the implicit
    float64 upcast that np.clip((arr - lo) / (hi - lo), ...) would trigger
    when lo/hi are plain Python floats — at 36864x36864 that upcast alone
    needs ~10GB, which can exhaust memory on machines with 16GB RAM.
    """
    valid = arr[np.isfinite(arr)]
    if valid.size == 0:
        return np.zeros_like(arr, dtype=np.uint8)

    lo, hi = np.percentile(valid, [low_pct, high_pct])
    lo, hi = np.float32(lo), np.float32(hi)

    stretched = arr.astype(np.float32, copy=True)
    stretched -= lo
    stretched /= (hi - lo)
    np.clip(stretched, 0, 1, out=stretched)
    np.nan_to_num(stretched, nan=0.0, copy=False)
    return (stretched * 255).astype(np.uint8)


def build_rgb_composite(bands_dict, color_bands):
    """
    Build an (rows, cols, 4) RGBA uint8 array directly from raw reflectance
    bands (true color — no ratio/index math). Alpha channel is 0 where the
    source band is NaN, 255 elsewhere.

    Each channel array is deleted from bands_dict once stretched, to
    reduce peak memory on large rasters (e.g. the 36864x36864 Sentinel-2
    cube, where each band is ~5GB in float32).
    """
    channels = {}
    nodata_mask = None

    for channel, band_name in color_bands.items():
        arr = bands_dict[band_name]

        chan_nodata = ~np.isfinite(arr)
        nodata_mask = chan_nodata if nodata_mask is None else (nodata_mask | chan_nodata)

        channels[channel] = stretch_to_uint8(arr)
        del bands_dict[band_name]  # free the source band once stretched

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

def build_truecolor_preview():
    needed_bands = sorted(set(TRUE_COLOR_BANDS.values()))
    bands_dict, x, y, crs_wkt = load_bands(NC_PATH, VAR_NAME, needed_bands)

    rgba = build_rgb_composite(bands_dict, TRUE_COLOR_BANDS)

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

    legend_text = (
        f"R = {TRUE_COLOR_BANDS['R']}\n"
        f"G = {TRUE_COLOR_BANDS['G']}\n"
        f"B = {TRUE_COLOR_BANDS['B']}"
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
    ax.set_title("Sentinel-2 True Color Composite", fontsize=13, fontweight="bold")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=0.06, right=0.995, bottom=0.09, top=0.96)
    fig.savefig(OUT_PATH, dpi=DPI)
    plt.close(fig)
    print(f"[✓] Saved → {OUT_PATH}")


if __name__ == "__main__":
    build_truecolor_preview()