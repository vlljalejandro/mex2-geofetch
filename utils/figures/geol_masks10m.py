import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib_scalebar.scalebar import ScaleBar
from pathlib import Path
from pyproj import Transformer
import contextily.plotting as cx_plot


# =============================================================================
# Config
# =============================================================================
NC_PATH    = Path("04_cube_data/geol_masks10m.nc")
OUT_PATH   = Path("figures/geol_masks10m_rfam_preview.png")

VAR_NAME   = "rfam"

DPI             = 300
FIGSIZE         = (10, 10)     # legend now sits inside the map, no extra width needed
CATEGORY_ALPHA  = 0.8          # categorical overlay transparency over basemap
BASEMAP_ALPHA   = 0.7

# Manual color overrides by class label (RGBA, 0-1 floats). Used to force
# geologically meaningful colors regardless of the qualitative palette.
COLOR_OVERRIDES = {
    "MAFIC_VOLC": (0.15, 0.15, 0.15, 1.0),   # dark grey/near-black
}

BASEMAP_SOURCE = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
GRID_INTERVAL_DEG = 0.5      # lat/lon tick spacing in decimal degrees


# =============================================================================
# Load data
# =============================================================================

def load_categorical(nc_path, var_name):
    """
    Load a categorical variable along with its flag_values/flag_meanings.

    Returns
    -------
    arr      : 2D array, float32 codes with NaN as nodata
    x, y     : coordinate arrays
    crs_wkt  : CRS WKT string
    codes    : list of int, valid class codes (excludes nodata)
    labels   : list of str, matching class names
    """
    ds = xr.open_dataset(nc_path)
    da = ds[var_name]

    arr = da.values  # float32 in memory due to _FillValue decoding; NaN = nodata
    x = da.coords["x"].values
    y = da.coords["y"].values
    crs_wkt = ds["spatial_ref"].attrs["crs_wkt"]

    codes  = [int(c) for c in da.attrs["flag_values"]]
    labels = da.attrs["flag_meanings"].split()

    ds.close()
    return arr, x, y, crs_wkt, codes, labels


# =============================================================================
# Discrete color palette + manual RGBA construction
# =============================================================================

def build_color_palette(codes, cmap_name="tab20"):
    """
    Build a list of RGBA tuples (0-1 floats), one per class code, used both
    for the manually-rendered raster and for the legend patches.
    """
    n = len(codes)
    base_cmap = plt.get_cmap(cmap_name)

    if n <= base_cmap.N:
        colors = [base_cmap(i) for i in range(n)]
    else:
        colors = [plt.get_cmap("nipy_spectral")(i / (n - 1)) for i in range(n)]

    return colors


def apply_color_overrides(colors, labels, overrides):
    """
    Override specific class colors by label name, e.g. to force a
    geologically meaningful color (MAFIC_VOLC -> dark grey/black) instead
    of whatever the qualitative palette happened to assign.

    overrides : dict {label: (r, g, b, a)} with values in 0-1 floats
    """
    colors = list(colors)  # don't mutate the caller's list
    for label, rgba in overrides.items():
        if label in labels:
            colors[labels.index(label)] = rgba
    return colors


def build_rgba_from_codes(arr, codes, colors, alpha=0.8):
    """
    Manually map a categorical code array to a uint8 RGBA image, avoiding
    matplotlib's imshow(cmap=..., norm=...) path entirely.

    matplotlib's colormap application internally builds a float64 RGBA
    array the same shape as the input — at 36864x36864 that's ~40GB and
    will exhaust memory on most machines. Building uint8 RGBA ourselves
    keeps peak memory at roughly 1/8th of that.

    Parameters
    ----------
    arr    : 2D float32 array of class codes, NaN = nodata
    codes  : list of int, valid class codes (must match colors order)
    colors : list of RGBA tuples (0-1 floats), one per code
    alpha  : overlay transparency applied to valid (non-nodata) pixels

    Returns
    -------
    (rows, cols, 4) uint8 RGBA array
    """
    rows, cols = arr.shape
    rgba = np.zeros((rows, cols, 4), dtype=np.uint8)

    nodata_mask = ~np.isfinite(arr)

    # Round to nearest int and clip to a safe range before indexing, since
    # the array is float32 (NaN-capable) rather than integer.
    arr_int = np.where(nodata_mask, 0, np.round(arr)).astype(np.int32)

    code_to_index = {code: i for i, code in enumerate(codes)}
    alpha_uint8 = np.uint8(round(alpha * 255))

    for code, idx in code_to_index.items():
        mask = (arr_int == code) & ~nodata_mask
        r, g, b, _ = colors[idx]
        rgba[..., 0][mask] = np.uint8(round(r * 255))
        rgba[..., 1][mask] = np.uint8(round(g * 255))
        rgba[..., 2][mask] = np.uint8(round(b * 255))
        rgba[..., 3][mask] = alpha_uint8

    # nodata pixels stay (0, 0, 0, 0) — fully transparent, basemap shows through
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
# Lat/lon tick labels (no interior grid lines)
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

def build_rfam_preview():
    arr, x, y, crs_wkt, codes, labels = load_categorical(NC_PATH, VAR_NAME)

    dx = abs(x[1] - x[0])
    dy = abs(y[1] - y[0])
    extent = [x.min() - dx / 2, x.max() + dx / 2, y.min() - dy / 2, y.max() + dy / 2]

    colors = build_color_palette(codes)
    colors = apply_color_overrides(colors, labels, COLOR_OVERRIDES)
    rgba = build_rgba_from_codes(arr, codes, colors, alpha=CATEGORY_ALPHA)
    del arr  # free the original float32 array, no longer needed

    fig, ax = plt.subplots(figsize=FIGSIZE)

    # 1. Satellite basemap
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal")
    add_basemap(ax, crs_wkt, BASEMAP_SOURCE, alpha=BASEMAP_ALPHA)

    # 2. Pre-rendered categorical RGBA overlay (uint8, no colormap math at
    #    draw time — avoids the float64 blowup from imshow(cmap=..., norm=...))
    ax.imshow(
        rgba,
        extent=extent,
        zorder=3,
        origin="upper",
        interpolation="nearest",  # never interpolate categorical codes
    )

    # 3. Lat/lon ticks
    add_latlon_ticks(ax, crs_wkt, (extent[0], extent[1]), (extent[2], extent[3]), GRID_INTERVAL_DEG)

    # 4. North arrow
    add_north_arrow(ax)

    # 5. Scale bar
    ax.add_artist(ScaleBar(1, location="lower left", box_alpha=0.7))

    # 6. Legend — one color patch per class, positioned inside the map
    #    above the scale bar (consistent with the spectral composite scripts)
    legend_handles = [
        mpatches.Patch(facecolor=colors[i], edgecolor="black", linewidth=0.3, label=labels[i])
        for i in range(len(codes))
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower left",
        bbox_to_anchor=(0.0, 0.05),
        fontsize=7,
        title="Rock Family",
        title_fontsize=8,
        frameon=True,
        framealpha=0.85,
        facecolor="white",
        ncol=1 if len(codes) <= 10 else 2,
    )

    # 7. Cosmetics
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Harmonised Rock Family Classification", fontsize=13, fontweight="bold")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=0.06, right=0.995, bottom=0.09, top=0.96)
    fig.savefig(OUT_PATH, dpi=DPI)
    plt.close(fig)
    print(f"[✓] Saved → {OUT_PATH}")


if __name__ == "__main__":
    build_rfam_preview()