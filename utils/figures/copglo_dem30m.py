import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LightSource
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import FancyBboxPatch
from matplotlib_scalebar.scalebar import ScaleBar
from pathlib import Path
from pyproj import Transformer
import contextily.plotting as cx_plot


# =============================================================================
# Config
# =============================================================================
NC_PATH    = Path("04_cube_data/copglo_dem30m.nc")
OUT_PATH   = Path("figures/copglo_dem30m_hillshade.png")

VAR_NAME   = "dem"

DPI            = 300
FIGSIZE        = (10, 10)
DEM_ALPHA      = 0.65          # DEM color overlay transparency (over hillshade)
HILLSHADE_AZIMUTH  = 315        # degrees, sun direction
HILLSHADE_ALTITUDE = 45         # degrees, sun elevation
VERT_EXAG          = 1.0        # hillshade vertical exaggeration

CMAP       = "terrain"
BASEMAP_SOURCE = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"

GRID_INTERVAL_DEG = 0.5         # lat/lon tick spacing in decimal degrees


# =============================================================================
# Load data
# =============================================================================

def load_dem(nc_path, var_name):
    ds = xr.open_dataset(nc_path)
    da = ds[var_name]

    x = da.coords["x"].values
    y = da.coords["y"].values
    arr = da.values  # (rows, cols), float32, NaN = nodata

    crs_wkt = ds["spatial_ref"].attrs["crs_wkt"]
    ds.close()
    return arr, x, y, crs_wkt


# =============================================================================
# Hillshade
# =============================================================================

def compute_hillshade(arr, dx, dy, azimuth, altitude, vert_exag):
    """
    Compute a hillshade array using matplotlib's LightSource.
    NaNs are filled with the array mean for the shading calculation only
    (so edge/nodata pixels don't break the gradient), then masked back out.
    """
    nan_mask = np.isnan(arr)
    filled = np.where(nan_mask, np.nanmean(arr), arr)

    ls = LightSource(azdeg=azimuth, altdeg=altitude)
    # dx/dy must be positive cell sizes; LightSource expects them as such
    hillshade = ls.hillshade(filled, vert_exag=vert_exag, dx=abs(dx), dy=abs(dy))

    hillshade = np.where(nan_mask, np.nan, hillshade)
    return hillshade


# =============================================================================
# Basemap helper — fetch + reproject-aware extent
# =============================================================================

def add_basemap(ax, crs_wkt, source, alpha=0.7):
    """
    Add a satellite basemap to an axis already in the data's native CRS.
    Uses contextily's plotting submodule directly (bypasses the package's
    top-level __init__, which otherwise pulls in geopy/aiohttp).
    """
    cx_plot.add_basemap(ax, crs=crs_wkt, source=source, attribution_size=6, alpha=alpha)


# =============================================================================
# Graticule (lat/lon grid) over a projected axis
# =============================================================================

def add_latlon_ticks(ax, crs_wkt, x_bounds, y_bounds, interval_deg=0.5):
    """
    Label the bottom and left axes with lat/lon coordinates instead of
    UTM easting/northing, without drawing any interior grid lines.

    Approach: find the lon/lat range covering the UTM extent, snap to
    round degree intervals, then transform each tick value back into UTM
    along a representative edge (bottom edge for longitude ticks, left
    edge for latitude ticks) to get its correct UTM tick position.
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
        np.ceil(lon_min / interval_deg) * interval_deg,
        lon_max,
        interval_deg,
    )
    lat_ticks_deg = np.arange(
        np.ceil(lat_min / interval_deg) * interval_deg,
        lat_max,
        interval_deg,
    )

    # Longitude ticks: transform (lon, lat_min) pairs to UTM, take x
    lon_tick_x, _ = transformer_fwd.transform(lon_ticks_deg, np.full_like(lon_ticks_deg, lat_min))
    # Latitude ticks: transform (lon_min, lat) pairs to UTM, take y
    _, lat_tick_y = transformer_fwd.transform(np.full_like(lat_ticks_deg, lon_min), lat_ticks_deg)

    def fmt_lon(v):
        hemi = "E" if v >= 0 else "W"
        return f"{abs(v):.1f}°{hemi}"

    def fmt_lat(v):
        hemi = "N" if v >= 0 else "S"
        return f"{abs(v):.1f}°{hemi}"

    ax.set_xticks(lon_tick_x)
    ax.set_xticklabels([fmt_lon(v) for v in lon_ticks_deg])

    ax.set_yticks(lat_tick_y)
    ax.set_yticklabels([fmt_lat(v) for v in lat_ticks_deg])

    return lon_ticks_deg, lat_ticks_deg


# =============================================================================
# North arrow
# =============================================================================

def add_north_arrow(ax, loc=(0.98, 0.98), size=0.06):
    """Simple north arrow drawn in axis-fraction coordinates."""
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

def build_dem_preview():
    arr, x, y, crs_wkt = load_dem(NC_PATH, VAR_NAME)

    dx = abs(x[1] - x[0])
    dy = abs(y[1] - y[0])

    hillshade = compute_hillshade(
        arr, dx, dy, HILLSHADE_AZIMUTH, HILLSHADE_ALTITUDE, VERT_EXAG
    )

    extent = [x.min() - dx / 2, x.max() + dx / 2, y.min() - dy / 2, y.max() + dy / 2]

    fig, ax = plt.subplots(figsize=FIGSIZE)

    # 1. Satellite basemap (drawn first, bottom layer)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal")
    add_basemap(ax, crs_wkt, BASEMAP_SOURCE)

    # 2. Hillshade (greyscale, multiplies visually with basemap via alpha)
    ax.imshow(
        hillshade,
        extent=extent,
        cmap="gray",
        alpha=0.5,
        zorder=2,
        origin="upper",
    )

    # 3. Colored DEM on top, semi-transparent so hillshade + basemap show through
    im = ax.imshow(
        arr,
        extent=extent,
        cmap=CMAP,
        alpha=DEM_ALPHA,
        zorder=3,
        origin="upper",
    )

    # 4. Graticule
    add_latlon_ticks(ax, crs_wkt, (extent[0], extent[1]), (extent[2], extent[3]), GRID_INTERVAL_DEG)

    # 5. North arrow
    add_north_arrow(ax)

    # 6. Scale bar
    ax.add_artist(ScaleBar(1, location="lower left", box_alpha=0.7))

    # 7. Colorbar — placed inside the axes (lower-left, above the scale
    #    bar), matching the legend position used in the composite scripts.
    #    A semi-transparent white box is drawn behind it, matching
    #    ScaleBar's box_alpha look, since inset_axes has no such option.
    fig.subplots_adjust(left=0.06, right=0.995, bottom=0.09, top=0.96)
    cax = inset_axes(
        ax,
        width="27%",   # ~10% smaller than a typical 30% inset width
        height="3.6%", # ~10% smaller than a typical 4% inset height
        loc="lower left",
        bbox_to_anchor=(0.03, 0.12, 1, 1),
        bbox_transform=ax.transAxes,
        borderpad=0,
    )

    # Force a draw pass so all axes positions (including this inset) are
    # finalized before reading cax's bounding box below — otherwise the
    # background box can end up misaligned.
    fig.canvas.draw()

    cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
    cbar.set_label("Elevation (m, EGM2008)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    # Background box sized from the FULL tight bounding box of the
    # colorbar axes — this includes tick labels and the axis label text,
    # not just the bare axes rectangle (cax.get_position() alone would
    # leave the tick/label text uncovered below the box).
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    full_bbox = cax.get_tightbbox(renderer).transformed(fig.transFigure.inverted())
    pad_x, pad_y = 0.012, 0.012
    bg_box = FancyBboxPatch(
        (full_bbox.x0 - pad_x, full_bbox.y0 - pad_y),
        full_bbox.width + 2 * pad_x,
        full_bbox.height + 2 * pad_y,
        boxstyle="round,pad=0,rounding_size=0.01",
        transform=fig.transFigure,
        facecolor="white",
        edgecolor="none",
        alpha=0.7,
        zorder=5,
    )
    fig.add_artist(bg_box)
    cax.set_zorder(6)

    # 8. Cosmetics
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Copernicus GLO-30 DEM", fontsize=13, fontweight="bold")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=DPI)
    plt.close(fig)
    print(f"[✓] Saved → {OUT_PATH}")


if __name__ == "__main__":
    build_dem_preview()