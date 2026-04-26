import yaml
import numpy as np
import geopandas as gpd
from pyproj import CRS
from shapely.ops import transform, unary_union
from shapely.geometry import box
import pyproj
from osgeo import gdal, osr
gdal.UseExceptions()
from pathlib import Path


# =============================================================================
# Shared Utilities
# =============================================================================

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def auto_utm_epsg(geom_4326):
    """
    Detects the best UTM zone EPSG code from a geometry in EPSG:4326.
    Uses the centroid longitude/latitude to determine zone and hemisphere.
    """
    centroid = geom_4326.centroid
    lon, lat = centroid.x, centroid.y
    zone = int((lon + 180) / 6) + 1
    epsg = 32600 + zone if lat >= 0 else 32700 + zone
    return epsg


def to_crs(geom, src_epsg, dst_epsg):
    """Reprojects a shapely geometry between two EPSG codes."""
    transformer = pyproj.Transformer.from_crs(src_epsg, dst_epsg, always_xy=True)
    return transform(transformer.transform, geom)


def best_grid_dims(width_m, height_m, pixel_size, target_px=256):
    """
    Given a fixed pixel size, finds the nearest nx and ny (number of pixels)
    that are both exactly divisible by 256 (target_px).

    Strategy:
      - Compute the raw pixel counts: nx_raw = width_m / pixel_size
      - Round each UP to the nearest multiple of 256
      - Recompute the adjusted physical extent from the rounded dims
      - The grid is then centred on the eroded AOI centroid in recommend()

    Returns (nx, ny, adj_width_m, adj_height_m).
    """
    block = target_px   # one 256-pixel block

    nx = int(np.floor(width_m  / pixel_size / block) * block)
    ny = int(np.floor(height_m / pixel_size / block) * block)

    adj_w = nx * pixel_size
    adj_h = ny * pixel_size

    return nx, ny, adj_w, adj_h


# =============================================================================
# Step 1 — Recommend pixel size
# =============================================================================

def recommend(config, project_root):
    """
    Reads the AOI, erodes it by the configured buffer, projects to UTM,
    computes grid dimensions, and recommends the best pixel size.
    Returns all computed parameters needed for Step 2.
    """
    # -- AOI --
    aoi_rel  = config['paths']['aoi']
    aoi_file = Path(aoi_rel) if Path(aoi_rel).is_absolute() else project_root / aoi_rel

    if not aoi_file.exists():
        print(f"[!] AOI file missing: {aoi_file}")
        return None

    aoi_uri = f"zip://{aoi_file.as_posix()}" if aoi_file.suffix.lower() == '.zip' else aoi_file.as_posix()
    print(f"[*] Reading AOI: {aoi_uri}")

    gdf = gpd.read_file(aoi_uri, engine='fiona').to_crs(epsg=4326)
    if hasattr(gdf.geometry, 'union_all'):
        aoi_geom_4326 = gdf.geometry.union_all()
    else:
        aoi_geom_4326 = gdf.geometry.unary_union

    # -- CRS: auto UTM or override --
    override_epsg = config['grid'].get('override_epsg')
    if override_epsg:
        utm_epsg = int(override_epsg)
        print(f"[*] CRS override: EPSG:{utm_epsg}")
    else:
        utm_epsg = auto_utm_epsg(aoi_geom_4326)
        utm_crs  = CRS.from_epsg(utm_epsg)
        print(f"[*] Auto-detected UTM: EPSG:{utm_epsg} ({utm_crs.name})")

    # -- Project to UTM --
    # Always reproject from EPSG:4326 (we already called .to_crs(epsg=4326) above)
    aoi_utm = to_crs(aoi_geom_4326, 4326, utm_epsg)

    # -- Erode inward/outward --
    buffer_m   = config['grid'].get('buffer_m', 1000)   # positive = inward, negative = outward
    eroded_utm = aoi_utm.buffer(-buffer_m)

    if eroded_utm.is_empty or eroded_utm.area == 0:
        direction = "inner" if buffer_m > 0 else "outer"
        print(f"[!] AOI collapsed after {abs(buffer_m)}m {direction} buffer.")
        print(f"    AOI area (UTM): {aoi_utm.area:,.0f} m²")
        return None

    minx, miny, maxx, maxy = eroded_utm.bounds
    width_m  = maxx - minx
    height_m = maxy - miny

    print(f"\n[*] Eroded AOI extent (UTM):")
    print(f"    Width  : {width_m:,.1f} m")
    print(f"    Height : {height_m:,.1f} m")

    # -- Fixed pixel size from config --
    ps        = config['grid']['pixel_size_m']
    target_px = config['grid'].get('tile_divisor', 256)
    print(f"[*] Pixel size (from config): {ps} m")
    print(f"[*] Tile divisor            : {target_px} px")

    nx, ny, adj_w, adj_h = best_grid_dims(width_m, height_m, pixel_size=ps, target_px=target_px)

    # Centre the adjusted extent on the eroded AOI centroid
    cx = (minx + maxx) / 2
    cy = (miny + maxy) / 2
    snap_minx = cx - adj_w / 2
    snap_miny = cy - adj_h / 2
    snap_maxx = cx + adj_w / 2
    snap_maxy = cy + adj_h / 2

    # -- Snap origin to whole-unit metres --
    # Floor minx/miny to the nearest whole pixel boundary so the geotransform
    # origin has no decimal component. nx/ny are then recomputed to fully
    # cover the centred extent after snapping.
    snap_minx = np.floor(snap_minx / ps) * ps
    snap_miny = np.floor(snap_miny / ps) * ps
    snap_maxx = snap_minx + adj_w
    snap_maxy = snap_miny + adj_h

    # Recompute dims after snap (should be identical, but guards against
    # any floating-point residual from the floor operation)
    nx = int(round((snap_maxx - snap_minx) / ps))
    ny = int(round((snap_maxy - snap_miny) / ps))

    print(f"\n{'='*55}")
    direction = "inward" if buffer_m >= 0 else "outward"
    print(f"  Buffer                 : {abs(buffer_m)} m {direction}")
    print(f"  Pixel size             : {ps} m  (from config)")
    print(f"  Tile divisor           : {target_px} px")
    print(f"  Grid dimensions        : {nx} x {ny} pixels")
    print(f"  Origin (snapped)       : {snap_minx:.0f}, {snap_miny:.0f} m  (whole-unit)")
    print(f"  Extent                 : {snap_maxx:.0f}, {snap_maxy:.0f} m")
    print(f"  Coverage               : {(snap_maxx-snap_minx)/1000:.2f} x {(snap_maxy-snap_miny)/1000:.2f} km")
    print(f"  Tile blocks            : {nx // target_px} x {ny // target_px} ({target_px}px each)")
    print(f"  Rounding removed       : -{width_m  - (snap_maxx-snap_minx):.1f} m W  -{height_m - (snap_maxy-snap_miny):.1f} m H")
    print(f"{'='*55}\n")

    return {
        "utm_epsg":  utm_epsg,
        "pixel_size": ps,
        "nx": nx,
        "ny": ny,
        "minx": snap_minx,
        "miny": snap_miny,
        "maxx": snap_maxx,
        "maxy": snap_maxy,
    }


# =============================================================================
# Step 2 — Create master grid
# =============================================================================

def create_grid(params, config, project_root):
    """
    Creates a GeoTIFF master grid populated entirely with 1s (uint8),
    using the parameters confirmed in Step 1.

    The grid:
      - Is in the auto-detected UTM CRS
      - Has exactly nx * ny pixels, both divisible by 256
      - Is filled with 1 (valid) — downstream align_raster.py reads
        0 as masked and 1 as valid, so this grid marks the full extent
        as valid with no interior masking.
    """
    out_rel  = config['paths']['output']
    out_file = Path(out_rel) if Path(out_rel).is_absolute() else project_root / out_rel
    out_file.parent.mkdir(parents=True, exist_ok=True)

    ps   = params['pixel_size']
    nx   = params['nx']
    ny   = params['ny']
    minx = params['minx']
    maxy = params['maxy']

    # Geotransform: (top-left x, pixel width, 0, top-left y, 0, -pixel height)
    geotransform = (minx, ps, 0, maxy, 0, -ps)

    driver = gdal.GetDriverByName("GTiff")
    ds = driver.Create(
        str(out_file),
        nx, ny,
        1,                      # single band
        gdal.GDT_Byte,          # uint8 — values are 0 or 1
        ["COMPRESS=LZW", "TILED=YES", "BLOCKXSIZE=256", "BLOCKYSIZE=256"]
    )

    ds.SetGeoTransform(geotransform)

    srs = osr.SpatialReference()
    srs.ImportFromEPSG(params['utm_epsg'])
    ds.SetProjection(srs.ExportToWkt())

    band = ds.GetRasterBand(1)
    band.Fill(1)                # all pixels = 1 (valid)
    band.SetNoDataValue(0)
    band.FlushCache()

    ds = None

    print(f"[+] Master grid saved: {out_file}")
    print(f"    EPSG:{params['utm_epsg']} | {nx} x {ny} px | {ps} m/px")
    print(f"    Tile blocks: {nx // 256} x {ny // 256} (256px each)")


# =============================================================================
# Entry Point
# =============================================================================

def main(config_file):
    config_path = Path(config_file).resolve()
    if not config_path.exists():
        print(f"[!] Config not found: {config_path}")
        return

    config       = load_config(config_path)
    project_root = config_path.parent.parent

    # Step 1 — Recommend
    params = recommend(config, project_root)
    if params is None:
        return

    # Step 2 — Confirm
    print("Proceed with this grid? [y/n]: ", end="", flush=True)
    answer = input().strip().lower()

    if answer != 'y':
        print("[*] Cancelled. Adjust config and re-run.")
        return

    # Step 3 — Create
    create_grid(params, config, project_root)


if __name__ == "__main__":
    current_dir = Path(__file__).resolve().parent
    config_path = current_dir.parent / "config" / "00_base_grid.yaml"
    main(config_file=str(config_path))