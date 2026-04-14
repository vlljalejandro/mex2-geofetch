import yaml
import requests
import geopandas as gpd
import numpy as np
import rasterio
import shutil
from rasterio.merge import merge
from rasterio.transform import from_origin
from pathlib import Path
from tqdm import tqdm
from math import floor


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
TILE_DEG        = 5          # GGMplus tiles are 5° × 5°
POINTS_PER_AXIS = 2500       # grid points per tile side at 0.002° resolution
GRID_RES        = 0.002      # degrees per cell

# Maps the server subdirectory name → actual file extension.
# (They differ for 'geoid': subdir=geoid, ext=ha)
FUNCTIONAL_EXT = {
    "ga":    "ga",   # free-fall gravity accelerations
    "dg":    "dg",   # gravity disturbances
    "xi":    "xi",   # North-South deflection of the vertical
    "eta":   "eta",  # East-West deflection of the vertical
    "geoid": "ha",   # geoid / quasigeoid heights
}

# Divisors to convert raw integer values → physical units
SCALE = {
    "ga":    10.0,    # 0.1 mGal   → mGal
    "dg":    10.0,    # 0.1 mGal   → mGal
    "xi":    10.0,    # 0.1 arcsec → arcsec
    "eta":   10.0,    # 0.1 arcsec → arcsec
    "geoid": 1000.0,  # mm         → m
}
UNITS = {
    "ga":    "mGal",
    "dg":    "mGal",
    "xi":    "arcsec",
    "eta":   "arcsec",
    "geoid": "m",
}
NODATA_FLOAT = -9999.0
# nodata sentinels are detected per-file in binary_to_geotiff:
#   int32 files → -2147483648,  int16 files → -32768


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

def load_config(config_path: Path) -> dict:
    with open(config_path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# -----------------------------------------------------------------------------
# Tile helpers
# -----------------------------------------------------------------------------

def overlapping_tiles(
    west: float, south: float, east: float, north: float
) -> list[tuple[int, int]]:
    """Return (lat_sw, lon_sw) pairs for every 5°×5° GGMplus tile that
    overlaps the WGS-84 bounding box.

    GGMplus coverage is limited to latitudes ±60° and land / near-coastal areas.
    """
    south = max(south, -60.0)
    north = min(north,  60.0)

    lat0 = int(floor(south / TILE_DEG) * TILE_DEG)
    lat1 = int(floor(north / TILE_DEG) * TILE_DEG)
    lon0 = int(floor(west  / TILE_DEG) * TILE_DEG)
    lon1 = int(floor(east  / TILE_DEG) * TILE_DEG)

    tiles = []
    lat = lat0
    while lat <= lat1:
        lon = lon0
        while lon <= lon1:
            tiles.append((lat, lon))
            lon += TILE_DEG
        lat += TILE_DEG
    return tiles


def tile_stem(lat_sw: int, lon_sw: int) -> str:
    """Build the GGMplus filename stem, e.g. (50, 5) → 'N50E005'."""
    ns = "N" if lat_sw >= 0 else "S"
    ew = "E" if lon_sw >= 0 else "W"
    return f"{ns}{abs(lat_sw):02d}{ew}{abs(lon_sw):03d}"


def tile_transform(lat_sw: int, lon_sw: int):
    """Rasterio affine transform for a tile whose SW corner sits at (lat_sw, lon_sw).

    GGMplus grid points run from (lon_sw+0.001, lat_sw+0.001) to
    (lon_sw+4.999, lat_sw+4.999) — i.e. they do NOT include the tile edges.

    rasterio's from_origin expects the upper-left *corner* of the upper-left
    pixel, which is:
        lon_sw + 0.001 − 0.001 = lon_sw
        lat_sw + 4.999 + 0.001 = lat_sw + 5
    """
    return from_origin(lon_sw, lat_sw + TILE_DEG, GRID_RES, GRID_RES)


# -----------------------------------------------------------------------------
# Download
# -----------------------------------------------------------------------------

def download_binary(url: str, save_path: Path) -> bool:
    """Fetch a binary file to disk with a progress bar. Returns True on success."""
    try:
        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        with open(save_path, "wb") as fh, tqdm(
            desc=save_path.name,
            total=total,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
            leave=False,
        ) as bar:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                fh.write(chunk)
                bar.update(len(chunk))
        return True
    except requests.HTTPError as exc:
        print(f"    [!] HTTP {exc.response.status_code}: {url}")
    except Exception as exc:
        print(f"    [!] {exc}")
    return False


# -----------------------------------------------------------------------------
# Binary → GeoTIFF conversion
# -----------------------------------------------------------------------------

def binary_to_geotiff(
    bin_path: Path,
    tif_path: Path,
    lat_sw: int,
    lon_sw: int,
    functional: str,
) -> None:
    """Convert a GGMplus raw binary tile to a float32 GeoTIFF.

    Binary layout:
      • dtype  : auto-detected from file size — int32 (e.g. ga) or int16 (e.g. dg)
      • endian : little-endian (data produced on x86-64 Linux)
      • order  : row-major, rows run S → N, columns W → E
      • size   : 2500 × 2500 values per tile  (5° / 0.002° = 2500)
    """
    scale     = SCALE[functional]
    n_vals    = POINTS_PER_AXIS * POINTS_PER_AXIS   # 6 250 000 expected values
    file_size = bin_path.stat().st_size

    if file_size == n_vals * 4:
        dtype    = "<i4"        # int32 — e.g. ga
        nodata_r = -2147483648
    elif file_size == n_vals * 2:
        dtype    = "<i2"        # int16 — e.g. dg
        nodata_r = -32768
    else:
        raise ValueError(
            f"Unexpected file size {file_size:,} B for {bin_path.name}. "
            f"Expected {n_vals*2:,} (int16) or {n_vals*4:,} (int32)."
        )

    raw  = np.fromfile(bin_path, dtype=dtype)
    # Data is stored column-major: latitude varies fastest (S→N), then longitude (W→E).
    # Reshape with Fortran order so rows=lat, cols=lon, then flip N→S for rasterio.
    grid = raw.reshape(POINTS_PER_AXIS, POINTS_PER_AXIS, order="F").astype(np.float32)

    nodata_mask = (raw == nodata_r).reshape(POINTS_PER_AXIS, POINTS_PER_AXIS, order="F")
    grid = grid / scale
    grid[nodata_mask] = NODATA_FLOAT

    # Rows run S→N after Fortran reshape; rasterio wants N→S → flip vertically
    grid = np.flipud(grid)

    profile = {
        "driver":    "GTiff",
        "dtype":     "float32",
        "width":     POINTS_PER_AXIS,
        "height":    POINTS_PER_AXIS,
        "count":     1,
        "crs":       "EPSG:4326",
        "transform": tile_transform(lat_sw, lon_sw),
        "nodata":    NODATA_FLOAT,
        "compress":  "lzw",
        "tiled":     True,
        "blockxsize": 512,
        "blockysize": 512,
    }
    with rasterio.open(tif_path, "w", **profile) as dst:
        dst.write(grid, 1)


# -----------------------------------------------------------------------------
# Mosaic
# -----------------------------------------------------------------------------

def mosaic_tiles(tile_paths: list[Path], output_path: Path) -> None:
    """Merge GeoTIFF tiles into a single file."""
    print(f"[*] Mosaicking {len(tile_paths)} tile(s) → {output_path.name} ...")
    datasets = [rasterio.open(p) for p in tile_paths]
    try:
        mosaic_arr, mosaic_transform = merge(datasets, nodata=NODATA_FLOAT)
        profile = datasets[0].profile.copy()
        profile.update({
            "height":     mosaic_arr.shape[1],
            "width":      mosaic_arr.shape[2],
            "transform":  mosaic_transform,
            "compress":   "lzw",
            "tiled":      True,
            "blockxsize":  512,
            "blockysize":  512,
            "BIGTIFF":    "YES",   # needed when output exceeds ~4 GB
        })
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(mosaic_arr)
        print(f"[*] Mosaic saved: {output_path}")
    finally:
        for ds in datasets:
            ds.close()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def download_ggmplus(config_file: str) -> None:
    # 1. Config & paths
    config_path = Path(config_file).resolve()
    if not config_path.exists():
        print(f"[!] Config not found: {config_path}")
        return

    config       = load_config(config_path)
    project_root = config_path.parent.parent

    # 2. AOI → WGS-84 bounding box
    aoi_rel  = config["paths"]["aoi"]
    aoi_file = Path(aoi_rel) if Path(aoi_rel).is_absolute() else project_root / aoi_rel
    if not aoi_file.exists():
        print(f"[!] AOI file not found: {aoi_file}")
        return

    print(f"[*] Reading AOI: {aoi_file}")
    aoi_gdf  = gpd.read_file(aoi_file).to_crs(epsg=4326)
    aoi_geom = (
        aoi_gdf.geometry.union_all()
        if hasattr(aoi_gdf.geometry, "union_all")
        else aoi_gdf.geometry.unary_union
    )
    west, south, east, north = aoi_geom.bounds
    print(f"[*] AOI bbox  : W{west:.3f}  S{south:.3f}  E{east:.3f}  N{north:.3f}")

    # 3. Output directories
    out_base  = Path(config["paths"]["output_base"])
    out_base  = out_base if out_base.is_absolute() else project_root / out_base
    bin_dir   = out_base / "raw_binary"   # raw binary tiles
    png_dir   = out_base / "raw_png"      # preview PNGs
    tiles_dir = out_base / "raw_tif"   # per-tile GeoTIFFs
    final_dir = out_base / "mosaic"        # mosaicked outputs
    for d in (bin_dir, png_dir, tiles_dir, final_dir):
        d.mkdir(parents=True, exist_ok=True)

    # 4. Determine which tiles are needed
    params      = config["ggmplus_params"]
    base_url    = params.get("base_url", "http://ddfe.curtin.edu.au/gravitymodels/GGMplus/data")
    functionals = params["functionals"]
    tiles       = overlapping_tiles(west, south, east, north)

    print(f"[*] Tiles needed : {len(tiles)}")
    print(f"[*] Functionals  : {functionals}")
    print(f"[*] Total files  : {len(tiles) * len(functionals)}\n")

    # 5. Download & convert each functional
    for functional in functionals:
        ext = FUNCTIONAL_EXT[functional]
        print(f"── {functional}  ({UNITS[functional]}) ──────────────────────────")
        tif_paths = []

        for lat_sw, lon_sw in tiles:
            stem     = tile_stem(lat_sw, lon_sw)
            filename = f"{stem}.{ext}"
            url      = f"{base_url}/{functional}/{filename}"
            bin_path = bin_dir   / functional / filename
            png_path = png_dir   / functional / f"{filename}.png"
            tif_path = tiles_dir / functional / f"{stem}.tif"
            for p in (bin_path, png_path, tif_path):
                p.parent.mkdir(parents=True, exist_ok=True)

            # --- download binary (skip if cached) ---
            if bin_path.exists() and bin_path.stat().st_size > 0:
                print(f"  [cached] {filename}")
            else:
                print(f"  [↓]      {url}")
                if not download_binary(url, bin_path):
                    print(f"  [!]      Skipping {filename} (download failed)")
                    continue

            # --- download PNG preview (skip if cached, non-fatal if missing) ---
            if png_path.exists() and png_path.stat().st_size > 0:
                print(f"  [cached] {png_path.name}")
            else:
                png_url = f"{url}.png"
                print(f"  [↓]      {png_url}")
                download_binary(png_url, png_path)   # failure is non-fatal

            # --- convert to GeoTIFF (skip if cached) ---
            if not (tif_path.exists() and tif_path.stat().st_size > 0):
                print(f"  [→]      Converting → {tif_path.name}")
                binary_to_geotiff(bin_path, tif_path, lat_sw, lon_sw, functional)

            tif_paths.append(tif_path)

        if not tif_paths:
            print(f"  [!] No valid tiles for '{functional}'. Skipping.\n")
            continue

        # 6. Mosaic (or just copy if only one tile)
        final_path = final_dir / f"GGMplus_{functional}.tif"
        if len(tif_paths) == 1:
            shutil.copy2(tif_paths[0], final_path)
            print(f"[*] Single tile copied → {final_path}")
        else:
            mosaic_tiles(tif_paths, final_path)
        print()

    print("[✓] Done.")


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    current_dir = Path(__file__).resolve().parent
    config_path = current_dir.parent / "config" / "01_ggmplus_download.yaml"
    download_ggmplus(str(config_path))