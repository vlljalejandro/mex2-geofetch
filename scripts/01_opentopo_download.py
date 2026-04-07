import yaml
import requests
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.merge import merge
from pathlib import Path
from tqdm import tqdm
from math import ceil


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
# OpenTopography hard limit per request (km²). Kept slightly under 450 000
# so floating-point bbox arithmetic never accidentally nudges us over.
OT_AREA_LIMIT_KM2 = 440_000


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def load_config(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def bbox_area_km2(west: float, south: float, east: float, north: float) -> float:
    """Rough surface area of a WGS-84 bounding box in km²."""
    lat_mid = (south + north) / 2.0
    km_per_deg_lat = 111.32
    km_per_deg_lon = 111.32 * np.cos(np.radians(lat_mid))
    return abs(east - west) * km_per_deg_lon * abs(north - south) * km_per_deg_lat


def subdivide_bbox(
    west: float, south: float, east: float, north: float, limit_km2: float
) -> list[tuple[float, float, float, float]]:
    """
    Split a bounding box into the smallest regular grid whose individual
    cells each fit under *limit_km2*.  Returns a flat list of
    (west, south, east, north) tuples.
    """
    total_area = bbox_area_km2(west, south, east, north)
    if total_area <= limit_km2:
        return [(west, south, east, north)]

    # Number of cells needed along each axis (keep tiles roughly square)
    n_cells = ceil(total_area / limit_km2)
    n_cols = ceil(np.sqrt(n_cells * (east - west) / max(north - south, 1e-9)))
    n_rows = ceil(n_cells / n_cols)

    lon_edges = np.linspace(west, east, n_cols + 1)
    lat_edges = np.linspace(south, north, n_rows + 1)

    tiles = []
    for i in range(n_rows):
        for j in range(n_cols):
            tile = (
                lon_edges[j],      # west
                lat_edges[i],      # south
                lon_edges[j + 1],  # east
                lat_edges[i + 1],  # north
            )
            tiles.append(tile)

    return tiles


def download_tile(
    base_url: str,
    api_key: str,
    demtype: str,
    output_format: str,
    west: float,
    south: float,
    east: float,
    north: float,
    save_path: Path,
) -> bool:
    """Download a single DEM tile.  Returns True on success."""
    params = {
        "demtype":      demtype,
        "south":        south,
        "north":        north,
        "west":         west,
        "east":         east,
        "outputFormat": output_format,
        "API_Key":      api_key,
    }

    try:
        response = requests.get(base_url, params=params, stream=True, timeout=600)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))
        with open(save_path, "wb") as fh, tqdm(
            desc=save_path.name,
            total=total_size,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
            leave=False,
        ) as bar:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                fh.write(chunk)
                bar.update(len(chunk))
        return True

    except requests.exceptions.HTTPError as exc:
        print(f"    [!] HTTP error: {exc.response.text}")
    except Exception as exc:
        print(f"    [!] Unexpected error: {exc}")

    return False


def mosaic_tiles(tile_paths: list[Path], output_path: Path) -> None:
    """Merge GeoTIFF tiles into a single file using rasterio."""
    print(f"[*] Mosaicking {len(tile_paths)} tile(s) → {output_path.name} ...")
    datasets = [rasterio.open(p) for p in tile_paths]
    try:
        mosaic_data, mosaic_transform = merge(datasets)
        profile = datasets[0].profile.copy()
        profile.update(
            {
                "height":    mosaic_data.shape[1],
                "width":     mosaic_data.shape[2],
                "transform": mosaic_transform,
                "compress":  "lzw",        # lossless, keeps file size down
                "tiled":     True,
                "blockxsize": 512,
                "blockysize": 512,
                "BIGTIFF":   "YES",        # required when output exceeds ~4 GB
            }
        )
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(mosaic_data)
        print(f"[*] Mosaic saved: {output_path}")
    finally:
        for ds in datasets:
            ds.close()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def download_opentopo_dem(config_file: str) -> None:
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
        print(f"[!] AOI file missing: {aoi_file}")
        return

    print(f"[*] Reading AOI: {aoi_file}")
    try:
        aoi_gdf  = gpd.read_file(aoi_file).to_crs(epsg=4326)
        aoi_geom = (
            aoi_gdf.geometry.union_all()
            if hasattr(aoi_gdf.geometry, "union_all")
            else aoi_gdf.geometry.unary_union
        )
        west, south, east, north = aoi_geom.bounds
    except Exception as exc:
        print(f"[!] GeoPandas error: {exc}")
        return

    total_area = bbox_area_km2(west, south, east, north)
    print(f"[*] AOI bounding box area ≈ {total_area:,.0f} km²")

    # 3. Output directory
    dem_params  = config["dem_params"]
    out_dir     = Path(config["paths"]["output_base"])
    out_dir     = out_dir if out_dir.is_absolute() else project_root / out_dir
    tiles_dir   = out_dir / "tiles"
    tiles_dir.mkdir(parents=True, exist_ok=True)

    final_path  = out_dir / f"{dem_params['demtype']}_subset.{dem_params['output_format'].lower()}"

    # 4. Build tile grid
    tiles = subdivide_bbox(west, south, east, north, OT_AREA_LIMIT_KM2)
    print(f"[*] Splitting into {len(tiles)} tile(s) "
          f"(limit {OT_AREA_LIMIT_KM2:,} km² each)")

    # 5. Download each tile
    tile_paths   = []
    failed_tiles = []

    for idx, (t_west, t_south, t_east, t_north) in enumerate(tiles, start=1):
        tile_name = (
            f"{dem_params['demtype']}_tile_{idx:03d}"
            f"_W{t_west:.4f}_S{t_south:.4f}_E{t_east:.4f}_N{t_north:.4f}"
            f".{dem_params['output_format'].lower()}"
        )
        tile_path = tiles_dir / tile_name

        # Skip already-downloaded tiles (useful for resuming interrupted runs)
        if tile_path.exists() and tile_path.stat().st_size > 0:
            print(f"  [{idx}/{len(tiles)}] Skipping (cached): {tile_name}")
            tile_paths.append(tile_path)
            continue

        tile_area = bbox_area_km2(t_west, t_south, t_east, t_north)
        print(f"  [{idx}/{len(tiles)}] Downloading ≈{tile_area:,.0f} km²: {tile_name}")

        success = download_tile(
            base_url      = config["api"]["base_url"],
            api_key       = config["auth"]["api_key"],
            demtype       = dem_params["demtype"],
            output_format = dem_params["output_format"],
            west          = t_west,
            south         = t_south,
            east          = t_east,
            north         = t_north,
            save_path     = tile_path,
        )

        if success:
            tile_paths.append(tile_path)
        else:
            failed_tiles.append(idx)

    if failed_tiles:
        print(f"\n[!] {len(failed_tiles)} tile(s) failed: {failed_tiles}")
        print("[!] Mosaicking will proceed with the tiles that downloaded successfully.")

    if not tile_paths:
        print("[!] No tiles downloaded. Aborting.")
        return

    # 6. Mosaic (skip if only one tile)
    if len(tile_paths) == 1:
        import shutil
        shutil.copy2(tile_paths[0], final_path)
        print(f"[*] Single tile copied to: {final_path}")
    else:
        mosaic_tiles(tile_paths, final_path)

    print(f"\n[✓] Done. Final DEM → {final_path}")


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    current_dir = Path(__file__).resolve().parent
    config_path = current_dir.parent / "config" / "01_opentopo_download.yaml"
    download_opentopo_dem(str(config_path))