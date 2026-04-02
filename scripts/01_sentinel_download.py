import yaml
import os
import pystac_client
import planetary_computer
import geopandas as gpd
from shapely.geometry import shape, Polygon
from shapely.ops import transform
import pyproj
from tqdm import tqdm
import requests
from pathlib import Path


def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)


def to_equal_area(geom, src_crs="EPSG:4326", dst_crs="EPSG:6933"):
    """
    Reprojects a shapely geometry to an equal-area CRS for reliable area comparisons.
    Uses the modern pyproj Transformer API (pyproj >= 2).
    """
    transformer = pyproj.Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    return transform(transformer.transform, geom)


def main(config_file):
    # 1. Path Setup
    config_path = Path(config_file).resolve()
    if not config_path.exists():
        print(f"[!] Error: Config not found at {config_path}")
        return

    config = load_config(config_path)
    project_root = config_path.parent.parent

    # 2. AOI Resolution
    aoi_rel = config['paths']['aoi']
    aoi_file = Path(aoi_rel) if os.path.isabs(aoi_rel) else project_root / aoi_rel

    if not aoi_file.exists():
        print(f"[!] AOI file missing: {aoi_file}")
        return

    # Handle zipped shapefiles via fiona URI scheme
    aoi_uri = f"zip://{aoi_file.as_posix()}" if aoi_file.suffix.lower() == '.zip' else aoi_file.as_posix()

    print(f"[*] Reading AOI: {aoi_uri}")
    try:
        aoi_gdf = gpd.read_file(aoi_uri, engine='fiona').to_crs(epsg=4326)

        if hasattr(aoi_gdf.geometry, 'union_all'):
            aoi_geom = aoi_gdf.geometry.union_all()
        else:
            aoi_geom = aoi_gdf.geometry.unary_union

    except Exception as e:
        print(f"[!] GeoPandas read failed: {e}")
        return

    # 3. Connect to Microsoft Planetary Computer
    print("[*] Connecting to Microsoft Planetary Computer...")
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )

    # 4. Search Collection
    params = config['sensors']['landsat']
    max_cloud = params['max_cloud_cover']
    print(f"[*] Searching {params['short_name']} | Platform: {params['platform']} | Cloud < {max_cloud}%")

    search = catalog.search(
        collections=[params['short_name']],
        bbox=list(aoi_geom.bounds),
        datetime=f"{params['start_date']}/{params['end_date']}",
        query={
            "eo:cloud_cover": {"lt": max_cloud},
            "platform":       {"eq": params['platform']}
        }
    )

    items = list(search.items())
    if not items:
        print("[!] No scenes found matching criteria.")
        return

    print(f"[*] Found {len(items)} scenes. Optimizing coverage...")

    # 5. Greedy Spatial Optimization
    # Sort by cloud cover (lowest first)
    sorted_items = sorted(items, key=lambda x: x.properties.get("eo:cloud_cover", 100))

    # Deduplicate by path/row — keep only the lowest-cloud scene per tile footprint
    seen_tiles = set()
    unique_items = []
    for item in sorted_items:
        tile_id = item.properties.get("s2:mgrs_tile")  # e.g. "37QDD"
        if tile_id not in seen_tiles:
            seen_tiles.add(tile_id)
            unique_items.append(item)
    sorted_items = unique_items

    # Project AOI to equal-area CRS once; all spatial comparisons done here
    aoi_geom_ea = to_equal_area(aoi_geom)
    accumulated_ea = Polygon()
    min_contribution = aoi_geom_ea.area * 0.0001

    selected_items = []

    for item in sorted_items:
        # Stop early if AOI is fully covered
        if accumulated_ea.covers(aoi_geom_ea):
            break

        item_geom = shape(item.geometry)
        intersection_ea = to_equal_area(item_geom.intersection(aoi_geom))

        if intersection_ea.is_empty:
            continue

        new_area = intersection_ea.difference(accumulated_ea)

        # 1 m² minimum threshold in EPSG:6933 equal-area projection
        if not new_area.is_empty and new_area.area > min_contribution:
            selected_items.append(item)
            accumulated_ea = accumulated_ea.union(intersection_ea).buffer(0)

    if not selected_items:
        print("[!] Optimization resulted in 0 scenes selected.")
        return
    
    # Calculate and print AOI coverage percentage
    covered_area = accumulated_ea.intersection(aoi_geom_ea).area
    total_area = aoi_geom_ea.area
    coverage_pct = (covered_area / total_area) * 100 if total_area > 0 else 0.0
    print(f"[*] Selected scenes cover {coverage_pct:.1f}% of the AOI.")

    # 6. Download (Signed URLs via Planetary Computer)
    out_dir = Path(config['paths']['output_base'])
    out_dir = out_dir if out_dir.is_absolute() else project_root / out_dir
    final_dir = out_dir / params['output_folder']
    final_dir.mkdir(parents=True, exist_ok=True)

    # Asset keys to download — configurable per collection
    band_keys   = set(params['band_keys'])
    qa_prefixes = params['qa_prefixes']

    print(f"[*] Selected {len(selected_items)} scenes for optimal coverage.")
    print(f"[*] Downloading to: {final_dir}")

    for item in tqdm(selected_items, desc=f"Downloading {params['short_name']}", unit="scene"):
        scene_dir = final_dir / item.id
        scene_dir.mkdir(exist_ok=True)

        for key, asset in item.assets.items():
            is_band = key in band_keys
            is_meta = any(pref in key.lower() for pref in qa_prefixes)

            if is_band or is_meta:
                filename = Path(asset.href.split('?')[0]).name
                save_path = scene_dir / filename

                if not save_path.exists():
                    try:
                        with requests.get(asset.href, stream=True) as r:
                            r.raise_for_status()
                            with open(save_path, 'wb') as f:
                                for chunk in r.iter_content(chunk_size=1024 * 1024):
                                    f.write(chunk)
                    except Exception as e:
                        print(f"\n[!] Error downloading {filename}: {e}")

    print(f"\n[*] Download complete: {final_dir}")


if __name__ == "__main__":
    current_dir = Path(__file__).resolve().parent
    config_path = current_dir.parent / "config" / "01_sentinel_download.yaml"
    main(config_file=str(config_path))