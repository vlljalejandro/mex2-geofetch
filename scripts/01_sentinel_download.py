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
from datetime import datetime, timezone


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


def scene_sort_key(item, reference_date):
    """
    Sort key that balances cloud cover with recency.
    Penalizes older scenes by 1 cloud-cover point per 30 days of age,
    so a cloud-free 2016 tile only wins if nothing newer can match it.
    """
    cloud = item.properties.get("eo:cloud_cover", 100)
    dt = datetime.fromisoformat(
        item.properties["datetime"].replace("Z", "+00:00")
    )
    age_days = (reference_date - dt).days
    age_penalty = age_days / 30  # 1 point per month of age
    return cloud + age_penalty


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
    params = config['sensors']['sentinel2']
    max_cloud = params['max_cloud_cover']
    platform_label = params.get('platform', 'Any')
    print(f"[*] Searching {params['short_name']} | Platform: {platform_label} | Cloud < {max_cloud}%")

    query_filter = {"eo:cloud_cover": {"lt": max_cloud}}
    if params.get('platform'):
        query_filter["platform"] = {"eq": params['platform']}

    start_date = str(params['start_date']).replace('/', '-')
    end_date = str(params['end_date']).replace('/', '-')

    search = catalog.search(
        collections=[params['short_name']],
        bbox=list(aoi_geom.bounds),
        datetime=f"{start_date}/{end_date}",
        query=query_filter
    )

    items = list(search.items())
    if not items:
        print("[!] No scenes found matching criteria.")
        return

    print(f"[*] Found {len(items)} scenes. Optimizing coverage...")

    # 5. Greedy Spatial Optimization
    # Sort by cloud cover + recency penalty so recent scenes are preferred
    # over older scenes with marginally lower cloud cover.
    reference_date = datetime.now(timezone.utc)
    sorted_items = sorted(items, key=lambda x: scene_sort_key(x, reference_date))

    aoi_geom_ea = to_equal_area(aoi_geom)
    accumulated_ea = Polygon()
    min_contribution = aoi_geom_ea.area * 0.0001

    selected_items = []

    for item in sorted_items:
        if accumulated_ea.covers(aoi_geom_ea):
            break

        # Use proj:geometry (actual valid-data footprint) instead of
        # item.geometry (theoretical grid cell) to avoid overclaiming
        # coverage on edge/cropped tiles.
        proj_geom = item.properties.get("proj:geometry")
        proj_epsg = item.properties.get("proj:epsg")

        if proj_geom and proj_epsg:
            item_geom_ea = to_equal_area(
                shape(proj_geom),
                src_crs=f"EPSG:{proj_epsg}",
                dst_crs="EPSG:6933"
            )
        else:
            item_geom_ea = to_equal_area(shape(item.geometry))

        intersection_ea = item_geom_ea.intersection(aoi_geom_ea)

        if intersection_ea.is_empty:
            continue

        new_area = intersection_ea.difference(accumulated_ea)

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
    print(f"[*] Selected {len(selected_items)} scenes covering {coverage_pct:.1f}% of the AOI.")

    # 6. Download (Signed URLs via Planetary Computer)
    out_dir = Path(config['paths']['output_base'])
    out_dir = out_dir if out_dir.is_absolute() else project_root / out_dir
    final_dir = out_dir / params['output_folder']
    final_dir.mkdir(parents=True, exist_ok=True)

    band_keys   = set(params['band_keys'])
    qa_prefixes = params['qa_prefixes']

    print(f"[*] Downloading to: {final_dir}")

    for item in tqdm(selected_items, desc=f"Downloading {params['short_name']}", unit="scene"):
        scene_dir = final_dir / item.id
        scene_dir.mkdir(exist_ok=True)

        for key, asset in item.assets.items():
            is_band = key in band_keys
            is_meta = any(pref in key.lower() for pref in qa_prefixes)

            if is_band or is_meta:
                extension = Path(asset.href.split('?')[0]).suffix
                if not extension:
                    extension = ".tif"
                filename = f"{item.id}_{key}{extension}"
                save_path = scene_dir / filename

                if not save_path.exists():
                    try:
                        signed_href = planetary_computer.sign(asset.href)

                        with requests.get(signed_href, stream=True) as r:
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