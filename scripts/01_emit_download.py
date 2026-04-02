import yaml
import os
import earthaccess
import geopandas as gpd
from shapely.geometry import shape, Polygon
from shapely.ops import transform
import pyproj
from tqdm import tqdm
from pathlib import Path


def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def get_cloud_cover(granule):
    """
    Extracts cloud cover percentage from NASA UMM-G metadata.
    CloudCover is a direct float in UMM-G, not a nested dict.
    """
    try:
        return float(granule['umm']['CloudCover'])
    except (KeyError, TypeError, ValueError):
        return 100.0


def get_geometry(granule):
    """
    Extracts shapely geometry from earthaccess granule.
    Applies buffer(0) to fix any invalid winding-order polygons from UMM-G.
    """
    if 'geometry' in granule:
        geom = shape(granule['geometry'])
        return geom.buffer(0) if not geom.is_valid else geom

    # Fallback: parse UMM-G SpatialExtent directly
    try:
        spatial = granule['umm']['SpatialExtent']['HorizontalSpatialDomain']['Geometry']['GPolygons'][0]
        points = spatial['Boundary']['Points']
        coords = [(p['Longitude'], p['Latitude']) for p in points]
        geom = Polygon(coords)
        return geom.buffer(0) if not geom.is_valid else geom
    except (KeyError, TypeError, IndexError):
        return None


def to_equal_area(geom, src_crs="EPSG:4326", dst_crs="EPSG:6933"):
    """
    Reprojects a shapely geometry to an equal-area CRS for reliable area comparisons.
    Uses the modern pyproj Transformer API (pyproj >= 2).
    """
    transformer = pyproj.Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    return transform(transformer.transform, geom)


def granule_is_on_disk(granule_ur: str, directory: Path) -> bool:
    """
    Returns True if at least one .nc file in directory has a stem that
    starts with the given GranuleUR, meaning the granule was previously downloaded.
    """
    return any(f.stem.startswith(granule_ur) for f in directory.glob("EMIT_L2A_*.nc"))


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

        bbox = tuple(aoi_geom.bounds)  # (xmin, ymin, xmax, ymax)
    except Exception as e:
        print(f"[!] GeoPandas read failed: {e}")
        return

    # 3. Authenticate with NASA Earthdata
    print("[*] Authenticating with NASA Earthdata...")
    auth = config.get('auth', {})
    os.environ["EARTHDATA_USERNAME"] = auth.get('username', '')
    os.environ["EARTHDATA_PASSWORD"] = auth.get('password', '')
    earthaccess.login(strategy="environment")

    # 4. Search dataset
    params = config['sensors']['emit']
    short_name = params['short_name']
    max_cloud = params['max_cloud_cover']
    print(f"[*] Searching {short_name} | Cloud <= {max_cloud}%")

    # NOTE: earthaccess cloud_cover param is not a reliable server-side filter;
    # we apply our own strict filter after retrieval.
    results = earthaccess.search_data(
        short_name=short_name,
        bounding_box=bbox,
        temporal=(params['start_date'], params['end_date']),
    )

    if not results:
        print("[!] No granules found matching criteria.")
        return

    # 5. Cloud cover filter
    filtered_results = [g for g in results if get_cloud_cover(g) <= max_cloud]
    if not filtered_results:
        print(f"[!] No granules passed cloud cover filter (<= {max_cloud}%).")
        return

    print(f"[*] {len(filtered_results)}/{len(results)} granules passed cloud filter.")

    # 6. Sort by cloud cover ascending so the greedy loop prefers cleaner scenes
    sorted_items = sorted(filtered_results, key=get_cloud_cover)

    # Deduplicate by acquisition (orbit + scene), keeping the lowest-cloud
    # processing version when CMR returns multiple VVV variants of the same
    # physical capture. EMIT orbit numbers are sequential and non-repeating,
    # so orbit+scene is a unique acquisition ID, not a repeating tile grid.
    seen_acquisitions = set()
    unique_items = []
    for item in sorted_items:
        granule_ur = item['umm'].get('GranuleUR', '')
        parts = granule_ur.split('_')
        acquisition_id = f"{parts[5]}_{parts[6]}" if len(parts) >= 7 else granule_ur
        if acquisition_id not in seen_acquisitions:
            seen_acquisitions.add(acquisition_id)
            unique_items.append(item)

    print(f"[*] {len(unique_items)} unique acquisitions after version dedup. Optimizing coverage...")

    # 7. Greedy spatial coverage optimization
    # Project AOI to equal-area CRS once; all area maths done in EPSG:6933.
    aoi_geom_ea = to_equal_area(aoi_geom)
    accumulated_ea = Polygon()

    # 1 m² absolute floor — rejects only degenerate near-zero slivers,
    # never a legitimate small fill granule.
    min_contribution = 1.0  # m² in EPSG:6933

    selected_items = []

    for item in unique_items:
        if accumulated_ea.covers(aoi_geom_ea):
            break

        item_geom = get_geometry(item)
        if item_geom is None:
            continue

        intersection_ea = to_equal_area(item_geom.intersection(aoi_geom))
        if intersection_ea.is_empty:
            continue

        new_area = intersection_ea.difference(accumulated_ea)

        if not new_area.is_empty and new_area.area > min_contribution:
            selected_items.append(item)
            accumulated_ea = accumulated_ea.union(intersection_ea).buffer(0)

    if not selected_items:
        print("[!] Optimization resulted in 0 granules selected.")
        return

    covered_area = accumulated_ea.intersection(aoi_geom_ea).area
    total_area = aoi_geom_ea.area
    coverage_pct = (covered_area / total_area) * 100 if total_area > 0 else 0.0
    print(f"[*] Selected {len(selected_items)} granules covering {coverage_pct:.1f}% of the AOI.")

    # 8. Resolve output directory
    out_dir = Path(config['paths']['output_base'])
    out_dir = out_dir if out_dir.is_absolute() else project_root / out_dir
    final_dir = out_dir / params['output_folder']
    final_dir.mkdir(parents=True, exist_ok=True)

    # 9. Build set of GranuleURs that belong in the output directory after this run
    selected_granule_urs = {
        item['umm'].get('GranuleUR', '')
        for item in selected_items
        if item['umm'].get('GranuleUR', '')
    }

    # 10. Delete stale files — on disk but not in current selection
    existing_files = list(final_dir.glob("EMIT_L2A_*.nc"))
    stale_files = [
        f for f in existing_files
        if not any(f.stem.startswith(ur) for ur in selected_granule_urs)
    ]

    if stale_files:
        print(f"[*] Removing {len(stale_files)} stale file(s) no longer in selection:")
        for f in stale_files:
            print(f"    [-] {f.name}")
            f.unlink()
    else:
        print("[*] No stale files to remove.")

    # 11. Classify each selected granule as cached or needing download
    items_to_download = []
    items_cached      = []

    for item in selected_items:
        granule_ur = item['umm'].get('GranuleUR', '')
        if granule_ur and granule_is_on_disk(granule_ur, final_dir):
            items_cached.append(item)
        else:
            items_to_download.append(item)

    # 12. Print full selection table with per-granule status
    print(f"\n{'#':<5} {'GranuleUR':<55} {'CloudCover':>10}  {'Status'}")
    print("─" * 85)
    for i, item in enumerate(selected_items, 1):
        granule_ur  = item['umm'].get('GranuleUR', 'N/A')
        cloud_cover = get_cloud_cover(item)
        cc_str      = f"{cloud_cover:.1f}%" if cloud_cover < 100.0 else "N/A"
        status      = "cached" if item in items_cached else "to download"
        print(f"{i:<5} {granule_ur:<55} {cc_str:>10}  {status}")
    print("─" * 85 + "\n")

    if not items_to_download:
        print("[*] All selected granules already on disk. Nothing to download.")
        return

    # 13. Download only new granules
    print(f"[*] {len(items_cached)} granule(s) already on disk — skipping.")
    print(f"[*] Downloading {len(items_to_download)} new granule(s) to: {final_dir}")

    for item in tqdm(items_to_download, unit="granule", desc=f"Downloading {short_name} granules"):
        earthaccess.download([item], local_path=str(final_dir))

    print("[*] Download complete.")


if __name__ == "__main__":
    current_dir = Path(__file__).resolve().parent
    config_path = current_dir.parent / "config" / "01_emit_download.yaml"
    main(config_file=str(config_path))