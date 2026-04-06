import sys
import re
import yaml
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.merge import merge
from scipy.ndimage import distance_transform_edt, label
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from rasterio.warp import calculate_default_transform, reproject, Resampling



def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)


def distance_weighted_merge(old_data, new_data, old_nodata, new_nodata, index=None, roff=None, coff=None):
    """
    Custom rasterio merge function using distance-weighted feather blending.
    Smoothly blends overlapping regions based on each pixel's distance to its
    nearest nodata edge — avoids hard seams between tiles.
    """
    for b in range(old_data.shape[0]):
        mask_old = ~old_nodata[b]
        mask_new = ~new_nodata[b]

        dist_old = distance_transform_edt(mask_old)
        dist_new = distance_transform_edt(mask_new)

        overlap = mask_old & mask_new
        total_dist = dist_old[overlap] + dist_new[overlap] + 1e-7
        weight_old = dist_old[overlap] / total_dist
        weight_new = dist_new[overlap] / total_dist

        # FIX #1: cast to float64 before blending to prevent silent integer truncation
        old_data[b][overlap] = (
            old_data[b][overlap].astype(np.float64) * weight_old +
            new_data[b][overlap].astype(np.float64) * weight_new
        )

        only_new = mask_new & ~mask_old
        old_data[b][only_new] = new_data[b][only_new]

        # A pixel is nodata only if both tiles have nodata at that location
        old_nodata[b] = old_nodata[b] & new_nodata[b]


def fill_small_nodata(data, max_size=20):
    """
    Fill NaN holes using nearest valid pixel (per band).

    PERF: Previously called distance_transform_edt once per hole in a Python
    loop, which is O(n_holes) scipy launches. Now runs a single EDT over the
    entire NaN mask and restricts the fill to only small-enough regions using
    vectorised label counting — O(1) EDT calls per band regardless of hole count.
    """
    filled = data.copy()
    for b in range(data.shape[0]):
        band = filled[b]
        nan_mask = np.isnan(band)
        if not nan_mask.any():
            continue

        labeled, _ = label(nan_mask)

        # Count pixels per label with a single bincount; label 0 is background
        region_sizes = np.bincount(labeled.ravel())
        region_sizes[0] = max_size + 1  # exclude background from fill candidates

        # Build a boolean mask covering only small holes
        small_mask = region_sizes[labeled] <= max_size

        if not small_mask.any():
            continue

        # Single EDT over the full NaN mask → nearest valid pixel indices for
        # every NaN pixel; apply only where the hole qualifies as small
        _, indices = distance_transform_edt(nan_mask, return_indices=True)
        band[small_mask] = band[tuple(ind[small_mask] for ind in indices)]

        filled[b] = band

    return filled

from rasterio.warp import calculate_default_transform, reproject, Resampling


def reproject_to_crs(src_path, target_crs, tmp_suffix="_reproj.tif"):
    """
    Reprojects a GeoTIFF to target_crs in-place (writes a sibling temp file,
    then replaces the original). Returns the path (unchanged) for chaining.
    """
    reproj_path = Path(str(src_path).replace(".tif", tmp_suffix))

    with rasterio.open(src_path) as src:
        if src.crs == target_crs:
            return src_path  # nothing to do

        transform, width, height = calculate_default_transform(
            src.crs, target_crs, src.width, src.height, *src.bounds
        )
        profile = src.profile.copy()
        profile.update(
            crs=target_crs,
            transform=transform,
            width=width,
            height=height,
        )

        with rasterio.open(reproj_path, "w", **profile) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=target_crs,
                    resampling=Resampling.bilinear,
                )

    # Replace original with reprojected version
    src_path.unlink()
    reproj_path.rename(src_path)
    return src_path


def mosaic_and_save(tif_paths, output_path, target_crs=None):
    """
    Opens a list of GeoTIFF paths, reprojects all to a common CRS if needed,
    mosaics them with feather blending, and saves the result.
    """
    # ── Normalise CRS ────────────────────────────────────────────────────────
    if target_crs is None:
        with rasterio.open(tif_paths[0]) as ref:
            target_crs = ref.crs

    aligned_paths = []
    for p in tif_paths:
        with rasterio.open(p) as src:
            needs_reproj = src.crs != target_crs
        if needs_reproj:
            print(f"  [~] Reprojecting {p.name} → {target_crs}")
            reproject_to_crs(p, target_crs)
        aligned_paths.append(p)
    # ─────────────────────────────────────────────────────────────────────────

    open_files = [rasterio.open(p) for p in aligned_paths]

    try:
        mosaic, out_transform = merge(open_files, method=distance_weighted_merge)
        mosaic = fill_small_nodata(mosaic, max_size=20)

        out_meta = open_files[0].meta.copy()
        out_meta.update({
            "driver":    "GTiff",
            "height":    mosaic.shape[1],
            "width":     mosaic.shape[2],
            "transform": out_transform,
            "dtype":     rasterio.float32,
            "nodata":    np.nan,
        })

        with rasterio.open(output_path, "w", **out_meta) as dst:
            dst.write(mosaic)

    finally:
        for f in open_files:
            f.close()

    if output_path.exists():
        for p in aligned_paths:
            Path(p).unlink(missing_ok=True)
    else:
        print(f"[!] Output not confirmed at {output_path} — temp files preserved.")

# =============================================================================
# Landsat Reader
# =============================================================================

def parse_mtl_metadata(mtl_path):
    """
    Parses a Landsat MTL.txt file for Path/Row, cloud cover,
    and per-band reflectance scaling coefficients.
    """
    metadata = {"reflectance_mult": {}, "reflectance_add": {}}
    content  = mtl_path.read_text()

    path_match = re.search(r'WRS_PATH\s*=\s*(\d+)', content)
    row_match  = re.search(r'WRS_ROW\s*=\s*(\d+)', content)
    if path_match and row_match:
        metadata["path_row"] = f"{int(path_match.group(1)):03d}{int(row_match.group(1)):03d}"

    cloud_match = re.search(r'CLOUD_COVER\s*=\s*([0-9.]+)', content)
    if cloud_match:
        metadata["cloud_cover"] = float(cloud_match.group(1))

    for match in re.finditer(r'REFLECTANCE_MULT_BAND_(\d+)\s*=\s*([-eE0-9.]+)', content):
        metadata["reflectance_mult"][int(match.group(1))] = float(match.group(2))

    for match in re.finditer(r'REFLECTANCE_ADD_BAND_(\d+)\s*=\s*([-eE0-9.]+)', content):
        metadata["reflectance_add"][int(match.group(1))] = float(match.group(2))

    return metadata


def select_best_landsat_scenes(input_dir):
    """
    Scans scene directories for MTL files and selects the scene with the
    lowest cloud cover for each unique WRS Path/Row combination.
    """
    scenes_by_pr = {}
    scene_dirs   = [d for d in input_dir.iterdir() if d.is_dir()]

    print(f"[*] Scanning {len(scene_dirs)} directories for MTL metadata...")
    for scene_dir in tqdm(scene_dirs, desc="  Parsing MTL"):
        mtl_files = list(scene_dir.glob("*_MTL.txt"))
        if not mtl_files:
            continue

        meta = parse_mtl_metadata(mtl_files[0])
        pr   = meta.get("path_row")

        # FIX (minor): warn instead of silently skipping scenes with missing path/row
        if not pr:
            print(f"[!] Warning: Could not parse WRS Path/Row from {scene_dir.name} — skipping.")
            continue

        scenes_by_pr.setdefault(pr, []).append({"dir": scene_dir, "meta": meta})

    best_scenes = [
        min(scenes, key=lambda x: x["meta"].get("cloud_cover", 100))
        for scenes in scenes_by_pr.values()
    ]

    print(f"[*] Selected {len(best_scenes)} best scene(s) across {len(scenes_by_pr)} Path/Row(s).")
    return best_scenes


def extract_landsat_band(scene, band_number, temp_path):
    """
    Applies Landsat C2 L2 reflectance scaling to a raw band TIF and writes
    a float32 GeoTIFF. Fill pixels (DN == 0) are set to NaN.
    """
    scene_dir = scene["dir"]
    meta      = scene["meta"]

    # FIX #5: try both uppercase and lowercase extensions for cross-platform compatibility
    band_files = (
        list(scene_dir.glob(f"*_B{band_number}.TIF")) or
        list(scene_dir.glob(f"*_b{band_number}.TIF")) or
        list(scene_dir.glob(f"*_B{band_number}.tif")) or
        list(scene_dir.glob(f"*_b{band_number}.tif"))
    )
    if not band_files:
        return False

    mult = meta["reflectance_mult"].get(band_number, 2.75e-05)
    add  = meta["reflectance_add"].get(band_number, -0.2)

    with rasterio.open(band_files[0]) as src:
        profile = src.profile.copy()
        data    = src.read(1).astype(np.float32)

    fill_mask       = (data == 0)
    data            = (data * mult) + add
    data[fill_mask] = np.nan

    profile.update(dtype=rasterio.float32, nodata=np.nan)
    with rasterio.open(temp_path, 'w', **profile) as dst:
        dst.write(data, 1)

    return True


def _extract_scene_band(args):
    """
    Worker thunk for parallel extraction: unpacks args and delegates to
    extract_landsat_band. Returns (temp_path, success) so the caller can
    collect results in completion order without needing a shared list.
    """
    scene, band_number = args
    temp_path = scene["dir"] / f"temp_sr_b{band_number}.tif"
    success   = extract_landsat_band(scene, band_number=band_number, temp_path=temp_path)
    return temp_path, success


def process_landsat(config, project_root):
    """
    Landsat C2 L2 mosaic pipeline:
      1. Scan input_dir for scene subdirectories with MTL files
      2. Select best scene per Path/Row (lowest cloud cover)
      3. For each configured band, apply SR scaling → temp GeoTIFF → mosaic → save

    PERF: scene extraction is parallelised with a ThreadPoolExecutor.
    Tune max_workers in the config (processing.max_workers) or leave unset
    to default to min(32, n_scenes).
    """
    params     = config['processing']
    input_dir  = Path(params['input_dir'])
    input_dir  = input_dir if input_dir.is_absolute() else project_root / input_dir
    output_dir = Path(params['output_dir'])
    output_dir = output_dir if output_dir.is_absolute() else project_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    target_crs_str = params.get("target_crs")
    target_crs = CRS.from_string(target_crs_str) if target_crs_str else None
    # pass target_crs into mosaic_and_save(temp_paths, out_path, target_crs=target_crs)

    if not input_dir.exists():
        print(f"[!] Input directory missing: {input_dir}")
        return

    best_scenes = select_best_landsat_scenes(input_dir)
    if not best_scenes:
        print("[!] No valid Landsat scenes found with MTL metadata.")
        return

    band_numbers = params['bands']
    max_workers  = params.get('max_workers', min(32, len(best_scenes)))
    print(f"[*] Processing {len(band_numbers)} band(s): {band_numbers}  |  workers: {max_workers}")

    for band in band_numbers:
        print(f"\n[*] Extracting Band {band}...")

        args       = [(scene, band) for scene in best_scenes]
        temp_paths = []

        # PERF: fan out scene reads/writes across threads; rasterio releases the
        # GIL during I/O so ThreadPoolExecutor gives real concurrency here.
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_extract_scene_band, a): a for a in args}
            with tqdm(as_completed(futures), total=len(futures),
                      desc=f"  Band {band} — scaling SR", unit="scene") as pbar:
                for future in pbar:
                    temp_path, success = future.result()
                    if success:
                        temp_paths.append(temp_path)

        if not temp_paths:
            print(f"[!] No valid tiles found for Band {band} — skipping.")
            continue

        out_path = output_dir / f"landsat_mosaic_b{band}.tif"

        if len(temp_paths) == 1:
            temp_paths[0].rename(out_path)
            print(f"[+] Single tile saved: {out_path.name}")
            continue

        print(f"[*] Mosaicking Band {band} with feathering...")
        try:
            mosaic_and_save(temp_paths, out_path, target_crs=target_crs)
            print(f"[+] Saved: {out_path.name}")
        except Exception as e:
            print(f"[!] Mosaic failed for Band {band}: {e}")

    print(f"\n[*] Landsat mosaic complete: {output_dir}")


# =============================================================================
# Entry Point
# =============================================================================

def main(config_file):
    config_path = Path(config_file).resolve()
    if not config_path.exists():
        print(f"[!] Error: Config not found at {config_path}")
        return

    config       = load_config(config_path)
    project_root = config_path.parent.parent

    print("[*] Starting Landsat mosaic pipeline...")
    process_landsat(config, project_root)


if __name__ == "__main__":
    current_dir = Path(__file__).resolve().parent

    if len(sys.argv) > 1:
        config_path = Path(sys.argv[1])
    else:
        config_path = current_dir.parent / "config" / "02_landsat_mosaic.yaml"

    main(config_file=str(config_path))