import sys
import re
import yaml
import numpy as np
import rasterio
from rasterio.merge import merge
from scipy.ndimage import distance_transform_edt, label
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)


# =============================================================================
# Shared Mosaic Utilities  (unchanged from Landsat pipeline)
# =============================================================================

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

        old_data[b][overlap] = (
            old_data[b][overlap].astype(np.float64) * weight_old +
            new_data[b][overlap].astype(np.float64) * weight_new
        )

        only_new = mask_new & ~mask_old
        old_data[b][only_new] = new_data[b][only_new]
        old_nodata[b] = old_nodata[b] & new_nodata[b]


def fill_small_nodata(data, max_size=20):
    """
    Fill NaN holes using nearest valid pixel (per band).
    Runs a single EDT over the entire NaN mask and restricts fill to only
    small-enough regions using vectorised label counting.
    """
    filled = data.copy()
    for b in range(data.shape[0]):
        band = filled[b]
        nan_mask = np.isnan(band)
        if not nan_mask.any():
            continue

        labeled, _ = label(nan_mask)
        region_sizes = np.bincount(labeled.ravel())
        region_sizes[0] = max_size + 1

        small_mask = region_sizes[labeled] <= max_size
        if not small_mask.any():
            continue

        _, indices = distance_transform_edt(nan_mask, return_indices=True)
        band[small_mask] = band[tuple(ind[small_mask] for ind in indices)]
        filled[b] = band

    return filled


def mosaic_and_save(tif_paths, output_path):
    """
    Opens a list of GeoTIFF paths, mosaics them with feather blending,
    and saves the result. Temp files are deleted only after output is confirmed.
    """
    open_files = [rasterio.open(p) for p in tif_paths]

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
            "nodata":    np.nan
        })

        with rasterio.open(output_path, "w", **out_meta) as dst:
            dst.write(mosaic)

    finally:
        for f in open_files:
            f.close()

    if output_path.exists():
        for p in tif_paths:
            Path(p).unlink(missing_ok=True)
    else:
        print(f"[!] Output not confirmed at {output_path} — temp files preserved.")


# =============================================================================
# Sentinel-2 Reader
# =============================================================================

def parse_s2_scene_info(scene_dir):
    """
    Extracts MGRS tile ID and acquisition datetime from a Sentinel-2 scene.

    Tries the directory name first (ESA standard product format):
        S2A_MSIL2A_20160702T080612_R078_T37QCG_20210212T011119
    Falls back to scanning filenames inside the directory:
        T37RFH_20251108T075141_B02_10m.tif
    """
    # --- 1. Try the directory name (most reliable) ---
    dir_match = re.search(r'_(T[A-Z0-9]{5})_', scene_dir.name)
    dt_match  = re.search(r'_(\d{8}T\d{6})_', scene_dir.name)

    if dir_match and dt_match:
        return {
            "tile_id":  dir_match.group(1),
            "datetime": dt_match.group(1),
            "dir":      scene_dir,
        }

    # --- 2. Fall back to scanning files inside the directory ---
    tif_files = sorted(scene_dir.glob("*.tif")) + sorted(scene_dir.glob("*.TIF"))
    for f in tif_files:
        match = re.match(r'(T[A-Z0-9]{5})_(\d{8}T\d{6})_', f.name)
        if match:
            return {
                "tile_id":  match.group(1),
                "datetime": match.group(2),
                "dir":      scene_dir,
            }

    return None


def extract_sentinel2_band(scene, band_key, temp_path):
    """
    Converts Sentinel-2 L2A uint16 DN to surface reflectance (float32).

    Scaling (processing baseline >= 04.00, applicable to 2022+ acquisitions):
        SR = (DN - 1000) / 10000

    Nodata handling:
        DN == 0      → NaN  (no-data fill)
        DN == 65535  → NaN  (saturated / invalid)

    Result is clamped to [0.0, 1.0] after scaling.
    """
    scene_dir = scene["dir"]
    band_name = band_key.split("_")[0]

    band_files = (
        list(scene_dir.glob(f"*_{band_name}.tif")) +
        list(scene_dir.glob(f"*_{band_name}.TIF"))
    )
    if not band_files:
        return False

    with rasterio.open(band_files[0]) as src:
        profile = src.profile.copy()
        data    = src.read(1).astype(np.float32)

    nodata_mask = (data == 0) | (data == 65535)
    data        = (data - 1000.0) / 10000.0
    data        = np.clip(data, 0.0, 1.0)
    data[nodata_mask] = np.nan

    profile.update(dtype=rasterio.float32, nodata=np.nan)
    with rasterio.open(temp_path, "w", **profile) as dst:
        dst.write(data, 1)

    return True


def _extract_scene_band(args):
    """
    Worker thunk for parallel extraction. Returns (temp_path, success).
    """
    scene, band_key = args
    temp_path = scene["dir"] / f"temp_sr_{band_key}.tif"
    success   = extract_sentinel2_band(scene, band_key=band_key, temp_path=temp_path)
    return temp_path, success


def process_sentinel2(config, project_root):
    """
    Sentinel-2 L2A mosaic pipeline:
      1. Scan input_dir for scene subdirectories
      2. Parse all available scenes
      3. For each configured band key, scale DN → SR → temp GeoTIFF → mosaic → save

    Band keys must match the resolution suffix in filenames, e.g.:
      B02_10m, B03_10m, B04_10m, B08_10m   (10m)
      B05_20m, B06_20m, B11_20m, B8A_20m   (20m)
      B01_60m, B09_60m                      (60m)
    """
    params     = config['processing']
    input_dir  = Path(params['input_dir'])
    input_dir  = input_dir if input_dir.is_absolute() else project_root / input_dir
    output_dir = Path(params['output_dir'])
    output_dir = output_dir if output_dir.is_absolute() else project_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        print(f"[!] Input directory missing: {input_dir}")
        return

    # Process all scenes found in the input directory
    scene_dirs = [d for d in input_dir.iterdir() if d.is_dir()]
    print(f"[*] Scanning {len(scene_dirs)} directories for Sentinel-2 scenes...")
    
    all_scenes = []
    for scene_dir in scene_dirs:
        info = parse_s2_scene_info(scene_dir)
        if info is not None:
            all_scenes.append(info)
        else:
            print(f"[!] Warning: Could not parse tile ID from {scene_dir.name} — skipping.")

    if not all_scenes:
        print("[!] No valid Sentinel-2 scenes found.")
        return

    print(f"[*] Queued {len(all_scenes)} scene(s) for processing.")

    band_keys   = params['bands']
    max_workers = params.get('max_workers', min(32, len(all_scenes)))
    print(f"[*] Processing {len(band_keys)} band(s): {band_keys}  |  workers: {max_workers}")

    for band_key in band_keys:
        print(f"\n[*] Extracting {band_key}...")

        args       = [(scene, band_key) for scene in all_scenes]
        temp_paths = []

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_extract_scene_band, a): a for a in args}
            with tqdm(as_completed(futures), total=len(futures),
                      desc=f"  {band_key} — scaling SR", unit="scene") as pbar:
                for future in pbar:
                    temp_path, success = future.result()
                    if success:
                        temp_paths.append(temp_path)

        if not temp_paths:
            print(f"[!] No valid tiles found for {band_key} — skipping.")
            continue

        out_path = output_dir / f"sentinel2_mosaic_{band_key}.tif"

        if len(temp_paths) == 1:
            temp_paths[0].rename(out_path)
            print(f"[+] Single tile saved: {out_path.name}")
            continue

        print(f"[*] Mosaicking {band_key} with feathering...")
        try:
            mosaic_and_save(temp_paths, out_path)
            print(f"[+] Saved: {out_path.name}")
        except Exception as e:
            print(f"[!] Mosaic failed for {band_key}: {e}")

    print(f"\n[*] Sentinel-2 mosaic complete: {output_dir}")


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

    print("[*] Starting Sentinel-2 mosaic pipeline...")
    process_sentinel2(config, project_root)


if __name__ == "__main__":
    current_dir = Path(__file__).resolve().parent

    if len(sys.argv) > 1:
        config_path = Path(sys.argv[1])
    else:
        config_path = current_dir.parent / "config" / "02_sentinel_mosaic.yaml"

    main(config_file=str(config_path))