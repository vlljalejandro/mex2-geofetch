import sys
import numpy as np
import netCDF4 as nc
import rasterio
from rasterio.transform import Affine
from rasterio.merge import merge
from scipy.ndimage import distance_transform_edt
from pathlib import Path
from tqdm import tqdm
import yaml


# =============================================================================
# Config
# =============================================================================

def load_config(config_path: Path) -> dict:
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def resolve_path(raw: str, project_root: Path) -> Path:
    p = Path(raw)
    return p if p.is_absolute() else project_root / p


def get_band_indices(bands_cfg) -> list[int]:
    if isinstance(bands_cfg, dict) and 'range' in bands_cfg:
        start, end = bands_cfg['range']
        return list(range(start, end + 1))
    return list(bands_cfg)


# =============================================================================
# EMIT Granule Reading
# =============================================================================

def _orthorectify(data: np.ndarray, glt_x: np.ndarray, glt_y: np.ndarray) -> np.ndarray:
    """
    Apply the Geographic Lookup Table (GLT) to map raw detector-space data
    into geographic space. GLT arrays are 1-indexed; 0 means no-data.
    """
    valid   = (glt_x > 0) & (glt_y > 0)
    gridded = np.full(glt_x.shape, np.nan, dtype=np.float32)
    gridded[valid] = data[glt_y[valid] - 1, glt_x[valid] - 1]
    return gridded


def _read_glt_and_transform(ds) -> tuple[np.ndarray, np.ndarray, Affine]:
    loc   = ds.groups['location']
    glt_x = np.array(loc.variables['glt_x'][:])
    glt_y = np.array(loc.variables['glt_y'][:])
    gt    = ds.getncattr('geotransform')   # [lon_min, res, 0, lat_max, 0, -res]
    transform = Affine(gt[1], gt[2], gt[0],
                       gt[4], gt[5], gt[3])
    return glt_x, glt_y, transform


def _write_geotiff(data: np.ndarray, transform: Affine, path: Path) -> None:
    profile = {
        "driver":    "GTiff",
        "dtype":     rasterio.float32,
        "nodata":    np.nan,
        "width":     data.shape[1],
        "height":    data.shape[0],
        "count":     1,
        "crs":       "EPSG:4326",
        "transform": transform,
    }
    with rasterio.open(path, 'w', **profile) as dst:
        dst.write(data, 1)


def extract_reflectance_band(nc_path: Path, band_index: int, out_path: Path) -> None:
    """
    Read one spectral band from an EMIT L2A reflectance NetCDF granule,
    orthorectify it with the embedded GLT, and save as a GeoTIFF.
    """
    with nc.Dataset(nc_path, 'r') as ds:
        band = np.array(ds.variables['reflectance'][:, :, band_index], dtype=np.float32)
        glt_x, glt_y, transform = _read_glt_and_transform(ds)

    band[band <= -9999] = np.nan
    band[band <  0    ] = np.nan

    gridded = _orthorectify(band, glt_x, glt_y)
    _write_geotiff(gridded, transform, out_path)


def extract_cloud_mask(mask_nc_path: Path, out_path: Path) -> None:
    with nc.Dataset(mask_nc_path, 'r') as ds:
        var = ds.variables['mask']
        fill_value = getattr(var, '_FillValue', None)
        raw = np.array(var[:], dtype=np.float32)   # float first — preserves fill values
        glt_x, glt_y, transform = _read_glt_and_transform(ds)

    if raw.ndim == 3:
        raw = raw[:, :, 0]

    # Null out fill values before bit operations
    if fill_value is not None:
        raw[raw == float(fill_value)] = np.nan
    raw[raw < 0] = np.nan   # catch any other sentinels

    # Safe integer cast — NaN pixels become 0 (clear) temporarily,
    # but we track them separately and restore as NaN at the end
    nan_mask   = np.isnan(raw)
    raw_int    = np.where(nan_mask, 0, raw).astype(np.uint8)

    cloud_flag = (raw_int & 0b00000001).astype(np.float32)  # 1 = cloudy
    validity   = np.where(cloud_flag == 0, 1.0, np.nan).astype(np.float32)
    validity[nan_mask] = np.nan   # fill pixels → NaN regardless of bit value

    gridded = _orthorectify(validity, glt_x, glt_y)
    _write_geotiff(gridded, transform, out_path)


def apply_mask_to_reflectance(refl_path: Path, mask_path: Path, out_path: Path) -> None:
    """
    Zero-out (NaN) any reflectance pixel that is flagged as cloudy.
    The resulting GeoTIFF feeds Pass 1 of the mosaic.
    """
    with rasterio.open(refl_path)  as r_src, \
         rasterio.open(mask_path)  as m_src:
        refl    = r_src.read(1).astype(np.float32)
        validity = m_src.read(1).astype(np.float32)
        profile = r_src.meta.copy()

    refl[np.isnan(validity)] = np.nan

    with rasterio.open(out_path, 'w', **profile) as dst:
        dst.write(refl, 1)


# =============================================================================
# Merging
# =============================================================================

def distance_weighted_merge(
    old_data, new_data, old_nodata, new_nodata,
    index=None, roff=None, coff=None
) -> None:
    """
    Rasterio custom merge callback.

    In overlap zones, each pixel is blended proportionally to its distance
    from the nearest NaN edge in each contributing tile. This eliminates hard
    seams without discarding information from either tile.

    Note: rasterio calls this pairwise (accumulated mosaic vs. one new tile),
    so the result in 3-way overlaps is path-dependent on tile sort order.
    """
    for b in range(old_data.shape[0]):
        mask_old = ~old_nodata[b]
        mask_new = ~new_nodata[b]

        dist_old = distance_transform_edt(mask_old)
        dist_new = distance_transform_edt(mask_new)

        overlap     = mask_old & mask_new
        total_dist  = dist_old[overlap] + dist_new[overlap] + 1e-7
        weight_old  = dist_old[overlap] / total_dist
        weight_new  = dist_new[overlap] / total_dist

        old_data[b][overlap]  = (old_data[b][overlap] * weight_old +
                                  new_data[b][overlap] * weight_new)
        old_data[b][mask_new & ~mask_old] = new_data[b][mask_new & ~mask_old]
        old_nodata[b] = old_nodata[b] & new_nodata[b]


def _merge_tiles(tif_paths: list[Path]) -> tuple[np.ndarray, Affine, dict]:
    """Open a list of GeoTIFFs, merge them, return (array, transform, meta)."""
    handles = [rasterio.open(p) for p in tif_paths]
    mosaic, transform = merge(handles, method=distance_weighted_merge)
    meta = handles[0].meta.copy()
    for h in handles:
        h.close()
    return mosaic, transform, meta


def two_pass_mosaic(
    raw_paths:    list[Path],
    masked_paths: list[Path],
    output_path:  Path,
) -> None:
    """
    Pass 1 — Feather-blend only clear-sky pixels (masked_paths).
              Cloud-contaminated pixels are NaN and therefore skipped.
    Pass 2 — Fill any holes left by Pass 1 (areas where every tile was
              cloudy or masked) with the unfiltered reflectance mosaic.

    Result: clouds are avoided in overlapping regions, but no data is
    permanently lost — even aggressively over-masked pixels (e.g. bright
    snow or sand falsely flagged as cloud) are recovered via Pass 2
    wherever no clear-sky alternative exists.
    """
    # Pass 1: cloud-free blend
    mosaic_clear, transform, meta = _merge_tiles(masked_paths)

    # Pass 2: fill NaN holes from unmasked tiles
    holes = np.isnan(mosaic_clear)
    if holes.any():
        mosaic_raw, _, _ = _merge_tiles(raw_paths)
        mosaic_clear[holes] = mosaic_raw[holes]

    meta.update({
        "driver":    "GTiff",
        "height":    mosaic_clear.shape[1],
        "width":     mosaic_clear.shape[2],
        "transform": transform,
        "dtype":     rasterio.float32,
        "nodata":    np.nan,
    })

    with rasterio.open(output_path, "w", **meta) as dst:
        dst.write(mosaic_clear)


def _cleanup(paths: list[Path]) -> None:
    for p in paths:
        try:
            p.unlink(missing_ok=True)
        except Exception:
            pass


# =============================================================================
# Per-band orchestration
# =============================================================================

def _find_mask_granule(refl_nc: Path) -> Path | None:
    """
    Derive the MASK granule path from a reflectance granule path.
    EMIT naming convention: replace 'RFL' with 'MASK' in the filename.
    Returns None if no mask file is found.
    """
    candidate = refl_nc.parent / refl_nc.name.replace('RFL', 'MASK')
    return candidate if candidate.exists() else None


def process_band(
    band_idx:  int,
    nc_files:  list[Path],
    output_dir: Path,
) -> None:
    """
    Full pipeline for a single spectral band:
      1. Extract orthorectified reflectance tile  (temp GeoTIFF)
      2. Extract clear-sky mask tile              (temp GeoTIFF)
      3. Apply mask to reflectance                (temp masked GeoTIFF)
      4. Two-pass mosaic → final output GeoTIFF
      5. Clean up all temp files
    """
    raw_paths    = []
    masked_paths = []
    temp_files   = []

    for nc_path in tqdm(nc_files, desc=f"  Band {band_idx:03d} — extracting", unit="granule"):
        stem     = nc_path.stem
        raw_tif  = nc_path.parent / f"_tmp_refl_b{band_idx:03d}_{stem}.tif"
        mask_tif = nc_path.parent / f"_tmp_mask_b{band_idx:03d}_{stem}.tif"
        mskd_tif = nc_path.parent / f"_tmp_mskd_b{band_idx:03d}_{stem}.tif"

        # --- reflectance ---
        try:
            extract_reflectance_band(nc_path, band_idx, raw_tif)
            temp_files.append(raw_tif)
        except Exception as e:
            print(f"\n[!] Reflectance extraction failed for {nc_path.name}: {e}")
            continue

        # --- cloud mask ---
        mask_nc = _find_mask_granule(nc_path)
        if mask_nc is not None:
            try:
                extract_cloud_mask(mask_nc, mask_tif)
                apply_mask_to_reflectance(raw_tif, mask_tif, mskd_tif)
                temp_files += [mask_tif, mskd_tif]
                masked_paths.append(mskd_tif)
            except Exception as e:
                print(f"\n[!] Mask extraction failed for {nc_path.name}, using unmasked: {e}")
                masked_paths.append(raw_tif)   # fallback: treat tile as cloud-free
        else:
            print(f"\n[~] No mask granule found for {nc_path.name}, using unmasked.")
            masked_paths.append(raw_tif)

        raw_paths.append(raw_tif)

    if not raw_paths:
        print(f"[!] No tiles extracted for Band {band_idx}.")
        return

    # Single tile: skip mosaic, just rename
    name_number = band_idx + 1
    if len(raw_paths) == 1:
        out_path = output_dir / f"emit_b{name_number:03d}.tif"
        raw_paths[0].rename(out_path)
        temp_files = [p for p in temp_files if p != raw_paths[0]]
        _cleanup(temp_files)
        print(f"[+] Single tile → {out_path.name}")
        return

    out_path = output_dir / f"emit_mosaic_b{name_number:03d}.tif"
    print(f"[*] Mosaicking Band {name_number:03d} ({len(raw_paths)} tiles)...")
    try:
        two_pass_mosaic(raw_paths, masked_paths, out_path)
        print(f"[+] Saved: {out_path.name}")
    except Exception as e:
        print(f"[!] Mosaic failed for Band {band_idx}: {e}")
    finally:
        _cleanup(temp_files)


# =============================================================================
# Pipeline entry
# =============================================================================

def process_emit(config: dict, project_root: Path) -> None:
    params     = config['processing']
    input_dir  = resolve_path(params['input_dir'],  project_root)
    output_dir = resolve_path(params['output_dir'], project_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        print(f"[!] Input directory not found: {input_dir}")
        return

    nc_files = sorted(input_dir.rglob("*.nc"))
    nc_files = [
        f for f in nc_files
        if 'RFL' in f.name.upper() and 'RFLUNCERT' not in f.name.upper()
    ]

    if not nc_files:
        print("[!] No EMIT L2A reflectance (.nc) files found.")
        return

    print(f"[*] Found {len(nc_files)} EMIT reflectance granule(s).")

    band_indices = get_band_indices(params['band_indices'])
    print(f"[*] Bands to process: {band_indices[0]}–{band_indices[-1]} ({len(band_indices)} total)")

    for band_idx in band_indices:
        print(f"\n{'─'*60}")
        print(f"[*] Band {band_idx:03d}")
        process_band(band_idx, nc_files, output_dir)

    print(f"\n[*] Pipeline complete → {output_dir}")


def main(config_file: str) -> None:
    config_path = Path(config_file).resolve()
    if not config_path.exists():
        print(f"[!] Config not found: {config_path}")
        return

    config       = load_config(config_path)
    project_root = config_path.parent.parent

    print("[*] EMIT mosaic pipeline starting...")
    process_emit(config, project_root)


if __name__ == "__main__":
    default_config = Path(__file__).resolve().parent.parent / "config" / "02_emit_mosaic.yaml"
    config_path    = Path(sys.argv[1]) if len(sys.argv) > 1 else default_config
    main(str(config_path))