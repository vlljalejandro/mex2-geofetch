import yaml
import numpy as np
from osgeo import gdal
from pathlib import Path
gdal.UseExceptions()


# =============================================================================
# Shared Utilities
# =============================================================================

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


# =============================================================================
# Core Alignment Function
# =============================================================================

def align_raster(
    input_path,
    reference_path,
    output_path,
    resample_alg=gdal.GRA_NearestNeighbour,
    apply_mask=False,
) -> None:
    """
    Reprojects and aligns a raster to perfectly match a reference (master) grid,
    then optionally applies the master grid as a binary mask.

    Workflow
    --------
    1. Reproject the input raster to the master CRS.
    2. Snap the reprojected grid exactly to the master pixel grid
       (same origin, pixel size, dimensions, and bounds).
    3. Detect master grid type and apply mask accordingly:
         - Master filled with 1s only → no masking (grid defines extent only)
         - Master contains 0s         → set output pixels where master == 0 to NaN

    The mask is the invariant ground truth. It is read once from the master
    raster and never warped, resampled, or transformed in any way.

    Parameters
    ----------
    input_path     : Path to the input raster to be aligned.
    reference_path : Path to the master reference raster defining the target grid.
    output_path    : Path where the aligned output raster will be saved.
    resample_alg   : GDAL resampling algorithm for reprojecting the input raster.
                     Default: GRA_NearestNeighbour.
                     Use GRA_Bilinear for continuous data like DEMs.
    apply_mask     : If True, pixels where master == 0 are set to NaN.
                     Auto-skipped if the master grid contains no zeros.
    """

    in_file  = Path(input_path)
    ref_file = Path(reference_path)
    out_file = Path(output_path)

    if not in_file.is_file():
        raise FileNotFoundError(f"Input raster not found:     {in_file}")
    if not ref_file.is_file():
        raise FileNotFoundError(f"Reference raster not found: {ref_file}")

    out_file.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Read master grid parameters
    # ------------------------------------------------------------------
    ref_ds     = gdal.Open(str(ref_file))
    projection = ref_ds.GetProjection()
    gt         = ref_ds.GetGeoTransform()
    cols       = ref_ds.RasterXSize
    rows       = ref_ds.RasterYSize

    # Derive exact bounds from geotransform — no rounding
    min_x = gt[0]
    max_y = gt[3]
    max_x = min_x + cols * gt[1]
    min_y = max_y + rows * gt[5]   # gt[5] is negative
    target_bounds = (min_x, min_y, max_x, max_y)

    print(f"[*] Aligning : {in_file.name}")
    print(f"    Master   : {cols} x {rows} px | {gt[1]:.1f} m/px")

    # ------------------------------------------------------------------
    # 2. Read master array once — used for mask detection and masking
    # ------------------------------------------------------------------
    master_data = ref_ds.GetRasterBand(1).ReadAsArray()

    # Detect whether master grid contains any 0s (masked pixels)
    has_zeros   = bool(np.any(master_data == 0))

    if apply_mask and not has_zeros:
        print(f"    Mask     : skipped (master grid is all 1s — extent-only grid)")
    elif apply_mask and has_zeros:
        zero_count = int(np.sum(master_data == 0))
        print(f"    Mask     : {zero_count:,} pixels will be set to NaN")
    else:
        print(f"    Mask     : disabled")

    # ------------------------------------------------------------------
    # 3. Reproject and snap input to master grid
    #
    #    targetAlignedPixels=False: prevents GDAL from snapping the grid
    #    to its own internally computed alignment, which would shift the
    #    output origin and break the 1:1 correspondence with the mask.
    # ------------------------------------------------------------------
    warp_options = gdal.WarpOptions(
        format="GTiff",
        outputType=gdal.GDT_Float32,
        dstSRS=projection,
        outputBounds=target_bounds,
        width=cols,
        height=rows,
        resampleAlg=resample_alg,
        dstNodata=np.nan,
        targetAlignedPixels=False,
        warpOptions=["INIT_DEST=NO_DATA"],
        creationOptions=["COMPRESS=LZW", "TILED=YES", "BLOCKXSIZE=256", "BLOCKYSIZE=256"]
    )

    out_ds = gdal.Warp(str(out_file), str(in_file), options=warp_options)
    if out_ds is None:
        raise RuntimeError("GDAL Warp failed.")

    # ------------------------------------------------------------------
    # 4. Apply mask if requested and master contains zeros
    #    Array index [i, j] in master maps 1:1 to output — no resampling.
    # ------------------------------------------------------------------
    if apply_mask and has_zeros:
        mask = (master_data == 0)
        for band_index in range(1, out_ds.RasterCount + 1):
            band = out_ds.GetRasterBand(band_index)
            data = band.ReadAsArray()
            data[mask]       = np.nan
            data[data < -1e20] = np.nan    # remove extreme float warp artifacts
            band.WriteArray(data)
            band.SetNoDataValue(np.nan)
            band.FlushCache()

    out_ds = None
    ref_ds = None

    print(f"    Output   : {out_file.name}\n")


# =============================================================================
# Batch Runner
# =============================================================================

RESAMPLE_ALGORITHMS = {
    "nearest":     gdal.GRA_NearestNeighbour,
    "bilinear":    gdal.GRA_Bilinear,
    "cubic":       gdal.GRA_Cubic,
    "cubicspline": gdal.GRA_CubicSpline,
    "lanczos":     gdal.GRA_Lanczos,
    "average":     gdal.GRA_Average,
    "mode":        gdal.GRA_Mode,
    "max":         gdal.GRA_Max,
    "min":         gdal.GRA_Min,
    "med":         gdal.GRA_Med,
}


def main(config_file):
    config_path = Path(config_file).resolve()
    if not config_path.exists():
        print(f"[!] Config not found: {config_path}")
        return

    config       = load_config(config_path)
    project_root = config_path.parent.parent

    # Master grid
    master_rel  = config['paths']['master_grid']
    master_file = Path(master_rel) if Path(master_rel).is_absolute() else project_root / master_rel
    if not master_file.exists():
        print(f"[!] Master grid missing: {master_file}")
        return

    # Output directory
    out_rel = config['paths']['output_dir']
    out_dir = Path(out_rel) if Path(out_rel).is_absolute() else project_root / out_rel
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resampling algorithm
    alg_name   = config['processing'].get('resample_alg', 'nearest').lower()
    resample   = RESAMPLE_ALGORITHMS.get(alg_name, gdal.GRA_NearestNeighbour)
    apply_mask = config['processing'].get('apply_mask', True)

    if alg_name not in RESAMPLE_ALGORITHMS:
        print(f"[!] Unknown resample_alg '{alg_name}'. Falling back to nearest.")

    # Input files — glob pattern or explicit list
    inputs_cfg = config['inputs']
    input_files = []

    if 'glob' in inputs_cfg:
        base = Path(inputs_cfg['glob']['dir'])
        base = base if base.is_absolute() else project_root / base
        input_files = sorted(base.glob(inputs_cfg['glob']['pattern']))
    elif 'files' in inputs_cfg:
        for f in inputs_cfg['files']:
            p = Path(f) if Path(f).is_absolute() else project_root / Path(f)
            input_files.append(p)

    if not input_files:
        print("[!] No input files found. Check config inputs section.")
        return

    print(f"[*] Master grid : {master_file.name}")
    print(f"[*] Resample    : {alg_name}")
    print(f"[*] Apply mask  : {apply_mask}")
    print(f"[*] Files to align: {len(input_files)}\n")

    for input_file in input_files:
        output_file = out_dir / input_file.name.replace("_b", "_snapped_b")
        try:
            align_raster(
                input_path=input_file,
                reference_path=master_file,
                output_path=output_file,
                resample_alg=resample,
                apply_mask=apply_mask,
            )
        except Exception as e:
            print(f"[!] Failed: {input_file.name} — {e}\n")

    print(f"[*] Alignment complete. Output: {out_dir}")


if __name__ == "__main__":
    current_dir = Path(__file__).resolve().parent
    config_path = current_dir.parent / "config" / "03_snap_raster.yaml"
    main(config_file=str(config_path))