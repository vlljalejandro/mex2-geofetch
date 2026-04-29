import yaml
import numpy as np
from osgeo import gdal, osr
from pathlib import Path

gdal.UseExceptions()


# =============================================================================
# Shared Utilities
# =============================================================================

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _read_master(reference_path):
    """
    Read master grid parameters and array.

    Returns (ds, gt, projection, cols, rows, array).

    The master's own NoData value — if set — is replaced with NaN in the
    returned array so it is never mistaken for a valid 0 (mask) pixel.
    """
    ref_ds     = gdal.Open(str(reference_path))
    projection = ref_ds.GetProjection()
    gt         = ref_ds.GetGeoTransform()
    cols       = ref_ds.RasterXSize
    rows       = ref_ds.RasterYSize
    band       = ref_ds.GetRasterBand(1)
    master_arr = band.ReadAsArray().astype(np.float32)
    master_nd  = band.GetNoDataValue()
    if master_nd is not None:
        master_arr[master_arr == master_nd] = np.nan
    return ref_ds, gt, projection, cols, rows, master_arr


def _target_bounds(gt, cols, rows):
    min_x = gt[0]
    max_y = gt[3]
    max_x = min_x + cols * gt[1]
    min_y = max_y + rows * gt[5]   # gt[5] is negative
    return (min_x, min_y, max_x, max_y)


def _filter_overlapping(input_files, master_projection, target_bounds):
    """
    Return only those files whose geographic extent overlaps the master grid.

    Each file's bounding box is reprojected to the master CRS before the
    overlap test so mixed-CRS inputs are handled correctly.
    """
    master_srs = osr.SpatialReference()
    master_srs.ImportFromWkt(master_projection)
    master_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

    m_xmin, m_ymin, m_xmax, m_ymax = target_bounds
    valid = []

    for f in input_files:
        ds = gdal.Open(str(f))
        if ds is None:
            print(f"    [!] Cannot open {Path(f).name} — skipping")
            continue

        src_gt   = ds.GetGeoTransform()
        src_cols = ds.RasterXSize
        src_rows = ds.RasterYSize
        src_srs  = osr.SpatialReference()
        src_srs.ImportFromWkt(ds.GetProjection())
        src_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        ds = None

        src_xmin = src_gt[0]
        src_ymax = src_gt[3]
        src_xmax = src_xmin + src_cols * src_gt[1]
        src_ymin = src_ymax + src_rows * src_gt[5]

        if not src_srs.IsSame(master_srs):
            transform = osr.CoordinateTransformation(src_srs, master_srs)
            corners   = [
                transform.TransformPoint(src_xmin, src_ymin),
                transform.TransformPoint(src_xmin, src_ymax),
                transform.TransformPoint(src_xmax, src_ymin),
                transform.TransformPoint(src_xmax, src_ymax),
            ]
            xs = [c[0] for c in corners]
            ys = [c[1] for c in corners]
            src_xmin, src_xmax = min(xs), max(xs)
            src_ymin, src_ymax = min(ys), max(ys)

        overlaps = (
            src_xmax > m_xmin and src_xmin < m_xmax and
            src_ymax > m_ymin and src_ymin < m_ymax
        )

        if overlaps:
            valid.append(Path(f))
        else:
            print(f"    [!] {Path(f).name} lies entirely outside master grid — skipped")

    return valid


def _base_warp_options(projection, target_bounds, cols, rows, resample_alg, creation_opts):
    return gdal.WarpOptions(
        format="GTiff",
        outputType=gdal.GDT_Float32,
        dstSRS=projection,
        outputBounds=target_bounds,
        width=cols,
        height=rows,
        resampleAlg=resample_alg,
        dstNodata=np.nan,
        targetAlignedPixels=False,
        warpOptions=[
            "INIT_DEST=NO_DATA",
            "UNIFIED_SRC_NODATA=YES",   # prevents NoData bleeding at data edges
        ],
        creationOptions=creation_opts,
    )


def _apply_mask(out_ds, master_arr):
    """
    Set pixels to NaN wherever master == 0.

    The extreme-value cleanup (GDAL warp sentinel ~-3.4e38) is restricted to
    pixels that are already masked out.  This prevents valid source-edge pixels
    — where GDAL simply had no coverage — from being wrongly set to NaN when
    the master marks them as valid (== 1).
    """
    already_invalid = (master_arr == 0)

    for band_index in range(1, out_ds.RasterCount + 1):
        band = out_ds.GetRasterBand(band_index)
        data = band.ReadAsArray()

        # Warp sentinel: extreme float values can only legitimately appear in
        # pixels that are masked out, so restrict cleanup to that region.
        warp_artifact = (data < -1e20)
        invalid = already_invalid | (warp_artifact & already_invalid)
        data[invalid] = np.nan

        band.WriteArray(data)
        band.SetNoDataValue(np.nan)
        band.FlushCache()


# =============================================================================
# Per-file Alignment
# =============================================================================

def align_raster(
    input_path,
    reference_path,
    output_path,
    resample_alg=gdal.GRA_NearestNeighbour,
    apply_mask=False,
    creation_opts=None,
) -> None:
    """
    Reproject and align a single raster to perfectly match a reference grid,
    then optionally apply the master as a binary mask.

    Parameters
    ----------
    input_path     : Path to the input raster to be aligned.
    reference_path : Path to the master reference raster defining the target grid.
    output_path    : Path where the aligned output raster will be saved.
    resample_alg   : GDAL resampling algorithm. Default: GRA_NearestNeighbour.
    apply_mask     : If True, pixels where master == 0 are set to NaN.
    creation_opts  : List of GDAL creation option strings for the output GeoTIFF.
    """
    if creation_opts is None:
        creation_opts = [
            "COMPRESS=LZW", "BIGTIFF=IF_SAFER",
            "TILED=YES", "BLOCKXSIZE=256", "BLOCKYSIZE=256",
        ]

    in_file  = Path(input_path)
    ref_file = Path(reference_path)
    out_file = Path(output_path)

    if not in_file.is_file():
        raise FileNotFoundError(f"Input raster not found:     {in_file}")
    if not ref_file.is_file():
        raise FileNotFoundError(f"Reference raster not found: {ref_file}")

    out_file.parent.mkdir(parents=True, exist_ok=True)

    ref_ds, gt, projection, cols, rows, master_arr = _read_master(ref_file)
    has_zeros     = bool(np.any(master_arr == 0))
    target_bounds = _target_bounds(gt, cols, rows)

    print(f"[*] Aligning : {in_file.name}")
    print(f"    Master   : {cols} x {rows} px | {gt[1]:.1f} m/px")

    if apply_mask and not has_zeros:
        print("    Mask     : skipped (master grid is all 1s — extent-only grid)")
    elif apply_mask:
        print(f"    Mask     : {int(np.nansum(master_arr == 0)):,} pixels will be set to NaN")
    else:
        print("    Mask     : disabled")

    if not _filter_overlapping([in_file], projection, target_bounds):
        print("    [!] Lies entirely outside master grid — skipping\n")
        ref_ds = None
        return

    warp_opts = _base_warp_options(projection, target_bounds, cols, rows, resample_alg, creation_opts)
    out_ds    = gdal.Warp(str(out_file), str(in_file), options=warp_opts)
    if out_ds is None:
        raise RuntimeError("GDAL Warp failed.")

    if apply_mask and has_zeros:
        _apply_mask(out_ds, master_arr)

    out_ds = ref_ds = None
    print(f"    Output   : {out_file.name}\n")


# =============================================================================
# Merge Mode
# =============================================================================

def merge_rasters(
    input_files,
    reference_path,
    output_path,
    resample_alg=gdal.GRA_NearestNeighbour,
    overlap_strategy="first",
    apply_mask=False,
    creation_opts=None,
    strip_height=1024,
) -> None:
    """
    Reproject and mosaic multiple rasters onto a master grid.

    Overlap strategies
    ------------------
    first   Build a VRT then warp once; first file wins in overlapping areas.
    last    Same as 'first' but last file wins.
    average Per-tile warp into strip-sized vsimem buffers; NaN-aware mean.
            Memory cost: O(strip_height × cols) regardless of tile count.

    Parameters
    ----------
    input_files      : Iterable of paths to input rasters.
    reference_path   : Path to the master reference GeoTIFF.
    output_path      : Path where the merged output raster will be saved.
    resample_alg     : GDAL resampling algorithm applied when warping inputs.
    overlap_strategy : "first", "last", or "average".
    apply_mask       : If True, pixels where master == 0 are set to NaN.
    creation_opts    : GDAL creation options for the output GeoTIFF.
    strip_height     : Row height of each processing strip (average mode only).
    """
    if creation_opts is None:
        creation_opts = [
            "COMPRESS=LZW", "BIGTIFF=IF_SAFER",
            "TILED=YES", "BLOCKXSIZE=256", "BLOCKYSIZE=256",
        ]

    ref_file = Path(reference_path)
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    ref_ds, gt, projection, cols, rows, master_arr = _read_master(ref_file)
    has_zeros     = bool(np.any(master_arr == 0))
    target_bounds = _target_bounds(gt, cols, rows)

    input_files = _filter_overlapping(input_files, projection, target_bounds)
    if not input_files:
        raise RuntimeError("No input files overlap the master grid — nothing to merge.")

    n_files = len(input_files)
    print(f"[*] Merging  : {n_files} file(s) → {out_file.name}")
    print(f"    Master   : {cols} x {rows} px | {gt[1]:.1f} m/px")
    print(f"    Strategy : {overlap_strategy}")

    if apply_mask and not has_zeros:
        print("    Mask     : skipped (master grid is all 1s — extent-only grid)")
    elif apply_mask:
        print(f"    Mask     : {int(np.nansum(master_arr == 0)):,} pixels will be set to NaN")
    else:
        print("    Mask     : disabled")

    # ------------------------------------------------------------------
    # Strategy A — first / last
    # ------------------------------------------------------------------
    if overlap_strategy in ("first", "last"):
        files = list(input_files)
        if overlap_strategy == "first":
            files = files[::-1]   # GDAL last-wins → reverse for first-wins

        # Check CRS homogeneity.
        # Note: we deliberately avoid ExportToProj4() here — for compound,
        # vertical, or some newer CRSs PROJ4 cannot represent the SRS and
        # raises "OGR Error: Unsupported SRS" under gdal.UseExceptions().
        # IsSame() compares the SRS objects directly and is robust.
        ref_srs = None
        all_same_crs = True
        for f in files:
            ds = gdal.Open(str(f))
            if ds is None:
                continue
            src_srs = osr.SpatialReference()
            src_srs.ImportFromWkt(ds.GetProjection())
            src_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
            ds = None
            if ref_srs is None:
                ref_srs = src_srs
            elif not src_srs.IsSame(ref_srs):
                all_same_crs = False
                break

        warp_opts = _base_warp_options(
            projection, target_bounds, cols, rows, resample_alg, creation_opts
        )

        if all_same_crs:
            print("    CRS      : homogeneous — VRT + single Warp")
            vrt_path = "/vsimem/merge_input.vrt"
            vrt_opts = gdal.BuildVRTOptions(resampleAlg=resample_alg, srcNodata=np.nan)
            vrt_ds   = gdal.BuildVRT(vrt_path, [str(f) for f in files], options=vrt_opts)
            if vrt_ds is None:
                raise RuntimeError("gdal.BuildVRT failed.")
            out_ds = gdal.Warp(str(out_file), vrt_ds, options=warp_opts)
            vrt_ds = None
            gdal.Unlink(vrt_path)
        else:
            print(f"    CRS      : mixed — multi-input Warp")
            out_ds = gdal.Warp(str(out_file), [str(f) for f in files], options=warp_opts)

        if out_ds is None:
            raise RuntimeError("GDAL Warp failed.")

    # ------------------------------------------------------------------
    # Strategy B — average (strip-based, constant memory)
    # ------------------------------------------------------------------
    elif overlap_strategy == "average":
        _probe  = gdal.Open(str(input_files[0]))
        n_bands = _probe.RasterCount
        _probe  = None

        driver = gdal.GetDriverByName("GTiff")
        out_ds = driver.Create(
            str(out_file), cols, rows, n_bands,
            gdal.GDT_Float32, options=creation_opts,
        )
        out_ds.SetProjection(projection)
        out_ds.SetGeoTransform(gt)
        for b in range(1, n_bands + 1):
            out_ds.GetRasterBand(b).SetNoDataValue(np.nan)

        mem_creation_opts = ["COMPRESS=NONE", "TILED=NO"]
        pixel_h  = abs(gt[5])
        n_strips = (rows + strip_height - 1) // strip_height
        print(f"    Strips   : {n_strips} × {strip_height} rows "
              f"({n_strips * n_files} warp calls total)")

        for strip_idx, y_off in enumerate(range(0, rows, strip_height)):
            actual_h = min(strip_height, rows - y_off)

            # Compute strip bounds from pixel size to avoid floating-point
            # drift from repeated gt[5] multiplications.
            # Clamp the first and last strip to the exact global bounds so
            # there are no hairline gaps or double-counted rows at seams.
            strip_max_y = gt[3] - y_off              * pixel_h
            strip_min_y = gt[3] - (y_off + actual_h) * pixel_h

            if y_off == 0:
                strip_max_y = target_bounds[3]
            if y_off + actual_h >= rows:
                strip_min_y = target_bounds[1]

            strip_bounds = (target_bounds[0], strip_min_y,
                            target_bounds[2], strip_max_y)

            acc_sum   = np.zeros((n_bands, actual_h, cols), dtype=np.float64)
            acc_count = np.zeros((actual_h, cols),          dtype=np.uint16)

            for f in input_files:
                strip_warp_opts = gdal.WarpOptions(
                    format="GTiff",
                    outputType=gdal.GDT_Float32,
                    dstSRS=projection,
                    outputBounds=strip_bounds,
                    width=cols,
                    height=actual_h,
                    resampleAlg=resample_alg,
                    dstNodata=np.nan,
                    targetAlignedPixels=False,
                    warpOptions=[
                        "INIT_DEST=NO_DATA",
                        "UNIFIED_SRC_NODATA=YES",
                    ],
                    creationOptions=mem_creation_opts,
                )
                ds = gdal.Warp("/vsimem/strip_tmp.tif", str(f), options=strip_warp_opts)
                if ds is None:
                    gdal.Unlink("/vsimem/strip_tmp.tif")
                    continue

                valid_mask = None
                for b in range(n_bands):
                    arr = ds.GetRasterBand(b + 1).ReadAsArray().astype(np.float32)
                    arr[arr < -1e20] = np.nan
                    if valid_mask is None:
                        valid_mask = np.isfinite(arr)
                    acc_sum[b] += np.where(np.isfinite(arr), arr, 0.0)

                acc_count += valid_mask.astype(np.uint16)
                ds = None
                gdal.Unlink("/vsimem/strip_tmp.tif")

            with np.errstate(invalid="ignore"):
                for b in range(n_bands):
                    mean_strip = np.where(
                        acc_count > 0,
                        acc_sum[b] / acc_count,
                        np.nan,
                    ).astype(np.float32)
                    out_ds.GetRasterBand(b + 1).WriteArray(mean_strip, 0, y_off)

            if (strip_idx + 1) % 5 == 0 or (strip_idx + 1) == n_strips:
                print(f"    Progress : strip {strip_idx + 1}/{n_strips}")

        for b in range(1, n_bands + 1):
            out_ds.GetRasterBand(b).FlushCache()

    else:
        raise ValueError(
            f"Unknown overlap_strategy '{overlap_strategy}'. "
            "Choose 'first', 'last', or 'average'."
        )

    if apply_mask and has_zeros:
        _apply_mask(out_ds, master_arr)

    out_ds = ref_ds = None
    print(f"    Output   : {out_file}\n")


# =============================================================================
# Resampling algorithm lookup
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


# =============================================================================
# Batch Runner
# =============================================================================

def main(config_file):
    config_path = Path(config_file).resolve()
    if not config_path.exists():
        print(f"[!] Config not found: {config_path}")
        return

    config       = load_config(config_path)
    project_root = config_path.parent.parent

    # Master grid
    master_rel  = config["paths"]["master_grid"]
    master_file = (
        Path(master_rel)
        if Path(master_rel).is_absolute()
        else project_root / master_rel
    )
    if not master_file.exists():
        print(f"[!] Master grid missing: {master_file}")
        return

    # Output directory
    out_rel = config["paths"]["output_dir"]
    out_dir = (
        Path(out_rel)
        if Path(out_rel).is_absolute()
        else project_root / out_rel
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resampling algorithm
    alg_name = config["processing"].get("resample_alg", "nearest").lower()
    resample  = RESAMPLE_ALGORITHMS.get(alg_name, gdal.GRA_NearestNeighbour)
    if alg_name not in RESAMPLE_ALGORITHMS:
        print(f"[!] Unknown resample_alg '{alg_name}'. Falling back to nearest.")

    apply_mask = config["processing"].get("apply_mask", True)

    # Merge / per-file mode
    merge_mode       = config["processing"].get("merge_output", False)
    overlap_strategy = config["processing"].get("overlap_strategy", "first").lower()
    output_name      = config["processing"].get("output_name", "merged_output.tif")
    strip_height     = config["processing"].get("strip_height", 1024)

    # Output format
    fmt        = config.get("output_format", {})
    bigtiff    = str(fmt.get("bigtiff",    "IF_SAFER")).upper()
    compress   = str(fmt.get("compress",   "LZW")).upper()
    predictor  = fmt.get("predictor",  1)
    block_size = fmt.get("block_size", 256)

    creation_opts = [
        f"COMPRESS={compress}",
        f"BIGTIFF={bigtiff}",
        "TILED=YES",
        f"BLOCKXSIZE={block_size}",
        f"BLOCKYSIZE={block_size}",
    ]
    if compress == "DEFLATE":
        creation_opts.append(f"PREDICTOR={predictor}")

    # Collect input files
    inputs_cfg  = config["inputs"]
    input_files = []

    if "glob" in inputs_cfg:
        base = Path(inputs_cfg["glob"]["dir"])
        base = base if base.is_absolute() else project_root / base
        input_files = sorted(base.glob(inputs_cfg["glob"]["pattern"]))
    elif "files" in inputs_cfg:
        for f in inputs_cfg["files"]:
            p = Path(f) if Path(f).is_absolute() else project_root / Path(f)
            input_files.append(p)

    if not input_files:
        print("[!] No input files found. Check config inputs section.")
        return

    print(f"[*] Master grid      : {master_file.name}")
    print(f"[*] Resample         : {alg_name}")
    print(f"[*] Apply mask       : {apply_mask}")
    print(f"[*] Mode             : {'merge → ' + overlap_strategy if merge_mode else 'per-file'}")
    print(f"[*] BigTIFF          : {bigtiff}")
    print(f"[*] Compress         : {compress}"
          + (f" (predictor={predictor})" if compress == "DEFLATE" else ""))
    print(f"[*] Block size       : {block_size} px")
    print(f"[*] Files to process : {len(input_files)}\n")

    if merge_mode:
        output_file = out_dir / output_name
        try:
            merge_rasters(
                input_files=input_files,
                reference_path=master_file,
                output_path=output_file,
                resample_alg=resample,
                overlap_strategy=overlap_strategy,
                strip_height=strip_height,
                apply_mask=apply_mask,
                creation_opts=creation_opts,
            )
        except Exception as e:
            print(f"[!] Merge failed — {e}\n")
    else:
        for input_file in input_files:
            output_file = out_dir / f"snapped_{input_file.name}"
            try:
                align_raster(
                    input_path=input_file,
                    reference_path=master_file,
                    output_path=output_file,
                    resample_alg=resample,
                    apply_mask=apply_mask,
                    creation_opts=creation_opts,
                )
            except Exception as e:
                print(f"[!] Failed: {input_file.name} — {e}\n")

    print(f"[*] Done. Output dir: {out_dir}")


if __name__ == "__main__":
    current_dir = Path(__file__).resolve().parent
    config_path = current_dir.parent / "config" / "03_snap_raster.yaml"
    main(config_file=str(config_path))