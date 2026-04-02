import yaml
import numpy as np
import xarray as xr
from osgeo import gdal, osr
from pathlib import Path
gdal.UseExceptions()


# =============================================================================
# Utilities
# =============================================================================

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def geotransform_to_coords(gt, cols, rows):
    """
    Derive 1-D pixel-centre coordinate arrays from a GDAL GeoTransform.

    The GeoTransform convention:
        gt[0] = west edge of leftmost column (NOT pixel centre)
        gt[3] = north edge of topmost row    (NOT pixel centre)
        gt[1] = pixel width  (positive)
        gt[5] = pixel height (negative for north-up rasters)

    Pixel centres are therefore:
        x[i] = gt[0] + (i + 0.5) * gt[1]
        y[j] = gt[3] + (j + 0.5) * gt[5]
    """
    xs = gt[0] + (np.arange(cols) + 0.5) * gt[1]
    ys = gt[3] + (np.arange(rows) + 0.5) * gt[5]
    return xs, ys


def get_crs_wkt(ds):
    """Return the WKT CRS string from a GDAL dataset."""
    proj = ds.GetProjection()
    if not proj:
        raise ValueError("Reference raster has no CRS.")
    srs = osr.SpatialReference()
    srs.ImportFromWkt(proj)
    return srs.ExportToWkt()


def extract_band_label(path):
    """
    Extract a band label from a filename by taking the last '_'-separated
    segment of the stem.

    Examples
    --------
    emit_mosaic_snapped_B281.tif  →  B281
    gravity.tif                   →  gravity
    """
    return Path(path).stem.split("_")[-1]


# =============================================================================
# Per-file loader
# =============================================================================

def load_tif_bands(tif_path, expected_shape):
    """
    Load all bands from a GeoTIFF as a list of 2-D float32 arrays.

    Parameters
    ----------
    tif_path       : Path to the GeoTIFF.
    expected_shape : (rows, cols) — validated against the file.

    Returns
    -------
    list of np.ndarray, each shape (rows, cols)
    """
    ds = gdal.Open(str(tif_path))
    if ds is None:
        raise RuntimeError(f"GDAL could not open: {tif_path}")

    actual_shape = (ds.RasterYSize, ds.RasterXSize)
    if actual_shape != expected_shape:
        raise ValueError(
            f"Shape mismatch for {Path(tif_path).name}: "
            f"expected {expected_shape}, got {actual_shape}"
        )

    bands = []
    for b in range(1, ds.RasterCount + 1):
        band_obj = ds.GetRasterBand(b)
        arr      = band_obj.ReadAsArray().astype(np.float32)
        nodata   = band_obj.GetNoDataValue()
        if nodata is not None:
            arr[arr == nodata] = np.nan
        arr[arr < -1e20] = np.nan
        bands.append(arr)

    ds = None
    return bands


# =============================================================================
# Resolve input files from config
# =============================================================================

def resolve_input_files(input_cfg, project_root):
    """
    Resolve a list of input file Paths from the variable's 'input' config block.

    Supports two modes:
      file : single explicit file path
      glob : dir + pattern scan
    """
    def resolve(raw):
        p = Path(raw)
        return p if p.is_absolute() else project_root / p

    if 'file' in input_cfg:
        return [resolve(input_cfg['file'])]

    if 'glob' in input_cfg:
        base = resolve(input_cfg['glob']['dir'])
        return sorted(base.glob(input_cfg['glob']['pattern']))

    raise ValueError(f"Unrecognised input mode in config. Use 'file' or 'glob'.")


# =============================================================================
# Stack input files into a DataArray
# =============================================================================

def stack_variable(var_name, input_paths, expected_shape, mask):
    """
    Load one or more GeoTIFFs and return a DataArray.

    Shape is determined automatically:
      - One file, one internal band  →  (y, x)
      - Everything else              →  (band, y, x)

    Parameters
    ----------
    var_name      : Variable name.
    input_paths   : List of Path objects.
    expected_shape: (rows, cols) for shape validation.
    mask          : Boolean 2-D array (True = mask out) or None.

    Returns
    -------
    xr.DataArray, or None if all files failed.
    """
    all_bands       = []
    all_band_labels = []

    for tif_path in input_paths:
        band_label = extract_band_label(tif_path)
        print(f"    {tif_path.name}  →  band '{band_label}'")
        try:
            bands = load_tif_bands(tif_path, expected_shape)
        except Exception as e:
            print(f"    [!] Skipped {tif_path.name} — {e}")
            continue

        if len(bands) > 1:
            for i, arr in enumerate(bands, start=1):
                all_bands.append(arr)
                all_band_labels.append(f"{band_label}_{i}")
        else:
            all_bands.append(bands[0])
            all_band_labels.append(band_label)

    if not all_bands:
        return None

    data = np.stack(all_bands, axis=0)   # (n_bands, rows, cols)

    if mask is not None:
        data[:, mask] = np.nan

    n_bands = len(all_bands)

    if n_bands == 1:
        # True 2-D — no band dimension
        da = xr.DataArray(
            data[0],
            dims=["y", "x"],
            name=var_name,
            attrs={"band_labels": all_band_labels[0]},
        )
    else:
        da = xr.DataArray(
            data,
            dims=["band", "y", "x"],
            coords={"band": np.arange(1, n_bands + 1, dtype=np.int32)},
            name=var_name,
            attrs={"band_labels": ", ".join(all_band_labels)},
        )

    return da


# =============================================================================
# Core writer
# =============================================================================

def build_nc_variable(config_file):
    """
    Write a single-variable NetCDF4 file from one or more GeoTIFFs.

    Shape is determined automatically:
      - One input file, one band  →  (y, x)
      - Everything else           →  (band, y, x)
    """
    config_path  = Path(config_file).resolve()
    config       = load_config(config_path)
    project_root = config_path.parent.parent

    def resolve(raw):
        p = Path(raw)
        return p if p.is_absolute() else project_root / p

    # ── Paths ────────────────────────────────────────────────────────
    master_file = resolve(config['paths']['master_grid'])
    out_path    = resolve(config['paths']['output_nc'])

    if not master_file.exists():
        print(f"[!] Master grid missing: {master_file}")
        return

    if out_path.suffix != ".nc":
        out_path = out_path.with_suffix(".nc")

    # ── Variable config ──────────────────────────────────────────────
    var_cfg    = config['variable']
    var_name   = var_cfg['var_name']
    apply_mask = var_cfg.get('apply_mask', True)

    input_paths = resolve_input_files(var_cfg['input'], project_root)

    # ── Processing options ───────────────────────────────────────────
    compress_level = config.get('processing', {}).get('compress_level', 4)
    extra_attrs    = config.get('metadata', {})

    # ── Master grid ──────────────────────────────────────────────────
    ref_ds     = gdal.Open(str(master_file))
    gt         = ref_ds.GetGeoTransform()
    cols       = ref_ds.RasterXSize
    rows       = ref_ds.RasterYSize
    crs_wkt    = get_crs_wkt(ref_ds)
    xs, ys     = geotransform_to_coords(gt, cols, rows)
    master_arr = ref_ds.GetRasterBand(1).ReadAsArray()
    ref_ds     = None

    mask_full = (master_arr == 0) if bool(np.any(master_arr == 0)) else None
    mask      = mask_full if (apply_mask and mask_full is not None) else None

    print(f"[*] Master grid : {master_file.name}  ({cols} x {rows} px, {gt[1]:.1f} m/px)")
    print(f"[*] Variable    : '{var_name}'  ({len(input_paths)} file(s), mask={apply_mask})")

    # ── Stack ────────────────────────────────────────────────────────
    da = stack_variable(var_name, input_paths, (rows, cols), mask)
    if da is None:
        print(f"[!] No data loaded for '{var_name}' — aborting.")
        return

    shape_str = (
        f"(y={rows}, x={cols})"
        if da.ndim == 2
        else f"(band={da.sizes['band']}, y={rows}, x={cols})"
    )
    print(f"     → shape : {shape_str}")

    # ── Build Dataset ────────────────────────────────────────────────
    ds = xr.Dataset({var_name: da}, coords={"y": ys, "x": xs})

    ds = ds.assign_coords(spatial_ref=0)
    ds["spatial_ref"].attrs.update({
        "crs_wkt":           crs_wkt,
        "grid_mapping_name": "transverse_mercator",
    })

    ds.attrs.update({
        "Conventions":  "CF-1.8",
        "master_grid":  master_file.name,
        "pixel_size_m": float(gt[1]),
        "ncols":        cols,
        "nrows":        rows,
        "x_min":        float(gt[0]),
        "y_max":        float(gt[3]),
    })
    if extra_attrs:
        ds.attrs.update(extra_attrs)

    ds["x"].attrs.update({"units": "metre", "axis": "X", "long_name": "Easting"})
    ds["y"].attrs.update({"units": "metre", "axis": "Y", "long_name": "Northing"})

    # ── Write ────────────────────────────────────────────────────────
    encoding = {
        var_name: {
            "dtype":      "float32",
            "zlib":       True,
            "complevel":  compress_level,
            "shuffle":    True,
            "_FillValue": np.nan,
        }
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()
        print(f"  [*] Removed existing: {out_path.name}")

    print(f"[*] Writing → {out_path}")
    ds.to_netcdf(str(out_path), format="NETCDF4", encoding=encoding)

    file_size = out_path.stat().st_size
    labels    = da.attrs.get("band_labels", "").split(", ")
    n_bands   = 1 if da.ndim == 2 else da.sizes['band']
    print(f"\n[✓] Done.  {n_bands} band(s) [{labels[0]} → {labels[-1]}]"
          f"  |  {file_size / 1e6:.1f} MB  |  {out_path}")


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    import sys
    current_dir = Path(__file__).resolve().parent
    config_path = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else current_dir.parent / "config" / "04_cube_raster.yaml"
    )
    build_nc_variable(config_file=str(config_path))