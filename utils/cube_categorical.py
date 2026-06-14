import csv
import yaml
import numpy as np
import xarray as xr
from osgeo import gdal, osr
from pathlib import Path
gdal.UseExceptions()


# =============================================================================
# Categorical Cube Writer
# =============================================================================
# Writes a NetCDF4 file containing one or more *categorical* variables
# (uint8 classification rasters), each paired with a code/label lookup
# CSV that becomes CF-1.8 'flag_values' / 'flag_meanings' attributes.
#
# Conventions:
#   - Input rasters are single-band uint8 (or castable to uint8).
#   - Code 0 = NO_DATA → written as _FillValue, excluded from
#     flag_values / flag_meanings (CF convention: fill value should not
#     also be a valid flag value).
#   - Lookup CSV format:
#       code,label
#       0,NO_DATA
#       1,SOME_CLASS
#       ...
#
# Usage:
#   python 04_cube_categorical.py [config_path]
#   (defaults to config/04_cube_categorical.yaml)
# =============================================================================


# =============================================================================
# Utilities
# =============================================================================

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def geotransform_to_coords(gt, cols, rows):
    """
    Derive 1-D pixel-centre coordinate arrays from a GDAL GeoTransform.
    See 04_cube_raster.py for full convention notes.
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


def load_lookup_table(csv_path):
    """
    Read a code/label lookup CSV.

    Returns
    -------
    codes  : list of int,    sorted ascending, EXCLUDING code 0 (NO_DATA)
    labels : list of str,    matching order
    """
    codes, labels = [], []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            code = int(row['code'])
            if code == 0:
                continue  # NO_DATA — handled as _FillValue, not a flag value
            codes.append(code)
            labels.append(row['label'].strip())

    # sort by code, in case the CSV isn't ordered
    order = np.argsort(codes)
    codes  = [codes[i] for i in order]
    labels = [labels[i] for i in order]
    return codes, labels


# =============================================================================
# Per-file loader
# =============================================================================

def load_categorical_band(tif_path, expected_shape):
    """
    Load a single-band categorical raster as a uint8 array.

    Parameters
    ----------
    tif_path       : Path to the GeoTIFF.
    expected_shape : (rows, cols) — validated against the file.

    Returns
    -------
    np.ndarray, dtype=uint8, shape (rows, cols)
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

    band = ds.GetRasterBand(1)
    arr  = band.ReadAsArray()
    nodata = band.GetNoDataValue()
    ds = None

    arr = arr.astype(np.uint8)

    # Any source nodata value (if not already 0) is remapped to 0.
    if nodata is not None and nodata != 0:
        arr[arr == np.uint8(nodata)] = 0

    return arr


# =============================================================================
# Resolve input file from config
# =============================================================================

def resolve_path(raw, project_root):
    p = Path(raw)
    return p if p.is_absolute() else project_root / p


# =============================================================================
# Core writer
# =============================================================================

def build_categorical_cube(config_file):
    """
    Write a NetCDF4 file containing one or more uint8 categorical variables,
    each annotated with CF-1.8 flag_values / flag_meanings derived from a
    lookup CSV.
    """
    config_path  = Path(config_file).resolve()
    config       = load_config(config_path)
    project_root = config_path.parent.parent

    # ── Paths ────────────────────────────────────────────────────────
    master_file = resolve_path(config['paths']['master_grid'], project_root)
    out_path    = resolve_path(config['paths']['output_nc'], project_root)

    if not master_file.exists():
        print(f"[!] Master grid missing: {master_file}")
        return

    if out_path.suffix != ".nc":
        out_path = out_path.with_suffix(".nc")

    # ── Variables ────────────────────────────────────────────────────
    var_configs = config['variables']
    extra_attrs = config.get('metadata', {})

    # ── Master grid ──────────────────────────────────────────────────
    ref_ds  = gdal.Open(str(master_file))
    gt      = ref_ds.GetGeoTransform()
    cols    = ref_ds.RasterXSize
    rows    = ref_ds.RasterYSize
    crs_wkt = get_crs_wkt(ref_ds)
    xs, ys  = geotransform_to_coords(gt, cols, rows)
    ref_ds  = None

    print(f"[*] Master grid : {master_file.name}  ({cols} x {rows} px, {gt[1]:.1f} m/px)")
    print(f"[*] Variables   : {', '.join(var_configs.keys())}")

    # ── Build each variable ──────────────────────────────────────────
    data_vars = {}
    summary   = []  # (var_name, n_classes)

    for var_name, var_cfg in var_configs.items():
        tif_path    = resolve_path(var_cfg['input'], project_root)
        lookup_path = resolve_path(var_cfg['lookup'], project_root)

        print(f"\n[*] Variable    : '{var_name}'")
        print(f"    raster : {tif_path.name}")
        print(f"    lookup : {lookup_path.name}")

        try:
            arr = load_categorical_band(tif_path, (rows, cols))
        except Exception as e:
            print(f"    [!] Skipped {tif_path.name} — {e}")
            continue

        codes, labels = load_lookup_table(lookup_path)

        da = xr.DataArray(
            arr,
            dims=["y", "x"],
            name=var_name,
        )

        da.attrs.update({
            "long_name":     var_cfg.get('long_name', var_name),
            "flag_values":   np.array(codes, dtype=np.uint8),
            "flag_meanings": " ".join(labels),
        })

        # Per-variable metadata overrides/additions
        var_attrs = extra_attrs.get(var_name, {})
        if var_attrs:
            da.attrs.update(var_attrs)

        print(f"    → {len(codes)} classes  [{labels[0]} .. {labels[-1]}]")

        data_vars[var_name] = da
        summary.append((var_name, len(codes)))

    if not data_vars:
        print("[!] No variables loaded — aborting.")
        return

    # ── Build Dataset ────────────────────────────────────────────────
    ds = xr.Dataset(data_vars, coords={"y": ys, "x": xs})

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

    # Shared/global metadata (top-level keys not matching a variable name)
    shared_attrs = {k: v for k, v in extra_attrs.items() if k not in var_configs}
    if shared_attrs:
        ds.attrs.update(shared_attrs)

    ds["x"].attrs.update({"units": "metre", "axis": "X", "long_name": "Easting"})
    ds["y"].attrs.update({"units": "metre", "axis": "Y", "long_name": "Northing"})

    # ── Write ────────────────────────────────────────────────────────
    compress_level = config.get('processing', {}).get('compress_level', 4)

    encoding = {
        var_name: {
            "dtype":      "uint8",
            "zlib":       True,
            "complevel":  compress_level,
            "shuffle":    True,
            "_FillValue": 0,
        }
        for var_name in data_vars
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()
        print(f"\n[*] Removed existing: {out_path.name}")

    print(f"[*] Writing → {out_path}")
    ds.to_netcdf(str(out_path), format="NETCDF4", encoding=encoding)

    file_size = out_path.stat().st_size
    print(f"\n[✓] Done.  {file_size / 1e6:.1f} MB  |  {out_path}")
    for var_name, n_classes in summary:
        print(f"     - {var_name}: {n_classes} classes")


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    import sys
    current_dir = Path(__file__).resolve().parent
    config_path = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else current_dir.parent / "utils" / "config" / "cube_categorical.yaml"
    )
    build_categorical_cube(config_file=str(config_path))
