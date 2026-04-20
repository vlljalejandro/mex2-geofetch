import os
import numpy as np
from osgeo import gdal

# ── Configure here ──────────────────────────────────────────────
folder = "data/03_snap_data/cubicspline_interpolated"
# folder = snap_magnetic_nearest
# ────────────────────────────────────────────────────────────────

gdal.UseExceptions()

tifs = sorted(
    f for f in os.listdir(folder) if f.lower().endswith((".tif", ".tiff"))
)

if not tifs:
    print(f"No TIF files found in: {folder}")
else:
    header = f"{'File':<40} {'Rows':>8} {'Cols':>8} {'Valid':>12} {'Nulls':>12}"
    print(header)
    print("-" * len(header))

    for fname in tifs:
        path = os.path.join(folder, fname)
        ds = gdal.Open(path)

        rows = ds.RasterYSize
        cols = ds.RasterXSize
        band = ds.GetRasterBand(1)
        nodata = band.GetNoDataValue()

        arr = band.ReadAsArray().astype(np.float64)

        # # Null pixels are NaN only
        # null_mask = np.isnan(arr)

        # Several null targets
        target_nulls = [0]  # List all values considered as null
        null_mask = np.isin(arr, target_nulls) | np.isnan(arr)

        nulls = int(null_mask.sum())
        valid = int(arr.size - nulls)
        print(f"{fname:<40} {rows:>8} {cols:>8} {valid:>12,} {nulls:>12,}")

# Running code:
# python utils/print_tif_statistics.py