import os
import numpy as np
from osgeo import gdal

# ── Configure here ──────────────────────────────────────────────
folder = "dem"
zero_is_null = True  # Set True for binary masks where 0 = null, 1 = data
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

        null_mask = np.isnan(arr)

        if nodata is not None and not np.isnan(nodata):
            null_mask |= (arr == nodata)

        if zero_is_null:
            null_mask |= (arr == 0)

        # Extra domain-specific null values
        extra_nulls = []  # e.g. [-9999, -32768]
        if extra_nulls:
            null_mask |= np.isin(arr, extra_nulls)

        nulls = int(null_mask.sum())
        valid = int(arr.size - nulls)
        print(f"{fname:<40} {rows:>8} {cols:>8} {valid:>12,} {nulls:>12,}")

# Running code:
# python print_tif_statistics.py