import numpy as np
import rasterio
from scipy.interpolate import griddata
from pathlib import Path

# ── CONFIG ────────────────────────────────────────────────────────────────────
MASK_PATH   = "data/00_base_grid/magnetic_base_grid180m.tif"          # 1 = should be filled, 0 = ignore
INPUT_DIR   = Path("data/03_snap_data/cubicspline")  # folder with the other TIFs
OUTPUT_DIR  = Path("data/03_snap_data/cubicspline_interpolated") # filled TIFs written here
# ─────────────────────────────────────────────────────────────────────────────

OUTPUT_DIR.mkdir(exist_ok=True)

# Load mask
with rasterio.open(MASK_PATH) as src:
    mask = src.read(1)          # shape: (rows, cols)
    profile = src.profile       # reuse for output

tif_paths = sorted(INPUT_DIR.glob("*.tif"))
print(f"Found {len(tif_paths)} TIF(s) to process.")

for tif_path in tif_paths:
    with rasterio.open(tif_path) as src:
        data    = src.read(1).astype(np.float32)
        nodata  = src.nodata
        tif_profile = src.profile

    # Treat nodata as NaN if applicable
    if nodata is not None:
        data[data == nodata] = np.nan

    rows, cols = np.indices(data.shape)

    # ── Identify valid (known) pixels ────────────────────────────────────────
    valid_mask = ~np.isnan(data)
    valid_rows = rows[valid_mask]
    valid_cols = cols[valid_mask]
    valid_values = data[valid_mask]

    # ── Identify fill targets: mask==1 AND value is NaN ──────────────────────
    fill_mask = (mask == 1) & np.isnan(data)
    fill_rows = rows[fill_mask]
    fill_cols = cols[fill_mask]

    filled_data = data.copy()

    if fill_rows.size > 0 and valid_values.size > 0:
        interpolated = griddata(
            points     = np.column_stack([valid_rows, valid_cols]),
            values     = valid_values,
            xi         = np.column_stack([fill_rows, fill_cols]),
            method     = "linear",
            fill_value = np.nan,   # flag any still-unfilled after linear pass
        )

        # Optional: nearest-neighbour fallback for points outside convex hull
        still_nan = np.isnan(interpolated)
        if still_nan.any():
            interpolated_nn = griddata(
                points     = np.column_stack([valid_rows, valid_cols]),
                values     = valid_values,
                xi         = np.column_stack([fill_rows[still_nan],
                                              fill_cols[still_nan]]),
                method     = "nearest",
            )
            interpolated[still_nan] = interpolated_nn

        filled_data[fill_mask] = interpolated
        n_filled = fill_rows.size
    else:
        n_filled = 0

    # ── Write output ──────────────────────────────────────────────────────────
    out_path = OUTPUT_DIR / tif_path.name
    tif_profile.update(dtype=rasterio.float32, nodata=np.nan)
    with rasterio.open(out_path, "w", **tif_profile) as dst:
        dst.write(filled_data.astype(np.float32), 1)

    print(f"  {tif_path.name}: {n_filled} NaN pixels filled under mask.")

print("Done.")