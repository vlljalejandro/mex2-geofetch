"""
Geological Map Rasterisation
Implements the method described in Section 2.3.4 of the technical report:
 - integer encoding of categorical attributes (code 0 = no-data)
 - polygon prep: HA_CONF==0 polygons -> code 0, burned first
 - valid polygons sorted ascending by area, small polygons burned last (priority)
 - center-point rasterisation onto the 10m master grid
 - boundary gap filling via nearest-neighbour (Euclidean distance transform)
 - companion CSV lookup tables

Per-field GeoTIFFs are written to the output directory. NetCDF cube
assembly is handled separately (e.g. 04_cube_raster.py).

Config-driven: see config_rasterize_geology.yaml
"""

import os
import csv
import yaml
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from scipy.ndimage import distance_transform_edt

# === PATHS ===
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
CONFIG_PATH = os.path.join(THIS_DIR, "config/rasterize_geology.yaml")

# === LOAD CONFIG ===
with open(CONFIG_PATH, "r") as f:
    cfg = yaml.safe_load(f)

SHAPEFILE_PATH = os.path.join(PROJECT_ROOT, cfg["paths"]["shapefile"])
MASTER_GRID_PATH = os.path.join(PROJECT_ROOT, cfg["paths"]["master_grid"])
OUT_DIR = os.path.join(PROJECT_ROOT, cfg["paths"]["output_dir"])

PROJECT_CRS = cfg["crs"]["project_crs"]

HA_FIELDS = cfg["fields"]["harmonised_fields"]
HA_CONF_FIELD = cfg["fields"]["confidence_field"]
UNCLASSIFIED_VALUE = cfg["fields"]["unclassified_value"]

NODATA_CODE = cfg["rasterisation"]["nodata_code"]
RASTER_DTYPE = cfg["rasterisation"]["dtype"]
ALL_TOUCHED = cfg["rasterisation"]["all_touched"]

CONSTRAIN_TO_LAND_MASK = cfg["gapfill"]["constrain_to_land_mask"]

GTIFF_PROFILE_EXTRA = dict(
    compress=cfg["output_geotiff"]["compress"],
    tiled=cfg["output_geotiff"]["tiled"],
    blockxsize=cfg["output_geotiff"]["blockxsize"],
    blockysize=cfg["output_geotiff"]["blockysize"],
)


# === HELPERS ===

def build_class_lookup(gdf, field):
    """
    Sort unique class labels alphabetically and assign consecutive integer
    codes starting at 1. Code 0 is reserved as the no-data sentinel.
    Returns: dict {label: code}, list of (code, label) rows for CSV.
    """
    labels = sorted(gdf[field].dropna().unique().tolist())
    label_to_code = {label: i + 1 for i, label in enumerate(labels)}
    csv_rows = [(NODATA_CODE, "NO_DATA")] + [(code, label) for label, code in label_to_code.items()]
    return label_to_code, csv_rows


def write_lookup_csv(csv_rows, field, out_dir):
    out_path = os.path.join(out_dir, f"{field}_lookup.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["code", "label"])
        writer.writerows(csv_rows)
    return out_path


def prepare_polygons(gdf, field, label_to_code):
    """
    Assign integer code per polygon for this field:
      - HA_CONF == unclassified_value -> code 0 for all fields
      - otherwise -> mapped code from label_to_code
    Returns gdf sorted in burn order: unclassified polygons first (burned
    first), then valid polygons ascending by area (small polygons burned
    last, so they overwrite larger enclosing units).
    """
    g = gdf.copy()

    is_unclassified = g[HA_CONF_FIELD] == UNCLASSIFIED_VALUE

    g["burn_code"] = NODATA_CODE
    valid_mask = ~is_unclassified
    g.loc[valid_mask, "burn_code"] = (
        g.loc[valid_mask, field].map(label_to_code).fillna(NODATA_CODE).astype(int)
    )

    g["_area"] = g.geometry.area
    g["_group"] = np.where(is_unclassified, 0, 1)
    g = g.sort_values(by=["_group", "_area"], ascending=[True, True], kind="stable")

    return g


def rasterize_field(gdf_sorted, field_code_col, master_transform, master_shape):
    """
    Center-point rasterisation (all_touched=False per config): a pixel is
    assigned the burn value of the containing polygon iff the pixel center
    falls within that polygon.

    Polygons are passed in burn order (unclassified first, then valid
    polygons ascending by area), so later (smaller) polygons overwrite
    earlier ones at shared/overlapping pixels.
    """
    shapes = (
        (geom, int(code))
        for geom, code in zip(gdf_sorted.geometry, gdf_sorted[field_code_col])
    )

    out = rasterize(
        shapes=shapes,
        out_shape=master_shape,
        transform=master_transform,
        fill=NODATA_CODE,
        all_touched=ALL_TOUCHED,
        dtype=RASTER_DTYPE,
        merge_alg=rasterio.enums.MergeAlg.replace,
    )
    return out


def gapfill_nearest(arr, land_mask):
    """
    Nearest-neighbour fill of code-0 pixels within the AOI land mask,
    via a 2D Euclidean distance transform. Ocean / out-of-AOI pixels
    (land_mask == 0) retain code 0.
    """
    if CONSTRAIN_TO_LAND_MASK:
        unclassified = (arr == NODATA_CODE) & (land_mask == 1)
    else:
        unclassified = arr == NODATA_CODE

    if not unclassified.any():
        return arr

    # Pixels that ARE classified (i.e. valid "seed" pixels for the EDT).
    # If this is empty, EDT cannot fill anything.
    classified = ~unclassified
    if CONSTRAIN_TO_LAND_MASK:
        classified_within_land = classified & (land_mask == 1)
    else:
        classified_within_land = classified

    if not classified_within_land.any():
        print("  [!] WARNING: no classified pixels within land mask — "
              "cannot gap-fill, all land pixels would remain 0.")
        return arr

    # distance_transform_edt(input): for each True pixel in `input`, finds
    # nearest False pixel. We want: for each unclassified (True) pixel,
    # nearest classified (False) pixel -> input array should be `unclassified`,
    # but EDT measures distance to nearest ZERO/False, i.e. nearest pixel
    # NOT in `unclassified`. That includes pixels OUTSIDE the land mask too,
    # which could be the bug: a pixel near the AOI edge might get its
    # "nearest classified" from an out-of-land-mask pixel (which is also 0
    # in `arr`, code NODATA_CODE) -- but that's fine for `indices` since we
    # then look up arr at that index, which would be NODATA_CODE again,
    # leaving the pixel at 0 even after "filling".
    #
    # Fix: restrict the EDT's notion of "classified" to pixels that are
    # BOTH non-zero in `arr` AND within the land mask, by setting the EDT
    # input mask to `~classified_within_land` (True = not a valid seed).
    not_seed = ~classified_within_land

    _, indices = distance_transform_edt(not_seed, return_indices=True)

    filled = arr.copy()
    nearest_vals = arr[tuple(indices)]
    filled[unclassified] = nearest_vals[unclassified]

    if CONSTRAIN_TO_LAND_MASK:
        filled[land_mask == 0] = NODATA_CODE

    residual = (filled == NODATA_CODE) & (land_mask == 1)
    if residual.any():
        print(f"  [!] WARNING: {residual.sum()} land pixels still code 0 "
              f"after gap-fill.")

    return filled


def write_band_geotiff(arr, transform, crs, out_path):
    profile = dict(
        driver="GTiff",
        height=arr.shape[0],
        width=arr.shape[1],
        count=1,
        dtype=RASTER_DTYPE,
        crs=crs,
        transform=transform,
        nodata=None,
        **GTIFF_PROFILE_EXTRA,
    )
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(arr, 1)


# === MAIN ===

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # --- Load master grid (land/valid mask, 10m) ---
    print("Loading master grid...")
    with rasterio.open(MASTER_GRID_PATH) as src:
        master_transform = src.transform
        master_crs = src.crs
        master_shape = (src.height, src.width)
        land_mask = src.read(1)  # 1 = valid/land, 0 = null/water

    land_mask = (land_mask == 1).astype(np.uint8)

    # --- Load and validate polygons ---
    print("Loading geology shapefile...")
    if SHAPEFILE_PATH.lower().endswith(".zip"):
        # geopandas/fiona can read shapefiles directly from a zip archive
        # using the "zip://" VFS prefix. If the zip contains the .shp at
        # its root, this works as-is; if nested in a subfolder, append
        # "!/<subfolder>" to the path.
        gdf = gpd.read_file(f"zip://{SHAPEFILE_PATH}")
    else:
        gdf = gpd.read_file(SHAPEFILE_PATH)

    if gdf.crs is None or str(gdf.crs).upper() != PROJECT_CRS.upper():
        print(f"Reprojecting polygons to {PROJECT_CRS}...")
        gdf = gdf.to_crs(PROJECT_CRS)

    invalid = ~gdf.geometry.is_valid
    if invalid.any():
        print(f"Fixing {invalid.sum()} invalid geometries...")
        gdf.loc[invalid, "geometry"] = gdf.loc[invalid, "geometry"].buffer(0)

    # buffer(0) can collapse degenerate geometries to empty/None;
    # drop these so rasterize doesn't emit ShapeSkipWarning for them.
    empty_or_none = gdf.geometry.isna() | gdf.geometry.is_empty
    if empty_or_none.any():
        print(f"Dropping {empty_or_none.sum()} empty/invalid geometries "
              f"at index: {gdf.index[empty_or_none].tolist()}")
        gdf = gdf.loc[~empty_or_none].reset_index(drop=True)

    if HA_CONF_FIELD not in gdf.columns:
        raise ValueError(f"Expected confidence field '{HA_CONF_FIELD}' not found in shapefile attributes")

    for field in HA_FIELDS:
        print(f"\n=== Processing field: {field} ===")

        # 1. Integer encoding (alphabetical, starting at 1; 0 = no-data)
        label_to_code, csv_rows = build_class_lookup(gdf, field)
        write_lookup_csv(csv_rows, field, OUT_DIR)
        print(f"  {len(label_to_code)} classes encoded, lookup CSV written.")

        # 2. Polygon prep + priority ordering
        gdf_sorted = prepare_polygons(gdf, field, label_to_code)

        # 3. Rasterise (center-point rule)
        print("  Rasterising to 10m master grid...")
        arr = rasterize_field(gdf_sorted, "burn_code", master_transform, master_shape)

        # 4. Gap fill (nearest-neighbour EDT, constrained to AOI land mask)
        print("  Gap-filling boundary no-data pixels...")
        arr_filled = gapfill_nearest(arr, land_mask)

        # 5. Write per-field GeoTIFF
        tif_path = os.path.join(OUT_DIR, f"{field}.tif")
        write_band_geotiff(arr_filled, master_transform, master_crs, tif_path)
        print(f"  Wrote {tif_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
