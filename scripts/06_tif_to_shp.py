"""
raster_to_smooth_gpkg.py
────────────────────────
Pipeline for large classified rasters (e.g. 36864×36864, 8-bit, 8 classes, nodata=255):
  1. SIEVE FILTER   – remove small isolated pixel groups (blending)
  2. POLYGONIZE     – raster → vector polygons (GDAL, streamed)
  3. DISSOLVE       – merge adjacent polygons of the same class  ← KEY FIX
  4. SMOOTH         – Chaikin corner-cutting to remove jagged edges
  5. EXPORT         – write final GeoPackage (no 4 GB limit)

Dependencies:
    pip install gdal numpy shapely geopandas tqdm
    (GDAL must also be installed system-wide or via conda)
"""

import json
import os
import sys

import numpy as np

try:
    from osgeo import gdal, ogr, osr
    gdal.UseExceptions()
except ImportError:
    sys.exit("ERROR: GDAL Python bindings not found.")

try:
    import geopandas as gpd
    from shapely.geometry import shape, MultiPolygon, Polygon
    from shapely.validation import make_valid
except ImportError:
    sys.exit("ERROR: geopandas / shapely not found.")

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION  ← edit these values
# ══════════════════════════════════════════════════════════════════════════════

INPUT_TIF   = "data/05_quaternary_output/quaternary_cover.tif"
OUTPUT_GPKG = "data/06_output/smooth_output.gpkg"   # ← GeoPackage, no 4 GB limit

# -- Sieve filter -------------------------------------------------------------
# For a 36864² raster (≈1.36 billion pixels, ~8 classes) a threshold of 500 is
# tiny. Rule of thumb: aim to remove features smaller than ~1 ha of your map.
# If one pixel = 10 m → 1 ha = 1000 px. Start at 2000–10000.
SIEVE_THRESHOLD  = 50000   # ↑ raise this aggressively to reduce polygon count
CONNECTEDNESS    = 8      # 8 = include diagonals (recommended)

# -- Minimum polygon area AFTER dissolve (map units²) -------------------------
# Removes any tiny leftover polygons before smoothing.
# If CRS is in metres and pixel ≈ 10 m: 10000 = 1 ha. Set 0 to disable.
MIN_AREA = 10_000

# -- Smoothing ----------------------------------------------------------------
SIMPLIFY_TOL      = 5.0   # Douglas-Peucker tolerance (map units ≈ pixels).
                           # Raised from 2 → 5 to cut vertex count further.
SMOOTH_ITERATIONS = 3     # Chaikin corner-cutting passes.

# -- Nodata -------------------------------------------------------------------
NODATA_VALUE = 255

# -- Misc ---------------------------------------------------------------------
KEEP_SIEVED_TIF = False   # True → keep intermediate blended TIF for QGIS inspection

# ══════════════════════════════════════════════════════════════════════════════


# ──────────────────────────────────────────────────────────────────────────────
# Step 1 – Sieve filter
# ──────────────────────────────────────────────────────────────────────────────

def sieve_raster(src_path: str, dst_path: str) -> None:
    print(f"\n[1/4] Sieve filter  (threshold={SIEVE_THRESHOLD} px, connectedness={CONNECTEDNESS})")

    src_ds = gdal.Open(src_path, gdal.GA_ReadOnly)
    if src_ds is None:
        sys.exit(f"Cannot open input raster: {src_path}")

    driver      = gdal.GetDriverByName("GTiff")
    create_opts = ["COMPRESS=LZW", "TILED=YES", "BLOCKXSIZE=512", "BLOCKYSIZE=512", "BIGTIFF=YES"]
    dst_ds      = driver.CreateCopy(dst_path, src_ds, strict=0, options=create_opts)
    dst_ds.FlushCache()

    src_band = src_ds.GetRasterBand(1)
    dst_band = dst_ds.GetRasterBand(1)

    nodata = src_band.GetNoDataValue()
    if nodata is not None:
        dst_band.SetNoDataValue(nodata)

    print("    Running sieve filter (may take several minutes for large rasters)…")
    gdal.SieveFilter(
        src_band, None, dst_band,
        SIEVE_THRESHOLD, CONNECTEDNESS,
        callback=gdal.TermProgress_nocb,
    )

    dst_ds.FlushCache()
    dst_ds = None
    src_ds = None
    print("    Sieve filter complete.")


# ──────────────────────────────────────────────────────────────────────────────
# Step 2 – Polygonize
# ──────────────────────────────────────────────────────────────────────────────

def polygonize_raster(raster_path: str) -> gpd.GeoDataFrame:
    print(f"\n[2/4] Polygonize raster → vector")

    rds  = gdal.Open(raster_path, gdal.GA_ReadOnly)
    band = rds.GetRasterBand(1)

    mem_driver = ogr.GetDriverByName("Memory")
    mem_ds     = mem_driver.CreateDataSource("memdata")

    srs = osr.SpatialReference()
    wkt = rds.GetProjectionRef()
    if wkt:
        srs.ImportFromWkt(wkt)

    layer = mem_ds.CreateLayer("polygons", srs=srs, geom_type=ogr.wkbPolygon)
    layer.CreateField(ogr.FieldDefn("class_id", ogr.OFTInteger))

    print("    Running polygonize (streaming)…")
    gdal.Polygonize(band, None, layer, 0, [], callback=gdal.TermProgress_nocb)

    features   = []
    feat_count = layer.GetFeatureCount()
    layer.ResetReading()

    iter_obj = tqdm(range(feat_count), desc="    Reading features", unit="feat") \
               if HAS_TQDM else range(feat_count)

    for _ in iter_obj:
        feat = layer.GetNextFeature()
        if feat is None:
            break
        geom     = shape(json.loads(feat.GetGeometryRef().ExportToJson()))
        class_id = feat.GetField("class_id")
        features.append({"class_id": class_id, "geometry": geom})

    rds = None

    if not features:
        sys.exit("Polygonize produced no features. Check your input raster.")

    gdf = gpd.GeoDataFrame(features, crs=wkt if wkt else None)
    gdf = gdf[gdf["class_id"] != NODATA_VALUE].copy()
    print(f"    {len(gdf):,} raw polygons extracted (nodata={NODATA_VALUE} excluded).")
    return gdf


# ──────────────────────────────────────────────────────────────────────────────
# Step 3 – DISSOLVE by class_id  ← the critical missing step
# ──────────────────────────────────────────────────────────────────────────────

def dissolve_and_filter(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Merge all touching polygons that share the same class_id into a single
    (Multi)Polygon. This is the single most effective step to cut polygon count:
    from millions of pixel-derived pieces down to tens of thousands or fewer.

    Then drop any remaining tiny polygons smaller than MIN_AREA.
    """
    print(f"\n[3/4] Dissolve by class_id + area filter (min={MIN_AREA} map-units²)")

    # dissolve – unions all geometries per class, then explode to single-part
    dissolved = (
        gdf.dissolve(by="class_id", as_index=False)
           .explode(index_parts=False)
           .reset_index(drop=True)
    )
    print(f"    After dissolve  : {len(dissolved):,} polygons")

    # make valid before area calculation
    dissolved["geometry"] = dissolved["geometry"].apply(make_valid)

    if MIN_AREA > 0:
        dissolved = dissolved[dissolved.geometry.area >= MIN_AREA].copy()
        print(f"    After area filter: {len(dissolved):,} polygons")

    return dissolved


# ──────────────────────────────────────────────────────────────────────────────
# Step 4 – Smooth geometries
# ──────────────────────────────────────────────────────────────────────────────

def chaikin_smooth(coords: np.ndarray) -> np.ndarray:
    new_coords = []
    for i in range(len(coords) - 1):
        A, B = coords[i], coords[i + 1]
        new_coords.extend([0.75 * A + 0.25 * B,
                           0.25 * A + 0.75 * B])
    new_coords.append(new_coords[0])
    return np.array(new_coords)


def smooth_ring(ring_coords: list) -> list:
    arr = np.array(ring_coords)
    for _ in range(SMOOTH_ITERATIONS):
        arr = chaikin_smooth(arr)
    return arr.tolist()


def smooth_polygon(poly: Polygon) -> Polygon:
    poly = poly.simplify(SIMPLIFY_TOL, preserve_topology=True)
    poly = make_valid(poly)

    if poly.geom_type == "MultiPolygon":
        parts = [
            Polygon(smooth_ring(list(p.exterior.coords)),
                    [smooth_ring(list(h.coords)) for h in p.interiors])
            for p in poly.geoms
        ]
        return MultiPolygon(parts)

    if poly.geom_type == "Polygon":
        return Polygon(
            smooth_ring(list(poly.exterior.coords)),
            [smooth_ring(list(h.coords)) for h in poly.interiors],
        )

    return poly


def smooth_geodataframe(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    print(f"\n[4/4] Smoothing  (simplify={SIMPLIFY_TOL}, Chaikin passes={SMOOTH_ITERATIONS})")

    iter_obj = tqdm(gdf.iterrows(), total=len(gdf), desc="    Smoothing", unit="poly") \
               if HAS_TQDM else gdf.iterrows()

    smoothed = []
    for _, row in iter_obj:
        try:
            geom = make_valid(row.geometry)
            geom = smooth_polygon(geom)
            geom = make_valid(geom)
        except Exception:
            geom = row.geometry
        smoothed.append(geom)

    gdf             = gdf.copy()
    gdf["geometry"] = smoothed
    gdf             = gdf[~gdf.geometry.is_empty].copy()
    print(f"    Done. {len(gdf):,} valid polygons.")
    return gdf


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    if not os.path.exists(INPUT_TIF):
        sys.exit(f"Input file not found: {INPUT_TIF}")

    os.makedirs(os.path.dirname(os.path.abspath(OUTPUT_GPKG)) or ".", exist_ok=True)

    base        = os.path.splitext(OUTPUT_GPKG)[0]
    sieved_path = base + ("_sieved.tif" if KEEP_SIEVED_TIF else "_sieved_tmp.tif")

    print("=" * 60)
    print("  Raster → Smooth GeoPackage Pipeline")
    print("=" * 60)
    print(f"  Input              : {INPUT_TIF}")
    print(f"  Output             : {OUTPUT_GPKG}")
    print(f"  Sieve threshold    : {SIEVE_THRESHOLD} px")
    print(f"  Min polygon area   : {MIN_AREA} map-units²")
    print(f"  Simplify tolerance : {SIMPLIFY_TOL}")
    print(f"  Chaikin iterations : {SMOOTH_ITERATIONS}")
    print(f"  No-data value      : {NODATA_VALUE}")

    # 1 – sieve
    sieve_raster(INPUT_TIF, sieved_path)

    # 2 – polygonize
    gdf = polygonize_raster(sieved_path)

    # 3 – dissolve + filter  ← KEY new step
    gdf = dissolve_and_filter(gdf)

    # 4 – smooth
    gdf = smooth_geodataframe(gdf)

    # 5 – export as GeoPackage (no size limit, single file, widely supported)
    print(f"\n[+] Writing GeoPackage → {OUTPUT_GPKG}")
    gdf.to_file(OUTPUT_GPKG, driver="GPKG", layer="classes")

    if not KEEP_SIEVED_TIF and os.path.exists(sieved_path):
        os.remove(sieved_path)
        print("    Temporary sieved TIF removed.")

    print("\n" + "=" * 60)
    print("  Pipeline complete!")
    print(f"  Output : {OUTPUT_GPKG}")
    print("=" * 60)


if __name__ == "__main__":
    main()