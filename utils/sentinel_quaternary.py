"""
Sentinel-2 L2A Quaternary Cover Mapping
========================================
Assumptions (based on user's preprocessing):
  • Bands are already snapped to a common base grid — NO resampling needed.
  • Masked pixels are already NaN — NO SCL cloud masking needed.
  • Values are surface reflectance [0,1] — NO DN scaling needed.

Outputs:
  - NDVI, EVI, SAVI, NDWI, MNDWI, NDBI, BSI, NBR,
    Clay_Minerals, Ferrous_Oxide, Iron_Oxide  (GeoTIFF + PNG each)
  - Unsupervised K-Means quaternary cover classification (GeoTIFF + overview PNG)
  - classification_legend.csv   (class IDs + per-class spectral means)
  - index_statistics.csv

Dependencies:
  pip install rasterio numpy scikit-learn matplotlib pandas tqdm
"""

import os
import glob
import warnings
import numpy as np
import pandas as pd
import rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# USER CONFIGURATION
# ─────────────────────────────────────────────
CONFIG = {
    # Directory containing pre-snapped band GeoTIFFs (or JP2s)
    "input_path": "data/03_snap_data/snap_sentinel_refl",

    # Output directory
    "output_dir": "./quaternary_output",

    # Number of quaternary / land-cover classes
    # Typical quaternary covers: water, alluvial, aeolian/sand, colluvial,
    # rock outcrop, dense vegetation, sparse vegetation, urban
    "n_classes": 8,

    # PCA compression before clustering? (speeds up large scenes)
    "use_pca": True,
    "pca_components": 6,

    # Random seed for reproducibility
    "random_seed": 42,

    # Number of valid pixels sampled to FIT the scaler/PCA/KMeans.
    # 500k is plenty for stable cluster centroids; raise if you have RAM to spare.
    "sample_size": 500_000,

    # Number of image rows processed at once during full-scene prediction.
    # Lower this if you still run out of memory during prediction.
    "chunk_rows": 256,
}

# Band filename patterns — adjust to match your file naming convention
BAND_PATTERNS = {
    "B02": "*B02*.tif",   # Blue
    "B03": "*B03*.tif",   # Green
    "B04": "*B04*.tif",   # Red
    "B05": "*B05*.tif",   # Red Edge 1  (optional)
    "B06": "*B06*.tif",   # Red Edge 2  (optional)
    "B07": "*B07*.tif",   # Red Edge 3  (optional)
    "B08": "*B08*.tif",   # NIR broad
    "B8A": "*B8A*.tif",   # NIR narrow  (optional)
    "B11": "*B11*.tif",   # SWIR1
    "B12": "*B12*.tif",   # SWIR2
}

# Class names — rename after inspecting classification_legend.csv
QUATERNARY_CLASSES = {
    0: "Water / Wetland",
    1: "Dense Vegetation",
    2: "Sparse Vegetation / Grassland",
    3: "Alluvial / Floodplain",
    4: "Aeolian / Sandy Deposit",
    5: "Colluvial / Slope Deposit",
    6: "Rock Outcrop / Bare Soil",
    7: "Urban / Built-up",
}

CLASS_COLORS = [
    "#1a78c2",  # Water
    "#228b22",  # Dense Vegetation
    "#90ee90",  # Sparse Vegetation
    "#c8a96e",  # Alluvial
    "#f5deb3",  # Aeolian/Sand
    "#b8860b",  # Colluvial
    "#808080",  # Rock Outcrop
    "#ff4500",  # Urban
]


# ─────────────────────────────────────────────
# UTILITY FUNCTIONS
# ─────────────────────────────────────────────

def find_band(base_path: str, pattern: str) -> str | None:
    """Recursively search for a band file matching the glob pattern."""
    matches = glob.glob(os.path.join(base_path, "**", pattern), recursive=True)
    if not matches:
        jp2_pattern = pattern.replace(".tif", ".jp2")
        matches = glob.glob(os.path.join(base_path, "**", jp2_pattern), recursive=True)
    return matches[0] if matches else None


def load_band(path: str) -> np.ndarray:
    """
    Load a pre-snapped band directly as float32.

    - Data is already on a common grid: no resampling or reprojection.
    - Masked pixels are already NaN: preserved as-is.
    - Any nodata value declared in file metadata is also converted to NaN.
    - Values assumed to be surface reflectance [0,1]: no DN scaling applied.
    """
    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float32)
        if src.nodata is not None:
            data[data == src.nodata] = np.nan
    return data


def read_profile(band_path: str) -> dict:
    """Read the spatial profile from a pre-snapped band (no transforms applied)."""
    with rasterio.open(band_path) as src:
        profile = src.profile.copy()
    profile.update(
        dtype="float32",
        count=1,
        nodata=np.nan,
        compress="lzw",
        tiled=True,
        blockxsize=256,
        blockysize=256,
    )
    return profile


def resample_to_reference(data: np.ndarray, src_path: str, ref_profile: dict) -> np.ndarray:
    """
    Resample a band to match the reference profile shape and transform.

    10m bands (B02/03/04/08) and 20m bands (B05-B8A/B11/B12) share the same
    snapped grid origin and extent but have different pixel counts. This function
    upsamples coarser bands to the 10m reference using bilinear interpolation,
    which preserves reflectance values. NaN pixels are propagated correctly.
    """
    ref_h = ref_profile["height"]
    ref_w = ref_profile["width"]

    if data.shape == (ref_h, ref_w):
        return data  # already at target resolution — nothing to do

    from rasterio.warp import reproject, Resampling as RS

    with rasterio.open(src_path) as src:
        out = np.full((ref_h, ref_w), np.nan, dtype=np.float32)
        reproject(
            source=data,
            destination=out,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=ref_profile["transform"],
            dst_crs=ref_profile["crs"],
            resampling=RS.bilinear,
            src_nodata=np.nan,
            dst_nodata=np.nan,
        )
    return out


def save_raster(array: np.ndarray, profile: dict, path: str):
    """Save a 2D float array as a single-band GeoTIFF."""
    out = profile.copy()
    out.update(dtype="float32", count=1)
    with rasterio.open(path, "w", **out) as dst:
        dst.write(array.astype(np.float32), 1)


def save_classification(array: np.ndarray, profile: dict, path: str):
    """Save integer classification raster (255 = no data / NaN)."""
    out = profile.copy()
    out.update(dtype="uint8", count=1, nodata=255)
    with rasterio.open(path, "w", **out) as dst:
        dst.write(array.astype(np.uint8), 1)


def thumb(array: np.ndarray, max_px: int = 2048) -> np.ndarray:
    """
    Downsample a 2D array to at most max_px on its longest side for plotting.
    Uses simple stride-based slicing — fast, zero dependencies, no smoothing.
    The full-resolution data is never modified; this only affects visualisation.
    """
    h, w = array.shape
    step = max(1, max(h, w) // max_px)
    return array[::step, ::step]


# ─────────────────────────────────────────────
# SPECTRAL INDEX CALCULATIONS
# ─────────────────────────────────────────────

def safe_divide(a, b, fill=np.nan):
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(b != 0, a / b, fill)
    return result.astype(np.float32)


def calc_ndvi(nir, red):
    """Normalized Difference Vegetation Index"""
    return safe_divide(nir - red, nir + red)

def calc_evi(nir, red, blue):
    """Enhanced Vegetation Index"""
    return 2.5 * safe_divide(nir - red, nir + 6 * red - 7.5 * blue + 1)

def calc_savi(nir, red, L=0.5):
    """Soil-Adjusted Vegetation Index"""
    return safe_divide((nir - red) * (1 + L), nir + red + L)

def calc_ndwi(green, nir):
    """Normalized Difference Water Index (Gao 1996)"""
    return safe_divide(green - nir, green + nir)

def calc_mndwi(green, swir1):
    """Modified NDWI (Xu 2006) — better open-water extraction"""
    return safe_divide(green - swir1, green + swir1)

def calc_ndbi(swir1, nir):
    """Normalized Difference Built-up Index"""
    return safe_divide(swir1 - nir, swir1 + nir)

def calc_bsi(blue, red, nir, swir1):
    """Bare Soil Index — key for quaternary deposit discrimination"""
    return safe_divide((swir1 + red) - (nir + blue),
                       (swir1 + red) + (nir + blue))

def calc_nbr(nir, swir2):
    """Normalized Burn Ratio — rock outcrop vs burned area discrimination"""
    return safe_divide(nir - swir2, nir + swir2)

def calc_clay_minerals(swir1, swir2):
    """Clay Minerals Ratio (Sabins 1987) — alluvial / clay deposit indicator"""
    return safe_divide(swir1, swir2)

def calc_ferrous_oxide(swir1, nir):
    """Ferrous Oxide Index — laterite / iron-rich quaternary surfaces"""
    return safe_divide(swir1, nir)

def calc_iron_oxide(red, blue):
    """Iron Oxide Ratio — aeolian / desert surface indicator"""
    return safe_divide(red, blue)


# ─────────────────────────────────────────────
# MAIN WORKFLOW
# ─────────────────────────────────────────────

def main():
    cfg = CONFIG
    os.makedirs(cfg["output_dir"], exist_ok=True)
    print("=" * 60)
    print("  Sentinel-2 L2A  |  Quaternary Cover Extractor")
    print("=" * 60)

    # ── 1. Locate band files ────────────────────────────────────
    print("\n[1/5] Locating band files...")
    bands_paths = {}
    for band, pattern in BAND_PATTERNS.items():
        p = find_band(cfg["input_path"], pattern)
        if p:
            bands_paths[band] = p
            print(f"  {band:4s} → {os.path.basename(p)}")
        else:
            print(f"  {band:4s} → not found (optional bands skipped)")

    required = ["B02", "B03", "B04", "B08", "B11", "B12"]
    missing = [b for b in required if b not in bands_paths]
    if missing:
        raise FileNotFoundError(
            f"Required bands missing: {missing}\n"
            f"Check CONFIG['input_path'] = '{cfg['input_path']}'"
        )

    # ── 2. Load bands, resampling coarser resolutions to 10m ──────
    # Bands share the same snapped grid origin/extent but differ in pixel count:
    #   10m → B02, B03, B04, B08          (e.g. 36864 × 36864)
    #   20m → B05, B06, B07, B8A, B11, B12 (e.g. 18432 × 18432)
    # The 10m NIR (B08) is used as the reference; all other bands are
    # bilinearly upsampled to match its shape before any further processing.
    print("\n[2/5] Loading bands (upsampling coarser bands to 10m reference)...")

    # Build reference profile from the 10m NIR band
    profile = read_profile(bands_paths["B08"])
    H, W = profile["height"], profile["width"]
    print(f"  Reference grid : {W} × {H} pixels  (10m NIR)")
    print(f"  CRS            : {profile['crs']}")

    ref = {}
    for band, bpath in tqdm(bands_paths.items(), unit="band"):
        raw = load_band(bpath)
        ref[band] = resample_to_reference(raw, bpath, profile)
        if raw.shape != ref[band].shape:
            print(f"    {band}: {raw.shape[1]}×{raw.shape[0]} → {W}×{H}")

    # ── 3. Build valid-pixel mask from existing NaN pattern ─────
    print("\n[3/5] Deriving valid-pixel mask from existing NaN pattern...")
    core = ["B02", "B03", "B04", "B08", "B11", "B12"]
    valid_mask = np.ones((H, W), dtype=bool)
    for b in core:
        valid_mask &= np.isfinite(ref[b]) & (ref[b] > 0)
    n_valid = int(valid_mask.sum())
    pct     = 100.0 * n_valid / valid_mask.size
    print(f"  Valid pixels : {n_valid:,} / {valid_mask.size:,}  ({pct:.1f}%)")
    print(f"  NaN-masked   : {valid_mask.size - n_valid:,} pixels (preserved from input)")

    # ── 4. Compute spectral indices ─────────────────────────────
    print("\n[4/5] Computing spectral indices...")

    blue, green, red = ref["B02"], ref["B03"], ref["B04"]
    nir   = ref["B08"]
    nir2  = ref.get("B8A", nir)   # fall back to B08 if B8A absent
    swir1 = ref["B11"]
    swir2 = ref["B12"]

    indices = {
        "NDVI":          calc_ndvi(nir, red),
        "EVI":           calc_evi(nir, red, blue),
        "SAVI":          calc_savi(nir, red),
        "NDWI":          calc_ndwi(green, nir),
        "MNDWI":         calc_mndwi(green, swir1),
        "NDBI":          calc_ndbi(swir1, nir),
        "BSI":           calc_bsi(blue, red, nir, swir1),
        "NBR":           calc_nbr(nir2, swir2),
        "Clay_Minerals": calc_clay_minerals(swir1, swir2),
        "Ferrous_Oxide": calc_ferrous_oxide(swir1, nir),
        "Iron_Oxide":    calc_iron_oxide(red, blue),
    }

    # Propagate the NaN mask into all index outputs
    for name in indices:
        indices[name][~valid_mask] = np.nan

    # Clip indices that are ratio-based (unbounded denominator → extreme outliers).
    # Clipping is done to the per-index P2–P98 range so genuine variation is
    # preserved while division-by-near-zero spikes are suppressed.
    # The same bounds are printed so you know exactly what was clipped.
    CLIP_INDICES = ["EVI", "Ferrous_Oxide", "Iron_Oxide"]
    print("  Clipping outlier-prone indices to P2–P98:")
    for name in CLIP_INDICES:
        if name not in indices:
            continue
        arr = indices[name]
        valid_vals = arr[np.isfinite(arr)]
        lo, hi = np.percentile(valid_vals, [2, 98])
        n_clipped = int(np.sum(np.isfinite(arr) & ((arr < lo) | (arr > hi))))
        indices[name] = np.clip(arr, lo, hi)
        indices[name][~valid_mask] = np.nan   # re-apply mask after clip
        print(f"    {name:<16s}  [{lo:.4f}, {hi:.4f}]  ({n_clipped:,} pixels clipped)")

    # Save index GeoTIFFs
    index_dir = os.path.join(cfg["output_dir"], "indices")
    os.makedirs(index_dir, exist_ok=True)
    print("  Saving index GeoTIFFs...")
    for name, arr in tqdm(indices.items(), unit="index"):
        save_raster(arr, profile, os.path.join(index_dir, f"{name}.tif"))

    # Index statistics
    stats_rows = []
    for name, arr in indices.items():
        v = arr[np.isfinite(arr)]
        if len(v) == 0:
            continue
        stats_rows.append({
            "Index": name,
            "Min":  float(v.min()),  "Max":  float(v.max()),
            "Mean": float(v.mean()), "Std":  float(v.std()),
            "P10":  float(np.percentile(v, 10)),
            "P50":  float(np.percentile(v, 50)),
            "P90":  float(np.percentile(v, 90)),
        })
    stats_df = pd.DataFrame(stats_rows)
    stats_path = os.path.join(cfg["output_dir"], "index_statistics.csv")
    stats_df.to_csv(stats_path, index=False, float_format="%.4f")
    print(f"\n  Index statistics → {stats_path}")
    print(stats_df.to_string(index=False))

    # Individual index plots
    ind_fig_dir = os.path.join(cfg["output_dir"], "index_plots")
    os.makedirs(ind_fig_dir, exist_ok=True)
    cmaps = {
        "NDVI": "RdYlGn",   "EVI": "YlGn",       "SAVI": "YlGn",
        "NDWI": "Blues_r",  "MNDWI": "Blues_r",   "NDBI": "hot_r",
        "BSI": "YlOrBr",    "NBR": "RdYlBu",
        "Clay_Minerals": "copper", "Ferrous_Oxide": "Oranges",
        "Iron_Oxide": "Reds",
    }
    for name, arr in tqdm(indices.items(), unit="plot", desc="  Index plots"):
        fig, ax = plt.subplots(figsize=(8, 6))
        fig.patch.set_facecolor("#0f1117")
        ax.set_facecolor("#0f1117")
        t = thumb(arr)  # downsample for plotting only — GeoTIFFs are full-res
        p5, p95 = np.nanpercentile(arr, [5, 95])
        im = ax.imshow(t, cmap=cmaps.get(name, "viridis"),
                       vmin=p5, vmax=p95, interpolation="nearest")
        ax.set_title(name, color="white", fontsize=12)
        ax.axis("off")
        cb = plt.colorbar(im, ax=ax, fraction=0.05, pad=0.02)
        cb.ax.tick_params(colors="white")
        plt.tight_layout()
        plt.savefig(os.path.join(ind_fig_dir, f"{name}.png"),
                    dpi=150, bbox_inches="tight", facecolor="#0f1117")
        plt.close()

    # ── 5. Quaternary cover classification ──────────────────────
    n_cls = min(cfg["n_classes"], len(QUATERNARY_CLASSES))
    print(f"\n[5/5] Quaternary cover classification ({n_cls} classes)...")

    # Build feature layers — keep everything float32 to avoid 2× memory from
    # NumPy's default float64 upcast. At 36864² × 15 bands float32 = ~76 GB,
    # so we never materialise the full stack. Instead we:
    #   1. Sample a random subset of valid pixels to FIT the scaler/PCA/KMeans.
    #   2. Predict labels for the full scene in row-chunks to stay within RAM.
    SAMPLE_SIZE  = cfg.get("sample_size",  500_000)   # pixels used to fit models
    CHUNK_ROWS   = cfg.get("chunk_rows",   256)        # rows predicted at a time

    feature_names = ["B02","B03","B04","B08","B11","B12",
                     "NDVI","BSI","MNDWI","NDBI","Clay_Minerals","Iron_Oxide"]
    for opt in ["B05","B06","B07"]:
        if opt in ref:
            feature_names.append(opt)
    n_features = len(feature_names)

    def get_pixel_features(row_slice) -> np.ndarray:
        """Extract feature matrix for a horizontal slice of rows, float32."""
        layers = []
        for name in feature_names:
            arr = indices[name] if name in indices else ref[name]
            layers.append(arr[row_slice].astype(np.float32))
        block = np.stack(layers, axis=-1)          # (rows, W, F)
        return block.reshape(-1, n_features)       # (rows*W, F)

    # ── Sample a manageable subset for fitting ──────────────────
    print(f"  Sampling {SAMPLE_SIZE:,} valid pixels to fit models "
          f"(scene has {valid_mask.sum():,} valid pixels)...")
    rng = np.random.default_rng(cfg["random_seed"])
    valid_idx = np.flatnonzero(valid_mask.ravel())
    sample_idx = rng.choice(valid_idx,
                            size=min(SAMPLE_SIZE, len(valid_idx)),
                            replace=False)
    sample_rows = sample_idx // W
    sample_cols = sample_idx  % W

    X_sample = np.stack(
        [(indices[n] if n in indices else ref[n])[sample_rows, sample_cols].astype(np.float32)
         for n in feature_names],
        axis=1,
    )
    X_sample = np.nan_to_num(X_sample, nan=0.0, posinf=0.0, neginf=0.0)

    print("  Fitting scaler on sample...")
    scaler = StandardScaler()
    X_sample_scaled = scaler.fit_transform(X_sample)

    pca = None
    if cfg["use_pca"] and cfg["pca_components"] < n_features:
        print(f"  Fitting PCA ({n_features} → {cfg['pca_components']} components) on sample...")
        pca = PCA(n_components=cfg["pca_components"], random_state=cfg["random_seed"])
        X_sample_scaled = pca.fit_transform(X_sample_scaled)
        explained = pca.explained_variance_ratio_.cumsum()[-1] * 100
        print(f"  Explained variance: {explained:.1f}%")

    print(f"  Fitting MiniBatchKMeans (k={n_cls}) on sample...")
    km = MiniBatchKMeans(
        n_clusters=n_cls,
        random_state=cfg["random_seed"],
        batch_size=min(10_000, len(X_sample_scaled)),
        n_init=5,
        max_iter=300,
    )
    km.fit(X_sample_scaled)

    # ── Predict full scene in row-chunks ────────────────────────
    print(f"  Predicting full scene in chunks of {CHUNK_ROWS} rows...")
    class_map_2d = np.full((H, W), 255, dtype=np.uint8)

    for row_start in tqdm(range(0, H, CHUNK_ROWS), unit="chunk"):
        row_end   = min(row_start + CHUNK_ROWS, H)
        row_slice = slice(row_start, row_end)

        X_chunk = get_pixel_features(row_slice)          # (chunk*W, F) float32
        mask_chunk = valid_mask[row_slice].ravel()        # (chunk*W,)

        X_chunk = np.nan_to_num(X_chunk, nan=0.0, posinf=0.0, neginf=0.0)
        X_chunk_scaled = scaler.transform(X_chunk)
        if pca is not None:
            X_chunk_scaled = pca.transform(X_chunk_scaled)

        labels_chunk = np.full(X_chunk.shape[0], 255, dtype=np.uint8)
        if mask_chunk.any():
            labels_chunk[mask_chunk] = km.predict(
                X_chunk_scaled[mask_chunk]
            ).astype(np.uint8)

        class_map_2d[row_slice] = labels_chunk.reshape(row_end - row_start, W)

    valid_flat = valid_mask.ravel()
    class_map  = class_map_2d.ravel()

    cls_path = os.path.join(cfg["output_dir"], "quaternary_cover.tif")
    save_classification(class_map_2d, profile, cls_path)
    print(f"  Classification raster → {cls_path}")

    # Per-class statistics for legend / renaming
    ndvi_flat  = indices["NDVI"].ravel()
    bsi_flat   = indices["BSI"].ravel()
    mndwi_flat = indices["MNDWI"].ravel()
    legend = []
    for c in range(n_cls):
        px = valid_flat & (class_map == c)
        legend.append({
            "Class_ID":       c,
            "Suggested_Name": QUATERNARY_CLASSES.get(c, f"Class_{c}"),
            "Pixel_Count":    int(px.sum()),
            "Mean_NDVI":      float(np.nanmean(ndvi_flat[px]))  if px.any() else np.nan,
            "Mean_BSI":       float(np.nanmean(bsi_flat[px]))   if px.any() else np.nan,
            "Mean_MNDWI":     float(np.nanmean(mndwi_flat[px])) if px.any() else np.nan,
        })
    legend_df = pd.DataFrame(legend)
    legend_path = os.path.join(cfg["output_dir"], "classification_legend.csv")
    legend_df.to_csv(legend_path, index=False, float_format="%.4f")
    print(f"  Class legend → {legend_path}")
    print("\n  Per-class summary (use NDVI/BSI/MNDWI to rename classes):")
    print(legend_df.to_string(index=False))

    # Overview figure
    print("\n  Generating overview figure...")
    colors_used = CLASS_COLORS[:n_cls]
    cmap_cls = ListedColormap(colors_used)

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.patch.set_facecolor("#0f1117")
    fig.suptitle("Sentinel-2 L2A  |  Quaternary Cover & Spectral Indices",
                 color="white", fontsize=15, fontweight="bold", y=0.99)

    def show(ax, data, title, cmap_name="RdYlGn", vmin=None, vmax=None):
        ax.set_facecolor("#0f1117")
        im = ax.imshow(thumb(data), cmap=cmap_name, vmin=vmin, vmax=vmax,
                       interpolation="nearest")
        ax.set_title(title, color="white", fontsize=10, pad=4)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02,
                     orientation="horizontal").ax.tick_params(colors="white", labelsize=7)

    ax0 = axes[0, 0]
    ax0.set_facecolor("#0f1117")
    masked_cls = np.ma.masked_where(class_map_2d == 255, class_map_2d)
    ax0.imshow(thumb(masked_cls), cmap=cmap_cls, vmin=0, vmax=n_cls - 1,
               interpolation="nearest")
    patches = [mpatches.Patch(color=colors_used[i],
                               label=f"{i}: {QUATERNARY_CLASSES.get(i, 'Class '+str(i))}")
               for i in range(n_cls)]
    ax0.legend(handles=patches, loc="lower left", fontsize=6,
               framealpha=0.6, facecolor="#1a1a2e", labelcolor="white",
               title="Quaternary Cover", title_fontsize=7)
    ax0.set_title("Quaternary Cover Classification", color="white", fontsize=10, pad=4)
    ax0.axis("off")

    show(axes[0, 1], indices["NDVI"],          "NDVI",                "RdYlGn",  -0.2, 0.9)
    show(axes[0, 2], indices["BSI"],           "Bare Soil Index",     "YlOrBr",  -0.5, 0.5)
    show(axes[1, 0], indices["MNDWI"],         "MNDWI (Water)",       "Blues_r", -0.5, 0.5)
    show(axes[1, 1], indices["NDBI"],          "NDBI (Built-up)",     "hot_r",   -0.5, 0.5)
    show(axes[1, 2], indices["Clay_Minerals"], "Clay Minerals Ratio", "copper",   0.5, 2.5)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig_path = os.path.join(cfg["output_dir"], "quaternary_cover_overview.png")
    plt.savefig(fig_path, dpi=180, bbox_inches="tight", facecolor="#0f1117")
    plt.close()
    print(f"  Overview figure → {fig_path}")

    print("\n" + "=" * 60)
    print("  ✓  All outputs saved to:", cfg["output_dir"])
    print("=" * 60)
    print("""
  Output structure:
  ├── quaternary_cover.tif           ← classification raster (uint8, 255=NaN)
  ├── quaternary_cover_overview.png
  ├── classification_legend.csv      ← rename classes using spectral means
  ├── index_statistics.csv
  ├── indices/
  │   ├── NDVI.tif   EVI.tif   SAVI.tif
  │   ├── NDWI.tif   MNDWI.tif
  │   ├── NDBI.tif   BSI.tif   NBR.tif
  │   ├── Clay_Minerals.tif   Ferrous_Oxide.tif   Iron_Oxide.tif
  └── index_plots/  *.png

  NEXT STEP: open classification_legend.csv and rename "Suggested_Name"
  values based on Mean_NDVI (vegetation), Mean_BSI (bare/deposits),
  and Mean_MNDWI (water) to match your actual quaternary deposit types.
    """)


if __name__ == "__main__":
    main()