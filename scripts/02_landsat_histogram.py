import sys
import re
import yaml
import numpy as np
import rasterio
from rasterio.transform import array_bounds
from rasterio.warp import reproject, Resampling
from rasterio.crs import CRS
from shapely.geometry import box
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from tqdm import tqdm


# =============================================================================
# Config / helpers
# =============================================================================

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def get_tile_bbox(src):
    """Return a shapely box for the valid (non-nodata) extent of a rasterio dataset."""
    b = src.bounds
    return box(b.left, b.bottom, b.right, b.top)


def tiles_overlap(src_a, src_b):
    """Return True if the two open rasterio datasets have any spatial overlap."""
    return get_tile_bbox(src_a).intersects(get_tile_bbox(src_b))


def read_band_as_float(src, band=1):
    """Read a single band, applying nodata mask → NaN, returning float32 array."""
    data = src.read(band).astype(np.float32)
    nd   = src.nodata
    if nd is not None:
        data[data == nd] = np.nan
    return data


# =============================================================================
# Overlap extraction
# =============================================================================

def extract_overlap_pixels(path_a, path_b, band=1):
    """
    Reproject tile B onto tile A's grid, clip to the intersection window,
    and return two flat float32 arrays of valid (both non-NaN) pixels.

    Returns
    -------
    pixels_a, pixels_b : 1-D np.ndarray  — matched valid overlap pixels
    overlap_pct_a      : float           — % of tile-A pixels that overlap
    n_overlap          : int             — raw pixel count in overlap
    """
    with rasterio.open(path_a) as src_a, rasterio.open(path_b) as src_b:

        if not tiles_overlap(src_a, src_b):
            return None, None, 0.0, 0

        # ── intersection in src_a CRS ──────────────────────────────────────
        bbox_a = get_tile_bbox(src_a)
        # reproject src_b bounds to src_a CRS if needed
        if src_a.crs and src_b.crs and src_a.crs != src_b.crs:
            from rasterio.warp import transform_bounds
            b_bounds_reproj = transform_bounds(src_b.crs, src_a.crs, *src_b.bounds)
            bbox_b = box(*b_bounds_reproj)
        else:
            bbox_b = get_tile_bbox(src_b)

        intersection = bbox_a.intersection(bbox_b)
        if intersection.is_empty:
            return None, None, 0.0, 0

        # ── window for the intersection inside src_a ───────────────────────
        from rasterio.windows import from_bounds
        win_a = from_bounds(*intersection.bounds, transform=src_a.transform)
        win_a = win_a.intersection(rasterio.windows.Window(0, 0, src_a.width, src_a.height))

        win_h = int(round(win_a.height))
        win_w = int(round(win_a.width))
        if win_h <= 0 or win_w <= 0:
            return None, None, 0.0, 0

        # ── read src_a in overlap window ───────────────────────────────────
        data_a = src_a.read(band, window=win_a).astype(np.float32)
        nd_a   = src_a.nodata
        if nd_a is not None:
            data_a[data_a == nd_a] = np.nan

        # ── reproject src_b into src_a overlap window ──────────────────────
        win_transform = src_a.window_transform(win_a)
        data_b = np.full((win_h, win_w), np.nan, dtype=np.float32)

        reproject(
            source      = rasterio.band(src_b, band),
            destination = data_b,
            src_transform   = src_b.transform,
            src_crs         = src_b.crs,
            dst_transform   = win_transform,
            dst_crs         = src_a.crs,
            resampling      = Resampling.bilinear,
            src_nodata      = src_b.nodata,
            dst_nodata      = np.nan,
        )

        # ── mask: valid in BOTH tiles ──────────────────────────────────────
        valid = np.isfinite(data_a) & np.isfinite(data_b)
        n_overlap = int(valid.sum())

        if n_overlap == 0:
            return None, None, 0.0, 0

        total_a   = int(np.isfinite(src_a.read(band).astype(np.float32)).sum())
        overlap_pct = 100.0 * n_overlap / max(total_a, 1)

        return data_a[valid].ravel(), data_b[valid].ravel(), overlap_pct, n_overlap


# =============================================================================
# Histogram plotting
# =============================================================================

PALETTE = {
    "bg":        "#0f1117",
    "panel":     "#1a1d27",
    "accent_a":  "#3a9dda",
    "accent_b":  "#e06c75",
    "overlap":   "#98c379",
    "text":      "#abb2bf",
    "grid":      "#2c3040",
    "border":    "#2e3247",
}

def _style_ax(ax, title=""):
    ax.set_facecolor(PALETTE["panel"])
    ax.tick_params(colors=PALETTE["text"], labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(PALETTE["border"])
    ax.xaxis.label.set_color(PALETTE["text"])
    ax.yaxis.label.set_color(PALETTE["text"])
    if title:
        ax.set_title(title, color=PALETTE["text"], fontsize=9, pad=6)
    ax.grid(True, color=PALETTE["grid"], linewidth=0.5, linestyle="--", alpha=0.6)


def plot_overlap_histogram(
    pixels_a, pixels_b,
    label_a, label_b,
    band, overlap_pct, n_overlap,
    output_path,
    n_bins=128,
):
    """
    Creates a 2-row figure:
      Row 1 — overlaid histograms of tile A vs tile B in the overlap zone
      Row 2 — normalized (density) histograms of A and B  +  scatter-plot A vs B
    """
    fig = plt.figure(figsize=(12, 8), facecolor=PALETTE["bg"])
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.32,
                            left=0.08, right=0.97, top=0.88, bottom=0.08)

    # ── shared bin range ───────────────────────────────────────────────────
    lo = float(np.nanpercentile(np.concatenate([pixels_a, pixels_b]), 0.5))
    hi = float(np.nanpercentile(np.concatenate([pixels_a, pixels_b]), 99.5))
    bins = np.linspace(lo, hi, n_bins + 1)

    # ── ax0: overlaid histograms ───────────────────────────────────────────
    ax0 = fig.add_subplot(gs[0, :])
    ax0.hist(pixels_a, bins=bins, color=PALETTE["accent_a"],
             alpha=0.55, label=f"Tile A  ({label_a})", linewidth=0)
    ax0.hist(pixels_b, bins=bins, color=PALETTE["accent_b"],
             alpha=0.55, label=f"Tile B  ({label_b})", linewidth=0)
    ax0.set_xlabel("Reflectance / DN value")
    ax0.set_ylabel("Pixel count")
    _style_ax(ax0, title=f"Overlap zone — Band {band}  "
                         f"({n_overlap:,} px · {overlap_pct:.1f}% of Tile A)")
    ax0.legend(facecolor=PALETTE["panel"], edgecolor=PALETTE["border"],
               labelcolor=PALETTE["text"], fontsize=8)

    # ── ax1: z-score normalized histograms ────────────────────────────────
    ax1 = fig.add_subplot(gs[1, 0])
    z_a = (pixels_a - pixels_a.mean()) / pixels_a.std()
    z_b = (pixels_b - pixels_b.mean()) / pixels_b.std()
    z_lo = float(np.nanpercentile(np.concatenate([z_a, z_b]), 0.5))
    z_hi = float(np.nanpercentile(np.concatenate([z_a, z_b]), 99.5))
    z_bins = np.linspace(z_lo, z_hi, n_bins + 1)
    ax1.hist(z_a, bins=z_bins, color=PALETTE["accent_a"],
             alpha=0.55, label=f"Tile A  ({label_a})", linewidth=0)
    ax1.hist(z_b, bins=z_bins, color=PALETTE["accent_b"],
             alpha=0.55, label=f"Tile B  ({label_b})", linewidth=0)
    ax1.axvline(0, color="#e5c07b", linewidth=1.0, linestyle="--", alpha=0.7)
    ax1.set_xlabel("Z-score  (μ=0, σ=1)")
    ax1.set_ylabel("Pixel count")
    ax1.legend(facecolor=PALETTE["panel"], edgecolor=PALETTE["border"],
               labelcolor=PALETTE["text"], fontsize=8)
    _style_ax(ax1, title="Normalized histograms  (zero mean, unit SD)")

    # ── ax2: scatter A vs B (density-sampled) ─────────────────────────────
    ax2 = fig.add_subplot(gs[1, 1])
    max_scatter = 15_000
    if len(pixels_a) > max_scatter:
        idx = np.random.choice(len(pixels_a), max_scatter, replace=False)
        sa, sb = pixels_a[idx], pixels_b[idx]
    else:
        sa, sb = pixels_a, pixels_b

    ax2.scatter(sa, sb, s=1.5, alpha=0.25, color=PALETTE["accent_a"],
                linewidths=0, rasterized=True)

    # 1:1 reference line
    ref = np.array([lo, hi])
    ax2.plot(ref, ref, color="#e5c07b", linewidth=1.0, linestyle="--",
             label="1:1 line", alpha=0.8)

    with np.errstate(invalid='ignore', divide='ignore'):
        r = np.corrcoef(sa, sb)[0, 1]
    r = float(r) if np.isfinite(r) else float('nan')
    ax2.set_xlabel("Tile A reflectance")
    ax2.set_ylabel("Tile B reflectance")
    ax2.legend(facecolor=PALETTE["panel"], edgecolor=PALETTE["border"],
               labelcolor=PALETTE["text"], fontsize=8)
    _style_ax(ax2, title=f"A vs B scatter  (r = {r:.4f})")

    # ── suptitle ───────────────────────────────────────────────────────────
    fig.suptitle(
        f"Overlap inspection — Band {band}\n"
        f"{label_a}  ×  {label_b}",
        color="#dcdfe4", fontsize=11, fontweight="bold",
        y=0.965,
    )

    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=PALETTE["bg"])
    plt.close(fig)


# =============================================================================
# Main pipeline
# =============================================================================

# =============================================================================
# Band-type grouping
# =============================================================================

# Matches spectral band suffixes only: SR_B1..B7, ST_B10, generic _Bx
_BAND_SUFFIX_RE = re.compile(r'(SR_B\d+|ST_B\d+|_B\d+)', re.IGNORECASE)

def is_band_tile(path: Path) -> bool:
    """Return True only for spectral band files. Excludes QA, RADSAT, MTL, ANG, etc."""
    return bool(_BAND_SUFFIX_RE.search(path.stem))


def infer_band_type(path: Path) -> str:
    """
    Extract the band-type suffix from a Landsat filename.
    e.g.  LC09_…_SR_B4.TIF  →  'SR_B4'
          mosaic_B3.tif      →  'B3'
    """
    m = _BAND_SUFFIX_RE.search(path.stem)
    return m.group(0).lstrip('_').upper() if m else path.stem


def group_tiles_by_band(tile_paths):
    """
    Returns a dict  {band_type: [Path, ...]}
    Only band types with ≥2 tiles (i.e., potential pairs) are kept.
    """
    groups = {}
    for p in tile_paths:
        bt = infer_band_type(p)
        groups.setdefault(bt, []).append(p)
    return {bt: paths for bt, paths in groups.items() if len(paths) >= 2}


# =============================================================================
# Fast spatial pre-filter using only bounds (no pixel I/O)
# =============================================================================

def get_bounds_index(tile_paths):
    """
    Read bounds for all tiles once and return a list of (path, bounds_box).
    Much faster than opening full datasets for every pair.
    """
    index = []
    for p in tile_paths:
        try:
            with rasterio.open(p) as src:
                index.append((p, get_tile_bbox(src), src.crs))
        except Exception as e:
            tqdm.write(f"  [!] Could not open {p.name}: {e}")
    return index


def fast_overlapping_pairs(bounds_index):
    """
    Return only pairs whose bounding boxes intersect — O(n²) on boxes,
    but no pixel I/O so very fast even for hundreds of tiles.
    """
    pairs = []
    for i, (pa, box_a, crs_a) in enumerate(bounds_index):
        for pb, box_b, crs_b in bounds_index[i+1:]:
            # rough check: if CRS match, use boxes directly; else assume overlap
            # (full reproject check happens later in extract_overlap_pixels)
            if crs_a and crs_b and crs_a == crs_b:
                if not box_a.intersects(box_b):
                    continue
            pairs.append((pa, pb))
    return pairs


# =============================================================================
# Main inspection loop
# =============================================================================

def run_overlap_inspection(tile_paths, output_dir, band=1, n_bins=128):
    """
    Groups tiles by band type, then for every same-band overlapping pair,
    extracts pixels in the shared area and exports a diagnostic histogram.

    Parameters
    ----------
    tile_paths : list[Path]   — GeoTIFF files to compare
    output_dir : Path         — where PNG figures are written
    band       : int          — raster band index to read (1-indexed)
    n_bins     : int          — histogram bin count
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── group by band type ─────────────────────────────────────────────────
    groups = group_tiles_by_band(tile_paths)
    if not groups:
        print("[!] No band groups with ≥2 tiles found.")
        return

    print(f"[*] Found {len(tile_paths)} tile(s) across {len(groups)} band type(s):")
    for bt, paths in sorted(groups.items()):
        print(f"      {bt:<20}  {len(paths):>3} tile(s)")

    total_exported   = 0
    total_skipped    = 0
    total_no_overlap = 0

    for band_type, bt_paths in sorted(groups.items()):

        # ── fast bounding-box pre-filter ───────────────────────────────────
        bounds_index = get_bounds_index(bt_paths)
        candidate_pairs = fast_overlapping_pairs(bounds_index)

        if not candidate_pairs:
            print(f"\n  [{band_type}] No spatially overlapping pairs — skipped.")
            continue

        bt_out = output_dir / band_type
        bt_out.mkdir(parents=True, exist_ok=True)

        print(f"\n  [{band_type}]  {len(bt_paths)} tiles  →  "
              f"{len(candidate_pairs)} candidate pair(s) after bbox pre-filter")

        exported   = 0
        skipped    = 0
        no_overlap = 0

        for path_a, path_b in tqdm(candidate_pairs,
                                   desc=f"    {band_type}", unit="pair"):
            label_a = path_a.stem
            label_b = path_b.stem

            pixels_a, pixels_b, overlap_pct, n_overlap = extract_overlap_pixels(
                path_a, path_b, band=band
            )

            if pixels_a is None or n_overlap == 0:
                no_overlap += 1
                continue

            if n_overlap < 50:
                tqdm.write(f"    [~] Overlap too small ({n_overlap} px), skipping: "
                           f"{label_a}  ×  {label_b}")
                skipped += 1
                continue

            fig_name = f"overlap_{label_a}_x_{label_b}.png"
            out_path = bt_out / fig_name

            plot_overlap_histogram(
                pixels_a, pixels_b,
                label_a=path_a.stem,
                label_b=path_b.stem,
                band=band,
                overlap_pct=overlap_pct,
                n_overlap=n_overlap,
                output_path=out_path,
                n_bins=n_bins,
            )
            tqdm.write(f"    [+] {fig_name}  ({n_overlap:,} px, {overlap_pct:.1f}%)")
            exported += 1

        print(f"    → {exported} saved, {skipped} tiny, {no_overlap} no real overlap")
        total_exported   += exported
        total_skipped    += skipped
        total_no_overlap += no_overlap

    print(f"\n[*] All done. Exported {total_exported} figure(s), "
          f"skipped {total_skipped} (tiny), {total_no_overlap} had no real overlap.")
    print(f"[*] Output directory: {output_dir}")


# =============================================================================
# Entry Point — can be used standalone OR imported by the mosaic pipeline
# =============================================================================

def main(config_file=None, tile_dir=None, output_dir=None, bands=None):
    """
    Usage modes
    -----------
    1. Pass a YAML config (same schema as the mosaic config):
         python tile_overlap_histograms.py path/to/config.yaml

    2. Pass keyword args directly when importing as a module:
         from tile_overlap_histograms import main
         main(tile_dir="data/tiles", output_dir="out/histograms", bands=[3, 4, 5])

    3. Point at a directory of TIFs from the command line:
         python tile_overlap_histograms.py --tile-dir data/tiles --band 4
    """
    # ── resolve sources ────────────────────────────────────────────────────
    if config_file:
        config_path = Path(config_file).resolve()
        if not config_path.exists():
            print(f"[!] Config not found: {config_path}")
            return
        config       = load_config(config_path)
        params       = config['processing']
        project_root = config_path.parent.parent

        _tile_dir = Path(params['input_dir'])
        _tile_dir = _tile_dir if _tile_dir.is_absolute() else project_root / _tile_dir
        _out_dir  = Path(params.get('histogram_output_dir',
                         str(project_root / 'output' / 'overlap_histograms')))
        _bands    = params.get('bands', [1])

    else:
        _tile_dir = Path(tile_dir).resolve() if tile_dir else Path('.').resolve()
        _out_dir  = Path(output_dir).resolve() if output_dir else _tile_dir / 'overlap_histograms'
        _bands    = bands or [1]

    if not _tile_dir.exists():
        print(f"[!] Tile directory not found: {_tile_dir}")
        return

    tif_paths = sorted(_tile_dir.rglob("*.tif")) + sorted(_tile_dir.rglob("*.TIF"))
    # deduplicate (rglob may return both on case-insensitive filesystems)
    seen      = set()
    tif_paths = [p for p in tif_paths if not (p in seen or seen.add(p))]

    # keep only spectral band files
    tif_paths = [p for p in tif_paths if is_band_tile(p)]

    if not tif_paths:
        print(f"[!] No spectral band TIF files found in: {_tile_dir}")
        return

    print(f"[*] Tile overlap histogram inspection")
    print(f"    Tiles  : {len(tif_paths)}  in  {_tile_dir}")
    print(f"    Bands  : {_bands}")
    print(f"    Output : {_out_dir}\n")

    for band in _bands:
        band_out = _out_dir / f"band_{band}"
        print(f"\n{'='*60}")
        print(f"  Band {band}")
        print(f"{'='*60}")
        run_overlap_inspection(tif_paths, band_out, band=band)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Export overlap-zone histograms for every pair of GeoTIFF tiles."
    )
    parser.add_argument("config", nargs="?", default=None,
                        help="Path to YAML config (same format as mosaic pipeline). "
                             "Mutually exclusive with --tile-dir.")
    parser.add_argument("--tile-dir", default=None,
                        help="Directory containing GeoTIFF tiles (no config needed).")
    parser.add_argument("--output-dir", default=None,
                        help="Where to save histogram PNGs. "
                             "Defaults to <tile-dir>/overlap_histograms.")
    parser.add_argument("--band", type=int, default=None,
                        help="Single band to inspect (overrides config). Default: 1.")
    parser.add_argument("--bands", type=int, nargs="+", default=None,
                        help="Multiple bands to inspect (e.g. --bands 3 4 5).")

    args = parser.parse_args()

    _bands = None
    if args.band:
        _bands = [args.band]
    elif args.bands:
        _bands = args.bands

    main(
        config_file = args.config,
        tile_dir    = args.tile_dir,
        output_dir  = args.output_dir,
        bands       = _bands,
    )