"""
snap_raster_batch.py
=======================
Runs snap/merge for every pattern listed under batch.patterns in the YAML,
using the same processing and output-format settings for all of them.

Each pattern is matched against paths.input_dir (and its subfolders when
paths.recursive is true).

Before processing, any input band without a declared NoData value gets
processing.src_nodata written to its header, so those cells are ignored
during resampling instead of being treated as real values.

Output name per pattern is auto-built as:
    snapped_{resample_alg}_{SUFFIX}.tif
where SUFFIX = pattern stripped of its leading '*_' and trailing '.tif'
    e.g. '*_MAG_AMF_RTP.tif' -> 'snapped_bilinear_MAG_AMF_RTP.tif'

Usage
-----
    python utils/snap_raster_batch.py
"""

import importlib
import sys
import time
import numpy as np
import yaml
from osgeo import gdal
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent   # utils/ -> project/
_SCRIPTS_DIR = _PROJECT_ROOT / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

_snap = importlib.import_module("03_snap_raster")
RESAMPLE_ALGORITHMS = _snap.RESAMPLE_ALGORITHMS
align_raster = _snap.align_raster
merge_rasters = _snap.merge_rasters

gdal.UseExceptions()


# =============================================================================
# Helpers
# =============================================================================

def _suffix_from_pattern(pattern: str) -> str:
    """
    Extract a clean suffix token from a glob pattern.

    '*_MAG_AMF_RTP.tif'  ->  'MAG_AMF_RTP'
    '*_DTM.tif'          ->  'DTM'
    'tile_*.tif'         ->  'tile_'   (graceful fallback for custom patterns)
    """
    stem = Path(pattern).stem          # drop .tif
    # strip leading wildcard + underscore(s) if present
    if stem.startswith("*"):
        stem = stem.lstrip("*").lstrip("_")
    return stem or pattern             # fallback: use whole pattern


def _build_creation_opts(fmt: dict) -> list[str]:
    bigtiff    = str(fmt.get("bigtiff",    "IF_SAFER")).upper()
    compress   = str(fmt.get("compress",   "LZW")).upper()
    predictor  = fmt.get("predictor",  1)
    block_size = fmt.get("block_size", 512)

    opts = [
        f"COMPRESS={compress}",
        f"BIGTIFF={bigtiff}",
        "TILED=YES",
        f"BLOCKXSIZE={block_size}",
        f"BLOCKYSIZE={block_size}",
    ]
    if compress in ("DEFLATE", "LZW", "ZSTD"):
        opts.append(f"PREDICTOR={predictor}")
    return opts


def _resolve(path_str: str, root: Path) -> Path:
    """Return path_str as an absolute Path, resolving relative paths against root."""
    p = Path(path_str)
    return p if p.is_absolute() else root / p


def _ensure_nodata(files: list[Path], nodata: float) -> tuple[int, list[tuple[Path, str]]]:
    """
    For every file, set `nodata` on bands that have no NoData declared.
    Only the header is changed; pixel values are untouched.
    Bands that already declare a NoData value are left as they are.

    Returns (number of files fixed, list of (file, error) that could not be fixed).
    """
    fixed, errors = 0, []
    for f in files:
        try:
            ds = gdal.Open(str(f), gdal.GA_ReadOnly)
            missing = [b for b in range(1, ds.RasterCount + 1)
                       if ds.GetRasterBand(b).GetNoDataValue() is None]
            ds = None
            if not missing:
                continue

            ds = gdal.Open(str(f), gdal.GA_Update)
            for b in missing:
                ds.GetRasterBand(b).SetNoDataValue(nodata)
            ds = None   # close / flush header
            fixed += 1
        except Exception as exc:
            errors.append((f, str(exc)))
    return fixed, errors


# =============================================================================
# Batch runner
# =============================================================================

def run_batch(config_file: str) -> None:
    config_path = Path(config_file).resolve()
    if not config_path.exists():
        print(f"[!] Config not found: {config_path}")
        return

    with open(config_path) as fh:
        config = yaml.safe_load(fh)

    # Relative paths in the YAML are resolved against the project root,
    # independent of where the config file itself lives.
    project_root = _PROJECT_ROOT
    paths_cfg    = config["paths"]

    # ------------------------------------------------------------------
    # Master grid
    # ------------------------------------------------------------------
    master_file = _resolve(paths_cfg["master_grid"], project_root)
    if not master_file.exists():
        print(f"[!] Master grid missing: {master_file}")
        return

    # ------------------------------------------------------------------
    # Input directory (searched by every batch pattern)
    # ------------------------------------------------------------------
    input_dir = _resolve(paths_cfg["input_dir"], project_root)
    if not input_dir.exists():
        print(f"[!] Input directory not found: {input_dir}")
        return

    recursive = paths_cfg.get("recursive", True)
    finder    = input_dir.rglob if recursive else input_dir.glob

    # ------------------------------------------------------------------
    # Output directory
    # ------------------------------------------------------------------
    out_dir = _resolve(paths_cfg["output_dir"], project_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Processing settings (shared across all patterns)
    # ------------------------------------------------------------------
    proc          = config["processing"]
    alg_name      = proc.get("resample_alg", "nearest").lower()
    resample      = RESAMPLE_ALGORITHMS.get(alg_name, gdal.GRA_NearestNeighbour)
    apply_mask    = proc.get("apply_mask",       True)
    merge_mode    = proc.get("merge_output",     False)
    overlap_strat = proc.get("overlap_strategy", "first").lower()
    strip_height  = proc.get("strip_height",     1024)
    src_nodata    = proc.get("src_nodata",       -9999)   # None in YAML disables the fix

    if alg_name not in RESAMPLE_ALGORITHMS:
        print(f"[!] Unknown resample_alg '{alg_name}'. Falling back to nearest.")

    # ------------------------------------------------------------------
    # Output format
    # ------------------------------------------------------------------
    creation_opts = _build_creation_opts(config.get("output_format", {}))

    # ------------------------------------------------------------------
    # Batch patterns
    # ------------------------------------------------------------------
    patterns = config.get("batch", {}).get("patterns", [])
    if not patterns:
        print("[!] No patterns found under batch.patterns in the config.")
        return

    # ------------------------------------------------------------------
    # Summary header
    # ------------------------------------------------------------------
    print("=" * 60)
    print(f"  Batch snap — {len(patterns)} pattern(s)")
    print(f"  Master     : {master_file.name}")
    print(f"  Resample   : {alg_name}")
    print(f"  Mode       : {'merge → ' + overlap_strat if merge_mode else 'per-file'}")
    print(f"  Src NoData : {src_nodata if src_nodata is not None else 'not enforced'}")
    print(f"  Input dir  : {input_dir}{'  (incl. subfolders)' if recursive else ''}")
    print(f"  Output dir : {out_dir}")
    print("=" * 60)
    print()

    batch_start = time.time()
    failures    = []

    for idx, pattern in enumerate(patterns, start=1):
        suffix      = _suffix_from_pattern(pattern)
        output_name = f"snapped_{alg_name}_{suffix}"
        input_files = sorted(finder(pattern))

        print(f"[{idx}/{len(patterns)}] Pattern  : {pattern}")
        print(f"           Output   : {output_name}.tif")
        print(f"           Files    : {len(input_files)} found")
        for f in input_files:
            print(f"             - {f.relative_to(input_dir)}")

        if not input_files:
            print(f"           [!] No files matched — skipping\n")
            failures.append((pattern, "no files matched"))
            continue

        # --------------------------------------------------------------
        # Make sure every input declares its NoData value
        # --------------------------------------------------------------
        if src_nodata is not None:
            n_fixed, nd_errors = _ensure_nodata(input_files, float(src_nodata))
            if n_fixed:
                print(f"           NoData   : set to {src_nodata} on {n_fixed} file(s)")
            for f, err in nd_errors:
                print(f"           [!] Could not set NoData on {f.name}: {err}")

        t0 = time.time()
        try:
            if merge_mode:
                output_path = out_dir / f"{output_name}.tif"
                merge_rasters(
                    input_files=input_files,
                    reference_path=master_file,
                    output_path=output_path,
                    resample_alg=resample,
                    overlap_strategy=overlap_strat,
                    strip_height=strip_height,
                    apply_mask=apply_mask,
                    creation_opts=creation_opts,
                )
            else:
                for input_file in input_files:
                    out_path = out_dir / f"{output_name}_{input_file.stem}.tif"
                    align_raster(
                        input_path=input_file,
                        reference_path=master_file,
                        output_path=out_path,
                        resample_alg=resample,
                        apply_mask=apply_mask,
                        creation_opts=creation_opts,
                    )

        except Exception as exc:
            print(f"           [!] FAILED — {exc}\n")
            failures.append((pattern, str(exc)))
            continue

        elapsed = time.time() - t0
        print(f"           Done in  : {elapsed:.1f}s\n")

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    total_elapsed = time.time() - batch_start
    n_ok = len(patterns) - len(failures)
    print("=" * 60)
    print(f"  Batch complete : {n_ok}/{len(patterns)} succeeded  |  {total_elapsed:.1f}s total")
    if failures:
        print(f"  Failures ({len(failures)}):")
        for pat, reason in failures:
            print(f"    - {pat}: {reason}")
    print("=" * 60)


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    config_path = _PROJECT_ROOT / "utils" / "config" / "snap_raster_batch.yaml"
    run_batch(config_file=str(config_path))