"""
03_snap_raster_batch.py
=======================
Runs snap/merge for every pattern listed under batch.patterns in the YAML,
using the same processing and output-format settings for all of them.

Output name per pattern is auto-built as:
    snapped_{resample_alg}_{SUFFIX}
where SUFFIX = pattern stripped of its leading '*_' and trailing '.tif'
    e.g. '*_MAG_AMF_RTP.tif' -> 'snapped_bilinear_MAG_AMF_RTP'

Usage
-----
    python scripts/03_snap_raster_batch.py
"""

import sys
import time
import numpy as np
import yaml
from osgeo import gdal
from pathlib import Path

# ---------------------------------------------------------------------------
# Make sure the script can import from the same directory regardless of how
# it is launched.
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from snap_raster import (          # noqa: E402  (import after path fix)
    RESAMPLE_ALGORITHMS,
    align_raster,
    merge_rasters,
)

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
    if compress == "DEFLATE":
        opts.append(f"PREDICTOR={predictor}")
    return opts


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

    project_root = config_path.parent.parent   # config/ lives one level below root

    # ------------------------------------------------------------------
    # Master grid
    # ------------------------------------------------------------------
    master_rel  = config["paths"]["master_grid"]
    master_file = Path(master_rel) if Path(master_rel).is_absolute() else project_root / master_rel
    if not master_file.exists():
        print(f"[!] Master grid missing: {master_file}")
        return

    # ------------------------------------------------------------------
    # Output directory
    # ------------------------------------------------------------------
    out_rel = config["paths"]["output_dir"]
    out_dir = Path(out_rel) if Path(out_rel).is_absolute() else project_root / out_rel
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Processing settings (shared across all patterns)
    # ------------------------------------------------------------------
    proc         = config["processing"]
    alg_name     = proc.get("resample_alg", "nearest").lower()
    resample     = RESAMPLE_ALGORITHMS.get(alg_name, gdal.GRA_NearestNeighbour)
    apply_mask   = proc.get("apply_mask",       True)
    merge_mode   = proc.get("merge_output",     False)
    overlap_strat = proc.get("overlap_strategy", "first").lower()
    strip_height  = proc.get("strip_height",     1024)

    if alg_name not in RESAMPLE_ALGORITHMS:
        print(f"[!] Unknown resample_alg '{alg_name}'. Falling back to nearest.")

    # ------------------------------------------------------------------
    # Output format
    # ------------------------------------------------------------------
    creation_opts = _build_creation_opts(config.get("output_format", {}))

    # ------------------------------------------------------------------
    # Input directory (glob mode only for batch)
    # ------------------------------------------------------------------
    glob_cfg = config["inputs"].get("glob")
    if not glob_cfg:
        print("[!] Batch mode requires inputs.glob.dir — 'files' list not supported here.")
        return

    input_dir = Path(glob_cfg["dir"])
    input_dir = input_dir if input_dir.is_absolute() else project_root / input_dir
    if not input_dir.exists():
        print(f"[!] Input directory not found: {input_dir}")
        return

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
    print(f"  Output dir : {out_dir}")
    print("=" * 60)
    print()

    batch_start = time.time()
    failures    = []

    for idx, pattern in enumerate(patterns, start=1):
        suffix      = _suffix_from_pattern(pattern)
        output_name = f"snapped_{alg_name}_{suffix}"
        input_files = sorted(input_dir.glob(pattern))

        print(f"[{idx}/{len(patterns)}] Pattern  : {pattern}")
        print(f"           Output   : {output_name}")
        print(f"           Files    : {len(input_files)} found")

        if not input_files:
            print(f"           [!] No files matched — skipping\n")
            failures.append((pattern, "no files matched"))
            continue

        t0 = time.time()
        try:
            if merge_mode:
                output_path = out_dir / output_name
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
    current_dir = Path(__file__).resolve().parent
    config_path = current_dir.parent / "config" / "03_snap_raster_batch.yaml"
    run_batch(config_file=str(config_path))