"""
inspect_nodata.py
=================
Read-only diagnostic for NoData problems.

Outputs (default): every *.tif in paths.output_dir
Inputs (--inputs): every *.tif under paths.input_dir — only problem files are listed

Usage
-----
    python utils/inspect_nodata.py             # analyze outputs
    python utils/inspect_nodata.py --inputs    # also scan raw input tiles
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import yaml
from osgeo import gdal

gdal.UseExceptions()

_PROJECT_ROOT = Path(__file__).resolve().parent.parent   # utils/ -> project/
CONFIG_FILE   = _PROJECT_ROOT / "utils" / "config" / "snap_raster_batch.yaml"

SENTINEL   = -9999.0
BLEND_LOW  = -9999.0    # values strictly between BLEND_LOW and BLEND_HIGH
BLEND_HIGH = -5000.0    # are most likely -9999 blended into real data
HUGE       = 1e30       # |value| >= HUGE -> other sentinel (e.g. -1e32, -3.4e38)


def _resolve(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else _PROJECT_ROOT / p


def _fmt(v) -> str:
    if v is None:
        return "None"
    if isinstance(v, float) and (abs(v) >= 1e6 or (v != 0 and abs(v) < 1e-3)):
        return f"{v:.4g}"
    return f"{v:,.3f}".rstrip("0").rstrip(".")


def analyze(path: Path, top_n: int = 5) -> dict:
    ds   = gdal.Open(str(path))
    band = ds.GetRasterBand(1)
    nd   = band.GetNoDataValue()
    arr  = band.ReadAsArray()
    dtype = arr.dtype

    is_float = np.issubdtype(dtype, np.floating)
    n_total  = arr.size
    n_nan    = int(np.isnan(arr).sum()) if is_float else 0
    finite   = arr[np.isfinite(arr)] if is_float else arr.ravel()

    nd_ok      = nd is not None and np.isfinite(nd)
    n_declared = int((finite == nd).sum()) if nd_ok else 0
    n_9999     = int((finite == SENTINEL).sum())
    n_blend    = int(((finite > BLEND_LOW) & (finite < BLEND_HIGH)).sum())
    n_huge     = int((np.abs(finite.astype(np.float64)) >= HUGE).sum())

    data = finite[(finite != SENTINEL) & (np.abs(finite.astype(np.float64)) < HUGE)]
    if nd_ok:
        data = data[data != nd]

    pct = (np.percentile(data, [0, 0.1, 50, 99.9, 100]).tolist()
           if data.size else [None] * 5)

    vals, counts = np.unique(finite, return_counts=True)
    order = np.argsort(counts)[::-1][:top_n]
    top = [(float(vals[i]), int(counts[i])) for i in order]

    ds = None
    return dict(dtype=str(dtype), nd=nd, total=n_total, nan=n_nan,
                declared=n_declared, s9999=n_9999, blend=n_blend, huge=n_huge,
                valid=int(data.size), pct=pct, top=top)


def verdict(r: dict) -> list[str]:
    issues = []
    nd = r["nd"]
    if r["s9999"] and nd != SENTINEL:
        issues.append(f"{r['s9999']:,} px are exactly -9999 but declared NoData is {_fmt(nd)} "
                      f"-> -9999 is read as real data")
    if r["blend"]:
        issues.append(f"{r['blend']:,} px between -9999 and -5000 "
                      f"-> -9999 was blended into data during resampling")
    if r["huge"]:
        issues.append(f"{r['huge']:,} px with |value| >= 1e30 -> another undeclared sentinel")
    if r["top"]:
        v, c = r["top"][0]
        if c > 0.05 * r["total"] and v != nd and not (nd is not None and np.isnan(nd)):
            issues.append(f"most frequent value {_fmt(v)} ({c:,} px) is not the declared NoData "
                          f"-> possible undeclared null value")
    return issues


def report_outputs(out_dir: Path) -> None:
    files = sorted(out_dir.glob("*.tif"))
    print("=" * 72)
    print(f"  OUTPUTS - {len(files)} file(s) in {out_dir}")
    print("=" * 72)

    for f in files:
        r = analyze(f)
        p0, p01, p50, p999, p100 = r["pct"]
        print(f"\n{f.name}")
        print(f"  dtype / NoData : {r['dtype']} / {_fmt(r['nd'])}")
        print(f"  pixels         : {r['total']:,} total | {r['valid']:,} valid | "
              f"{r['nan']:,} NaN | {r['declared']:,} = declared NoData")
        print(f"  -9999 exact    : {r['s9999']:,}")
        print(f"  -9999<v<-5000  : {r['blend']:,}")
        print(f"  |v| >= 1e30    : {r['huge']:,}")
        print(f"  valid range    : min {_fmt(p0)} | p0.1 {_fmt(p01)} | median {_fmt(p50)} | "
              f"p99.9 {_fmt(p999)} | max {_fmt(p100)}")
        print(f"  most frequent  : " + ", ".join(f"{_fmt(v)} ({c:,})" for v, c in r["top"]))

        issues = verdict(r)
        if issues:
            for msg in issues:
                print(f"  [!] {msg}")
        else:
            print("  [OK] no NoData problems detected")


def report_inputs(in_dir: Path) -> None:
    files = sorted(in_dir.rglob("*.tif"))
    print("\n" + "=" * 72)
    print(f"  INPUTS - {len(files)} file(s) under {in_dir}")
    print("=" * 72)

    nd_counter = Counter()
    problems   = []
    for f in files:
        try:
            r = analyze(f, top_n=1)
        except Exception as exc:
            problems.append((f, [f"could not read: {exc}"], None))
            continue
        nd_counter[_fmt(r["nd"])] += 1
        issues = verdict(r)
        if issues:
            problems.append((f, issues, r))

    print("\n  Declared NoData across inputs:")
    for nd, n in nd_counter.most_common():
        print(f"    {nd:>12} : {n} file(s)")

    print(f"\n  Problem files: {len(problems)}")
    for f, issues, r in problems:
        extra = f"  (NoData {_fmt(r['nd'])}, dtype {r['dtype']})" if r else ""
        print(f"\n  {f.relative_to(in_dir)}{extra}")
        for msg in issues:
            print(f"    [!] {msg}")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")

    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", action="store_true", help="also scan raw input tiles")
    args = ap.parse_args()

    with open(CONFIG_FILE) as fh:
        cfg = yaml.safe_load(fh)

    report_outputs(_resolve(cfg["paths"]["output_dir"]))
    if args.inputs:
        report_inputs(_resolve(cfg["paths"]["input_dir"]))


if __name__ == "__main__":
    main()