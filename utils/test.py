"""
rename_dem_to_dtm.py
====================
Renames every *_DEM.tif under the input folder (including subfolders)
to *_DTM.tif.

    4623-2_RGP_GEOPH1_UTM38N_MAG_DEM.tif -> 4623-2_RGP_GEOPH1_UTM38N_MAG_DTM.tif

Usage
-----
    python utils/rename_dem_to_dtm.py            # dry run: only lists changes
    python utils/rename_dem_to_dtm.py --apply    # actually renames
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent   # utils/ -> project/
INPUT_DIR     = _PROJECT_ROOT / "data" / "01_raw_data" / "raw_magnetic"


def main(apply: bool) -> None:
    if not INPUT_DIR.exists():
        print(f"[!] Input directory not found: {INPUT_DIR}")
        return

    candidates = sorted(
        p for p in INPUT_DIR.rglob("*.tif")
        if p.stem.upper().endswith("_DEM")
    )

    print(f"{'APPLYING' if apply else 'DRY RUN'} — {len(candidates)} file(s) found in {INPUT_DIR}\n")

    renamed, skipped = 0, 0
    for src in candidates:
        dst = src.with_name(src.stem[:-4] + "_DTM" + src.suffix)
        rel = src.relative_to(INPUT_DIR)

        if dst.exists():
            print(f"  [skip] {rel}\n         target already exists: {dst.name}")
            skipped += 1
            continue

        print(f"  {rel}\n    -> {dst.name}")
        if apply:
            src.rename(dst)
        renamed += 1

    print(f"\n{'Renamed' if apply else 'Would rename'}: {renamed}  |  Skipped: {skipped}")
    if not apply and renamed:
        print("Run again with --apply to rename.")


if __name__ == "__main__":
    main(apply="--apply" in sys.argv)