"""
02_prisma_export.py
===================
PRISMA L2D (.he5) granules -> float32 reflectance GeoTIFFs, one per granule.

For each PRS_L2D_*.he5 under input_dir:
  1. Band plan: every valid VNIR + SWIR band sorted by wavelength, or the band
     closest to each wavelength in target_wavelengths_nm (VNIR wins ties in the
     ~940-980 nm overlap). Invalid bands (wavelength 0 or flag 0) are skipped.
  2. Reflectance = ScaleMin + DN * (ScaleMax - ScaleMin) / 65535, per cube
     (L2ScaleVnirMin/Max, L2ScaleSwirMin/Max). DN = 0 -> NaN.
  3. Georeferencing from Product_UL/LRcorner_easting/northing, taken as pixel
     centres (as prismaread does), CRS from Epsg_Code (UTM).
  4. Bands are read one plane at a time (cubes are (lines, bands, samples),
     not chunked) and written with wavelength in the band description and tags.

Granules already exported are skipped. PRISMA data are ordered from the ASI
portal (https://prisma.asi.it); there is no download step.
"""

import yaml
import h5py
import numpy as np
import rasterio
from pathlib import Path
from rasterio.crs import CRS
from rasterio.transform import from_origin
from tqdm import tqdm

CUBE_PATH = "/HDFEOS/SWATHS/PRS_L2D_HCO/Data Fields/{}_Cube"
CUBES     = ("Vnir", "Swir")


def load_config(config_path: Path) -> dict:
    with open(config_path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def resolve(raw: str, project_root: Path) -> Path:
    p = Path(raw)
    return p if p.is_absolute() else project_root / p


def band_plan(attrs, targets) -> list[tuple[str, int, float]]:
    """(cube, band index, wavelength) for every output band."""
    candidates = []
    for cube in CUBES:
        wl    = np.asarray(attrs[f"List_Cw_{cube}"], dtype=float).ravel()
        flags = np.asarray(attrs.get(f"List_Cw_{cube}_Flags", np.ones_like(wl))).ravel()
        candidates += [(cube, i, w) for i, w in enumerate(wl) if w > 0 and flags[i] > 0]
    if not targets:
        return sorted(candidates, key=lambda c: c[2])
    return [min(candidates, key=lambda c: (abs(c[2] - t), c[0] != "Vnir")) for t in targets]


def georef(attrs, rows: int, cols: int):
    ul_e, ul_n = float(attrs["Product_ULcorner_easting"]), float(attrs["Product_ULcorner_northing"])
    lr_e, lr_n = float(attrs["Product_LRcorner_easting"]), float(attrs["Product_LRcorner_northing"])
    res_x, res_y = (lr_e - ul_e) / (cols - 1), (ul_n - lr_n) / (rows - 1)
    transform = from_origin(ul_e - res_x / 2, ul_n + res_y / 2, res_x, res_y)
    return transform, CRS.from_epsg(int(attrs["Epsg_Code"]))


def export_granule(he5_path: Path, out_path: Path, targets, compress: str) -> int:
    with h5py.File(he5_path, "r") as f:
        attrs = f.attrs
        cubes = {c: f[CUBE_PATH.format(c.upper())] for c in CUBES}
        scale = {c: (float(attrs[f"L2Scale{c}Min"]), float(attrs[f"L2Scale{c}Max"])) for c in CUBES}
        plan  = band_plan(attrs, targets)
        rows, _, cols = cubes["Vnir"].shape
        transform, crs = georef(attrs, rows, cols)

        profile = dict(driver="GTiff", dtype="float32", count=len(plan), width=cols, height=rows,
                       crs=crs, transform=transform, nodata=np.nan, compress=compress,
                       tiled=True, blockxsize=256, blockysize=256, interleave="band")
        if len(plan) * rows * cols * 4 > 3.5e9:
            profile["BIGTIFF"] = "YES"
        tmp = out_path.with_suffix(".part.tif")
        with rasterio.open(tmp, "w", **profile) as dst:
            for k, (cube, i, wl) in enumerate(plan, start=1):
                dn   = cubes[cube][:, i, :].astype(np.float32)
                smin, smax = scale[cube]
                refl = smin + dn * (smax - smin) / 65535.0
                refl[dn == 0] = np.nan
                dst.write(refl, k)
                dst.set_band_description(k, f"{wl:.2f} nm ({cube.upper()} {i + 1})")
                dst.update_tags(k, wavelength_nm=f"{wl:.2f}", source_cube=cube.upper(), source_band=str(i + 1))
            dst.update_tags(source=he5_path.name, units="reflectance")
        tmp.replace(out_path)
    return len(plan)


def main(config_file: str) -> None:
    config_path = Path(config_file).resolve()
    if not config_path.exists():
        print(f"[!] Config not found: {config_path}")
        return
    config       = load_config(config_path)
    project_root = config_path.parent.parent
    params       = config["processing"]

    in_dir  = resolve(params["input_dir"], project_root)
    out_dir = resolve(params["output_dir"], project_root)
    targets = params.get("target_wavelengths_nm") or None
    if not in_dir.exists():
        print(f"[!] Input directory missing: {in_dir}")
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(in_dir.rglob("*.he5") if params.get("recursive", True) else in_dir.glob("*.he5"))
    if not files:
        print(f"[!] No .he5 files in {in_dir}")
        return
    print(f"[*] {len(files)} granule(s) | bands: {'all valid' if targets is None else targets}")

    errors = []
    for he5 in tqdm(files, desc="  PRISMA", unit="granule"):
        out_path = out_dir / f"{he5.stem}.tif"
        if out_path.exists():
            tqdm.write(f"  [skip] {out_path.name}")
            continue
        try:
            n = export_granule(he5, out_path, targets, params.get("compress", "DEFLATE"))
            tqdm.write(f"  [+] {out_path.name} ({n} bands)")
        except Exception as exc:
            tqdm.write(f"  [!] {he5.name}: {exc}")
            errors.append(he5.name)

    print(f"\n[*] Done: {len(files) - len(errors)}/{len(files)} granules -> {out_dir}")


if __name__ == "__main__":
    current_dir = Path(__file__).resolve().parent
    config_path = current_dir.parent / "config" / "02_prisma_export.yaml"
    main(config_file=str(config_path))
