"""
01_eigen6c4_bouguer.py
======================
Gravity functionals of the EIGEN-6C4 model over the AOI, computed locally with
pyshtools from the spherical harmonic coefficients (no web service).

1. EIGEN-6C4.gfc (~90 MB) is read from gfc_path. If missing, its link is looked up
   in the ICGEM model table and downloaded once (manual instructions otherwise).
2. The GRS80 normal field is subtracted from the coefficients and the remainder
   is synthesised up to lmax_expand directly at the output pixels, on the
   ellipsoid (h = 0, exact radius per row): disturbing potential T and
   gravity disturbance dg = -dT/dr.
3. Functionals (mGal):
     gravity_disturbance : dg
     gravity_anomaly     : dg - 2T/r                     (free-air, spherical approx.)
     gravity_anomaly_bg  : gravity_anomaly - 2*pi*G*rho*h (simple Bouguer, h from DEM, h < 0 -> 0)
     gravity             : total gravity, GRS80 normal gravity + dg
4. The output grid is the AOI bounding box at grid_step; each functional is
   written as float32 GeoTIFF (EPSG:4326):
     <output_base>/mosaic/EIGEN6C4_<functional>.tif

Model: Foerste et al. (2014), EIGEN-6C4, doi:10.5880/icgem.2015.1
"""

import re
import yaml
import requests
import numpy as np
import geopandas as gpd
import pyshtools as pysh
import rasterio
from math import ceil, floor
from pathlib import Path
from rasterio.transform import from_origin
from rasterio.warp import reproject, Resampling
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
ICGEM_TABLE_URL = "https://icgem.gfz.de/tom_longtime"
ICGEM_BASE_URL  = "https://icgem.gfz.de"

# GRS80 (Moritz, 1980)
GRS80_A     = 6378137.0
GRS80_F     = 1.0 / 298.257222101
GRS80_GM    = 3.986005e14
GRS80_J2    = 108263e-8
GRS80_GAMMA_E = 9.7803267715     # normal gravity at the equator (m/s2)
GRS80_K       = 0.001931851353   # Somigliana constant

BOUGUER_FACTOR = 0.04193   # 2*pi*G in mGal per (g/cm3 * m)
TO_MGAL        = 1e5
NODATA         = -9999.0
FUNCTIONALS    = ("gravity_disturbance", "gravity_anomaly", "gravity_anomaly_bg", "gravity")


# -----------------------------------------------------------------------------
# Config / paths
# -----------------------------------------------------------------------------

def load_config(config_path: Path) -> dict:
    with open(config_path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def resolve(raw: str, project_root: Path) -> Path:
    p = Path(raw)
    return p if p.is_absolute() else project_root / p


# -----------------------------------------------------------------------------
# Model file
# -----------------------------------------------------------------------------

def find_gfc_url(model_name: str) -> str | None:
    """Look up the current download link of a model in the ICGEM table."""
    print(f"[*] Looking up {model_name}.gfc in the ICGEM model table ...")
    try:
        resp = requests.get(ICGEM_TABLE_URL, timeout=30)
        resp.raise_for_status()
    except Exception as exc:
        print(f"[!] Could not fetch the ICGEM table: {exc}")
        return None
    pattern = rf'href=["\'](/getmodel/gfc/[^"\']+/{re.escape(model_name)}\.gfc)["\']'
    match = re.search(pattern, resp.text, re.IGNORECASE)
    if not match:
        print(f"[!] {model_name}.gfc link not found in {ICGEM_TABLE_URL}")
        return None
    return ICGEM_BASE_URL + match.group(1)


def download_gfc(url: str, dest: Path) -> bool:
    """Stream a .gfc file to disk; rejects HTML pages served instead of the file."""
    print(f"[*] Downloading {url}")
    tmp = dest.with_suffix(".part")
    try:
        with requests.get(url, stream=True, timeout=600) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            with open(tmp, "wb") as fh, tqdm(total=total, unit="iB", unit_scale=True, desc=dest.name) as bar:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    fh.write(chunk)
                    bar.update(len(chunk))
        with open(tmp, "rb") as fh:
            if fh.read(512).lstrip().startswith(b"<"):
                print("[!] Server returned an HTML page, not the model file.")
                tmp.unlink()
                return False
        tmp.replace(dest)
        return True
    except Exception as exc:
        print(f"[!] Download failed: {exc}")
        tmp.unlink(missing_ok=True)
        return False


def ensure_gfc(gfc_path: Path) -> bool:
    if gfc_path.exists():
        return True
    gfc_path.parent.mkdir(parents=True, exist_ok=True)
    url = find_gfc_url("EIGEN-6C4")
    if url and download_gfc(url, gfc_path):
        return True
    print("[!] Get the file manually: https://icgem.gfz.de/home -> Table of Models -> EIGEN-6C4 -> gfc")
    print(f"    and save it as {gfc_path}")
    return False


# -----------------------------------------------------------------------------
# Normal field
# -----------------------------------------------------------------------------

def grs80_zonal(nmax: int = 5) -> dict[int, float]:
    """Fully normalised zonal coefficients C(2n,0) of the GRS80 normal potential."""
    e2 = 2 * GRS80_F - GRS80_F ** 2
    out = {}
    for n in range(1, nmax + 1):
        j2n = (-1) ** (n + 1) * 3 * e2 ** n / ((2 * n + 1) * (2 * n + 3)) * (1 - n + 5 * n * GRS80_J2 / e2)
        out[2 * n] = -j2n / np.sqrt(4 * n + 1)
    return out


def disturbing_coeffs(clm):
    """Model minus GRS80 normal potential, scaled to the model's GM and r0."""
    t = clm.copy()
    gm_ratio = GRS80_GM / clm.gm
    t.coeffs[0, 0, 0] -= gm_ratio
    for l, c in grs80_zonal().items():
        if l <= t.lmax:
            t.coeffs[0, l, 0] -= c * gm_ratio * (GRS80_A / clm.r0) ** l
    return t


# -----------------------------------------------------------------------------
# Grids
# -----------------------------------------------------------------------------

def target_grid(aoi_file: Path, step: float):
    """Pixel grid (EPSG:4326) covering the AOI bounding box, snapped to step."""
    gdf = gpd.read_file(aoi_file).to_crs(epsg=4326)
    w, s, e, n = gdf.total_bounds
    w, s = floor(w / step) * step, floor(s / step) * step
    e, n = ceil(e / step) * step, ceil(n / step) * step
    width, height = int(round((e - w) / step)), int(round((n - s) / step))
    transform = from_origin(w, n, step, step)
    lons = w + (np.arange(width) + 0.5) * step
    lats = n - (np.arange(height) + 0.5) * step
    return transform, width, height, lats, lons


def synthesize(t, lats_gd: np.ndarray, lons: np.ndarray, lmax: int):
    """T (m2/s2) and dg = -dT/dr (m/s2) of the coefficients t at points on the
    GRS80 ellipsoid. Rows share geocentric latitude and radius, so each row is
    one Legendre evaluation plus a Fourier sum over the order m."""
    if t.normalization != "4pi" or t.csphase != 1:
        raise ValueError("expected 4pi-normalised coefficients without Condon-Shortley phase (ICGEM)")
    e2  = GRS80_F * (2 - GRS80_F)
    lat = np.deg2rad(lats_gd)
    nu  = GRS80_A / np.sqrt(1 - e2 * np.sin(lat) ** 2)
    rho, z = nu * np.cos(lat), nu * (1 - e2) * np.sin(lat)
    r, psi = np.hypot(rho, z), np.arctan2(z, rho)                      # radius, geocentric latitude

    deg = np.arange(lmax + 1)
    ll, mm = np.tril_indices(lmax + 1)
    idx = ll * (ll + 1) // 2 + mm                                        # PlmBar packing
    lam = np.deg2rad(lons)
    cos_ml, sin_ml = np.cos(np.outer(deg, lam)), np.sin(np.outer(deg, lam))
    C, S = t.coeffs[0, :lmax + 1, :lmax + 1], t.coeffs[1, :lmax + 1, :lmax + 1]

    T  = np.empty((len(lat), len(lons)))
    dg = np.empty_like(T)
    P  = np.zeros((lmax + 1, lmax + 1))
    for i in tqdm(range(len(lat)), desc="  Synthesis", unit="row"):
        P[ll, mm] = pysh.legendre.PlmBar(lmax, np.sin(psi[i]))[idx]
        q  = t.gm / r[i] * (t.r0 / r[i]) ** deg                          # degree scaling at this radius
        qd = q * (deg + 1) / r[i]
        CP, SP = C * P, S * P
        T[i]  = (q  @ CP) @ cos_ml + (q  @ SP) @ sin_ml
        dg[i] = (qd @ CP) @ cos_ml + (qd @ SP) @ sin_ml
    return T, dg, r


def normal_gravity(lats_gd: np.ndarray) -> np.ndarray:
    """GRS80 normal gravity on the ellipsoid (Somigliana), m/s2."""
    e2 = GRS80_F * (2 - GRS80_F)
    s2 = np.sin(np.deg2rad(lats_gd)) ** 2
    return GRS80_GAMMA_E * (1 + GRS80_K * s2) / np.sqrt(1 - e2 * s2)


def dem_on_grid(dem_path: Path, transform, width: int, height: int) -> np.ndarray:
    """Mean DEM height per target pixel; sea and nodata -> 0 m."""
    out = np.full((height, width), np.nan, dtype=np.float32)
    with rasterio.open(dem_path) as src:
        reproject(
            source=rasterio.band(src, 1), destination=out,
            src_nodata=src.nodata, dst_nodata=np.nan,
            dst_transform=transform, dst_crs="EPSG:4326",
            resampling=Resampling.average,
        )
    return np.clip(np.nan_to_num(out, nan=0.0), 0, None)


def write_tif(data: np.ndarray, transform, path: Path, label: str) -> None:
    data = np.where(np.isfinite(data), data, NODATA).astype(np.float32)
    profile = dict(driver="GTiff", dtype="float32", count=1, height=data.shape[0], width=data.shape[1],
                   crs="EPSG:4326", transform=transform, nodata=NODATA,
                   compress="LZW", tiled=True, blockxsize=256, blockysize=256)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data, 1)
        dst.set_band_description(1, f"EIGEN-6C4 {label} (mGal)")
    print(f"[+] {path.name}  min {np.nanmin(data):.1f}  max {np.nanmax(data):.1f} mGal")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main(config_file: str) -> None:
    config_path = Path(config_file).resolve()
    if not config_path.exists():
        print(f"[!] Config not found: {config_path}")
        return
    config       = load_config(config_path)
    project_root = config_path.parent.parent
    params       = config["eigen6c4_params"]

    aoi_file = resolve(config["paths"]["aoi"], project_root)
    gfc_path = resolve(params["gfc_path"], project_root)
    out_dir  = resolve(config["paths"]["output_base"], project_root) / "mosaic"
    lmax     = int(params["lmax_expand"])
    wanted   = list(params["functionals"])

    unknown = [f for f in wanted if f not in FUNCTIONALS]
    if unknown:
        print(f"[!] Unknown functionals {unknown}; choose from {FUNCTIONALS}")
        return
    if not aoi_file.exists():
        print(f"[!] AOI file missing: {aoi_file}")
        return
    dem_path = resolve(params["dem_path"], project_root) if "gravity_anomaly_bg" in wanted else None
    if dem_path is not None and not dem_path.exists():
        print(f"[!] DEM missing (needed for gravity_anomaly_bg): {dem_path}")
        return
    if not ensure_gfc(gfc_path):
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    transform, width, height, lats_t, lons_t = target_grid(aoi_file, float(params["grid_step"]))
    print(f"[*] Output grid : {width} x {height} px at {params['grid_step']} deg")

    print(f"[*] Reading {gfc_path.name} up to degree {lmax} ...")
    clm = pysh.SHGravCoeffs.from_file(gfc_path, format="icgem", lmax=lmax)

    print(f"[*] Synthesising the disturbing field on {width} x {height} ellipsoid points ...")
    T, dg, r = synthesize(disturbing_coeffs(clm), lats_t, lons_t, lmax)
    fa = dg - 2 * T / r[:, None]
    results = {
        "gravity_disturbance": dg * TO_MGAL,
        "gravity_anomaly":     fa * TO_MGAL,
        "gravity":             (normal_gravity(lats_t)[:, None] + dg) * TO_MGAL,
    }
    if dem_path is not None:
        print(f"[*] Bouguer plate correction, rho = {params['density']} g/cm3, DEM {dem_path.name}")
        h = dem_on_grid(dem_path, transform, width, height)
        results["gravity_anomaly_bg"] = results["gravity_anomaly"] - BOUGUER_FACTOR * float(params["density"]) * h

    for f in wanted:
        write_tif(results[f], transform, out_dir / f"EIGEN6C4_{f}.tif", f)
    print(f"\n[*] Done: {out_dir}")


if __name__ == "__main__":
    current_dir = Path(__file__).resolve().parent
    config_path = current_dir.parent / "config" / "01_eigen6c4_bouguer.yaml"
    main(config_file=str(config_path))
