# mex2-geofetch

Acquisition and preprocessing of the remote sensing, elevation and geophysical rasters
of the Maaden Project (Arabian Shield, Saudi Arabia). Each source is downloaded for an
AOI, mosaicked, snapped onto one master grid (EPSG:32637) and stacked into NetCDF
cubes, the inputs of the downstream ML and prospectivity work.

| Source | Product | Native res. | Download (01) | Preprocessing (02) |
|---|---|---|---|---|
| Landsat 8/9 C2 L2 | surface reflectance, 7 bands | 30 m | Planetary Computer | QA_PIXEL fill mask, SR scaling, feathered mosaic |
| Sentinel-2 L2A | surface reflectance, 12 bands | 10-60 m | Planetary Computer | SR scaling, feathered mosaic |
| EMIT L2A | reflectance, 285 bands | 60 m | NASA Earthdata | GLT orthorectification, cloud mask, two-pass mosaic |
| PRISMA L2D | reflectance, ~230 bands | 30 m | ASI portal (manual) | DN to reflectance, one GeoTIFF per granule |
| Copernicus GLO-30 | DEM | 30 m | OpenTopography | tiles merged in 01 |
| Sandwell V33.1 | free-air gravity, error, vertical gravity gradient | 1' | Scripps (topex.ucsd.edu) | AOI clip of the global NetCDFs |
| GGMplus | gravity accelerations and disturbances | 0.002° (~200 m) | Curtin University | tiles converted and merged in 01 |
| EIGEN-6C4 | free-air and Bouguer anomaly, disturbance | to degree 2190 (~9 km) | ICGEM (coefficients) | computed in 01 with pyshtools |

```
00_base_grid.py           AOI                       -> master grid (UTM, nx and ny multiples of 256, 1 = valid)
01_<source>_download.py   AOI                       -> raw scenes, tiles or global grids
01_eigen6c4_bouguer.py    AOI + coefficients + DEM  -> gravity functionals on the AOI bounding box
02_<source>_mosaic.py     raw scenes                -> one mosaic per band
02_prisma_export.py       PRISMA granules           -> one reflectance GeoTIFF per granule
03_snap_raster.py         mosaics or tiles + grid   -> rasters aligned to the master grid, masked
04_cube_raster.py         snapped rasters + grid    -> NetCDF4 cube (band, y, x) with CRS and metadata
```

Each step is one script in `scripts/` with one YAML file of the same name in `config/`.
Paths are relative to the project root. No command-line arguments, except that
`02_landsat_mosaic.py`, `02_sentinel_mosaic.py` and `04_cube_raster.py` take an
optional config path, so one script can serve several configs (e.g. one cube per config).

## Run

```
conda activate maaden
python scripts/00_base_grid.py            # prints the fitted grid, asks y/n before writing

python scripts/01_landsat_download.py
python scripts/01_sentinel_download.py
python scripts/01_emit_download.py        # needs Earthdata credentials
python scripts/01_opentopo_download.py    # needs OPENTOPO_API_KEY
python scripts/01_gravity_download.py     # Sandwell V33.1, ~1.4 GB for the three default grids
python scripts/01_ggmplus_download.py
python scripts/01_eigen6c4_bouguer.py     # Bouguer needs the COP30 DEM from 01_opentopo

python scripts/02_landsat_mosaic.py
python scripts/02_sentinel_mosaic.py
python scripts/02_emit_mosaic.py
python scripts/02_gravity_mosaic.py
python scripts/02_prisma_export.py        # granules ordered from ASI, placed in raw_prisma_hyper30m

python scripts/03_snap_raster.py          # once per source, pointing the config at its inputs
python scripts/04_cube_raster.py          # or: python scripts/04_cube_raster.py config/<cube>.yaml
```

Finished downloads and tiles are skipped, so interrupted runs resume. The EMIT
download also deletes files in its output folder that are no longer in the selection.

### Credentials

| Service | Used by | Setup |
|---|---|---|
| OpenTopography | `01_opentopo_download.py` | API key from https://portal.opentopography.org/myopentopo in the environment variable `OPENTOPO_API_KEY` |
| NASA Earthdata | `01_emit_download.py` | `EARTHDATA_USERNAME` and `EARTHDATA_PASSWORD` in the environment, or `auth` in `config/01_emit_download.yaml` (local only, do not commit) |
| Planetary Computer | `01_landsat_download.py`, `01_sentinel_download.py` | none (anonymous, signed by `planetary-computer`) |

## Method notes

- **Master grid (00).** The AOI is buffered by `buffer_m`, projected to UTM (auto-detected
  from the centroid, or `override_epsg`) and the extent is rounded up to whole
  `tile_divisor` blocks (256 px) at `pixel_size_m`, centred on the AOI. Pixels inside the
  mask polygon are 1, outside 0. One grid per resolution (10, 30, 60 m ...).
- **Scene selection (01).** Greedy AOI coverage, areas in EPSG:6933: candidates are sorted
  and added while they still cover new ground. Landsat: lowest cloud cover, one scene per
  path/row. Sentinel-2: cloud cover plus 1 point per 30 days of age, footprint from
  `proj:geometry` (valid data, not the tile box). EMIT: lowest cloud cover, one version
  per orbit and scene.
- **OpenTopography (01).** The AOI bounding box is split below the 450,000 km² per-request
  limit; tiles overlap by 0.003° so the merged DEM has no seams.
- **Landsat (02).** SR = DN × mult + add from the MTL file. Fill pixels come from bit 0
  of QA_PIXEL (DN = 0 is valid reflectance, used only if QA_PIXEL is missing). Tiles are
  reprojected to `target_crs`.
- **Sentinel-2 (02).** SR = (DN - 1000) / 10000 (processing baseline 04.00 offset).
- **Feathered mosaics (02).** In overlaps each tile is weighted by its distance to its own
  nodata edge, so seams blend. Landsat and Sentinel-2 mosaics also get holes up to 20 px
  filled from the nearest valid pixel. Outputs are float32 reflectance with NaN as nodata.
- **EMIT (02).** Each band of each granule is orthorectified with its GLT (EPSG:4326) and
  masked with the granule's L2A mask. Pass 1 blends only clear pixels; pass 2 fills what
  is still empty from the unmasked mosaic, so bright surfaces falsely flagged as cloud
  are not lost.
- **EIGEN-6C4 (01).** The `.gfc` coefficients (downloaded once from ICGEM, manual fallback)
  minus the GRS80 normal field are synthesised up to `lmax_expand` at every output pixel,
  on the ellipsoid (h = 0), giving the disturbing potential T and the disturbance
  dg = -dT/dr. Free-air anomaly = dg - 2T/r; simple Bouguer = free-air - 0.04193 ρ h, with
  h the DEM averaged per pixel (sea set to 0). About 2 min for the Shield at 0.01° and
  degree 2190.
- **PRISMA (02).** Reflectance = ScaleMin + DN (ScaleMax - ScaleMin) / 65535 per cube, DN 0
  as nodata; corner coordinates are pixel centres (EPSG from the file). All valid VNIR +
  SWIR bands by wavelength, or the closest band to each `target_wavelengths_nm`.
- **Gravity (02).** The Sandwell NetCDFs are read with xarray (GDAL misreads their
  GMT/COARDS naming) and clipped to the AOI, with 0-360° longitudes handled.
- **Snapping (03).** GDAL warp to the master grid (`resample_alg`); cells where the grid
  is 0 become NaN (`apply_mask`). With `merge_output`, overlaps follow
  `overlap_strategy` (`first`, `last` or `average`); `average` runs as a running sum in
  strips of `strip_height` rows, so grids of ~40k × 40k px fit in memory. Outputs are
  tiled, LZW-compressed BigTIFF where needed.
- **Cubes (04).** Single- or multi-band GeoTIFFs become one NetCDF4 variable each, on the
  master grid coordinates, with the CRS in `spatial_ref`, band labels from the file
  names, zlib compression and the attributes under `metadata` in the config.

## Adjusting

- New AOI: add the zipped shapefile to `aoi/` and point `paths.aoi` to it.
- New resolution: run 00 with another `grid.pixel_size_m` and point the 03 and 04
  configs at the new grid.
- Dates, cloud limit, bands: `sensors.<name>` in the 01 configs.
- Many inputs with the same naming (e.g. aeromagnetic derivatives):
  `utils/snap_raster_batch.py` runs 03 once per pattern in `batch.patterns`.
- New cube: copy `config/04_cube_raster.yaml`, list the files under `variables`, run
  `python scripts/04_cube_raster.py config/<new>.yaml`.

## Utilities (`utils/`, configs in `utils/config/`)

| Script | Purpose |
|---|---|
| `snap_raster_batch.py` | step 03 for many file patterns; writes `src_nodata` to inputs without declared nodata |
| `rasterize_geology.py` | harmonised 1:250k geology (`data/01_raw_data/raw_geology/`) (HA_DOMAIN, HA_RFAM, HA_RTYPE, HA_TECTO, HA_AGEBIN) to uint8 GeoTIFFs on the 10 m grid, nearest-neighbour gap fill inside the land mask, CSV lookups |
| `cube_categorical.py` | categorical rasters + lookups to NetCDF with CF `flag_values` / `flag_meanings` |
| `sentinel_quaternary.py` | spectral indices and K-means Quaternary cover classes from snapped Sentinel-2 |
| `interpolate_nan_by_mask.py` | interpolates NaNs inside a mask (survey gaps) |
| `inspect_nodata.py`, `print_tif_statistics.py` | nodata and value checks on GeoTIFFs |
| `rename_dem_to_dtm.py` | renames `*_DEM.tif` aeromagnetic grids to `*_DTM.tif` (dry run unless `APPLY = True`) |
| `landsat_histogram.py` | value histograms in tile overlaps (mosaic QC) |
| `emitl2a_band_sweep.py` | per-band and sliding-RGB frames from the EMIT cube |
| `ngp_download_codes.py` | SGS NGDP aeromagnetic download URLs per 1:250k sheet |
| `figures/` | one map script per cube (`landsat_refl30m.py`, `aeromag_hr60m.py`, ...) |

## Repository layout

```
aoi/       zipped AOI shapefiles (Maaden study area, Shield basins, Saudi Arabia, ...)
config/    00_base_grid.yaml ... 04_cube_raster.yaml
scripts/   00_base_grid.py ... 04_cube_raster.py
utils/     helpers, QC and figures (see above)
data/      (not in git)
  00_base_grid/     master grids, <aoi>_base_grid<res>.tif
  01_raw_data/      downloads, one folder per product (raw_<sensor>_<product><res>)
  02_mosaic_data/   per-band mosaics
  03_snap_data/     rasters on the master grid
  04_cube_data/     NetCDF cubes (landsat_refl30m.nc, emitl2a_hyper60m.nc, aeromag_hr60m.nc, ...)
```

Products are named `<sensor>_<product><res>`, e.g. `copglo_dem30m`, `geophys_coarse90m`,
`geol_masks10m`.

## Requirements

Python ≥ 3.10. `pip install -r requirements.txt` into the `maaden` conda env. The GDAL
Python bindings (`osgeo`) are easiest from conda-forge (`conda install -c conda-forge gdal`).

## Data sources

- Landsat C2 L2, Sentinel-2 L2A: Microsoft Planetary Computer.
- EMIT L2A (`EMITL2ARFL`): NASA LP DAAC.
- Copernicus DEM GLO-30: ESA, via OpenTopography.
- PRISMA L2D: © Italian Space Agency (ASI), https://prisma.asi.it.
- Förste, C., et al. (2014). EIGEN-6C4, the latest combined global gravity field model.
  GFZ Data Services, doi:10.5880/icgem.2015.1.
- Sandwell, D. T., et al. (2014). New global marine gravity model from CryoSat-2 and
  Jason-1 reveals buried tectonic structure. *Science* 346, 65-67. V33.1 grids.
- Hirt, C., et al. (2013). New ultrahigh-resolution picture of Earth's gravity field.
  *GRL* 40, 4279-4283. GGMplus.
