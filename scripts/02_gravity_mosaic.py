import yaml
import os
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.windows import from_bounds, Window
from rasterio.features import geometry_mask
from rasterio.transform import from_origin
from rasterio.crs import CRS
from shapely.geometry import mapping
from shapely.affinity import translate
from pathlib import Path


def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)


def align_aoi_to_raster_east(aoi_geom, src_east):
    """
    Sandwell global grids use 0..360 longitude convention. If the AOI is in
    -180..180 with negative longitudes that fall west of the antimeridian,
    shift those parts by +360 so the clip lands in the right place.
    Returns the (possibly shifted) geometry and a status message.
    """
    raster_is_0_360 = src_east > 180.0
    if not raster_is_0_360:
        return aoi_geom, "no shift (raster is -180..180)"

    minx, _, maxx, _ = aoi_geom.bounds
    if minx >= 0:
        return aoi_geom, "no shift (AOI is in 0..360 range)"

    if maxx <= 0:
        return translate(aoi_geom, xoff=360.0), "shifted +360 (AOI was fully west of prime meridian)"

    return aoi_geom, "WARNING: AOI crosses longitude 0; no shift applied"


def write_clipped_tif(data, transform, src_meta, out_path, params):
    """Write the clipped array as a compressed, tiled GeoTIFF."""
    out_meta = src_meta.copy()
    out_meta.update({
        'driver':     'GTiff',
        'height':     data.shape[-2],
        'width':      data.shape[-1],
        'transform':  transform,
        'count':      data.shape[0] if data.ndim == 3 else 1,
        'dtype':      str(data.dtype),
        'compress':   params.get('compression', 'LZW'),
        'tiled':      True,
        'blockxsize': 512,
        'blockysize': 512,
    })

    # Predictor: 3 for float (LZW/DEFLATE benefit), 2 for integer
    if np.issubdtype(data.dtype, np.floating):
        out_meta['predictor'] = 3
    elif np.issubdtype(data.dtype, np.integer):
        out_meta['predictor'] = 2

    if 'nodata' in params and params['nodata'] is not None:
        out_meta['nodata'] = params['nodata']

    # BigTIFF: auto-flip near 4 GB uncompressed
    n_bytes = int(np.prod(data.shape)) * data.dtype.itemsize
    bigtiff = params.get('bigtiff', 'IF_SAFER')
    if bigtiff == 'IF_SAFER' and n_bytes > 3 * (1024 ** 3):
        out_meta['BIGTIFF'] = 'YES'
    elif bigtiff in ('YES', 'IF_NEEDED'):
        out_meta['BIGTIFF'] = bigtiff

    with rasterio.open(out_path, 'w', **out_meta) as dst:
        if data.ndim == 2:
            dst.write(data, 1)
        else:
            dst.write(data)


def _apply_geometry_mask(arr, transform, aoi_geom, nodata, params):
    """Apply polygon mask + NaN -> nodata in-place across all bands."""
    out_shape = arr.shape[-2:]
    poly_mask = geometry_mask(
        [mapping(aoi_geom)],
        out_shape=out_shape,
        transform=transform,
        invert=False,
        all_touched=params.get('all_touched', False),
    )

    nan_mask = None
    if np.issubdtype(arr.dtype, np.floating):
        nan_mask = np.isnan(arr)

    if arr.ndim == 2:
        arr[poly_mask] = nodata
        if nan_mask is not None:
            arr[nan_mask] = nodata
    else:
        for b in range(arr.shape[0]):
            arr[b][poly_mask] = nodata
            if nan_mask is not None:
                arr[b][nan_mask[b]] = nodata
    return arr


def _resolve_nodata(arr, attrs, params):
    """Pick a nodata value from source attrs, params override, or dtype default."""
    nodata = params.get('nodata') if params.get('nodata') is not None else None
    if nodata is None:
        nodata = attrs.get('_FillValue', attrs.get('missing_value'))
    if nodata is not None:
        return float(nodata)
    if np.issubdtype(arr.dtype, np.floating):
        return np.float32(-9999.0)
    return 0


def clip_netcdf_with_xarray(nc_path, aoi_geom, params, var=None):
    """
    Read a NetCDF with xarray (handles GMT/COARDS conventions that rasterio
    does not auto-detect), slice to the AOI bounding box, apply the polygon
    mask, return (data, transform, nodata, source_meta).
    """
    import xarray as xr  # local import — only needed for .nc

    ds = xr.open_dataset(nc_path)
    try:
        # 1. Locate the data variable
        if var is None:
            candidates = [n for n, da_ in ds.data_vars.items() if da_.ndim >= 2]
            if not candidates:
                raise ValueError(f"No 2D variable found in {nc_path.name}")
            var = candidates[0]
        da = ds[var]
        print(f"    NetCDF variable: '{var}'  dims: {da.dims}  shape: {da.shape}")
        print(f"    Source dtype: {da.dtype}")

        # 2. Locate lat/lon dims (common aliases + fallback to last two dims)
        lon_aliases = {'lon', 'longitude', 'x', 'X'}
        lat_aliases = {'lat', 'latitude', 'y', 'Y'}
        lon_dim = next((d for d in da.dims if d in lon_aliases), None)
        lat_dim = next((d for d in da.dims if d in lat_aliases), None)
        if lon_dim is None or lat_dim is None:
            lat_dim, lon_dim = da.dims[-2], da.dims[-1]
            print(f"    [!] Falling back to dim convention: lat='{lat_dim}', lon='{lon_dim}'")

        lons = da[lon_dim].values
        lats = da[lat_dim].values

        # 3. Source extent and resolution
        src_west,  src_east  = float(min(lons[0], lons[-1])), float(max(lons[0], lons[-1]))
        src_south, src_north = float(min(lats[0], lats[-1])), float(max(lats[0], lats[-1]))
        xres = abs(float(lons[1] - lons[0]))
        yres = abs(float(lats[1] - lats[0]))
        print(f"    Source extent: lon [{src_west:.4f}, {src_east:.4f}]  "
              f"lat [{src_south:.4f}, {src_north:.4f}]")
        print(f"    Resolution: x={xres:.6f}\u00b0  y={yres:.6f}\u00b0")

        # 4. AOI alignment for 0..360 grids
        aoi_aligned, shift_msg = align_aoi_to_raster_east(aoi_geom, src_east)
        print(f"    Longitude alignment: {shift_msg}")
        aoi_w, aoi_s, aoi_e, aoi_n = aoi_aligned.bounds

        # 5. Coordinate-based slice — xarray handles descending coords via slice
        lat_slice = slice(aoi_n, aoi_s) if lats[0] > lats[-1] else slice(aoi_s, aoi_n)
        lon_slice = slice(aoi_w, aoi_e) if lons[0] < lons[-1] else slice(aoi_e, aoi_w)

        sliced = da.sel({lat_dim: lat_slice, lon_dim: lon_slice})

        if sliced.size == 0:
            raise ValueError(
                f"AOI [{aoi_w:.4f}, {aoi_s:.4f}, {aoi_e:.4f}, {aoi_n:.4f}] doesn't overlap "
                f"raster [{src_west:.4f}, {src_south:.4f}, {src_east:.4f}, {src_north:.4f}]"
            )

        # 6. Force read (only the AOI window from disk via memmap/lazy backend)
        arr = np.asarray(sliced.values)
        sliced_lons = sliced[lon_dim].values
        sliced_lats = sliced[lat_dim].values
        print(f"    Sliced shape: {arr.shape}")

        # 7. Reorient to raster convention (lat descending, lon ascending)
        if sliced_lats[0] < sliced_lats[-1]:
            arr = arr[..., ::-1, :]
            sliced_lats = sliced_lats[::-1]
        if sliced_lons[0] > sliced_lons[-1]:
            arr = arr[..., :, ::-1]
            sliced_lons = sliced_lons[::-1]

        # 8. Build affine transform from cell-center coords
        top  = float(sliced_lats[0]) + yres / 2.0
        left = float(sliced_lons[0]) - xres / 2.0
        transform = from_origin(left, top, xres, yres)

        # 9. Promote to 3D (bands, rows, cols)
        if arr.ndim == 2:
            arr = arr[np.newaxis, ...]

        # 10. Resolve nodata, mask, return
        nodata = _resolve_nodata(arr, da.attrs, params)
        arr = _apply_geometry_mask(arr, transform, aoi_aligned, nodata, params)

        src_meta = {
            'driver':    'GTiff',
            'dtype':     str(arr.dtype),
            'crs':       CRS.from_epsg(4326),
            'count':     arr.shape[0],
            'height':    arr.shape[1],
            'width':     arr.shape[2],
            'transform': transform,
            'nodata':    nodata,
        }
        return arr, transform, nodata, src_meta

    finally:
        ds.close()


def clip_raster_with_rasterio(src_path, aoi_geom, params):
    """
    Windowed clip for georeferenced rasters (GeoTIFF, etc.) that rasterio can
    open with full georef. Reads only the AOI-bounding window from disk.
    """
    with rasterio.open(src_path) as src:
        print(f"    Source CRS:    {src.crs}")
        print(f"    Source bounds: {src.bounds}")
        print(f"    Source shape:  {src.height} x {src.width}, {src.count} band(s)")

        aoi_aligned, shift_msg = align_aoi_to_raster_east(aoi_geom, src.bounds.right)
        print(f"    Longitude alignment: {shift_msg}")

        win = from_bounds(*aoi_aligned.bounds, transform=src.transform)
        full = Window(0, 0, src.width, src.height)
        win = win.intersection(full).round_offsets().round_lengths()

        if win.width <= 0 or win.height <= 0:
            raise ValueError("AOI does not intersect the raster extent.")

        print(f"    Window: col_off={win.col_off}, row_off={win.row_off}, "
              f"width={win.width}, height={win.height}")

        arr = src.read(window=win, masked=False)
        win_transform = src.window_transform(win)

        nodata = _resolve_nodata(arr, {'_FillValue': src.nodata}, params)
        arr = _apply_geometry_mask(arr, win_transform, aoi_aligned, nodata, params)

        return arr, win_transform, nodata, src.meta


def main(config_file):
    # 1. Path Setup
    config_path = Path(config_file).resolve()
    if not config_path.exists():
        print(f"[!] Error: Config not found at {config_path}")
        return

    config = load_config(config_path)
    project_root = config_path.parent.parent

    # 2. AOI Resolution
    aoi_rel = config['paths']['aoi']
    aoi_file = Path(aoi_rel) if os.path.isabs(aoi_rel) else project_root / aoi_rel

    if not aoi_file.exists():
        print(f"[!] AOI file missing: {aoi_file}")
        return

    aoi_uri = f"zip://{aoi_file.as_posix()}" if aoi_file.suffix.lower() == '.zip' else aoi_file.as_posix()

    print(f"[*] Reading AOI: {aoi_uri}")
    try:
        aoi_gdf = gpd.read_file(aoi_uri, engine='fiona').to_crs(epsg=4326)
        if hasattr(aoi_gdf.geometry, 'union_all'):
            aoi_geom = aoi_gdf.geometry.union_all()
        else:
            aoi_geom = aoi_gdf.geometry.unary_union
    except Exception as e:
        print(f"[!] GeoPandas read failed: {e}")
        return

    print(f"[*] AOI bounds: {aoi_geom.bounds}")

    # 3. Source / Output Directories
    params = config['extraction']

    src_dir = Path(config['paths']['source_base'])
    src_dir = src_dir if src_dir.is_absolute() else project_root / src_dir

    out_dir = Path(config['paths']['output_base'])
    out_dir = out_dir if out_dir.is_absolute() else project_root / out_dir
    final_dir = out_dir / params['output_folder']
    final_dir.mkdir(parents=True, exist_ok=True)

    print(f"[*] Source dir: {src_dir}")
    print(f"[*] Output dir: {final_dir}")

    # 4. Process Each Input Grid
    inputs = params['inputs']
    print(f"[*] Inputs: {len(inputs)} grid(s) to clip\n")

    successes, failures = 0, 0
    netcdf_exts = {'.nc', '.nc4', '.netcdf'}

    for entry in inputs:
        src_file = src_dir / entry['file']
        if not src_file.exists():
            print(f"[!] Skip: {src_file.name} not found in {src_dir}")
            failures += 1
            continue

        subdataset = entry.get('subdataset')
        out_name   = entry.get('output_name', src_file.stem + '_aoi.tif')
        out_path   = final_dir / out_name

        if out_path.exists() and not params.get('overwrite', False):
            print(f"[=] {out_name}  (exists, skip - set overwrite: true to redo)\n")
            continue

        print(f"[+] Clipping {src_file.name}")

        try:
            if src_file.suffix.lower() in netcdf_exts:
                data, transform, nodata, src_meta = clip_netcdf_with_xarray(
                    src_file, aoi_geom, params, var=subdataset
                )
            else:
                data, transform, nodata, src_meta = clip_raster_with_rasterio(
                    src_file, aoi_geom, params
                )

            out_params = dict(params)
            out_params['nodata'] = nodata
            write_clipped_tif(data, transform, src_meta, out_path, out_params)

            mb = out_path.stat().st_size / (1024 ** 2)
            print(f"    [\u2713] {out_path.name}  ({data.shape}, {mb:.2f} MB, nodata={nodata})\n")
            successes += 1

        except Exception as e:
            print(f"    [!] Failed: {e}\n")
            failures += 1

    print(f"[*] Done. Success: {successes}  Failed: {failures}")
    print(f"[*] Output directory: {final_dir}")


if __name__ == "__main__":
    current_dir = Path(__file__).resolve().parent
    config_path = current_dir.parent / "config" / "02_gravity_mosaic.yaml"
    main(config_file=str(config_path))