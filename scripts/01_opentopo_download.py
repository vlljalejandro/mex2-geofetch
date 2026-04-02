import yaml
import requests
import geopandas as gpd
from pathlib import Path
from tqdm import tqdm


def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)

def download_opentopo_dem(config_file):
    # 1. Path Setup
    config_path = Path(config_file).resolve()
    if not config_path.exists():
        print(f"[!] Error: Config not found at {config_path}")
        return

    config = load_config(config_path)
    project_root = config_path.parent.parent

    # 2. AOI Resolution
    aoi_rel = config['paths']['aoi']
    aoi_file = Path(aoi_rel) if Path(aoi_rel).is_absolute() else project_root / aoi_rel

    if not aoi_file.exists():
        print(f"[!] AOI file missing: {aoi_file}")
        return

    print(f"[*] Reading AOI: {aoi_file}")
    try:
        aoi_gdf = gpd.read_file(aoi_file).to_crs(epsg=4326)

        if hasattr(aoi_gdf.geometry, 'union_all'):
            aoi_geom = aoi_gdf.geometry.union_all()
        else:
            aoi_geom = aoi_gdf.geometry.unary_union

        west, south, east, north = aoi_geom.bounds
    except Exception as e:
        print(f"[!] GeoPandas read failed: {e}")
        return

    # 3. Build Request Parameters
    dem_params = config['dem_params']
    params = {
        "demtype":      dem_params['demtype'],
        "south":        south,
        "north":        north,
        "west":         west,
        "east":         east,
        "outputFormat": dem_params['output_format'],
        "API_Key":      config['auth']['api_key']
    }

    # 4. Setup Output
    out_dir = Path(config['paths']['output_base'])
    out_dir = out_dir if out_dir.is_absolute() else project_root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{dem_params['demtype']}_subset.{dem_params['output_format'].lower()}"
    save_path = out_dir / filename

    # 5. Execute Download
    print(f"[*] Requesting {dem_params['demtype']} from OpenTopography...")
    try:
        response = requests.get(
            config['api']['base_url'],
            params=params,
            stream=True
        )
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))

        with open(save_path, 'wb') as f, tqdm(
            desc=filename,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                size = f.write(chunk)
                bar.update(size)

        print(f"[*] Download complete: {save_path}")

    except requests.exceptions.HTTPError as e:
        print(f"[!] API Error: {e.response.text}")
    except Exception as e:
        print(f"[!] Unexpected error: {e}")


if __name__ == "__main__":
    current_dir = Path(__file__).resolve().parent
    config_path = current_dir.parent / "config" / "01_opentopo_download.yaml"
    download_opentopo_dem(str(config_path))