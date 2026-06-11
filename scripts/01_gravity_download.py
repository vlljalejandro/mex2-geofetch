import yaml
import os
import time
from tqdm import tqdm
import requests
from pathlib import Path


def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)


def get_remote_size(session, url, timeout=30):
    """HEAD request to get Content-Length; returns 0 if unavailable."""
    try:
        r = session.head(url, allow_redirects=True, timeout=timeout)
        r.raise_for_status()
        return int(r.headers.get('Content-Length', 0))
    except Exception:
        return 0


def download_file(session, url, save_path, timeout, retries):
    """
    Stream a single file to disk with resume support and a byte-level progress bar.
    Writes to <name>.part and atomically renames on success.
    """
    part_path = save_path.with_suffix(save_path.suffix + '.part')
    total_size = get_remote_size(session, url, timeout=30)

    # Already-complete check
    if save_path.exists() and total_size > 0 and save_path.stat().st_size == total_size:
        return True, save_path.stat().st_size, "cached"

    # Resume from .part if present and server knows the total size
    existing = part_path.stat().st_size if part_path.exists() else 0
    headers, mode = {}, 'wb'
    if 0 < existing < total_size:
        headers['Range'] = f'bytes={existing}-'
        mode = 'ab'
    elif total_size > 0 and existing >= total_size:
        part_path.rename(save_path)
        return True, save_path.stat().st_size, "resumed-complete"

    for attempt in range(1, retries + 1):
        try:
            with session.get(url, headers=headers, stream=True, timeout=timeout) as r:
                r.raise_for_status()
                with open(part_path, mode) as f, tqdm(
                    total=total_size if total_size > 0 else None,
                    initial=existing if mode == 'ab' else 0,
                    unit='B', unit_scale=True, unit_divisor=1024,
                    desc=save_path.name, leave=False
                ) as pbar:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))

            actual = part_path.stat().st_size
            if total_size > 0 and actual != total_size:
                raise IOError(f"Size mismatch: expected {total_size}, got {actual}")

            part_path.rename(save_path)
            return True, actual, "downloaded"

        except Exception as e:
            if attempt == retries:
                print(f"\n[!] Failed after {retries} attempts: {save_path.name} — {e}")
                return False, 0, "failed"
            time.sleep(2 ** attempt)  # exponential backoff: 2s, 4s, 8s, ...
            # Recompute resume position in case bytes were written before the failure
            existing = part_path.stat().st_size if part_path.exists() else 0
            if 0 < existing < total_size:
                headers['Range'] = f'bytes={existing}-'
                mode = 'ab'

    return False, 0, "failed"


def human(nbytes):
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if nbytes < 1024.0:
            return f"{nbytes:.2f} {unit}"
        nbytes /= 1024.0
    return f"{nbytes:.2f} PB"


def main(config_file):
    # 1. Path Setup
    config_path = Path(config_file).resolve()
    if not config_path.exists():
        print(f"[!] Error: Config not found at {config_path}")
        return

    config = load_config(config_path)
    project_root = config_path.parent.parent

    # 2. Source Parameters
    params   = config['sensors']['marine_gravity']
    base_url = params['base_url'].rstrip('/')
    files    = params['files']

    if not files:
        print("[!] No files listed in config.")
        return

    # 3. Output Setup
    out_dir = Path(config['paths']['output_base'])
    out_dir = out_dir if out_dir.is_absolute() else project_root / out_dir
    final_dir = out_dir / params['output_folder']
    final_dir.mkdir(parents=True, exist_ok=True)

    print(f"[*] Source : {base_url}")
    print(f"[*] Output : {final_dir}")
    print(f"[*] Files  : {len(files)} to fetch")
    print()

    # 4. Download Loop
    timeout = params.get('timeout_seconds', 300)
    retries = params.get('retries', 3)

    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 (marine-gravity-pipeline)"})

    downloaded, cached, failed = 0, 0, 0
    total_bytes = 0

    for fname in files:
        url       = f"{base_url}/{fname}"
        save_path = final_dir / fname

        print(f"[+] {fname}")
        ok, size, status = download_file(session, url, save_path, timeout, retries)

        if ok:
            total_bytes += size
            if status == "cached":
                cached += 1
                print(f"    [=] already complete ({human(size)})")
            else:
                downloaded += 1
                print(f"    [\u2713] {human(size)}")
        else:
            failed += 1

    print()
    print(f"[*] Done. Downloaded: {downloaded}  Cached: {cached}  Failed: {failed}")
    print(f"[*] Total data: {human(total_bytes)}")
    print(f"[*] Output directory: {final_dir}")


if __name__ == "__main__":
    current_dir = Path(__file__).resolve().parent
    config_path = current_dir.parent / "config" / "01_gravity_download.yaml"
    main(config_file=str(config_path))
