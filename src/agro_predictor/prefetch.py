"""Download BDC imagery into the local MOD13Q1 cache."""

import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlencode

import rasterio

from agro_predictor.config import BDC_STAC_URL, PROJECT_ROOT, RunConfig
from agro_predictor.roi import load_roi

DEFAULT_CACHE_DIR = PROJECT_ROOT / "data" / "cache" / "mod13q1"

STAC_PAGE_SIZE = 50
DOWNLOAD_WORKERS = 4
DOWNLOAD_ATTEMPTS = 6
DOWNLOAD_TIMEOUT_SECONDS = 120
USER_AGENT = "agro-predictor"


def _bbox(config: RunConfig) -> tuple[float, float, float, float]:
    roi = load_roi(config.roi)
    if isinstance(roi, dict):
        return (
            float(roi["lon_min"]),
            float(roi["lat_min"]),
            float(roi["lon_max"]),
            float(roi["lat_max"]),
        )
    return tuple(float(value) for value in roi.total_bounds)


def _fetch_stac_page(
    config: RunConfig,
    bbox: tuple[float, float, float, float],
    page: int,
) -> dict:
    query = urlencode(
        {
            "bbox": ",".join(str(value) for value in bbox),
            "datetime": f"{config.start_date}/{config.end_date}",
            "limit": STAC_PAGE_SIZE,
            "page": page,
        }
    )
    url = f"{BDC_STAC_URL}/collections/{config.collection.lower()}/items?{query}"
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response:
        return json.load(response)


def _assets_from_item(item: dict, bands: tuple[str, ...]) -> list[dict]:
    properties = item["properties"]
    tile = properties["bdc:tiles"][0]
    date = properties["datetime"][:10]
    item_assets = item.get("assets", {})
    assets = []
    for band in bands:
        if band not in item_assets:
            continue
        asset = item_assets[band]
        assets.append(
            {
                "tile": tile,
                "band": band,
                "date": date,
                "href": asset["href"],
                "size": asset.get("bdc:size"),
            }
        )
    return assets


def list_run_assets(config: RunConfig) -> list[dict]:
    """List the requested band assets intersecting a run's ROI and dates."""
    bbox = _bbox(config)
    assets = []
    page = 1
    while True:
        features = _fetch_stac_page(config, bbox, page).get("features", [])
        for item in features:
            assets.extend(_assets_from_item(item, config.bands))
        if len(features) < STAC_PAGE_SIZE:
            break
        page += 1
    return assets


def _destination_path(asset: dict, dest_root: Path) -> Path:
    return dest_root / (f"TERRA_MODIS_{asset['tile']}_{asset['band']}_{asset['date']}.tif")


def _is_cached(asset: dict, destination: Path) -> bool:
    if not destination.exists():
        return False
    size = destination.stat().st_size
    expected_size = asset["size"]
    return size == expected_size if expected_size is not None else size > 0


def _download_asset(asset: dict, destination: Path) -> None:
    partial = destination.with_suffix(f"{destination.suffix}.part")
    expected_size = asset["size"]
    last_error = None

    for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
        try:
            if partial.exists() and expected_size is not None:
                partial_size = partial.stat().st_size
                if partial_size == expected_size:
                    partial.replace(destination)
                    return
                if partial_size > expected_size:
                    partial.unlink()

            offset = partial.stat().st_size if partial.exists() else 0
            headers = {"User-Agent": USER_AGENT}
            if offset:
                headers["Range"] = f"bytes={offset}-"
            request = urllib.request.Request(asset["href"], headers=headers)
            with urllib.request.urlopen(request, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response:
                mode = "ab" if offset and response.status == 206 else "wb"
                with partial.open(mode) as output:
                    while chunk := response.read(1024 * 1024):
                        output.write(chunk)

            downloaded_size = partial.stat().st_size
            if expected_size is not None and downloaded_size != expected_size:
                raise OSError(f"expected {expected_size} bytes, received {downloaded_size}")
            if downloaded_size == 0:
                raise OSError("downloaded file is empty")
            partial.replace(destination)
            return
        except Exception as error:  # noqa: BLE001 -- retry transient HTTP and I/O failures
            last_error = error
            if attempt < DOWNLOAD_ATTEMPTS:
                time.sleep(attempt)

    raise RuntimeError(
        f"Failed to download {destination.name} after {DOWNLOAD_ATTEMPTS} attempts: {last_error}"
    ) from last_error


def _download_missing_assets(assets: list[dict], dest_root: Path) -> None:
    dest_root.mkdir(parents=True, exist_ok=True)
    pending = []
    completed = 0
    total = len(assets)

    for asset in assets:
        destination = _destination_path(asset, dest_root)
        if _is_cached(asset, destination):
            completed += 1
            print(f"{completed}/{total}")
        else:
            pending.append((asset, destination))

    with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as executor:
        futures = {
            executor.submit(_download_asset, asset, destination): destination
            for asset, destination in pending
        }
        for future in as_completed(futures):
            future.result()
            completed += 1
            print(f"{completed}/{total}")

    if total == 0:
        print("0/0")


def _is_readable_geotiff(path: Path) -> bool:
    try:
        with rasterio.open(path) as dataset:
            if dataset.count < 1 or dataset.width < 1 or dataset.height < 1:
                return False
            dataset.read(1, window=((0, 1), (0, 1)))
    except Exception:  # noqa: BLE001 -- any raster failure means the cache file is unusable
        return False
    return True


def _remove_corrupt_files(assets: list[dict], dest_root: Path) -> list[dict]:
    corrupt = []
    for asset in assets:
        destination = _destination_path(asset, dest_root)
        if destination.exists() and _is_readable_geotiff(destination):
            continue
        if destination.exists():
            destination.unlink()
            print(f"Invalid GeoTIFF removed: {destination}")
        corrupt.append(asset)
    return corrupt


def prefetch_run(
    config: RunConfig,
    dest_root: Path = DEFAULT_CACHE_DIR,
) -> Path:
    """Download and verify every COG needed by a run."""
    dest_root = Path(dest_root)
    assets = list_run_assets(config)
    _download_missing_assets(assets, dest_root)

    corrupt = _remove_corrupt_files(assets, dest_root)
    if corrupt:
        print(f"Retrying {len(corrupt)} invalid GeoTIFF(s)")
        _download_missing_assets(corrupt, dest_root)

    still_corrupt = _remove_corrupt_files(corrupt, dest_root)
    if still_corrupt:
        names = ", ".join(_destination_path(asset, dest_root).name for asset in still_corrupt)
        raise RuntimeError(f"GeoTIFF integrity check failed after retry: {names}")
    return dest_root
