"""Region of interest: the Mato Grosso do Sul state boundary from IBGE."""

import gzip
import urllib.request
from pathlib import Path
from urllib.parse import urlencode

import geopandas as gpd

from agro_predictor.config import MS_BOUNDARY_PATH

# IBGE "malhas" v3 API; serves SIRGAS 2000 (EPSG:4674).
MS_BOUNDARY_URL = "https://servicodados.ibge.gov.br/api/v3/malhas/estados/MS?" + urlencode(
    {"formato": "application/vnd.geo+json", "qualidade": "intermediaria"}
)


def fetch_ms_boundary(dest: Path = MS_BOUNDARY_PATH) -> Path:
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(MS_BOUNDARY_URL, headers={"User-Agent": "agro-predictor"})
    with urllib.request.urlopen(request) as response:
        payload = response.read()
    if payload[:2] == b"\x1f\x8b":  # IBGE gzips the response regardless of Accept-Encoding
        payload = gzip.decompress(payload)
    dest.write_bytes(payload)
    return dest


def load_roi(roi: dict | Path) -> "dict | gpd.GeoDataFrame":
    """Return the ROI in a form sits_cube() accepts.

    A bbox dict (lon_min/lat_min/lon_max/lat_max) passes through unchanged; a
    GeoJSON path is fetched if missing, loaded, and reprojected to EPSG:4326.
    """
    if isinstance(roi, dict):
        return roi
    boundary = gpd.read_file(fetch_ms_boundary(Path(roi)))
    if boundary.crs is None:
        boundary = boundary.set_crs(epsg=4674)
    return boundary.to_crs(epsg=4326)


def describe_boundary(path: Path = MS_BOUNDARY_PATH) -> None:
    boundary = gpd.read_file(fetch_ms_boundary(path))
    lon_min, lat_min, lon_max, lat_max = boundary.total_bounds
    print(f"Boundary file: {path}")
    print(f"CRS: {boundary.crs}")
    print(f"Bounds: lon {lon_min:.3f}..{lon_max:.3f}, lat {lat_min:.3f}..{lat_max:.3f}")
