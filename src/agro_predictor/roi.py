"""Region of interest: Brazilian state boundaries from the IBGE malhas API."""

import gzip
import urllib.request
from pathlib import Path
from urllib.parse import urlencode

import geopandas as gpd

from agro_predictor.config import ROI_DIR

# IBGE "malhas" v3 API; serves SIRGAS 2000 (EPSG:4674).
BOUNDARY_URL = "https://servicodados.ibge.gov.br/api/v3/malhas/estados/{uf}?" + urlencode(
    {"formato": "application/vnd.geo+json", "qualidade": "intermediaria"}
)


def boundary_path(uf: str) -> Path:
    return ROI_DIR / f"{uf.lower()}_boundary.geojson"


def fetch_state_boundary(uf: str) -> Path:
    dest = boundary_path(uf)
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = BOUNDARY_URL.format(uf=uf.upper())
    request = urllib.request.Request(url, headers={"User-Agent": "agro-predictor"})
    with urllib.request.urlopen(request) as response:
        payload = response.read()
    if payload[:2] == b"\x1f\x8b":  # IBGE gzips the response regardless of Accept-Encoding
        payload = gzip.decompress(payload)
    dest.write_bytes(payload)
    return dest


def load_roi(roi: dict | Path) -> "dict | gpd.GeoDataFrame":
    """Return the ROI in a form sits_cube() accepts.

    A bbox dict (lon_min/lat_min/lon_max/lat_max) passes through unchanged; a
    boundary GeoJSON path (data/roi/<uf>_boundary.geojson) is fetched if
    missing, loaded, and reprojected to EPSG:4326.
    """
    if isinstance(roi, dict):
        return roi
    path = Path(roi)
    if not path.exists():
        fetch_state_boundary(path.stem.split("_")[0])
    boundary = gpd.read_file(path)
    if boundary.crs is None:
        boundary = boundary.set_crs(epsg=4674)
    return boundary.to_crs(epsg=4326)


def describe_boundary(uf: str) -> None:
    path = fetch_state_boundary(uf)
    boundary = gpd.read_file(path)
    lon_min, lat_min, lon_max, lat_max = boundary.total_bounds
    print(f"Boundary file: {path}")
    print(f"CRS: {boundary.crs}")
    print(f"Bounds: lon {lon_min:.3f}..{lon_max:.3f}, lat {lat_min:.3f}..{lat_max:.3f}")
