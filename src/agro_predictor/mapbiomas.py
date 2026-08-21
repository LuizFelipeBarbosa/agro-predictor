"""Era-calibrated training samples from MapBiomas Collection 10.

The canned sitsdata samples carry 2006-2016 MODIS signatures; Terra's orbital
drift makes recent MODIS years look different, so a model for a recent crop
year must be trained on series from that same era. MapBiomas provides the
labels: Brazil-wide 30 m annual land-cover COGs on public cloud storage,
readable by windowed range-requests with no full download.

A point qualifies as a training sample when its class is identical in the two
most recent MapBiomas years (stability) and uniform across the ~250 m MODIS
footprint (homogeneity). The output CSV feeds sits_get_data, which extracts
each point's MODIS time series for the requested crop-year window.
"""

import os

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from shapely.geometry import Point

from agro_predictor.roi import fetch_state_boundary

MAPBIOMAS_URL = (
    "https://storage.googleapis.com/mapbiomas-public/"
    "initiatives/brasil/collection_10/lulc/coverage/brazil_coverage_{year}.tif"
)
STABILITY_YEARS = (2023, 2024)

# MapBiomas class id -> training label. Mosaic/ambiguous classes are omitted
# on purpose: a 250 m MODIS pixel of them has no single-class signature.
CLASS_MAP = {
    3: "Forest",
    4: "Cerrado",
    9: "Planted_Forest",
    11: "Wetland",
    12: "Grassland",
    15: "Pasture",
    20: "Sugarcane",
    33: "Water",
    39: "Soybean",
}

# A MODIS pixel (231.66 m) spans ~7.7 MapBiomas pixels (30 m); an 8x8 block
# must be single-class for the point to be usable at MODIS scale.
BLOCK = 8

# Classes with fewer samples than this are dropped: too few points to teach
# the random forest a signature.
MIN_CLASS_SAMPLES = 30


def sample_state(
    uf: str,
    start_date: str,
    end_date: str,
    per_class: int = 200,
    max_windows: int = 60,
    window_px: int = 2048,
    seed: int = 42,
) -> pd.DataFrame:
    """Sample stable, homogeneous MapBiomas points across a state.

    Returns a sits-ready frame: longitude, latitude, start_date, end_date, label.
    """
    os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
    boundary = gpd.read_file(fetch_state_boundary(uf)).to_crs(epsg=4326)
    state_shape = boundary.union_all()
    lon_min, lat_min, lon_max, lat_max = boundary.total_bounds

    rng = np.random.default_rng(seed)
    collected: dict[str, list[tuple[float, float]]] = {v: [] for v in CLASS_MAP.values()}

    old, new = (rasterio.open(MAPBIOMAS_URL.format(year=y)) for y in STABILITY_YEARS)
    try:
        for i in range(max_windows):
            if all(len(v) >= per_class for v in collected.values()):
                break
            lon = rng.uniform(lon_min, lon_max)
            lat = rng.uniform(lat_min, lat_max)
            for label, point in _stable_blocks_in_window(new, old, lon, lat, window_px):
                if len(collected[label]) < per_class * 3:
                    collected[label].append(point)
            if (i + 1) % 10 == 0:
                done = {k: len(v) for k, v in collected.items()}
                print(f"window {i + 1}/{max_windows}: {done}")
    finally:
        old.close()
        new.close()

    rows = []
    for label, points in collected.items():
        inside = [p for p in points if state_shape.contains(Point(p))]
        if len(inside) < MIN_CLASS_SAMPLES:
            print(f"Dropping {label}: only {len(inside)} usable points (<{MIN_CLASS_SAMPLES})")
            continue
        chosen = rng.choice(len(inside), size=min(per_class, len(inside)), replace=False)
        rows += [
            {
                "longitude": round(inside[j][0], 6),
                "latitude": round(inside[j][1], 6),
                "start_date": start_date,
                "end_date": end_date,
                "label": label,
            }
            for j in chosen
        ]
    frame = pd.DataFrame(rows)
    if not frame.empty:
        print("Sampled per class:", frame["label"].value_counts().to_dict())
    return frame


def _stable_blocks_in_window(new_src, old_src, lon: float, lat: float, window_px: int):
    """Yield (label, (lon, lat)) for every stable single-class 8x8 block."""
    row, col = new_src.index(lon, lat)
    row = max(0, min(row, new_src.height - window_px))
    col = max(0, min(col, new_src.width - window_px))
    window = Window(col, row, window_px, window_px)
    a_new = new_src.read(1, window=window)
    a_old = old_src.read(1, window=window)

    side = (window_px // BLOCK) * BLOCK
    blocks_new = a_new[:side, :side].reshape(side // BLOCK, BLOCK, side // BLOCK, BLOCK)
    blocks_old = a_old[:side, :side].reshape(side // BLOCK, BLOCK, side // BLOCK, BLOCK)
    first = blocks_new[:, :1, :, :1]
    uniform = (
        (blocks_new == first).all(axis=(1, 3))
        & (blocks_old == first).all(axis=(1, 3))
    )
    classes = blocks_new[:, 0, :, 0]

    for class_id, label in CLASS_MAP.items():
        for br, bc in zip(*np.nonzero(uniform & (classes == class_id))):
            center_row = row + br * BLOCK + BLOCK // 2
            center_col = col + bc * BLOCK + BLOCK // 2
            x, y = new_src.xy(center_row, center_col)
            yield label, (x, y)
