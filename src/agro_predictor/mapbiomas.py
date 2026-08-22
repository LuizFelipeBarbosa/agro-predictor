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
    12: "Pasture",
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
    max_windows: int = 200,
    window_px: int = 2048,
    seed: int = 42,
    min_windows: int = 40,
) -> pd.DataFrame:
    """Sample stable, homogeneous MapBiomas points across a state.

    Returns a sits-ready frame: longitude, latitude, start_date, end_date, label.
    """
    os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
    boundary = gpd.read_file(fetch_state_boundary(uf)).to_crs(epsg=4326)
    state_shape = boundary.union_all()
    lon_min, lat_min, lon_max, lat_max = boundary.total_bounds

    rng = np.random.default_rng(seed)
    primary = {label: [] for label in CLASS_MAP.values()}
    overflow = {label: [] for label in CLASS_MAP.values()}
    primary_windows = {label: set() for label in CLASS_MAP.values()}
    per_window_cap = max(1, per_class // 5)

    old, new = (rasterio.open(MAPBIOMAS_URL.format(year=y)) for y in STABILITY_YEARS)
    try:
        for i in range(max_windows):
            if _should_stop(primary, per_class, i, min_windows):
                break
            lon = rng.uniform(lon_min, lon_max)
            lat = rng.uniform(lat_min, lat_max)
            window_points = {label: [] for label in CLASS_MAP.values()}
            for label, point in _stable_blocks_in_window(new, old, lon, lat, window_px):
                window_points[label].append(point)
            for label, points in window_points.items():
                _accept_window_points(
                    primary,
                    overflow,
                    primary_windows,
                    label,
                    points,
                    i,
                    per_window_cap,
                )
            if (i + 1) % 10 == 0:
                done = {label: len(points) for label, points in primary.items()}
                print(f"window {i + 1}/{max_windows}: {done}")
    finally:
        old.close()
        new.close()

    print(
        "windows per class:",
        {label: len(windows) for label, windows in primary_windows.items()},
    )
    rows = []
    for label, points in primary.items():
        primary_inside = [
            point for point in points if state_shape.contains(Point(point[:2]))
        ]
        overflow_inside = [
            point for point in overflow[label] if state_shape.contains(Point(point[:2]))
        ]
        selection_pool, used_overflow = _selection_pool(
            primary_inside, overflow_inside, per_class
        )
        if used_overflow:
            contributing_windows = {point[2] for point in selection_pool}
            print(
                f"{label} is spatially concentrated: relaxed per-window cap "
                f"(points from {len(contributing_windows)} window(s))"
            )
        if len(selection_pool) < MIN_CLASS_SAMPLES:
            print(
                f"Dropping {label}: only {len(selection_pool)} usable points "
                f"(<{MIN_CLASS_SAMPLES})"
            )
            continue
        chosen = rng.choice(
            len(selection_pool),
            size=min(per_class, len(selection_pool)),
            replace=False,
        )
        rows += [
            {
                "longitude": round(selection_pool[j][0], 6),
                "latitude": round(selection_pool[j][1], 6),
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


def _should_stop(
    primary: dict[str, list],
    per_class: int,
    windows_visited: int,
    min_windows: int,
) -> bool:
    """Return whether every pool is full after enough windows were visited."""
    return windows_visited >= min_windows and all(
        len(points) >= per_class for points in primary.values()
    )


def _accept_window_points(
    primary: dict[str, list[tuple[float, float, int]]],
    overflow: dict[str, list[tuple[float, float, int]]],
    primary_windows: dict[str, set[int]],
    label: str,
    points: list[tuple[float, float]],
    window_index: int,
    per_window_cap: int,
) -> None:
    """Retain capped primary and overflow points without overall pool limits."""
    primary_points = primary[label]
    overflow_points = overflow[label]
    primary_count = min(len(points), per_window_cap)
    overflow_count = min(len(points) - primary_count, per_window_cap)

    new_primary = [(*point, window_index) for point in points[:primary_count]]
    primary_points.extend(new_primary)
    if new_primary:
        primary_windows[label].add(window_index)

    overflow_points.extend(
        (*point, window_index)
        for point in points[primary_count : primary_count + overflow_count]
    )


def _selection_pool(
    primary: list[tuple[float, float, int]],
    overflow: list[tuple[float, float, int]],
    per_class: int,
) -> tuple[list[tuple[float, float, int]], list[tuple[float, float, int]]]:
    """Return primary points, topped up by overflow in deterministic window order."""
    if len(primary) >= per_class:
        return list(primary), []

    ordered_overflow = [
        point
        for _, point in sorted(
            enumerate(overflow), key=lambda item: (item[1][2], item[0])
        )
    ]
    used_overflow = ordered_overflow[: per_class - len(primary)]
    return [*primary, *used_overflow], used_overflow


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
