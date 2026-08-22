import json
import math
import os
import subprocess
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin
from shapely.geometry import box

matplotlib.use("Agg")

from agro_predictor import areas, config

PIXEL_SIZE = 231.65635826
RASTER_CRS = CRS.from_proj4(
    "+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +R=6371007.181 +units=m +no_defs"
)
TRANSFORM = from_origin(-5_000_000, -2_000_000, PIXEL_SIZE, PIXEL_SIZE)
RASTER_DATA = np.array(
    [
        [1, 1, 2, 255],
        [1, 2, 2, 255],
        [3, 3, 2, 255],
        [3, 3, 3, 255],
    ],
    dtype=np.uint8,
)


def _write_raster(path: Path) -> Path:
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=RASTER_DATA.shape[0],
        width=RASTER_DATA.shape[1],
        count=1,
        dtype="uint8",
        crs=RASTER_CRS,
        transform=TRANSFORM,
        nodata=255,
    ) as dest:
        dest.write(RASTER_DATA, 1)
    return path


def _legend() -> list[dict]:
    return [
        {"code": 1, "name": "Cerrado", "display": "Cerrado", "color": "#a1d99b"},
        {"code": 2, "name": "Forest", "display": "Forest", "color": "#00441b"},
        {"code": 3, "name": "Pasture", "display": "Pasture", "color": "#fee391"},
    ]


def _boundary() -> gpd.GeoDataFrame:
    left = TRANSFORM.c
    top = TRANSFORM.f
    right = left + 2 * PIXEL_SIZE
    bottom = top - 3 * PIXEL_SIZE
    return gpd.GeoDataFrame(geometry=[box(left, bottom, right, top)], crs=RASTER_CRS)


def test_pixel_area_km2_uses_transform():
    assert areas.pixel_area_km2(TRANSFORM) == pytest.approx(0.0536647, rel=1e-4)


def test_find_or_build_mosaic_rebuilds_stale_and_reuses_fresh(tmp_path):
    run_dir = tmp_path / "test-run"
    run_dir.mkdir()
    tile_a = _write_raster(run_dir / "TERRA_MODIS_A_class_v1.tif")
    tile_b = _write_raster(run_dir / "TERRA_MODIS_B_class_v1.tif")
    mosaic = _write_raster(run_dir / "test-run_class_mosaic.tif")

    stale_mtime = 1_700_000_000
    tile_mtime = stale_mtime + 10
    os.utime(mosaic, (stale_mtime, stale_mtime))
    os.utime(tile_a, (tile_mtime, tile_mtime))
    os.utime(tile_b, (tile_mtime, tile_mtime))

    assert areas.find_or_build_mosaic(run_dir) == mosaic
    assert mosaic.stat().st_mtime > stale_mtime

    fresh_mtime = tile_mtime + 20
    untouched_data = np.full_like(RASTER_DATA, 7)
    with rasterio.open(mosaic, "r+") as dest:
        dest.write(untouched_data, 1)
    os.utime(mosaic, (fresh_mtime, fresh_mtime))

    assert areas.find_or_build_mosaic(run_dir) == mosaic
    assert mosaic.stat().st_mtime == pytest.approx(fresh_mtime)
    with rasterio.open(mosaic) as src:
        np.testing.assert_array_equal(src.read(1), untouched_data)


def test_class_areas_excludes_nodata_and_totals_classified_pixels(tmp_path):
    raster = _write_raster(tmp_path / "classes.tif")

    result = areas.class_areas(raster, _legend())

    assert result.iloc[:-1]["pixels"].tolist() == [3, 4, 5]
    assert result.iloc[:-1]["pct"].sum() == pytest.approx(100.0, abs=1e-6)
    assert result.iloc[-1]["pixels"] == 12
    assert result.iloc[-1]["km2"] == pytest.approx(
        12 * areas.pixel_area_km2(TRANSFORM)
    )


def test_clip_to_boundary_drops_pixels_outside_rectangle(tmp_path):
    raster = _write_raster(tmp_path / "classes.tif")
    clipped = tmp_path / "classes_clipped.tif"

    assert areas.clip_to_boundary(raster, _boundary(), clipped) == clipped

    with rasterio.open(clipped) as src:
        clipped_data = src.read(1)
    assert clipped_data.size < RASTER_DATA.size
    np.testing.assert_array_equal(
        clipped_data[clipped_data != 255],
        RASTER_DATA[:3, :2].ravel(),
    )


@pytest.mark.parametrize(
    "metadata",
    [
        {
            "labels": ["Wetland", "Soybean"],
            "version": "v1",
            "start_date": "2025-09-14",
            "end_date": "2026-08-31",
        },
        {"labels": ["Wetland", "Soybean"], "version": "v1"},
    ],
)
def test_load_run_legend_accepts_label_metadata_variants(tmp_path, metadata):
    (tmp_path / "run_labels.json").write_text(json.dumps(metadata), encoding="utf-8")

    legend = areas.load_run_legend(tmp_path)

    assert [(entry["code"], entry["name"]) for entry in legend] == [
        (1, "Wetland"),
        (2, "Soybean"),
    ]
    assert legend[0]["display"] == "Wetland"
    assert legend[1]["display"] == "Soybean (any rotation)"


def test_load_run_legend_falls_back_to_sorted_canned_classes(tmp_path):
    legend = areas.load_run_legend(tmp_path)

    assert [entry["code"] for entry in legend] == list(range(1, 8))
    assert [entry["name"] for entry in legend] == sorted(config.CANNED_CLASSES)


def test_compare_to_benchmark_aggregates_and_keeps_uncategorized_class():
    frame = pd.DataFrame(
        [
            {"code": 1, "name": "Soybean", "display": "Soybean", "pct": 20.0},
            {"code": 2, "name": "Pasture", "display": "Pasture", "pct": 40.0},
            {"code": 3, "name": "Cerrado", "display": "Cerrado", "pct": 30.0},
            {"code": 4, "name": "Mystery", "display": "Mystery class", "pct": 10.0},
            {"code": pd.NA, "name": pd.NA, "display": pd.NA, "pct": 100.0},
        ]
    )

    comparison = areas.compare_to_benchmark(frame, "siga-ms-2023-24").set_index(
        "category"
    )

    assert comparison.loc["soybean", "delta_pp"] == pytest.approx(8.2)
    assert comparison.loc["pasture", "delta_pp"] == pytest.approx(-8.3)
    assert comparison.loc["Mystery class", "model_pct"] == 10.0
    assert math.isnan(comparison.loc["Mystery class", "benchmark_pct"])


def test_importing_areas_does_not_import_pysits():
    repo_root = Path(__file__).resolve().parents[1]
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, agro_predictor.areas; assert 'pysits' not in sys.modules",
        ],
        check=True,
        cwd=repo_root,
    )


def test_render_preview_writes_png_with_boundary_and_percentages(tmp_path):
    raster = _write_raster(tmp_path / "classes.tif")
    preview = tmp_path / "preview.png"
    summary = areas.class_areas(raster, _legend())

    areas.render_preview(
        raster,
        _legend(),
        preview,
        boundary=_boundary(),
        areas_df=summary,
    )

    assert preview.stat().st_size > 0
