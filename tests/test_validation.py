import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin, xy
from rasterio.warp import transform

from agro_predictor import validation

PIXEL_SIZE = 231.65635826
RASTER_CRS = CRS.from_proj4(
    "+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +R=6371007.181 +units=m +no_defs"
)
TRANSFORM = from_origin(-5_000_000, -2_000_000, PIXEL_SIZE, PIXEL_SIZE)
RASTER_DATA = np.array([[1, 2], [255, 1]], dtype=np.uint8)


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
        {"code": 2, "name": "Pasture", "display": "Pasture", "color": "#fee391"},
    ]


def _point_for_cell(row: int, column: int) -> tuple[float, float]:
    x, y = xy(TRANSFORM, row, column, offset="center")
    longitudes, latitudes = transform(RASTER_CRS, "EPSG:4326", [x], [y])
    return longitudes[0], latitudes[0]


def _outside_point() -> tuple[float, float]:
    x = TRANSFORM.c - PIXEL_SIZE
    y = TRANSFORM.f - PIXEL_SIZE / 2
    longitudes, latitudes = transform(RASTER_CRS, "EPSG:4326", [x], [y])
    return longitudes[0], latitudes[0]


def test_sample_class_at_points_maps_codes_and_marks_unsampled(tmp_path):
    raster = _write_raster(tmp_path / "classes.tif")
    points = pd.DataFrame(
        [
            (*_point_for_cell(0, 0), "Cerrado"),
            (*_point_for_cell(0, 1), "Pasture"),
            (*_point_for_cell(1, 0), "Pasture"),
            (*_outside_point(), "Cerrado"),
        ],
        columns=["longitude", "latitude", "label"],
    )

    sampled = validation.sample_class_at_points(raster, points, _legend())

    assert sampled.loc[:1, "predicted_label"].tolist() == ["Cerrado", "Pasture"]
    assert pd.isna(sampled.loc[2, "predicted_label"])
    assert pd.isna(sampled.loc[3, "predicted_label"])
    assert "predicted_label" not in points.columns
    assert sampled.columns.tolist() == [*points.columns, "predicted_label"]


def test_confusion_matrix_is_square_and_excludes_unsampled_rows():
    frame = pd.DataFrame(
        {
            "label": ["A", "A", "B", "B", "C"],
            "predicted_label": ["A", "B", "B", pd.NA, "A"],
        }
    )

    result = validation.confusion_matrix(frame)

    expected = pd.DataFrame(
        [[1, 1, 0], [0, 1, 0], [1, 0, 0]],
        index=["A", "B", "C"],
        columns=["A", "B", "C"],
        dtype=int,
    )
    expected.index.name = "reference"
    pd.testing.assert_frame_equal(result, expected)
    assert int(frame["predicted_label"].isna().sum()) == 1


def test_accuracy_metrics_matches_hand_calculation():
    cm = pd.DataFrame([[4, 1], [2, 3]], index=["A", "B"], columns=["A", "B"])

    metrics = validation.accuracy_metrics(cm)

    # n=10, OA=(4+3)/10=.7; expected=(5*6 + 5*4)/100=.5; kappa=(.7-.5)/(1-.5)=.4.
    assert metrics["overall_accuracy"] == pytest.approx(0.7)
    assert metrics["kappa"] == pytest.approx(0.4)
    assert metrics["per_class"]["A"]["producer_accuracy"] == pytest.approx(4 / 5)
    assert metrics["per_class"]["A"]["user_accuracy"] == pytest.approx(4 / 6)
    assert metrics["per_class"]["B"]["producer_accuracy"] == pytest.approx(3 / 5)
    assert metrics["per_class"]["B"]["user_accuracy"] == pytest.approx(3 / 4)


def test_validate_map_writes_all_artifacts(tmp_path):
    run_dir = tmp_path / "fake-run"
    run_dir.mkdir()
    _write_raster(run_dir / "fake-run_class_mosaic.tif")
    (run_dir / "run_labels.json").write_text(
        json.dumps({"labels": ["Cerrado", "Pasture"]}),
        encoding="utf-8",
    )
    points_csv = tmp_path / "mapbiomas_reference.csv"
    pd.DataFrame(
        [
            (*_point_for_cell(0, 0), "Cerrado"),
            (*_point_for_cell(0, 1), "Cerrado"),
            (*_point_for_cell(1, 1), "Cerrado"),
            (*_point_for_cell(1, 0), "Pasture"),
        ],
        columns=["longitude", "latitude", "label"],
    ).to_csv(points_csv, index=False)

    metrics = validation.validate_map(run_dir, points_csv, state="MS")

    validation_dir = run_dir / "validation"
    assert (validation_dir / "points.csv").exists()
    assert (validation_dir / "confusion_matrix.csv").exists()
    accuracy_path = validation_dir / "accuracy.json"
    assert accuracy_path.exists()
    payload = json.loads(accuracy_path.read_text(encoding="utf-8"))
    assert set(payload) == {
        "overall_accuracy",
        "kappa",
        "per_class",
        "n_points",
        "n_unsampled",
        "points_csv",
        "state",
        "generated_on",
        "reference",
        "caveat",
    }
    assert payload["n_points"] == 4
    assert payload["n_unsampled"] == 1
    assert payload["reference"] == "mapbiomas-c10"
    assert payload["overall_accuracy"] == pytest.approx(metrics["overall_accuracy"])


def test_importing_validation_does_not_import_pysits():
    repo_root = Path(__file__).resolve().parents[1]
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, agro_predictor.validation; assert 'pysits' not in sys.modules",
        ],
        check=True,
        cwd=repo_root,
    )
