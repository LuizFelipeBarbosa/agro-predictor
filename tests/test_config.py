import re
from datetime import date

import pytest

from agro_predictor import config


def test_presets_construct():
    smoke = config.smoke()
    full = config.full_state()
    assert smoke.collection == "MOD13Q1-6.1"
    assert smoke.source == "BDC"
    assert full.roi == config.ROI_DIR / "ms_boundary.geojson"


def test_full_state_takes_a_state_code():
    goias = config.full_state("GO")
    assert goias.name == "go-2023-2024"
    assert goias.roi == config.ROI_DIR / "go_boundary.geojson"


def test_dates_are_valid_and_ordered():
    for cfg in (config.smoke(), config.full_state()):
        assert date.fromisoformat(cfg.start_date) < date.fromisoformat(cfg.end_date)


def test_smoke_bbox_is_ordered():
    bbox = config.SMOKE_BBOX
    assert bbox["lon_min"] < bbox["lon_max"]
    assert bbox["lat_min"] < bbox["lat_max"]


def test_output_dir_is_per_run():
    assert config.smoke().output_dir == config.OUTPUT_ROOT / "smoke"
    assert config.full_state().output_dir == config.OUTPUT_ROOT / "ms-2023-2024"


def test_labels_csv_path_is_under_data_labels():
    assert config.LABELS_CSV_PATH.parent == config.PROJECT_ROOT / "data" / "labels"
    assert config.LABELS_CSV_PATH.name == "labels.csv"


def test_brazil_bbox_contains_smoke_bbox():
    brazil = config.BRAZIL_BBOX
    smoke = config.SMOKE_BBOX
    assert brazil["lon_min"] < brazil["lon_max"]
    assert brazil["lat_min"] < brazil["lat_max"]
    assert brazil["lon_min"] <= smoke["lon_min"] <= smoke["lon_max"] <= brazil["lon_max"]
    assert brazil["lat_min"] <= smoke["lat_min"] <= smoke["lat_max"] <= brazil["lat_max"]


def test_presets_use_labels_by_default():
    assert config.smoke().use_labels is True
    assert config.full_state().use_labels is True
    assert config.smoke().num_trees == 100
    assert config.smoke().mtry is None
    assert config.smoke().rebalance is None


def test_run_config_rejects_unknown_bands():
    with pytest.raises(ValueError, match=r"Unknown band\(s\): GREEN.*Known bands:.*NDVI.*MIR"):
        config.RunConfig(
            name="invalid",
            start_date="2023-09-14",
            end_date="2024-08-31",
            roi=config.SMOKE_BBOX,
            bands=("NDVI", "GREEN"),
        )


def test_canned_sample_band_validation_rejects_red_and_blue():
    with pytest.raises(
        ValueError,
        match=r"Canned samples do not support band\(s\): BLUE, RED.*NDVI, EVI, NIR, MIR",
    ):
        config.validate_canned_sample_bands(("NDVI", "RED", "BLUE"))


def test_class_registry_invariants():
    assert len(config.CLASSES) == 13
    assert len({style.name for style in config.CLASSES}) == len(config.CLASSES)
    assert len({style.color for style in config.CLASSES}) == len(config.CLASSES)
    assert all(re.fullmatch(r"#[0-9a-fA-F]{6}", style.color) for style in config.CLASSES)
    assert all(isinstance(style.display, str) and style.display for style in config.CLASSES)


def test_canned_classes_are_a_registry_subset():
    assert config.CANNED_CLASSES == (
        "Cerrado",
        "Forest",
        "Pasture",
        "Soy_Corn",
        "Soy_Cotton",
        "Soy_Fallow",
        "Soy_Millet",
    )
    assert set(config.CANNED_CLASSES) <= set(config.CLASS_NAMES)


def test_mapbiomas_classes_are_registered():
    from agro_predictor.mapbiomas import CLASS_MAP

    assert set(CLASS_MAP.values()) <= set(config.CLASS_NAMES)
