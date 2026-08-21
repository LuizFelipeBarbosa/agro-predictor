from datetime import date

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
