from datetime import date

from agro_predictor import config


def test_presets_construct():
    smoke = config.smoke()
    full = config.full_state()
    assert smoke.collection == "MOD13Q1-6.1"
    assert smoke.source == "BDC"
    assert full.roi == config.MS_BOUNDARY_PATH


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
