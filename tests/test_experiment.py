import os

import pytest

from agro_predictor.experiment import (
    era_cache_paths,
    reject_rebalance_in_kfold,
    variant_slug,
)


def test_era_cache_paths_are_band_order_insensitive(tmp_path):
    csv_path = tmp_path / "mapbiomas_ms_train.csv"

    first = era_cache_paths(csv_path, ("EVI", "NDVI"))
    second = era_cache_paths(csv_path, ("NDVI", "EVI"))

    assert first == second
    assert first == (
        tmp_path / "mapbiomas_ms_train.EVI-NDVI.rds",
        tmp_path / "mapbiomas_ms_train.EVI-NDVI.raw.rds",
    )


def test_era_cache_paths_migrate_current_legacy_caches(tmp_path):
    csv_path = tmp_path / "mapbiomas_ms_train.csv"
    csv_path.write_text("longitude,latitude,label\n", encoding="utf-8")
    legacy_rds = csv_path.with_suffix(".rds")
    legacy_raw = csv_path.with_suffix(".raw.rds")
    legacy_rds.write_bytes(b"filtered")
    legacy_raw.write_bytes(b"raw")
    os.utime(csv_path, (1, 1))
    os.utime(legacy_rds, (2, 2))
    os.utime(legacy_raw, (2, 2))

    rds_path, raw_path = era_cache_paths(csv_path, ("NDVI", "EVI"))

    assert rds_path.read_bytes() == b"filtered"
    assert raw_path.read_bytes() == b"raw"
    assert not legacy_rds.exists()
    assert not legacy_raw.exists()


@pytest.mark.parametrize(
    ("bands", "num_trees", "mtry", "rebalance", "expected"),
    [
        (("NDVI", "EVI"), 100, None, None, "2b-nt100"),
        (
            ("NDVI", "EVI", "RED", "NIR", "MIR"),
            500,
            None,
            (400, 400),
            "5b-nt500-rb400x400",
        ),
        (("NDVI", "EVI"), 100, 3, None, "2b-nt100-mtry3"),
    ],
)
def test_variant_slug(bands, num_trees, mtry, rebalance, expected):
    assert variant_slug(bands, num_trees, mtry, rebalance) == expected


def test_kfold_rebalance_guard():
    reject_rebalance_in_kfold(None)

    with pytest.raises(ValueError, match=r"not allowed in k-fold mode.*holdout mode"):
        reject_rebalance_in_kfold((400, 400))
