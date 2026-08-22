import subprocess
import sys
from copy import deepcopy
from dataclasses import replace

import pytest

from agro_predictor import config, prefetch

STAC_ITEM = {
    "id": "MOD13Q1.A2025273.h12v10.061.2025290004041",
    "properties": {
        "datetime": "2025-09-30T00:00:00.000000Z",
        "bdc:tiles": ["012010"],
    },
    "assets": {
        "EVI": {
            "href": "https://data.inpe.br/bdc/data/mod13q1/v6.1/h12/v10/2025/"
            "MOD13Q1.A2025273.h12v10.061.2025290004041/"
            "MOD13Q1.A2025273.h12v10.061.2025290004041_EVI.tif",
            "bdc:size": 50239954,
        },
        "NDVI": {
            "href": "https://data.inpe.br/bdc/data/mod13q1/v6.1/h12/v10/2025/"
            "MOD13Q1.A2025273.h12v10.061.2025290004041/"
            "MOD13Q1.A2025273.h12v10.061.2025290004041_NDVI.tif",
            "bdc:size": 50254212,
        },
        "thumbnail": {
            "href": "https://data.inpe.br/bdc/data/mod13q1/v6.1/h12/v10/2025/"
            "MOD13Q1.A2025273.h12v10.061.2025290004041/"
            "MOD13Q1.A2025273.h12v10.061.2025290004041.png",
            "bdc:size": 1085545,
        },
    },
}


def test_asset_parsing_and_destination_path_are_deterministic(tmp_path):
    assets = prefetch._assets_from_item(STAC_ITEM, ("NDVI", "EVI"))

    assert [asset["band"] for asset in assets] == ["NDVI", "EVI"]
    assert {asset["tile"] for asset in assets} == {"012010"}
    assert {asset["date"] for asset in assets} == {"2025-09-30"}
    assert {asset["size"] for asset in assets} == {50254212, 50239954}
    assert assets[0]["href"].endswith("_NDVI.tif")
    assert assets[1]["href"].endswith("_EVI.tif")
    assert "thumbnail" not in {asset["band"] for asset in assets}

    expected = tmp_path / "TERRA_MODIS_012010_NDVI_2025-09-30.tif"
    assert prefetch._destination_path(assets[0], tmp_path) == expected
    assert prefetch._destination_path(assets[0], tmp_path) == expected


def test_reflectance_asset_keys_keep_single_token_sits_band_filenames(tmp_path):
    item = deepcopy(STAC_ITEM)
    item["assets"].update(
        {
            "red_reflectance": {"href": "https://example.test/red.tif"},
            "NIR_reflectance": {"href": "https://example.test/nir.tif"},
            "MIR_reflectance": {"href": "https://example.test/mir.tif"},
        }
    )

    assets = prefetch._assets_from_item(item, ("RED", "NIR", "MIR"))

    assert prefetch.SITS_BAND_TO_STAC_ASSET["RED"] == "red_reflectance"
    assert [asset["band"] for asset in assets] == ["RED", "NIR", "MIR"]
    assert [asset["href"] for asset in assets] == [
        "https://example.test/red.tif",
        "https://example.test/nir.tif",
        "https://example.test/mir.tif",
    ]
    for asset in assets:
        destination = prefetch._destination_path(asset, tmp_path)
        assert destination.stem.split("_") == [
            "TERRA",
            "MODIS",
            "012010",
            asset["band"],
            "2025-09-30",
        ]


def test_asset_parsing_rejects_unknown_sits_band():
    with pytest.raises(ValueError, match=r"Unknown SITS band 'SWIR'.*known bands"):
        prefetch._assets_from_item(STAC_ITEM, ("SWIR",))


def test_list_run_assets_rejects_band_missing_across_all_pages(monkeypatch):
    cfg = replace(config.smoke(), bands=("NDVI", "RED"))
    pages = []

    def fake_fetch(config, bbox, page):
        pages.append(page)
        return {"features": [STAC_ITEM] if page == 1 else []}

    monkeypatch.setattr(prefetch, "STAC_PAGE_SIZE", 1)
    monkeypatch.setattr(prefetch, "_fetch_stac_page", fake_fetch)

    with pytest.raises(RuntimeError, match=r"No STAC assets matched.*RED"):
        prefetch.list_run_assets(cfg)

    assert pages == [1, 2]


def test_download_orchestration_skips_existing_file_at_expected_size(tmp_path, monkeypatch):
    asset = {
        "tile": "012010",
        "band": "NDVI",
        "date": "2025-09-30",
        "href": "https://example.test/ndvi.tif",
        "size": 4,
    }
    prefetch._destination_path(asset, tmp_path).write_bytes(b"data")
    calls = []

    def unexpected_download(asset, destination):
        calls.append((asset, destination))

    monkeypatch.setattr(prefetch, "_download_asset", unexpected_download)

    prefetch._download_missing_assets([asset], tmp_path)

    assert calls == []


def test_integrity_check_removes_corrupt_file_for_redownload(tmp_path):
    asset = {
        "tile": "012010",
        "band": "NDVI",
        "date": "2025-09-30",
        "href": "https://example.test/ndvi.tif",
        "size": 7,
    }
    destination = prefetch._destination_path(asset, tmp_path)
    destination.write_bytes(b"garbage")

    corrupt = prefetch._remove_corrupt_files([asset], tmp_path)

    assert corrupt == [asset]
    assert not destination.exists()


def test_importing_prefetch_does_not_import_pysits():
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, agro_predictor.prefetch; assert 'pysits' not in sys.modules",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
