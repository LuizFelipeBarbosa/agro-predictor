import sys

import pytest

from agro_predictor.__main__ import _parse_args


@pytest.mark.parametrize("command", ["prefetch", "run", "validate"])
def test_bands_cli_option_parses_uppercase_tuple(monkeypatch, command):
    monkeypatch.setattr(
        sys,
        "argv",
        ["agro-predictor", command, "--bands", "ndvi,Evi,red,nir,mir"],
    )

    args = _parse_args()

    assert args.bands == ("NDVI", "EVI", "RED", "NIR", "MIR")


def test_validate_model_options_parse(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "agro-predictor",
            "validate",
            "--num-trees",
            "500",
            "--mtry",
            "3",
            "--rebalance",
            "400,250",
        ],
    )

    args = _parse_args()

    assert args.num_trees == 500
    assert args.mtry == 3
    assert args.rebalance == (400, 250)
