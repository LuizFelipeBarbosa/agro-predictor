import re
from datetime import date

import pandas as pd
import pytest

from agro_predictor import config
from agro_predictor.labels import (
    LABELS_SCHEMA,
    PROVENANCE_COLUMNS,
    add_label,
    canonical_frame,
    extraction_cache_key,
    filter_labels_to_window,
    import_labels,
    load_labels,
    merge_review_frames,
    validate_labels,
)


def _valid_frame(**overrides):
    row = {
        "longitude": -54.5,
        "latitude": -20.5,
        "start_date": "2023-09-14",
        "end_date": "2024-08-31",
        "label": "Pasture",
        "source": "field",
        "note": "checked",
        "added_on": "2026-08-19",
    }
    row.update(overrides)
    return pd.DataFrame([row])


def test_validate_labels_accepts_valid_frame():
    assert validate_labels(_valid_frame()) == []


def test_validate_labels_rejects_longitude_outside_brazil():
    errors = validate_labels(_valid_frame(longitude=-80.0))
    assert any("line 2" in error and "longitude" in error for error in errors)


def test_validate_labels_rejects_reversed_dates():
    errors = validate_labels(_valid_frame(start_date="2024-08-31", end_date="2023-09-14"))
    assert any(
        "line 2" in error and "start_date must be before end_date" in error for error in errors
    )


def test_validate_labels_rejects_unparseable_date():
    errors = validate_labels(_valid_frame(start_date="not-a-date"))
    assert any("line 2" in error and "start_date" in error for error in errors)


def test_validate_labels_rejects_empty_label():
    errors = validate_labels(_valid_frame(label=""))
    assert any("line 2" in error and "must not be empty" in error for error in errors)


def test_validate_labels_rejects_reserved_label():
    errors = validate_labels(_valid_frame(label="NoClass"))
    assert any("line 2" in error and "NoClass" in error for error in errors)


def test_validate_labels_rejects_invalid_label_characters():
    errors = validate_labels(_valid_frame(label="Soy Corn"))
    assert any("line 2" in error and "must match" in error for error in errors)


def test_validate_labels_requires_canonical_class_case():
    errors = validate_labels(_valid_frame(label="pasture"))
    assert any("line 2" in error and "Pasture" in error for error in errors)


def test_validate_labels_reports_missing_schema_once():
    frame = _valid_frame().drop(columns="end_date")
    errors = validate_labels(frame)
    assert len(errors) == 1
    assert "end_date" in errors[0]


def test_add_label_creates_file_with_header(tmp_path):
    path = tmp_path / "nested" / "labels.csv"
    add_label(-54.5, -20.5, "Pasture", path=path)

    assert path.read_text().splitlines()[0] == ",".join(LABELS_SCHEMA + PROVENANCE_COLUMNS)
    assert len(load_labels(path)) == 1


def test_add_label_appends_without_duplicate_header(tmp_path):
    path = tmp_path / "labels.csv"
    add_label(-54.5, -20.5, "Pasture", path=path)
    add_label(-54.4, -20.4, "Cerrado", path=path)

    lines = path.read_text().splitlines()
    assert len(lines) == 3
    assert sum(line.startswith("longitude,latitude") for line in lines) == 1


def test_add_label_uses_crop_dates_and_stamps_added_on(tmp_path):
    path = tmp_path / "labels.csv"
    add_label(-54.5, -20.5, "Pasture", path=path)

    row = load_labels(path).iloc[0]
    assert row["start_date"] == config.CROP_YEAR_START
    assert row["end_date"] == config.CROP_YEAR_END
    assert row["added_on"] == date.today().isoformat()  # noqa: DTZ011


def test_add_label_rejects_invalid_coordinate_without_writing(tmp_path):
    path = tmp_path / "labels.csv"
    with pytest.raises(ValueError, match="longitude"):
        add_label(-80.0, -20.5, "Pasture", path=path)
    assert not path.exists()


def test_import_labels_adds_rows_and_fills_provenance(tmp_path):
    src = tmp_path / "incoming.csv"
    path = tmp_path / "stored" / "labels.csv"
    pd.DataFrame(
        [
            {
                "longitude": -54.5,
                "latitude": -20.5,
                "start_date": "2023-09-14",
                "end_date": "2024-08-31",
                "label": "Pasture",
            },
            {
                "longitude": -54.4,
                "latitude": -20.4,
                "start_date": "2023-09-14",
                "end_date": "2024-08-31",
                "label": "Cerrado",
            },
        ]
    ).to_csv(src, index=False)

    assert import_labels(src, path) == 2
    imported = load_labels(path)
    assert len(imported) == 2
    assert imported["source"].tolist() == ["", ""]
    assert imported["note"].tolist() == ["", ""]
    assert imported["added_on"].tolist() == [date.today().isoformat()] * 2  # noqa: DTZ011


def test_import_labels_replaces_existing_dedupe_key(tmp_path):
    path = tmp_path / "labels.csv"
    src = tmp_path / "incoming.csv"
    add_label(-54.5000001, -20.5000001, "Pasture", note="old", path=path)
    _valid_frame(
        longitude=-54.5000002,
        latitude=-20.5000002,
        label="Cerrado",
        note="corrected",
    ).to_csv(src, index=False)

    assert import_labels(src, path) == 1
    imported = load_labels(path)
    assert len(imported) == 1
    assert imported.iloc[0]["label"] == "Cerrado"
    assert imported.iloc[0]["note"] == "corrected"


def test_import_labels_bad_row_reports_source_line(tmp_path):
    src = tmp_path / "incoming.csv"
    path = tmp_path / "labels.csv"
    pd.concat([_valid_frame(), _valid_frame(latitude=-40.0)], ignore_index=True).to_csv(
        src, index=False
    )

    with pytest.raises(ValueError, match="CSV line 3"):
        import_labels(src, path)
    assert not path.exists()


def test_canonical_frame_has_schema_only_and_is_deterministic():
    first = _valid_frame(longitude=-54.4, label="Cerrado")
    second = _valid_frame(longitude=-54.5, label="Pasture")
    forward = pd.concat([first, second], ignore_index=True)
    reverse = pd.concat([second, first], ignore_index=True)

    canonical = canonical_frame(forward)
    assert tuple(canonical.columns) == LABELS_SCHEMA
    assert canonical["longitude"].tolist() == [-54.5, -54.4]
    pd.testing.assert_frame_equal(canonical, canonical_frame(reverse))


def test_filter_labels_to_window_keeps_overlaps_and_touching_boundaries():
    labels = pd.concat(
        [
            _valid_frame(note="overlap", start_date="2024-01-01", end_date="2024-02-01"),
            _valid_frame(note="before", start_date="2022-09-14", end_date="2023-09-13"),
            _valid_frame(note="after", start_date="2024-09-01", end_date="2025-08-31"),
            _valid_frame(note="touches-start", start_date="2022-09-14", end_date="2023-09-14"),
            _valid_frame(note="touches-end", start_date="2024-08-31", end_date="2025-08-31"),
        ],
        ignore_index=True,
    )

    filtered = filter_labels_to_window(labels, "2023-09-14", "2024-08-31")

    assert filtered["note"].tolist() == ["overlap", "touches-start", "touches-end"]
    assert filtered.index.tolist() == [0, 1, 2]


def test_extraction_cache_key_is_stable_and_sensitive():
    first = pd.concat(
        [_valid_frame(longitude=-54.4), _valid_frame(longitude=-54.5)], ignore_index=True
    )
    second = first.iloc[::-1].reset_index(drop=True)
    settings = ("2023-09-14", "2024-08-31", ("NDVI", "EVI"), "MOD13Q1-6.1")

    key = extraction_cache_key(first, *settings)
    assert key == extraction_cache_key(second, *settings)
    assert re.fullmatch(r"[0-9a-f]{16}", key)

    changed_row = first.copy()
    changed_row.loc[0, "label"] = "Cerrado"
    assert extraction_cache_key(changed_row, *settings) != key
    assert extraction_cache_key(first, settings[0], settings[1], ("NDVI",), settings[3]) != key
    assert extraction_cache_key(first, "2023-10-01", settings[1], settings[2], settings[3]) != key


def test_merge_review_frames_combines_sorts_and_preserves_missing_points():
    user = pd.DataFrame(
        [
            [-54.0, -20.0, "2023-09-14", "2024-08-31", "Pasture"],
            [-53.0, -21.0, "2023-09-14", "2024-08-31", "Cerrado"],
            [-55.0, -22.0, "2023-09-14", "2024-08-31", "Forest"],
        ],
        columns=LABELS_SCHEMA,
    )
    classes = pd.DataFrame(
        [
            {"longitude": -54.0000004, "latitude": -20.0000004, "label": "Pasture"},
            {"longitude": -53.0000004, "latitude": -21.0000004, "label": "Pasture"},
        ]
    )
    probabilities = pd.DataFrame(
        [
            {"longitude": -54.0000004, "latitude": -20.0000004, "Pasture": 0.87654, "Cerrado": 0.12346},
            {"longitude": -53.0000004, "latitude": -21.0000004, "Pasture": 0.65432, "Cerrado": 0.34568},
        ]
    )

    review = merge_review_frames(user, classes, probabilities)

    assert review.columns.tolist() == [
        "longitude",
        "latitude",
        "user_label",
        "predicted_label",
        "match",
        "prob_Pasture",
        "prob_Cerrado",
    ]
    assert review["match"].tolist() == [False, False, True]

    mismatch = review.loc[review["longitude"] == -53.0].iloc[0]
    assert mismatch["user_label"] == "Cerrado"
    assert mismatch["predicted_label"] == "Pasture"
    assert mismatch["prob_Pasture"] == 0.6543

    missing = review.loc[review["longitude"] == -55.0].iloc[0]
    assert pd.isna(missing["predicted_label"])
    assert not missing["match"]
    assert pd.isna(missing["prob_Pasture"])
