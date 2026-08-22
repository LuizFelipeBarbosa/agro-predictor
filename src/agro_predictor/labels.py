"""User-maintained ground-truth table using the sits-native CSV schema plus provenance.

The labels extend training data and support review of model predictions.
"""

import hashlib
import re
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from agro_predictor import config
from agro_predictor.config import CANNED_CLASSES, LABELS_CSV_PATH

LABELS_SCHEMA = ("longitude", "latitude", "start_date", "end_date", "label")
PROVENANCE_COLUMNS = ("source", "note", "added_on")

_ALL_COLUMNS = LABELS_SCHEMA + PROVENANCE_COLUMNS
_LABEL_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


def spatial_train_holdout_split(
    df: pd.DataFrame,
    holdout_fraction: float = 0.2,
    cell_deg: float = 0.5,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split sits-schema points by spatial grid cell into train and holdout sets."""
    longitudes = np.floor(df["longitude"].to_numpy(dtype=float) / cell_deg).astype(int)
    latitudes = np.floor(df["latitude"].to_numpy(dtype=float) / cell_deg).astype(int)
    row_cells = list(zip(longitudes, latitudes, strict=True))

    cell_rows: dict[tuple[int, int], list[int]] = {}
    for row_position, cell in enumerate(row_cells):
        cell_rows.setdefault(cell, []).append(row_position)

    cells = sorted(cell_rows)
    shuffled_cells = [
        cells[position]
        for position in np.random.default_rng(seed).permutation(len(cells))
    ]
    holdout_cells: set[tuple[int, int]] = set()
    holdout_count = 0
    holdout_target = holdout_fraction * len(df)
    for cell in shuffled_cells:
        if holdout_count >= holdout_target:
            break
        holdout_cells.add(cell)
        holdout_count += len(cell_rows[cell])

    labels = df["label"].to_numpy()
    label_counts = df["label"].value_counts().to_dict()
    eligible_labels = sorted(label for label, count in label_counts.items() if count >= 5)
    label_positions = {label: labels == label for label in eligible_labels}
    point_assignments = np.full(len(df), -1, dtype=np.int8)

    def current_holdout_mask(
        cells_in_holdout: set[tuple[int, int]],
    ) -> np.ndarray:
        mask = np.fromiter(
            (cell in cells_in_holdout for cell in row_cells),
            dtype=bool,
            count=len(df),
        )
        overridden = point_assignments >= 0
        mask[overridden] = point_assignments[overridden].astype(bool)
        return mask

    for label in eligible_labels:
        holdout_mask = current_holdout_mask(holdout_cells)
        positions = label_positions[label]
        in_holdout = bool(holdout_mask[positions].any())
        in_train = bool((~holdout_mask[positions]).any())
        if in_holdout and in_train:
            continue

        label_cells = {
            cell
            for cell, positions in cell_rows.items()
            if any(labels[position] == label for position in positions)
        }
        candidate_cells = (
            label_cells - holdout_cells if not in_holdout else label_cells & holdout_cells
        )
        candidate_ranks = []
        for cell in candidate_cells:
            trial_holdout_cells = set(holdout_cells)
            if not in_holdout:
                trial_holdout_cells.add(cell)
            else:
                trial_holdout_cells.remove(cell)
            trial_mask = current_holdout_mask(trial_holdout_cells)
            if not (
                trial_mask[positions].any() and (~trial_mask[positions]).any()
            ):
                continue

            safe = True
            for other_label in eligible_labels:
                if other_label == label:
                    continue
                other_positions = label_positions[other_label]
                other_was_split = holdout_mask[other_positions].any() and (
                    ~holdout_mask[other_positions]
                ).any()
                other_stays_split = trial_mask[other_positions].any() and (
                    ~trial_mask[other_positions]
                ).any()
                if other_was_split and not other_stays_split:
                    safe = False
                    break
            if not safe:
                continue

            class_count = sum(labels[position] == label for position in cell_rows[cell])
            other_count = len(cell_rows[cell]) - class_count
            candidate_ranks.append(((-class_count, other_count, cell), cell))

        if candidate_ranks:
            _, best_cell = min(candidate_ranks)
            if not in_holdout:
                holdout_cells.add(best_cell)
            else:
                holdout_cells.remove(best_cell)
            continue

        row_positions = np.flatnonzero(positions)
        fallback_seed = int.from_bytes(
            hashlib.sha256(f"{seed}:{label}".encode()).digest()[:8], "little"
        )
        shuffled_positions = np.random.default_rng(fallback_seed).permutation(row_positions)
        holdout_size = max(
            1, min(len(row_positions) - 1, round(holdout_fraction * len(row_positions)))
        )
        point_assignments[row_positions] = 0
        point_assignments[shuffled_positions[:holdout_size]] = 1
        print(
            f"spatial independence not achieved for {label}: all points in "
            f"{len(label_cells)} cell(s); using point-level split"
        )

    holdout_mask = current_holdout_mask(holdout_cells)
    for label in sorted(label_counts):
        count = label_counts[label]
        if count < 5:
            print(f"{label}: only {count} points (<5); keeping all in train")
            holdout_mask[labels == label] = False

    train_df = df.loc[~holdout_mask].reset_index(drop=True)
    holdout_df = df.loc[holdout_mask].reset_index(drop=True)
    return train_df, holdout_df


def load_labels(path: Path = LABELS_CSV_PATH) -> pd.DataFrame:
    """Load and validate the user-maintained labels table."""
    if not path.exists():
        return pd.DataFrame(columns=_ALL_COLUMNS)

    labels = pd.read_csv(
        path,
        dtype={
            "longitude": float,
            "latitude": float,
            "start_date": str,
            "end_date": str,
            "label": str,
            "source": str,
            "note": str,
            "added_on": str,
        },
        keep_default_na=False,
    )
    errors = validate_labels(labels)
    if errors:
        raise ValueError("\n".join(errors))
    return labels


def validate_labels(df: pd.DataFrame) -> list[str]:
    """Return human-readable validation errors, using CSV-style line numbers."""
    missing_columns = [column for column in LABELS_SCHEMA if column not in df.columns]
    if missing_columns:
        return [f"Missing required column(s): {', '.join(missing_columns)}"]

    errors = []
    canonical_classes = {label.casefold(): label for label in config.CLASS_NAMES}
    for row_number, (_, row) in enumerate(df.iterrows(), start=2):
        _validate_coordinate(
            row["longitude"],
            "longitude",
            config.BRAZIL_BBOX["lon_min"],
            config.BRAZIL_BBOX["lon_max"],
            row_number,
            errors,
        )
        _validate_coordinate(
            row["latitude"],
            "latitude",
            config.BRAZIL_BBOX["lat_min"],
            config.BRAZIL_BBOX["lat_max"],
            row_number,
            errors,
        )

        parsed_dates = {}
        for column in ("start_date", "end_date"):
            try:
                parsed_dates[column] = date.fromisoformat(row[column])
            except (TypeError, ValueError):
                errors.append(f"CSV line {row_number}: {column} must be a valid ISO date")
        if len(parsed_dates) == 2 and parsed_dates["start_date"] >= parsed_dates["end_date"]:
            errors.append(f"CSV line {row_number}: start_date must be before end_date")

        label = row["label"]
        label_text = label if isinstance(label, str) else ""
        if not label_text:
            errors.append(f"CSV line {row_number}: label must not be empty")
        if label_text == "NoClass":
            errors.append(f"CSV line {row_number}: label must not be the reserved value NoClass")
        if not _LABEL_PATTERN.fullmatch(label_text):
            errors.append(
                f"CSV line {row_number}: label must match ^[A-Za-z][A-Za-z0-9_]*$"
            )

        canonical_label = canonical_classes.get(label_text.casefold())
        if canonical_label is not None and label_text != canonical_label:
            errors.append(
                f"CSV line {row_number}: use canonical label spelling {canonical_label} "
                f"instead of {label_text}"
            )

    return errors


def _validate_coordinate(
    value: object,
    name: str,
    minimum: float,
    maximum: float,
    row_number: int,
    errors: list[str],
) -> None:
    try:
        coordinate = float(value)
    except (TypeError, ValueError):
        errors.append(f"CSV line {row_number}: {name} must be a valid number")
        return

    if not minimum <= coordinate <= maximum:
        errors.append(
            f"CSV line {row_number}: {name} must be between {minimum} and {maximum}"
        )


def add_label(
    longitude: float,
    latitude: float,
    label: str,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
    source: str = "",
    note: str = "",
    path: Path = LABELS_CSV_PATH,
) -> None:
    """Validate and append one label to the ground-truth table."""
    row = pd.DataFrame(
        [
            {
                "longitude": longitude,
                "latitude": latitude,
                "start_date": config.CROP_YEAR_START if start_date is None else start_date,
                "end_date": config.CROP_YEAR_END if end_date is None else end_date,
                "label": label,
                "source": source,
                "note": note,
                "added_on": date.today().isoformat(),  # noqa: DTZ011
            }
        ],
        columns=_ALL_COLUMNS,
    )
    errors = validate_labels(row)
    if errors:
        raise ValueError("\n".join(errors))

    path.parent.mkdir(parents=True, exist_ok=True)
    row.to_csv(path, mode="a" if path.exists() else "w", header=not path.exists(), index=False)


def import_labels(src: Path, path: Path = LABELS_CSV_PATH) -> int:
    """Import labels, replacing existing rows that share the same spatial-period key."""
    incoming = pd.read_csv(src, dtype=str, keep_default_na=False)
    missing_columns = [column for column in LABELS_SCHEMA if column not in incoming.columns]
    if missing_columns:
        raise ValueError(f"Missing required column(s): {', '.join(missing_columns)}")

    for column in PROVENANCE_COLUMNS:
        if column not in incoming.columns:
            incoming[column] = ""
    incoming["added_on"] = date.today().isoformat()  # noqa: DTZ011
    incoming = incoming.loc[:, _ALL_COLUMNS].copy()

    errors = validate_labels(incoming)
    if errors:
        raise ValueError("\n".join(errors))

    incoming["longitude"] = incoming["longitude"].astype(float)
    incoming["latitude"] = incoming["latitude"].astype(float)
    existing = load_labels(path).loc[:, _ALL_COLUMNS]

    rows_by_key = {_dedupe_key(row): row for _, row in existing.iterrows()}
    for _, row in incoming.iterrows():
        rows_by_key[_dedupe_key(row)] = row

    merged = pd.DataFrame(rows_by_key.values(), columns=_ALL_COLUMNS)
    path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(path, index=False)
    return len(incoming)


def _dedupe_key(row: pd.Series) -> tuple[float, float, str, str]:
    return (
        round(float(row["longitude"]), 6),
        round(float(row["latitude"]), 6),
        str(row["start_date"]),
        str(row["end_date"]),
    )


def summarize_labels(path: Path = LABELS_CSV_PATH) -> None:
    """Print a compact summary of label counts."""
    labels = load_labels(path)
    if labels.empty:
        print(f"No labels yet: {path}")
        return

    print(f"Labels: {path}")
    print(f"Total rows: {len(labels)}")
    for label, count in labels["label"].value_counts().items():
        note = " (new class (not in canned training set))" if label not in CANNED_CLASSES else ""
        print(f"{label}: {count}{note}")


def canonical_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return the deterministic, provenance-free frame consumed by extraction."""
    canonical = df.loc[:, LABELS_SCHEMA].copy()
    for column in ("start_date", "end_date"):
        canonical[column] = pd.to_datetime(canonical[column], errors="raise").dt.strftime("%Y-%m-%d")
    return canonical.sort_values(
        ["longitude", "latitude", "start_date", "label"]
    ).reset_index(drop=True)


def filter_labels_to_window(
    df: pd.DataFrame, start_date: str, end_date: str
) -> pd.DataFrame:
    """Return labels whose periods overlap the given window."""
    overlaps = (df["end_date"] >= start_date) & (df["start_date"] <= end_date)
    return df.loc[overlaps].reset_index(drop=True).copy()


def extraction_cache_key(
    df: pd.DataFrame,
    start_date: str,
    end_date: str,
    bands: tuple[str, ...],
    collection: str,
) -> str:
    """Hash canonical labels and extraction settings into a short stable key."""
    canonical_csv = canonical_frame(df).to_csv(index=False).encode("utf-8")
    settings = repr((start_date, end_date, tuple(bands), collection)).encode("utf-8")
    return hashlib.sha256(canonical_csv + settings).hexdigest()[:16]


def merge_review_frames(
    user_df: pd.DataFrame,
    class_df: pd.DataFrame,
    probs_df: pd.DataFrame,
) -> pd.DataFrame:
    """Combine user labels with model predictions and class probabilities."""
    join_columns = ["_join_longitude", "_join_latitude"]

    review = user_df.loc[:, ["longitude", "latitude", "label"]].copy()
    review = review.rename(columns={"label": "user_label"})
    review[join_columns[0]] = review["longitude"].round(6)
    review[join_columns[1]] = review["latitude"].round(6)

    predictions = class_df.loc[:, ["longitude", "latitude", "label"]].copy()
    predictions = predictions.rename(columns={"label": "predicted_label"})
    predictions[join_columns[0]] = predictions["longitude"].round(6)
    predictions[join_columns[1]] = predictions["latitude"].round(6)
    review = review.merge(
        predictions.loc[:, join_columns + ["predicted_label"]],
        how="left",
        on=join_columns,
    )

    class_columns = [
        column for column in probs_df.columns if column not in ("longitude", "latitude")
    ]
    probabilities = probs_df.copy()
    probabilities[join_columns[0]] = probabilities["longitude"].round(6)
    probabilities[join_columns[1]] = probabilities["latitude"].round(6)
    probability_columns = [f"prob_{column}" for column in class_columns]
    probabilities = probabilities.rename(
        columns=dict(zip(class_columns, probability_columns, strict=True))
    )
    review = review.merge(
        probabilities.loc[:, join_columns + probability_columns],
        how="left",
        on=join_columns,
    )

    predicted_labels = review["predicted_label"]
    review["match"] = predicted_labels.notna() & review["user_label"].eq(
        predicted_labels.fillna("")
    )
    review["predicted_label"] = predicted_labels.where(predicted_labels.notna(), pd.NA)
    review[probability_columns] = review[probability_columns].round(4)

    output_columns = [
        "longitude",
        "latitude",
        "user_label",
        "predicted_label",
        "match",
        *probability_columns,
    ]
    return review.loc[:, output_columns].sort_values(
        ["match", "longitude", "latitude"]
    ).reset_index(drop=True)
