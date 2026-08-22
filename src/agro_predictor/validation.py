"""R-free validation of classified maps against reference points."""

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform

from agro_predictor import areas


def sample_class_at_points(
    class_tif: Path,
    points_df: pd.DataFrame,
    legend: list[dict],
) -> pd.DataFrame:
    """Sample a classified raster's class at each point; NA where unsampled."""
    code_to_name = {int(entry["code"]): entry["name"] for entry in legend}
    predicted_labels = []

    with rasterio.open(class_tif) as src:
        xs, ys = transform(
            "EPSG:4326",
            src.crs,
            points_df["longitude"].tolist(),
            points_df["latitude"].tolist(),
        )
        nodata = src.nodata if src.nodata is not None else 255
        for x, y in zip(xs, ys, strict=True):
            if not (
                src.bounds.left <= x <= src.bounds.right
                and src.bounds.bottom <= y <= src.bounds.top
            ):
                predicted_labels.append(pd.NA)
                continue

            code = int(next(src.sample([(x, y)]))[0])
            if code == nodata:
                predicted_labels.append(pd.NA)
            else:
                predicted_labels.append(code_to_name.get(code, pd.NA))

    sampled = points_df.copy()
    sampled["predicted_label"] = predicted_labels
    return sampled


def confusion_matrix(
    df: pd.DataFrame,
    reference_col: str = "label",
    predicted_col: str = "predicted_label",
) -> pd.DataFrame:
    """Build a reference x predicted confusion matrix, excluding unsampled rows."""
    sampled = df.loc[df[predicted_col].notna(), [reference_col, predicted_col]]
    classes = sorted(set(sampled[reference_col]) | set(sampled[predicted_col]))
    class_positions = {class_name: position for position, class_name in enumerate(classes)}
    counts = np.zeros((len(classes), len(classes)), dtype=int)
    for reference, predicted in sampled.itertuples(index=False, name=None):
        counts[class_positions[reference], class_positions[predicted]] += 1

    matrix = pd.DataFrame(counts, index=classes, columns=classes)
    matrix.index.name = "reference"
    return matrix


def accuracy_metrics(cm: pd.DataFrame) -> dict:
    """Overall accuracy, Cohen's kappa, and per-class producer/user accuracy."""
    values = cm.to_numpy(dtype=float)
    n = float(values.sum())
    if n == 0:
        overall_accuracy = float("nan")
        expected_agreement = float("nan")
        kappa = float("nan")
    else:
        overall_accuracy = float(np.trace(values) / n)
        row_totals = values.sum(axis=1)
        col_totals = values.sum(axis=0)
        expected_agreement = float(np.dot(row_totals, col_totals) / n**2)
        kappa = (
            float("nan")
            if expected_agreement == 1
            else float((overall_accuracy - expected_agreement) / (1 - expected_agreement))
        )

    row_totals = values.sum(axis=1)
    col_totals = values.sum(axis=0)
    per_class = {}
    for position, class_name in enumerate(cm.index):
        correct = values[position, position]
        per_class[class_name] = {
            "producer_accuracy": (
                float(correct / row_totals[position]) if row_totals[position] else None
            ),
            "user_accuracy": (
                float(correct / col_totals[position]) if col_totals[position] else None
            ),
        }

    return {
        "overall_accuracy": overall_accuracy,
        "kappa": kappa,
        "per_class": per_class,
    }


def validate_map(
    run_dir: Path,
    points_csv: Path,
    state: str | None = "MS",
) -> dict:
    """Validate a classified run against reference points; persist artifacts."""
    run_dir = Path(run_dir)
    points_csv = Path(points_csv)
    legend = areas.load_run_legend(run_dir)
    mosaic_tif = areas.find_or_build_mosaic(run_dir)
    points_df = pd.read_csv(points_csv)
    sampled = sample_class_at_points(mosaic_tif, points_df, legend)

    validation_dir = run_dir / "validation"
    validation_dir.mkdir(parents=True, exist_ok=True)
    sampled.to_csv(validation_dir / "points.csv", index=False)
    cm = confusion_matrix(sampled)
    cm.to_csv(validation_dir / "confusion_matrix.csv")

    metrics = accuracy_metrics(cm)
    n_points = len(sampled)
    n_unsampled = int(sampled["predicted_label"].isna().sum())
    reference = (
        "mapbiomas-c10" if points_csv.name.startswith("mapbiomas") else "user-labels"
    )
    payload = {
        **metrics,
        "n_points": n_points,
        "n_unsampled": n_unsampled,
        "points_csv": str(points_csv),
        "state": state,
        "generated_on": date.today().isoformat(),  # noqa: DTZ011
        "reference": reference,
        "caveat": (
            "reference labels are MapBiomas-derived weak labels, not field truth; "
            "independence holds only if the map was trained without these points"
        ),
    }
    (validation_dir / "accuracy.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Overall accuracy: {metrics['overall_accuracy']:.4f}")
    print(f"Kappa: {metrics['kappa']:.4f}")
    per_class = pd.DataFrame.from_dict(metrics["per_class"], orient="index")
    per_class.index.name = "class"
    print(per_class.to_string())
    return metrics
