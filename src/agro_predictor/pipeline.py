"""The classification pipeline: cube -> train -> classify -> smooth -> label.

Follows the canonical sits workflow. sits streams the Brasil Data Cube COGs
over HTTP during classification, so there is no separate download step.
"""

from pathlib import Path

from pysits import (
    plot,
    sits_accuracy_summary,
    sits_classify,
    sits_cube,
    sits_kfold_validate,
    sits_label_classification,
    sits_rfor,
    sits_smooth,
    sits_timeline,
    sits_train,
)

from agro_predictor.config import RunConfig
from agro_predictor.roi import load_roi
from agro_predictor.samples import EXPECTED_TIMELINE_STEPS, load_training_samples


def build_cube(config: RunConfig):
    return sits_cube(
        source=config.source,
        collection=config.collection,
        roi=load_roi(config.roi),
        start_date=config.start_date,
        end_date=config.end_date,
        bands=list(config.bands),
    )


def check_timeline(cube) -> None:
    steps = len(sits_timeline(cube))
    if steps != EXPECTED_TIMELINE_STEPS:
        raise RuntimeError(
            f"Cube timeline has {steps} instances but the training samples have "
            f"{EXPECTED_TIMELINE_STEPS}. Adjust start/end dates so the window "
            "contains exactly one crop year of MOD13Q1 composites — e.g. shift "
            "--start to the first composite on/after Sep 1 (2023-09-14)."
        )


def run(config: RunConfig) -> Path:
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Building {config.collection} cube: {config.start_date} .. {config.end_date}")
    cube = build_cube(config)
    check_timeline(cube)

    print("Training random forest on sitsdata samples...")
    samples = load_training_samples(config.bands)
    model = sits_train(samples, sits_rfor())

    print(f"Classifying (memsize={config.memsize_gb}GB, multicores={config.multicores})...")
    probs = sits_classify(
        cube,
        ml_model=model,
        roi=load_roi(config.roi),
        output_dir=str(output_dir),
        memsize=config.memsize_gb,
        multicores=config.multicores,
        version=config.version,
    )

    print("Applying Bayesian smoothing...")
    smoothed = sits_smooth(
        probs,
        output_dir=str(output_dir),
        memsize=config.memsize_gb,
        multicores=config.multicores,
        version=config.version,
    )

    print("Labeling final map...")
    class_map = sits_label_classification(
        smoothed,
        output_dir=str(output_dir),
        memsize=config.memsize_gb,
        multicores=config.multicores,
        version=config.version,
    )

    _save_preview(class_map, output_dir / "classified_preview.png")
    print(f"Done. Outputs in {output_dir}")
    return output_dir


def validate(multicores: int = 4) -> None:
    """5-fold cross-validation of the model on the training samples.

    Measures model accuracy on the (Mato Grosso) samples — not map accuracy
    for Mato Grosso do Sul, which would need local ground truth.
    """
    from agro_predictor.config import smoke

    samples = load_training_samples(smoke().bands)
    print("Running 5-fold cross-validation (random forest)...")
    assessment = sits_kfold_validate(
        samples, folds=5, ml_method=sits_rfor(), multicores=multicores
    )
    print(sits_accuracy_summary(assessment))


def _save_preview(class_map, path: Path) -> None:
    try:
        plot(class_map).save(path)
        print(f"Preview saved: {path}")
    except Exception as error:
        print(f"Preview PNG failed ({error}); open the *_class_*.tif in QGIS instead.")
