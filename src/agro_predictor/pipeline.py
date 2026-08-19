"""The classification pipeline: cube -> train -> classify -> smooth -> label.

Follows the canonical sits workflow. sits streams the Brasil Data Cube COGs
over HTTP during classification, so there is no separate download step.
"""

import os
from pathlib import Path

from pysits import (
    sits_accuracy_summary,
    sits_classify,
    sits_cube,
    sits_kfold_validate,
    sits_label_classification,
    sits_labels,
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


# GDAL streams the BDC imagery over HTTP; without these, a dropped connection
# stalls the classification forever instead of retrying. HTTP/1.1 avoids the
# HTTP/2 PROTOCOL_ERRORs some networks produce. os.environ is inherited by the
# R session and its workers. setdefault keeps user overrides in charge.
GDAL_HTTP_DEFAULTS = {
    "GDAL_HTTP_VERSION": "1.1",
    "GDAL_HTTP_CONNECTTIMEOUT": "30",
    "GDAL_HTTP_TIMEOUT": "120",
    "GDAL_HTTP_MAX_RETRY": "10",
    "GDAL_HTTP_RETRY_DELAY": "5",
}


def _configure_gdal_http() -> None:
    for key, value in GDAL_HTTP_DEFAULTS.items():
        os.environ.setdefault(key, value)


def run(config: RunConfig) -> Path:
    _configure_gdal_http()
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

    _save_preview(class_map, output_dir)
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


# Conventional land-cover colors for the sample-set classes; unknown labels
# fall back to matplotlib's tab10 palette.
PREVIEW_COLORS = {
    "Cerrado": "#a1d99b",
    "Forest": "#00441b",
    "Pasture": "#fee391",
    "Soy_Corn": "#ec7014",
    "Soy_Cotton": "#807dba",
    "Soy_Fallow": "#fdd0a2",
    "Soy_Millet": "#d94801",
}


def _save_preview(class_map, output_dir: Path) -> None:
    try:
        labels = [str(label) for label in sits_labels(class_map)]
        class_tif = max(output_dir.glob("*_class_*.tif"), key=lambda p: p.stat().st_mtime)
        render_preview(class_tif, labels, output_dir / "classified_preview.png")
    except Exception as error:  # noqa: BLE001 — the preview is cosmetic, never fail the run
        print(f"Preview PNG failed ({error}); open the *_class_*.tif in QGIS instead.")


def render_preview(class_tif: Path, labels: list[str], path: Path) -> None:
    """Render a classified GeoTIFF (pixel values 1..len(labels)) to a PNG."""
    import numpy as np
    import rasterio
    from matplotlib import colors as mcolors
    from matplotlib import patches
    from matplotlib import pyplot as plt

    with rasterio.open(class_tif) as src:
        data = src.read(1)

    fallback = plt.get_cmap("tab10")
    hex_colors = [
        PREVIEW_COLORS.get(label, mcolors.to_hex(fallback(i % 10)))
        for i, label in enumerate(labels)
    ]
    masked = np.ma.masked_outside(data, 1, len(labels))

    fig, ax = plt.subplots(figsize=(10, 7), dpi=150)
    ax.imshow(
        masked,
        cmap=mcolors.ListedColormap(hex_colors),
        vmin=1,
        vmax=len(labels),
        interpolation="nearest",
    )
    ax.set_axis_off()
    ax.set_title(class_tif.stem, fontsize=9)
    handles = [
        patches.Patch(color=color, label=label)
        for color, label in zip(hex_colors, labels)
    ]
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Preview saved: {path}")
