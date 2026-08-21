"""The classification pipeline: cube -> train -> classify -> smooth -> label.

Follows the canonical sits workflow. sits streams the Brasil Data Cube COGs
over HTTP during classification, so there is no separate download step.
"""

import dataclasses
import json
import os
import tempfile
from collections import Counter
from pathlib import Path

import pandas as pd
from pysits import (
    read_rds,
    sits_accuracy_summary,
    sits_classify,
    sits_cube,
    sits_get_class,
    sits_get_data,
    sits_get_probs,
    sits_kfold_validate,
    sits_label_classification,
    sits_labels,
    sits_rfor,
    sits_smooth,
    sits_timeline,
    sits_train,
)

from agro_predictor.config import LABELS_CACHE_DIR, OUTPUT_ROOT, RunConfig
from agro_predictor.labels import (
    canonical_frame,
    extraction_cache_key,
    filter_labels_to_window,
    load_labels,
    merge_review_frames,
)
from agro_predictor.roi import load_roi
from agro_predictor.samples import (
    EXPECTED_TIMELINE_STEPS,
    combine_samples,
    load_training_samples,
)


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


# Fewer steps than this and the series no longer covers the crop cycle well
# enough to classify.
MIN_TIMELINE_STEPS = 18


def align_samples_to_cube(samples, cube):
    """Trim training series to the cube's timeline length.

    Allows classifying a crop year whose final composites are not published
    yet. The random forest's features are positional, so BOTH timelines must
    start on the same composite of the crop calendar (day-of-year); trimming
    drops the same trailing dry-season steps from every training series. A
    window start shifted from the training calendar produces garbage instead
    of an error — keep --start on the Sep 14 composite.
    """
    cube_steps = len(sits_timeline(cube))
    sample_timeline = list(sits_timeline(samples))
    if cube_steps == len(sample_timeline):
        return samples
    if cube_steps > len(sample_timeline):
        raise RuntimeError(
            f"Cube has {cube_steps} composites but the training samples only "
            f"{len(sample_timeline)}; narrow the date window."
        )
    if cube_steps < MIN_TIMELINE_STEPS:
        raise RuntimeError(
            f"Cube has only {cube_steps} composites — too few to cover the crop "
            f"cycle (minimum {MIN_TIMELINE_STEPS}). Widen the date window."
        )
    trimmed = _trim_series_length(samples, cube_steps)
    print(
        f"Cube has {cube_steps} of {len(sample_timeline)} crop-year composites "
        "(recent ones not yet published); trimmed every training series to match."
    )
    return trimmed


def load_era_samples(config: RunConfig, cube):
    """Extract training series at config.samples_csv points from the run's cube.

    The extraction is the expensive step (one HTTP-streamed series per point),
    so the result is RDS-cached next to the CSV.
    """
    csv_path = Path(config.samples_csv)
    rds_path = csv_path.with_suffix(".rds")
    if rds_path.exists() and rds_path.stat().st_mtime >= csv_path.stat().st_mtime:
        print(f"Era-sample cache hit: {rds_path}")
        return read_rds(str(rds_path))

    raw_path = csv_path.with_suffix(".raw.rds")
    if raw_path.exists() and raw_path.stat().st_mtime >= csv_path.stat().st_mtime:
        print(f"Raw extraction cache hit: {raw_path}")
        samples = read_rds(str(raw_path))
    else:
        print(f"Extracting time series for {csv_path.name} (network-bound, be patient)...")
        samples = sits_get_data(
            cube,
            samples=str(csv_path),
            bands=list(config.bands),
            multicores=config.multicores,
        )
        _save_rds(samples, raw_path)

    cube_steps = len(sits_timeline(cube))
    kept = _filter_full_series(samples, cube_steps)
    if len(kept) < len(samples):
        print(f"{len(kept)} of {len(samples)} points yielded full {cube_steps}-step series")
    if len(kept) == 0:
        raise RuntimeError(f"no era samples yielded a full {cube_steps}-step time series")
    _save_rds(kept, rds_path)
    print("Era training counts:", dict(_class_counts(kept)))
    return kept


def _trim_series_length(samples, steps: int):
    """Keep the first `steps` rows of every sample's time series.

    Positional, not date-based: the samples span several crop years, so a
    date filter would drop whole samples instead of trimming each series.
    """
    import rpy2.robjects as ro
    from pysits.models.data.ts import SITSTimeSeriesModel

    trim = ro.r(
        """function(samples, steps) {
             samples$time_series <- lapply(samples$time_series,
                                           function(ts) ts[seq_len(steps), ])
             samples$end_date <- as.Date(vapply(samples$time_series,
               function(ts) as.character(max(ts$Index)), character(1)))
             samples
           }"""
    )
    # Deliberate private-attribute access is contained at the pysits/R boundary.
    return SITSTimeSeriesModel(trim(samples._instance, steps))


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


def _save_rds(samples, path: Path) -> None:
    import rpy2.robjects as ro

    # Deliberate private-attribute access is contained at the pysits/R boundary.
    ro.r["saveRDS"](samples._instance, str(path))


def _filter_full_series(samples, steps: int = EXPECTED_TIMELINE_STEPS):
    import rpy2.robjects as ro
    from pysits.models.data.ts import SITSTimeSeriesModel

    # Deliberate private-attribute access is contained at the pysits/R boundary.
    instance = samples._instance
    series_lengths = ro.r["vapply"](
        instance.rx2("time_series"), ro.r["nrow"], ro.IntVector([0])
    )
    keep = ro.r["=="](series_lengths, steps)
    return SITSTimeSeriesModel(instance.rx(keep, True))


def _class_counts(samples) -> Counter[str]:
    try:
        counts = samples["label"].value_counts()
        if int(counts.sum()) != len(samples):
            raise ValueError("label counts do not cover every sample")
        return Counter({str(label): int(count) for label, count in counts.items()})
    except Exception:  # noqa: BLE001 -- use R as a compatibility fallback for pysits models
        # Deliberate private-attribute access is contained at the pysits/R boundary.
        labels = list(samples._instance.rx2("label"))
        return Counter(str(label) for label in labels)


def _filter_samples_by_labels(samples, labels: list[str]):
    import rpy2.robjects as ro
    from pysits.models.data.ts import SITSTimeSeriesModel

    # Deliberate private-attribute access is contained at the pysits/R boundary.
    instance = samples._instance
    keep = ro.r["%in%"](instance.rx2("label"), ro.StrVector(labels))
    return SITSTimeSeriesModel(instance.rx(keep, True))


def extract_labeled_samples(config: RunConfig):
    """Extract time series for user labels from a dedicated cube; RDS-cached.

    Returns ``None`` when no labels exist or none yield a complete time series.
    """
    labels = load_labels()
    if labels.empty:
        return None

    key = extraction_cache_key(
        labels,
        config.start_date,
        config.end_date,
        config.bands,
        config.collection,
    )
    rds_path = LABELS_CACHE_DIR / f"{key}.rds"
    if rds_path.exists():
        print(f"Labeled-sample cache hit: {rds_path}")
        return read_rds(str(rds_path))

    _configure_gdal_http()
    canonical = canonical_frame(labels)
    bbox = {
        "lon_min": float(canonical["longitude"].min()) - 0.05,
        "lat_min": float(canonical["latitude"].min()) - 0.05,
        "lon_max": float(canonical["longitude"].max()) + 0.05,
        "lat_max": float(canonical["latitude"].max()) + 0.05,
    }
    cube = sits_cube(
        source=config.source,
        collection=config.collection,
        roi=bbox,
        start_date=config.start_date,
        end_date=config.end_date,
        bands=list(config.bands),
    )
    check_timeline(cube)

    LABELS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = LABELS_CACHE_DIR / f"{key}.csv"
    canonical.to_csv(csv_path, index=False)
    samples = sits_get_data(
        cube,
        samples=str(csv_path),
        bands=list(config.bands),
        multicores=config.multicores,
    )

    original_count = len(samples)
    samples = _filter_full_series(samples)
    filtered_count = len(samples)
    if filtered_count < original_count:
        print(
            f"Warning: {filtered_count} of {original_count} labels yielded full series; "
            "dropped points may sit outside cube coverage or use a different date window"
        )
    if filtered_count == 0:
        print("Warning: no labels yielded full series; labeled samples were not cached")
        return None

    _save_rds(samples, rds_path)
    print(
        f"Extracted and cached {filtered_count} labeled samples "
        f"({filtered_count} of {original_count} yielded full series)"
    )
    return samples


def load_run_samples(config: RunConfig):
    """Return canned samples plus user-labeled samples when present and enabled."""
    canned = load_training_samples(config.bands)
    if not config.use_labels:
        return canned

    user = extract_labeled_samples(config)
    if user is None:
        return canned

    merged = combine_samples(canned, user)
    counts = _class_counts(merged)
    warned_labels = set()
    print("Training sample counts:")
    for label, count in counts.items():
        print(f"{label}: {count}")
        if count < 20:
            print(f"Warning: {label} ({count}): too few to influence the map; add more points")
            warned_labels.add(label)
    for label, count in _class_counts(user).items():
        if count < 20 and label not in warned_labels:
            print(
                f"Warning: {label} ({count} user-labeled samples): "
                "too few to influence the map; add more points"
            )
    return merged


def run(config: RunConfig) -> Path:
    _configure_gdal_http()
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Building {config.collection} cube: {config.start_date} .. {config.end_date}")
    cube = build_cube(config)

    print("Loading training samples and training random forest...")
    labels_frame = load_labels()
    if config.samples_csv:
        samples = load_era_samples(config, cube)
    else:
        samples = load_run_samples(config)
        samples = align_samples_to_cube(samples, cube)
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

    labels = [str(label) for label in sits_labels(class_map)]
    (output_dir / "run_labels.json").write_text(
        json.dumps(
            {
                "labels": labels,
                "version": config.version,
                "start_date": config.start_date,
                "end_date": config.end_date,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    if config.use_labels and not labels_frame.empty:
        canonical_frame(labels_frame).to_csv(
            output_dir / "training_labels.csv", index=False
        )
        try:
            _write_review_csv(
                class_map,
                smoothed,
                labels_frame,
                output_dir / "review.csv",
                window=(config.start_date, config.end_date),
            )
        except Exception as error:  # noqa: BLE001 — review is optional after a finished run
            print(f"Review export failed ({error}); run labels review for this run later.")

    _mosaic_and_preview(output_dir, config.name, labels)
    print(f"Done. Outputs in {output_dir}")
    return output_dir


def _write_review_csv(
    class_cube,
    probs_cube,
    labels_df,
    dest: Path,
    window: tuple[str, str] | None = None,
) -> Path:
    if window is not None:
        filtered_labels = filter_labels_to_window(labels_df, window[0], window[1])
        dropped = len(labels_df) - len(filtered_labels)
        labels_df = filtered_labels
        if dropped:
            print(f"{dropped} label(s) outside the run window excluded from review")

    canonical = canonical_frame(labels_df)
    with tempfile.TemporaryDirectory() as scratch:
        samples_path = Path(scratch) / "labels.csv"
        canonical.to_csv(samples_path, index=False)
        predicted = pd.DataFrame(sits_get_class(class_cube, samples=str(samples_path)))
        probabilities = pd.DataFrame(sits_get_probs(probs_cube, samples=str(samples_path)))

    class_df = predicted.loc[:, ["longitude", "latitude", "label"]].copy()
    class_labels = [str(label) for label in sits_labels(probs_cube)]
    probs_df = probabilities.loc[
        :, ["longitude", "latitude", *class_labels]
    ].copy()
    review = merge_review_frames(canonical, class_df, probs_df)
    review.to_csv(dest, index=False)

    matches = int(review["match"].sum())
    total = len(review)
    print(f"review.csv: {matches} of {total} labeled points match the prediction")
    return dest


def export_review(run_name: str, version: str | None = None) -> Path:
    """Rebuild a run's classified outputs and write review.csv for the current labels."""
    output_dir = OUTPUT_ROOT / run_name
    if not output_dir.is_dir():
        raise RuntimeError(f"no such run under output/: {run_name}")

    legend_path = output_dir / "run_labels.json"
    if not legend_path.exists():
        raise RuntimeError(
            "re-run the classification to record the legend (run_labels.json)"
        )

    try:
        legend = json.loads(legend_path.read_text(encoding="utf-8"))
        labels = legend["labels"]
        recorded_version = legend["version"]
        if not isinstance(labels, list) or not all(isinstance(label, str) for label in labels):
            raise ValueError("labels must be a list of strings")
        if not isinstance(recorded_version, str):
            raise TypeError("version must be a string")
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        raise RuntimeError(
            f"invalid run_labels.json ({error}); re-run the classification to record the legend"
        ) from error

    if version is None:
        version = recorded_version
    window = (
        (legend["start_date"], legend["end_date"])
        if "start_date" in legend and "end_date" in legend
        else None
    )

    current_labels = load_labels()
    training_labels_path = output_dir / "training_labels.csv"
    if training_labels_path.exists():
        training_labels = pd.read_csv(training_labels_path, keep_default_na=False)
        if not canonical_frame(training_labels).equals(canonical_frame(current_labels)):
            print("Note: labels.csv has changed since this run was trained")

    label_map = {str(index): label for index, label in enumerate(labels, start=1)}
    with tempfile.TemporaryDirectory() as scratch:
        scratch_dir = Path(scratch)
        class_dir = _symlink_result_tifs(output_dir, scratch_dir, "class")
        try:
            class_cube = sits_cube(
                source="BDC",
                collection="MOD13Q1-6.1",
                data_dir=str(class_dir),
                bands="class",
                labels=label_map,
                version=version,
                multicores=1,
                progress=False,
            )
        except Exception as error:
            raise RuntimeError(
                f"could not rebuild the class cube ({error}); re-run the classification"
            ) from error

        try:
            bayes_dir = _symlink_result_tifs(output_dir, scratch_dir, "bayes")
            probs_cube = sits_cube(
                source="BDC",
                collection="MOD13Q1-6.1",
                data_dir=str(bayes_dir),
                bands="bayes",
                labels=label_map,
                version=version,
                multicores=1,
                progress=False,
            )
            return _write_review_csv(
                class_cube,
                probs_cube,
                current_labels,
                output_dir / "review.csv",
                window=window,
            )
        except Exception as bayes_error:  # noqa: BLE001 — raw probabilities are the fallback
            print(f"Bayes review rebuild failed ({bayes_error}); trying raw probabilities.")

        try:
            probs_dir = _symlink_result_tifs(output_dir, scratch_dir, "probs")
            probs_cube = sits_cube(
                source="BDC",
                collection="MOD13Q1-6.1",
                data_dir=str(probs_dir),
                bands="probs",
                labels=label_map,
                version=version,
                multicores=1,
                progress=False,
            )
            return _write_review_csv(
                class_cube,
                probs_cube,
                current_labels,
                output_dir / "review.csv",
                window=window,
            )
        except Exception as error:
            raise RuntimeError(
                f"could not rebuild review from the saved probability TIFFs ({error}); "
                "re-run the classification"
            ) from error


def _symlink_result_tifs(output_dir: Path, scratch_dir: Path, band: str) -> Path:
    result_tifs = sorted(output_dir.glob(f"TERRA_MODIS_*_{band}_*.tif"))
    if not result_tifs:
        raise RuntimeError(f"no saved {band} TIFFs found in {output_dir}")

    data_dir = scratch_dir / band
    data_dir.mkdir()
    for tif in result_tifs:
        os.symlink(tif, data_dir / tif.name)
    return data_dir


def validate(multicores: int = 4, use_labels: bool = True) -> None:
    """Run 5-fold cross-validation on the available training samples.

    User-labeled samples are included when present and enabled. Classes with
    fewer samples than folds are excluded because they cannot be validated.
    Accuracy on training samples is not map accuracy for the classified region;
    assessing that requires independent local ground truth.
    """
    from agro_predictor.config import smoke

    config = dataclasses.replace(
        smoke(), multicores=multicores, use_labels=use_labels
    )
    samples = load_run_samples(config)
    counts = _class_counts(samples)
    excluded = {label: count for label, count in counts.items() if count < 5}
    for label, count in excluded.items():
        print(f"Excluding {label} ({count} samples): fewer samples than folds")
    if excluded:
        included_labels = [label for label in counts if label not in excluded]
        samples = _filter_samples_by_labels(samples, included_labels)

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
    "Soybean": "#ec7014",
    "Planted_Forest": "#238b45",
    "Grassland": "#d9f0a3",
    "Wetland": "#41b6c4",
    "Water": "#225ea8",
    "Sugarcane": "#dd3497",
}


def _mosaic_and_preview(output_dir: Path, name: str, labels: list[str]) -> None:
    try:
        class_tifs = sorted(output_dir.glob("TERRA_MODIS_*_class_*.tif"))
        if len(class_tifs) == 1:
            mosaic_tif = class_tifs[0]
        else:
            mosaic_tif = output_dir / f"{name}_class_mosaic.tif"
            _merge_tiles(class_tifs, mosaic_tif)
            print(f"Mosaic saved: {mosaic_tif}")
        render_preview(mosaic_tif, labels, output_dir / "classified_preview.png")
    except Exception as error:  # noqa: BLE001 — mosaic/preview are cosmetic, never fail the run
        print(f"Mosaic/preview failed ({error}); open the *_class_*.tif in QGIS instead.")


def _merge_tiles(tifs: list[Path], dest: Path) -> None:
    import rasterio
    from rasterio.merge import merge

    sources = [rasterio.open(tif) for tif in tifs]
    try:
        mosaic, transform = merge(sources)
        meta = sources[0].meta | {
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": transform,
            "compress": "deflate",
        }
        with rasterio.open(dest, "w", **meta) as out:
            out.write(mosaic)
    finally:
        for src in sources:
            src.close()


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
