"""The classification pipeline: cube -> train -> classify -> smooth -> label.

Follows the canonical sits workflow. sits streams the Brasil Data Cube COGs
over HTTP during classification, so there is no separate download step.
"""

import json
import os
import tempfile
import traceback
from collections import Counter
from datetime import date
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
    sits_reduce_imbalance,
    sits_rfor,
    sits_smooth,
    sits_timeline,
    sits_train,
    sits_validate,
)

from agro_predictor import areas
from agro_predictor import validation as validation_metrics
from agro_predictor.config import LABELS_CACHE_DIR, OUTPUT_ROOT, RunConfig
from agro_predictor.experiment import (
    cache_is_current,
    era_cache_paths,
    reject_rebalance_in_kfold,
    variant_slug,
)
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
    if config.local_data_dir is not None:
        return sits_cube(
            source=config.source,
            collection=config.collection,
            data_dir=str(config.local_data_dir),
            start_date=config.start_date,
            end_date=config.end_date,
            bands=list(config.bands),
            parse_info=["satellite", "sensor", "tile", "band", "date"],
        )

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


def load_era_samples(config: RunConfig, cube, csv_path: Path | None = None):
    """Extract training series at CSV points from the run's cube.

    The extraction is the expensive step (one HTTP-streamed series per point),
    so the result is RDS-cached next to the CSV.
    """
    csv_path = Path(config.samples_csv) if csv_path is None else Path(csv_path)
    rds_path, raw_path = era_cache_paths(csv_path, config.bands)
    if cache_is_current(rds_path, csv_path):
        print(f"Era-sample cache hit: {rds_path}")
        return read_rds(str(rds_path))

    if cache_is_current(raw_path, csv_path):
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


def _save_rds(obj, path: Path) -> None:
    import rpy2.robjects as ro

    # Deliberate private-attribute access is contained at the pysits/R boundary.
    ro.r["saveRDS"](obj._instance, str(path))


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


def _make_ml_method(config: RunConfig):
    kwargs = {"num_trees": config.num_trees}
    if config.mtry is not None:
        kwargs["mtry"] = config.mtry
    return sits_rfor(**kwargs)


def _maybe_rebalance(samples, config: RunConfig):
    if config.rebalance is None:
        return samples
    return sits_reduce_imbalance(
        samples,
        n_samples_over=config.rebalance[0],
        n_samples_under=config.rebalance[1],
        multicores=config.multicores,
    )


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
    samples = _maybe_rebalance(samples, config)
    model = sits_train(samples, _make_ml_method(config))
    _save_rds(model, output_dir / "model.rds")

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
                "bands": list(config.bands),
                "num_trees": config.num_trees,
                "mtry": config.mtry,
                "rebalance": list(config.rebalance) if config.rebalance else None,
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

    _finalize(output_dir, config.name, config)
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


def validate(config: RunConfig, holdout_csv: Path | None = None) -> None:
    """Score configured samples with either 5-fold or independent holdout validation."""
    if holdout_csv is not None:
        _validate_holdout(config, Path(holdout_csv))
        return

    reject_rebalance_in_kfold(config.rebalance)
    if config.samples_csv is not None:
        samples = _load_kfold_era_cache(config)
        train_name = Path(config.samples_csv).stem
    else:
        samples = load_run_samples(config)
        train_name = "canned"

    counts = _class_counts(samples)
    excluded = {label: count for label, count in counts.items() if count < 5}
    for label, count in excluded.items():
        print(f"Excluding {label} ({count} samples): fewer samples than folds")
    if excluded:
        included_labels = [label for label in counts if label not in excluded]
        samples = _filter_samples_by_labels(samples, included_labels)

    print("Running 5-fold cross-validation (random forest)...")
    assessment = sits_kfold_validate(
        samples,
        folds=5,
        ml_method=_make_ml_method(config),
        multicores=config.multicores,
    )
    summary = sits_accuracy_summary(assessment)
    print(summary)

    summary_text = str(summary)
    assessment_text = str(assessment)
    if assessment_text.strip() and assessment_text.strip() != summary_text.strip():
        summary_text = f"{summary_text.rstrip()}\n\n{assessment_text.rstrip()}"

    validation_dir = _validation_output_dir(config, train_name)
    validation_dir.mkdir(parents=True, exist_ok=True)
    summary_path = validation_dir / "summary.txt"
    summary_path.write_text(f"{summary_text.rstrip()}\n", encoding="utf-8")
    print(f"Validation summary saved: {summary_path}")


def _load_kfold_era_cache(config: RunConfig):
    csv_path = Path(config.samples_csv)
    rds_path, _ = era_cache_paths(csv_path, config.bands)
    bands = ",".join(config.bands)
    if not rds_path.exists():
        raise RuntimeError(
            f"No band-aware sample cache at {rds_path} for bands {bands}; run "
            f"`agro-predictor run --samples-csv {csv_path} --bands {bands}` once first. "
            "Alternatively, holdout mode can auto-extract train and holdout caches with "
            "`--local-data`."
        )
    if not cache_is_current(rds_path, csv_path):
        raise RuntimeError(
            f"Band-aware sample cache {rds_path} is older than {csv_path}; run "
            f"`agro-predictor run --samples-csv {csv_path} --bands {bands}` again. "
            "Alternatively, holdout mode can refresh train and holdout caches with "
            "`--local-data`."
        )
    return read_rds(str(rds_path))


def _validate_holdout(config: RunConfig, holdout_csv: Path) -> None:
    if config.samples_csv is None:
        raise ValueError("--holdout-csv requires --samples-csv for the training samples")

    train_csv = Path(config.samples_csv)
    if config.local_data_dir is not None:
        cube = build_cube(config)
        train_samples = load_era_samples(config, cube, csv_path=train_csv)
        holdout_samples = load_era_samples(config, cube, csv_path=holdout_csv)
    else:
        train_samples, holdout_samples = _load_holdout_era_caches(
            config, train_csv, holdout_csv
        )

    train_samples = _maybe_rebalance(train_samples, config)
    print("Running independent holdout validation (random forest)...")
    assessment = sits_validate(
        train_samples,
        samples_validation=holdout_samples,
        ml_method=_make_ml_method(config),
    )
    cm = pd.DataFrame(assessment.table).T
    cm.index.name = "reference"
    cm.columns.name = "predicted"
    metrics = validation_metrics.accuracy_metrics(cm)

    caveat = (
        "This is a point-level model score that skips Bayesian smoothing, unlike a full "
        "classify+smooth run."
    )
    payload = {
        **metrics,
        "n_points": int(cm.to_numpy().sum()),
        "n_unsampled": 0,
        "points_csv": str(holdout_csv),
        "state": _config_state(config),
        "generated_on": date.today().isoformat(),  # noqa: DTZ011
        "reference": "mapbiomas-c10",
        "caveat": caveat,
        "bands": list(config.bands),
        "num_trees": config.num_trees,
        "mtry": config.mtry,
        "rebalance": list(config.rebalance) if config.rebalance else None,
    }

    validation_dir = _validation_output_dir(
        config,
        train_csv.stem,
        holdout_name=holdout_csv.stem,
    )
    validation_dir.mkdir(parents=True, exist_ok=True)
    cm.to_csv(validation_dir / "confusion_matrix.csv")
    (validation_dir / "accuracy.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )

    per_class = pd.DataFrame.from_dict(metrics["per_class"], orient="index")
    per_class.index.name = "class"
    summary_text = (
        "Independent holdout validation\n"
        f"Train samples: {train_csv}\n"
        f"Holdout samples: {holdout_csv}\n"
        f"Bands: {', '.join(config.bands)}\n"
        f"Random forest: num_trees={config.num_trees}, mtry={config.mtry}\n"
        f"Rebalance: {config.rebalance}\n"
        f"Overall accuracy: {metrics['overall_accuracy']:.4f}\n"
        f"Kappa: {metrics['kappa']:.4f}\n\n"
        f"{per_class.to_string()}\n\n"
        f"Caveat: {caveat}\n"
    )
    summary_path = validation_dir / "summary.txt"
    summary_path.write_text(summary_text, encoding="utf-8")
    print(f"Overall accuracy: {metrics['overall_accuracy']:.4f}")
    print(f"Kappa: {metrics['kappa']:.4f}")
    print(per_class.to_string())
    print(f"Validation artifacts saved: {validation_dir}")


def _load_holdout_era_caches(config: RunConfig, train_csv: Path, holdout_csv: Path):
    cache_pairs = []
    problems = []
    for csv_path in (train_csv, holdout_csv):
        rds_path, _ = era_cache_paths(csv_path, config.bands)
        cache_pairs.append((csv_path, rds_path))
        if not rds_path.exists():
            problems.append(f"{rds_path} (missing)")
        elif not cache_is_current(rds_path, csv_path):
            problems.append(f"{rds_path} (older than {csv_path.name})")

    if problems:
        bands = ",".join(config.bands)
        state = _config_state(config)
        preset_options = "--preset smoke" if state is None else f"--preset full --state {state}"
        command = (
            "agro-predictor validate "
            f"{preset_options} --start {config.start_date} --end {config.end_date} "
            f"--samples-csv {train_csv} --holdout-csv {holdout_csv} "
            f"--bands {bands} --local-data"
        )
        details = "\n  ".join(problems)
        raise RuntimeError(
            "Missing or stale band-aware sample cache(s):\n"
            f"  {details}\n"
            "Build both from the prefetched local cube with:\n"
            f"  {command}"
        )

    return tuple(read_rds(str(rds_path)) for _, rds_path in cache_pairs)


def _validation_output_dir(
    config: RunConfig,
    train_name: str,
    holdout_name: str | None = None,
) -> Path:
    name = train_name if holdout_name is None else f"{train_name}-vs-{holdout_name}"
    variant = variant_slug(
        config.bands,
        config.num_trees,
        config.mtry,
        config.rebalance,
    )
    return (
        OUTPUT_ROOT
        / "validation"
        / f"{name}-{date.today().isoformat()}-{variant}"  # noqa: DTZ011
    )


def _config_state(config: RunConfig) -> str | None:
    if not isinstance(config.roi, Path):
        return None
    return config.roi.stem.split("_")[0].upper()


def _finalize(output_dir: Path, name: str, config: RunConfig) -> None:
    if isinstance(config.roi, Path):
        state = Path(config.roi).stem.split("_")[0].upper()
        clip = True
    else:
        state = None
        clip = False
    try:
        areas.compute_run_areas(output_dir, state=state, clip=clip)
    except Exception:  # noqa: BLE001 — classification outputs are already safely written
        traceback.print_exc()
        if clip:
            command = f"agro-predictor areas --dir {output_dir} --state {state}"
        else:
            command = f"agro-predictor areas --dir {output_dir} --no-clip"
        print(
            "Areas/preview failed; outputs are safe. Re-run later: "
            f"{command}"
        )
