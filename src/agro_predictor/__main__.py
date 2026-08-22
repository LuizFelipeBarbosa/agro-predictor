"""Command-line interface.

Imports of pysits-dependent modules happen inside each command so that
`check` can diagnose a broken R bridge and pure-Python commands (fetch-roi,
tests) work before R is installed.
"""

import argparse
from dataclasses import replace
from pathlib import Path


def main() -> int:
    args = _parse_args()

    if args.command == "check":
        from agro_predictor.checkup import check_setup

        return 0 if check_setup() else 1

    if args.command == "labels":
        return _labels_command(args)

    if args.command == "fetch-roi":
        from agro_predictor.roi import describe_boundary

        describe_boundary(args.state)
        return 0

    if args.command == "samples-info":
        from agro_predictor.samples import describe_samples

        describe_samples()
        return 0

    if args.command == "split-samples":
        import pandas as pd

        frame = pd.read_csv(args.csv_path)
        _write_sample_split(
            frame,
            args.csv_path,
            holdout_fraction=args.holdout_fraction,
            cell_deg=args.cell_deg,
            seed=args.seed,
        )
        return 0

    if args.command == "validate":
        from agro_predictor import config
        from agro_predictor.pipeline import validate

        if args.preset == "smoke":
            cfg = config.smoke()
        else:
            cfg = config.full_state(
                args.state,
                start_date=args.start or config.CROP_YEAR_START,
                end_date=args.end or config.CROP_YEAR_END,
            )
        overrides = {
            "start_date": args.start,
            "end_date": args.end,
            "bands": args.bands,
            "multicores": args.multicores,
            "num_trees": args.num_trees,
            "mtry": args.mtry,
            "rebalance": args.rebalance,
        }
        cfg = replace(cfg, **{key: value for key, value in overrides.items() if value})
        cfg = replace(cfg, use_labels=not args.no_labels)
        if args.samples_csv:
            cfg = replace(cfg, samples_csv=Path(args.samples_csv))

        if args.local_data:
            from agro_predictor import prefetch

            assets = prefetch.list_run_assets(cfg)
            missing = [
                prefetch._destination_path(asset, prefetch.DEFAULT_CACHE_DIR)
                for asset in assets
                if not prefetch._destination_path(asset, prefetch.DEFAULT_CACHE_DIR).exists()
            ]
            if missing:
                command = (
                    "agro-predictor prefetch "
                    f"--preset {args.preset} --state {args.state} "
                    f"--start {cfg.start_date} --end {cfg.end_date} "
                    f"--bands {','.join(cfg.bands)}"
                )
                print(
                    f"Local cache is missing {len(missing)} of {len(assets)} "
                    "required files. Run this first:\n"
                    f"  {command}"
                )
                return 1
            cfg = replace(cfg, local_data_dir=prefetch.DEFAULT_CACHE_DIR)

        validate(cfg, holdout_csv=args.holdout_csv)
        return 0

    if args.command == "make-samples":
        from agro_predictor import config
        from agro_predictor.mapbiomas import sample_state

        start = args.start or config.CROP_YEAR_START
        end = args.end or config.CROP_YEAR_END
        frame = sample_state(args.state, start, end, per_class=args.per_class)
        if frame.empty:
            print("No usable samples found; try more windows or a larger per-class quota.")
            return 1
        dest = config.LABELS_DIR / f"mapbiomas_{args.state.lower()}_{start}_{end}.csv"
        dest.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(dest, index=False)
        print(f"Wrote {len(frame)} samples: {dest}")
        train_path, _ = _write_sample_split(frame, dest)
        print(
            f"Suggested: uv run agro-predictor run --preset full --state {args.state} "
            f"--samples-csv {train_path}"
        )
        return 0

    if args.command == "areas":
        from agro_predictor import areas, config

        if args.dir is not None:
            run_dir = args.dir
        elif args.preset == "smoke":
            run_dir = config.smoke().output_dir
        else:
            run_dir = config.full_state(args.state).output_dir
        areas.compute_run_areas(
            run_dir,
            state=args.state,
            clip=not args.no_clip,
            benchmark=args.compare,
        )
        return 0

    if args.command == "validate-map":
        from agro_predictor import validation

        validation.validate_map(
            run_dir=Path(args.dir),
            points_csv=Path(args.points),
            state=args.state,
        )
        return 0

    if args.command == "prefetch":
        from agro_predictor import config, prefetch

        if args.preset == "smoke":
            cfg = config.smoke()
        else:
            cfg = config.full_state(
                args.state,
                start_date=args.start or config.CROP_YEAR_START,
                end_date=args.end or config.CROP_YEAR_END,
            )
        date_overrides = {
            "start_date": args.start,
            "end_date": args.end,
            "bands": args.bands,
        }
        cfg = replace(
            cfg,
            **{key: value for key, value in date_overrides.items() if value},
        )
        cache_dir = prefetch.prefetch_run(cfg)
        tif_count = sum(1 for _ in cache_dir.glob("*.tif"))
        print(f"Cache directory: {cache_dir}")
        print(f"GeoTIFF files: {tif_count}")
        return 0

    if args.command == "run":
        from agro_predictor import config

        if args.preset == "smoke":
            cfg = config.smoke()
        else:
            cfg = config.full_state(
                args.state,
                start_date=args.start or config.CROP_YEAR_START,
                end_date=args.end or config.CROP_YEAR_END,
            )
        overrides = {
            "name": args.name,
            "start_date": args.start,
            "end_date": args.end,
            "memsize_gb": args.memsize,
            "multicores": args.multicores,
            "bands": args.bands,
            "num_trees": args.num_trees,
            "mtry": args.mtry,
            "rebalance": args.rebalance,
        }
        cfg = replace(cfg, **{key: value for key, value in overrides.items() if value})
        cfg = replace(cfg, use_labels=not args.no_labels)
        if args.samples_csv:
            cfg = replace(cfg, samples_csv=Path(args.samples_csv))

        if args.local_data:
            from agro_predictor import prefetch

            assets = prefetch.list_run_assets(cfg)
            missing = [
                prefetch._destination_path(asset, prefetch.DEFAULT_CACHE_DIR)
                for asset in assets
                if not prefetch._destination_path(asset, prefetch.DEFAULT_CACHE_DIR).exists()
            ]
            if missing:
                command = (
                    "agro-predictor prefetch "
                    f"--preset {args.preset} --state {args.state} "
                    f"--start {cfg.start_date} --end {cfg.end_date} "
                    f"--bands {','.join(cfg.bands)}"
                )
                print(
                    f"Local cache is missing {len(missing)} of {len(assets)} "
                    "required files. Run this first:\n"
                    f"  {command}"
                )
                return 1
            cfg = replace(cfg, local_data_dir=prefetch.DEFAULT_CACHE_DIR)

        from agro_predictor.pipeline import run

        run(cfg)
        return 0

    raise AssertionError(f"unhandled command: {args.command}")


def _write_sample_split(
    frame,
    csv_path: Path,
    *,
    holdout_fraction: float = 0.2,
    cell_deg: float = 0.5,
    seed: int = 42,
) -> tuple[Path, Path]:
    from agro_predictor import labels

    train, holdout = labels.spatial_train_holdout_split(
        frame,
        holdout_fraction=holdout_fraction,
        cell_deg=cell_deg,
        seed=seed,
    )
    train_path = csv_path.with_name(f"{csv_path.stem}_train.csv")
    holdout_path = csv_path.with_name(f"{csv_path.stem}_holdout.csv")
    train.to_csv(train_path, index=False)
    holdout.to_csv(holdout_path, index=False)

    counts = train["label"].value_counts().rename("train").to_frame()
    counts = counts.join(
        holdout["label"].value_counts().rename("holdout"),
        how="outer",
    ).fillna(0).astype(int).sort_index()
    counts.index.name = "label"
    print(counts.to_string())
    print(f"Train: {len(train)} samples -> {train_path}")
    print(f"Holdout: {len(holdout)} samples -> {holdout_path}")
    return train_path, holdout_path


def _labels_command(args: argparse.Namespace) -> int:
    try:
        if args.labels_command == "add":
            from agro_predictor.labels import add_label, load_labels

            add_label(
                args.lon,
                args.lat,
                args.label,
                start_date=args.start,
                end_date=args.end,
                source=args.source,
                note=args.note,
                path=args.file,
            )
            print(f"Added 1 label. Total rows: {len(load_labels(args.file))}")
            return 0

        elif args.labels_command == "import":
            from agro_predictor.labels import import_labels, load_labels

            imported = import_labels(args.csv_path, args.file)
            total = len(load_labels(args.file))
            print(f"Imported/merged {imported} rows. Total rows: {total}")
            return 0

        elif args.labels_command == "list":
            from agro_predictor.labels import load_labels

            labels = load_labels(args.file)
            if labels.empty:
                print(f"No labels yet: {args.file}")
            elif args.label:
                labels = labels[labels["label"] == args.label]
                if labels.empty:
                    print(f"No labels matching --label {args.label}")
                else:
                    print(labels.to_string(index=False))
            else:
                print(labels.to_string(index=False))
            return 0

        elif args.labels_command == "summary":
            from agro_predictor.labels import summarize_labels

            summarize_labels(args.file)
            return 0

        elif args.labels_command == "extract":
            from agro_predictor import config
            from agro_predictor.pipeline import extract_labeled_samples

            overrides = {
                "start_date": args.start,
                "end_date": args.end,
                "multicores": args.multicores,
            }
            cfg = replace(
                config.smoke(),
                **{key: value for key, value in overrides.items() if value},
            )
            if extract_labeled_samples(cfg) is None:
                print("No labels to extract")
            return 0

        elif args.labels_command == "review":
            from agro_predictor.pipeline import export_review

            print(export_review(args.run, version=args.version))
            return 0

        raise AssertionError(f"unhandled labels command: {args.labels_command}")
    except (ValueError, RuntimeError) as error:
        print(str(error))
        return 1


def _parse_bands(value: str) -> tuple[str, ...]:
    return tuple(band.strip().upper() for band in value.split(",") if band.strip())


def _parse_rebalance(value: str) -> tuple[int, int]:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("rebalance must be OVER,UNDER (for example 400,400)")
    try:
        return int(parts[0]), int(parts[1])
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "rebalance must be OVER,UNDER using integers (for example 400,400)"
        ) from error


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="agro-predictor",
        description="Agriculture classification for Mato Grosso do Sul (sits/pysits + BDC)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("check", help="verify the R/sits/pysits/BDC setup")

    from agro_predictor import areas as areas_module
    from agro_predictor import config

    labels_parser = sub.add_parser("labels", help="manage user-maintained labels")
    labels_sub = labels_parser.add_subparsers(dest="labels_command", required=True)

    labels_add = labels_sub.add_parser("add", help="add one label")
    labels_add.add_argument("--lon", type=float, required=True)
    labels_add.add_argument("--lat", type=float, required=True)
    labels_add.add_argument("--label", required=True)
    labels_add.add_argument("--start")
    labels_add.add_argument("--end")
    labels_add.add_argument("--source", default="")
    labels_add.add_argument("--note", default="")
    labels_add.add_argument("--file", type=Path, default=config.LABELS_CSV_PATH)

    labels_import = labels_sub.add_parser("import", help="import labels from CSV")
    labels_import.add_argument("csv_path", type=Path)
    labels_import.add_argument("--file", type=Path, default=config.LABELS_CSV_PATH)

    labels_list = labels_sub.add_parser("list", help="list labels")
    labels_list.add_argument("--label")
    labels_list.add_argument("--file", type=Path, default=config.LABELS_CSV_PATH)

    labels_summary = labels_sub.add_parser("summary", help="summarize labels")
    labels_summary.add_argument("--file", type=Path, default=config.LABELS_CSV_PATH)

    labels_extract = labels_sub.add_parser(
        "extract",
        help="extract time series for labels (warms the cache; ROI comes from the labels)",
    )
    labels_extract.add_argument("--start", help="override start date (YYYY-MM-DD)")
    labels_extract.add_argument("--end", help="override end date (YYYY-MM-DD)")
    labels_extract.add_argument("--multicores", type=int, help="override worker count")

    labels_review = labels_sub.add_parser("review", help="export review CSV for a past run")
    labels_review.add_argument("--run", required=True)
    labels_review.add_argument(
        "--version",
        default=None,
        help="cube version (defaults to the one recorded in run_labels.json)",
    )

    fetch_roi = sub.add_parser("fetch-roi", help="download and cache a state boundary")
    fetch_roi.add_argument("--state", default="MS", help="state code (UF), e.g. MS, GO")
    sub.add_parser("samples-info", help="describe the training sample set")

    validate = sub.add_parser("validate", help="score the model with k-fold or holdout data")
    validate.add_argument("--preset", choices=("smoke", "full"), default="smoke")
    validate.add_argument("--state", default="MS", help="state code (UF) for --preset full")
    validate.add_argument("--start", help="override start date (YYYY-MM-DD)")
    validate.add_argument("--end", help="override end date (YYYY-MM-DD)")
    validate.add_argument("--bands", type=_parse_bands, help="comma-separated SITS band ids")
    validate.add_argument("--multicores", type=int, default=4)
    validate.add_argument("--num-trees", type=int, help="random forest tree count")
    validate.add_argument("--mtry", type=int, help="random forest variables per split")
    validate.add_argument(
        "--rebalance",
        type=_parse_rebalance,
        help="oversampling and undersampling targets as OVER,UNDER",
    )
    validate.add_argument(
        "--no-labels", action="store_true", help="train on canned samples only"
    )
    validate.add_argument(
        "--samples-csv",
        help="validate cached era samples from this CSV instead of canned samples",
    )
    validate.add_argument("--holdout-csv", type=Path, help="independent holdout samples CSV")
    validate.add_argument(
        "--local-data",
        action="store_true",
        help="extract missing holdout caches from the prefetched local BDC cache",
    )

    split_samples = sub.add_parser(
        "split-samples",
        help="split a sits-schema CSV into spatially disjoint train and holdout sets",
    )
    split_samples.add_argument("csv_path", type=Path, metavar="CSV_PATH")
    split_samples.add_argument("--holdout-fraction", type=float, default=0.2)
    split_samples.add_argument("--cell-deg", type=float, default=0.5)
    split_samples.add_argument("--seed", type=int, default=42)

    make_samples = sub.add_parser(
        "make-samples",
        help="build era-calibrated training points from MapBiomas",
    )
    make_samples.add_argument("--state", default="MS", help="state code (UF)")
    make_samples.add_argument("--start", help="crop-year start (YYYY-MM-DD)")
    make_samples.add_argument("--end", help="crop-year end (YYYY-MM-DD)")
    make_samples.add_argument("--per-class", type=int, default=200)

    areas = sub.add_parser(
        "areas",
        help="compute class areas and regenerate the classified preview",
    )
    areas_source = areas.add_mutually_exclusive_group(required=True)
    areas_source.add_argument("--dir", type=Path, help="existing run directory")
    areas_source.add_argument("--preset", choices=("smoke", "full"))
    areas.add_argument("--state", default="MS", help="state code (UF)")
    areas.add_argument(
        "--no-clip",
        action="store_true",
        help="compute from the un-clipped mosaic",
    )
    areas.add_argument(
        "--compare",
        choices=list(areas_module.BENCHMARKS),
        help="compare the modeled percentages with a named benchmark",
    )

    validate_map = sub.add_parser(
        "validate-map",
        help="validate a classified run against reference points",
    )
    validate_map.add_argument("--dir", required=True, help="existing run directory")
    validate_map.add_argument("--points", required=True, help="reference points CSV")
    validate_map.add_argument("--state", default="MS", help="state code (UF)")

    prefetch = sub.add_parser("prefetch", help="download BDC imagery into the local cache")
    prefetch.add_argument("--preset", choices=("smoke", "full"), default="smoke")
    prefetch.add_argument("--state", default="MS", help="state code (UF) for --preset full")
    prefetch.add_argument("--start", help="override start date (YYYY-MM-DD)")
    prefetch.add_argument("--end", help="override end date (YYYY-MM-DD)")
    prefetch.add_argument("--bands", type=_parse_bands, help="comma-separated SITS band ids")

    run = sub.add_parser("run", help="run the classification pipeline")
    run.add_argument("--preset", choices=("smoke", "full"), default="smoke")
    run.add_argument("--state", default="MS", help="state code (UF) for --preset full")
    run.add_argument(
        "--samples-csv",
        help="train ONLY on series extracted at these points (see make-samples)",
    )
    run.add_argument("--name", help="override the run name (and output/ subdirectory)")
    run.add_argument("--start", help="override start date (YYYY-MM-DD)")
    run.add_argument("--end", help="override end date (YYYY-MM-DD)")
    run.add_argument("--memsize", type=int, help="override memory budget in GB")
    run.add_argument("--multicores", type=int, help="override worker count")
    run.add_argument("--bands", type=_parse_bands, help="comma-separated SITS band ids")
    run.add_argument("--num-trees", type=int, help="random forest tree count")
    run.add_argument("--mtry", type=int, help="random forest variables per split")
    run.add_argument(
        "--rebalance",
        type=_parse_rebalance,
        help="oversampling and undersampling targets as OVER,UNDER",
    )
    run.add_argument("--no-labels", action="store_true", help="train on canned samples only")
    run.add_argument(
        "--local-data",
        action="store_true",
        help="classify from the prefetched local BDC cache",
    )

    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
