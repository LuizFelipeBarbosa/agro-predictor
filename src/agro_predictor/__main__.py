"""Command-line interface.

Imports of pysits-dependent modules happen inside each command so that
`check` can diagnose a broken R bridge and pure-Python commands (fetch-roi,
tests) work before R is installed.
"""

import argparse
from dataclasses import replace


def main() -> int:
    args = _parse_args()

    if args.command == "check":
        from agro_predictor.checkup import check_setup

        return 0 if check_setup() else 1

    if args.command == "fetch-roi":
        from agro_predictor.roi import describe_boundary

        describe_boundary()
        return 0

    if args.command == "samples-info":
        from agro_predictor.samples import describe_samples

        describe_samples()
        return 0

    if args.command == "validate":
        from agro_predictor.pipeline import validate

        validate(multicores=args.multicores)
        return 0

    if args.command == "run":
        from agro_predictor import config
        from agro_predictor.pipeline import run

        cfg = config.smoke() if args.preset == "smoke" else config.full_state()
        overrides = {
            "start_date": args.start,
            "end_date": args.end,
            "memsize_gb": args.memsize,
            "multicores": args.multicores,
        }
        cfg = replace(cfg, **{key: value for key, value in overrides.items() if value})
        run(cfg)
        return 0

    raise AssertionError(f"unhandled command: {args.command}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="agro-predictor",
        description="Agriculture classification for Mato Grosso do Sul (sits/pysits + BDC)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("check", help="verify the R/sits/pysits/BDC setup")
    sub.add_parser("fetch-roi", help="download and cache the MS state boundary")
    sub.add_parser("samples-info", help="describe the training sample set")

    validate = sub.add_parser("validate", help="5-fold cross-validation of the model")
    validate.add_argument("--multicores", type=int, default=4)

    run = sub.add_parser("run", help="run the classification pipeline")
    run.add_argument("--preset", choices=("smoke", "full"), default="smoke")
    run.add_argument("--start", help="override start date (YYYY-MM-DD)")
    run.add_argument("--end", help="override end date (YYYY-MM-DD)")
    run.add_argument("--memsize", type=int, help="override memory budget in GB")
    run.add_argument("--multicores", type=int, help="override worker count")

    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
