"""Environment verification for the R -> sits -> pysits -> BDC chain.

Checks run in dependency order so the first FAIL points at the actual root
cause; each failure message includes the remedy.
"""

import shutil
import subprocess
import urllib.request

from agro_predictor.config import BDC_STAC_URL, SMOKE_BBOX


def check_setup() -> bool:
    checks = [
        ("R installation", _check_r),
        ("R packages (sits, sitsdata, arrow)", _check_r_packages),
        ("pysits import (rpy2 bridge)", _check_pysits),
        ("BDC STAC reachable", _check_bdc_stac),
        ("BDC cube creation", _check_cube_build),
        ("labels table (optional)", _check_labels),
    ]
    all_ok = True
    for label, check in checks:
        try:
            detail = check()
            print(f"PASS  {label}" + (f" — {detail}" if detail else ""))
        except Exception as error:  # noqa: BLE001 — every failure must print its remedy
            print(f"FAIL  {label}\n      {error}")
            all_ok = False
    return all_ok


def _check_r() -> str:
    if not shutil.which("Rscript"):
        raise RuntimeError(
            "Rscript not found on PATH. Install R with: brew install --cask r "
            "(or run scripts/bootstrap_macos.sh)"
        )
    version_line = subprocess.run(
        ["Rscript", "--version"], capture_output=True, text=True, check=True
    )
    return (version_line.stdout or version_line.stderr).strip().splitlines()[0]


def _check_r_packages() -> str:
    versions = []
    for package in ("sits", "sitsdata", "arrow"):
        result = subprocess.run(
            ["Rscript", "-e", f'cat(as.character(packageVersion("{package}")))'],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"R package '{package}' is not installed. "
                "Run: Rscript scripts/install_r_packages.R"
            )
        versions.append(f"{package} {result.stdout.strip()}")
    return ", ".join(versions)


def _check_pysits() -> str:
    try:
        import pysits
    except Exception as error:
        raise RuntimeError(
            f"import pysits failed: {error}\n"
            '      Remedies: export R_HOME="$(R RHOME)"; '
            "if R was upgraded, rebuild the bridge with: "
            "uv pip install --force-reinstall --no-cache rpy2"
        ) from error
    return f"pysits {getattr(pysits, '__version__', 'unknown')}"


def _check_bdc_stac() -> str:
    url = f"{BDC_STAC_URL}/collections/mod13q1-6.1"
    request = urllib.request.Request(url, headers={"User-Agent": "agro-predictor"})
    with urllib.request.urlopen(request, timeout=30) as response:
        if response.status != 200:
            raise RuntimeError(f"GET {url} returned HTTP {response.status}")
    return "mod13q1-6.1 collection metadata retrieved (no token needed)"


def _check_cube_build() -> str:
    from pysits import sits_cube

    # One 16-day window over the smoke bbox: proves the whole chain
    # (R, sits, GDAL, BDC STAC) without downloading rasters.
    sits_cube(
        source="BDC",
        collection="MOD13Q1-6.1",
        roi=SMOKE_BBOX,
        start_date="2024-01-01",
        end_date="2024-01-20",
        bands=["NDVI"],
    )
    return "built a small test cube from BDC"


def _check_labels() -> str:
    from agro_predictor import config, labels

    if not config.LABELS_CSV_PATH.exists():
        return "no labels.csv (optional)"

    try:
        labels_table = labels.load_labels()
    except ValueError as error:
        raise RuntimeError(
            f"{error}\n      Remedy: fix the rows above in data/labels/labels.csv"
        ) from error

    distinct_labels = set(labels_table["label"])
    beyond_canned = len(distinct_labels.difference(labels.CANNED_CLASSES))
    return (
        f"{len(labels_table)} points, {len(distinct_labels)} classes "
        f"({beyond_canned} beyond canned set)"
    )
