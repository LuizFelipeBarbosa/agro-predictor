#!/usr/bin/env bash
# Idempotent environment bootstrap for macOS (Apple Silicon). Safe to re-run.
#
# Installs the Homebrew formula R plus the system libraries its geospatial
# packages compile against, then the R packages and the Python environment.
# Note: formula R compiles every CRAN package from source (~1h first time).
# If you have admin rights, `brew install --cask r` + CRAN's binary packages
# is much faster — but the cask needs a password, so this script stays
# non-interactive and uses the formula.
set -euo pipefail

cd "$(dirname "$0")/.."

echo "==> Checking Xcode Command Line Tools (needed to compile rpy2 and R packages)"
if ! xcode-select -p >/dev/null 2>&1; then
  echo "Missing. Run: xcode-select --install   then re-run this script."
  exit 1
fi

echo "==> Installing R and geospatial system libraries"
brew install r gdal proj geos udunits pkg-config apache-arrow

echo "==> Installing R packages (sits, arrow, sitsdata, ML extras)"
Rscript scripts/install_r_packages.R

echo "==> Syncing Python environment"
uv sync

# The prebuilt rpy2 wheel links against CRAN's R.framework; rebuild the
# rinterface from source so it binds the Homebrew R in fast API mode.
echo "==> Rebuilding rpy2-rinterface against the installed R"
uv pip install --force-reinstall --no-deps --no-binary :all: rpy2-rinterface

echo "==> Done. (BDC data is open access — no token needed.)"
echo "Next: uv run agro-predictor check"
