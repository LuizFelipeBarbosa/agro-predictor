#!/usr/bin/env bash
# Idempotent environment bootstrap for macOS (Apple Silicon).
# Installs R (CRAN binary cask), the R packages sits/arrow/sitsdata, and the
# Python environment. Safe to re-run.
set -euo pipefail

cd "$(dirname "$0")/.."

echo "==> Checking Xcode Command Line Tools (needed to compile rpy2)"
if ! xcode-select -p >/dev/null 2>&1; then
  echo "Missing. Run: xcode-select --install   then re-run this script."
  exit 1
fi

echo "==> Checking R"
if ! command -v Rscript >/dev/null 2>&1; then
  # The cask installs CRAN's binary R, which gets prebuilt package binaries.
  # Do NOT use the 'r' formula: it forces source builds of sf/terra (1h+).
  echo "Installing R from the CRAN binary cask (may ask for your password)..."
  brew install --cask r
fi
Rscript --version

echo "==> Installing R packages (sits, arrow, remotes, sitsdata)"
Rscript scripts/install_r_packages.R

echo "==> Syncing Python environment (builds rpy2 against the installed R)"
uv sync

echo "==> Done. (BDC data is open access — no token needed.)"
echo "Next: uv run agro-predictor check"
