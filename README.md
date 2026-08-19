# agro-predictor

Agricultural land-use classification for the state of **Mato Grosso do Sul**, Brazil,
from satellite image time series served by the [Brasil Data Cube](https://data.inpe.br/bdc)
(INPE), using the [sits](https://github.com/e-sensing/sits) methodology through its
official Python API, [pysits](https://github.com/e-sensing/pysits).

The pipeline classifies a full crop year (Sep–Aug) of MODIS `MOD13Q1-6.1`
NDVI/EVI composites (250 m, 16-day) with a random forest trained on the public
`samples_matogrosso_mod13q1` sample set, then applies Bayesian smoothing and
produces a labeled land-use map as cloud-optimized GeoTIFFs.

## Prerequisites

- macOS (Apple Silicon) with [Homebrew](https://brew.sh) and [uv](https://docs.astral.sh/uv/)
- Xcode Command Line Tools (`xcode-select --install`)

No BDC account or access token is required: the imagery lives on INPE's
open-data portal (<https://data.inpe.br/dados/>) and is served openly. sits
discovers it via the BDC STAC API (`https://data.inpe.br/bdc/stac/v1`) and
streams the cloud-optimized GeoTIFFs during classification — no manual
downloads. (If you do have a `BDC_ACCESS_KEY` in the environment, sits will
use it; it is optional.)

## Setup

```sh
./scripts/bootstrap_macos.sh     # installs R (CRAN cask), sits/arrow/sitsdata, uv sync
uv run agro-predictor check      # verifies the whole chain — must be 5/5 PASS
```

The `check` command verifies, in dependency order: R, the R packages, the
pysits/rpy2 bridge, BDC STAC reachability, and that a small test cube can be
built from BDC.

pysits bridges Python to the R sits package via rpy2, which is why R is a
hard dependency.

## Usage

```sh
uv run agro-predictor fetch-roi           # cache the MS state boundary (IBGE)
uv run agro-predictor samples-info        # inspect the training samples
uv run agro-predictor validate            # 5-fold cross-validation (confusion matrix, kappa)
uv run agro-predictor run --preset smoke  # small bbox in the Dourados soy belt (~10-30 min)
uv run agro-predictor run --preset full   # whole state (several GB streamed, hours)
```

`run` accepts `--start/--end` (crop-year window), `--memsize` (GB) and
`--multicores` overrides.

## Outputs

Each run writes to `output/<run-name>/`:

| File | Content |
|---|---|
| `*_probs_v1.tif` | per-class probability cube |
| `*_bayes_v1.tif` | Bayesian-smoothed probabilities |
| `*_class_v1.tif` | final labeled map (open in QGIS) |
| `classified_preview.png` | quick-look rendering |

Classes: Cerrado, Forest, Pasture, Soy_Corn, Soy_Cotton, Soy_Fallow, Soy_Millet.

## Known limitations

- **Sample transfer**: training samples were collected in Mato Grosso (the
  neighbor state). Land-cover types absent from the training set — Pantanal
  wetlands, open water, sugarcane, planted forest — are forced into the nearest
  available class. Treat the map as a working pipeline baseline, not a
  publishable product.
- **Model vs map accuracy**: `validate` reports cross-validated model accuracy
  on the training samples; true map accuracy for MS would require local ground
  truth.
- Single crop year per run (change dates via `--start/--end`).

## Troubleshooting

- **`import pysits` fails / R_HOME errors** — `export R_HOME="$(R RHOME)"`; if
  R was upgraded, rebuild the bridge:
  `uv pip install --force-reinstall --no-deps --no-binary :all: rpy2-rinterface`.
- **Formula R vs cask R** — the bootstrap uses the Homebrew `r` formula, which
  compiles every CRAN package from source (~1 h first run; sf/terra need the
  gdal/proj/geos/udunits libraries the script installs). If you have admin
  rights, `brew install --cask r` + CRAN's prebuilt binary packages is much
  faster — but never mix the two R installations.
- **pandas 3 incompatibility** — pysits 1.5.4 imports `pandas._typing.Self`,
  removed in pandas 3; this project pins `pandas<3` (already handled in
  pyproject.toml).
- **Timeline mismatch error** — the cube window must contain exactly 23
  MOD13Q1 composites; shift `--start` to the first composite on/after Sep 1
  (e.g. `2023-09-14`).
- **Crash mid-classification** — `--memsize` larger than available RAM kills
  the R session with an opaque rpy2 error; lower it.
- **`sitsdata` install is slow** — it is a large data package fetched from GitHub once.
