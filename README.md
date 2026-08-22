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

On networks where COG streaming is unreliable, run
`agro-predictor prefetch --preset full --state MS` to download the required
files into the shared local cache, then use
`agro-predictor run --preset full --state MS --local-data` to classify from
that cache instead of streaming over HTTP.

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
uv run agro-predictor fetch-roi --state GO           # cache a state boundary (IBGE)
uv run agro-predictor samples-info                    # inspect the training samples
uv run agro-predictor validate                        # 5-fold cross-validation (confusion matrix, kappa)
uv run agro-predictor run --preset smoke              # small bbox in the Dourados soy belt (~10-30 min)
uv run agro-predictor run --preset full --state GO    # whole state (several GB streamed, hours)
```

`run` accepts `--start/--end` (crop-year window), `--memsize` (GB) and
`--multicores` overrides. `--state` defaults to `MS` for `fetch-roi` and the
`full` preset; the `smoke` preset always uses the Dourados-area bbox.

## Ground truth labels

- Add or import labels. They merge into training automatically.
- Run the pipeline, then inspect `output/<run>/review.csv` for the prediction
  and your label at each point, with mismatches first.
- Correct or extend the labels and re-run.

The git-tracked table is `data/labels/labels.csv`:

| Columns | Content |
|---|---|
| `longitude, latitude, start_date, end_date, label` | sits-native sample fields |
| `source, note, added_on` | provenance |

Omitted dates default to the crop-year window. Labels may use the seven canned
classes or add new classes such as `Sugarcane`; new classes appear in the map
legend and receive automatic preview colors.

```sh
uv run agro-predictor labels add --lon -55.3 --lat -21.5 --label Sugarcane
uv run agro-predictor labels import points.csv
uv run agro-predictor labels list
uv run agro-predictor labels summary
uv run agro-predictor labels extract          # warm the extraction cache (ROI = the labels' own bbox)
uv run agro-predictor labels review --run smoke
```

- When `labels.csv` has entries, `run` and `validate` use them automatically;
  `--no-labels` opts out.
- A class needs on the order of dozens of points to influence the random forest;
  the pipeline warns below 20 samples.
- A handful of points will appear as mismatches in `review.csv`; that is the
  correction loop working.
- Classes with fewer samples than the five folds are excluded from `validate`.
- `data/labels/cache/` holds extracted time series keyed by content hash. It is
  gitignored and safe to delete; re-extraction is automatic.

## Outputs

Each run writes to `output/<run-name>/`:

| File | Content |
|---|---|
| `*_probs_v1.tif` | per-class probability cube |
| `*_bayes_v1.tif` | Bayesian-smoothed probabilities |
| `*_class_v1.tif` | final labeled map (open in QGIS) |
| `*_class_mosaic.tif` | merged final map for multi-tile runs |
| `classified_preview.png` | quick-look rendering |
| `run_labels.json` | map legend, needed by `labels review` |
| `training_labels.csv` | labels table at training start (includes points later dropped for incomplete series) |
| `review.csv` | predicted vs. user label at each labeled point |

Classes: see `config.CLASSES` for the full registry (13 built-in classes with
display names and colors); user labels can extend this list.

### Areas and percentages

The `areas` command clips an existing run's mosaic to the state boundary,
computes per-class km² and percentages, writes `areas.csv` and `areas.json`,
and regenerates `classified_preview.png` with display names and percentages.
It can also compare the results with a named benchmark:

```sh
uv run agro-predictor areas --dir output/ms-2025-2026 --state MS --compare siga-ms-2023-24
```

### Spatial holdout validation

Run `split-samples` on the era-sample CSV, then retrain with `run --samples-csv ..._train.csv`.
Validate the retrained map with `validate-map --points ..._holdout.csv`.
The `accuracy.json` caveat notes these are MapBiomas-derived weak labels, not field truth;
independence holds only when retraining excluded the holdout points.
An irreducibly co-located class falls back to a point-level split with a loud warning.
Regenerating samples with the diversity-capped `make-samples` sampler is recommended for better
spatial coverage; because the CSV changes, extraction caches are invalidated and the next
`run`/`validate` re-extracts instead of reusing a cached RDS.

## Known limitations

- **Sample transfer**: the canned samples were collected in Mato Grosso (the
  neighbor state). Without local labels, types absent from that set — Pantanal
  wetlands, open water, sugarcane, planted forest — are forced into the nearest
  available class. Add missing classes through [Ground truth labels](#ground-truth-labels).
  Treat the map as a working pipeline baseline, not a publishable product.
- **Model vs map accuracy**: `validate` reports cross-validated model accuracy
  on the training samples; true map accuracy for the target state requires local
  ground truth.
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
