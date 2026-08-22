# agro-predictor

Agricultural land-use classification for Brazilian states — primarily
**Mato Grosso do Sul (MS)** — from satellite image time series served by the
[Brasil Data Cube](https://data.inpe.br/bdc) (BDC / INPE), using the
[sits](https://github.com/e-sensing/sits) methodology through its official
Python API, [pysits](https://github.com/e-sensing/pysits).

Given a crop year (September–August), the pipeline builds a data cube of MODIS
`MOD13Q1-6.1` NDVI/EVI composites (250 m, 16-day, 23 per year), trains a random
forest on labeled time series, classifies every pixel, applies Bayesian
smoothing, and produces a labeled land-use map as cloud-optimized GeoTIFFs,
plus per-class area statistics and accuracy reports.

## How it works

### The classification pipeline

`agro-predictor run` executes the canonical sits workflow
(`src/agro_predictor/pipeline.py`):

```
BDC STAC ──► data cube ──► random forest ──► per-pixel probs ──► Bayesian ──► labeled map
 (MOD13Q1)   (NDVI/EVI,      trained on        (sits_classify)    smoothing     (sits_label_
              23 dates)    labeled series                        (sits_smooth)  classification)
                                 ▲
                    training samples (three sources, below)
```

1. **Cube** — `sits_cube` assembles a virtual cube of every MOD13Q1 composite
   intersecting the region of interest (ROI) and date window. Nothing is
   downloaded up front; GDAL streams the cloud-optimized GeoTIFFs over HTTP
   during classification (or reads a prefetched local cache, see
   [Local data cache](#local-data-cache-flaky-networks)).
2. **Train** — a random forest (`sits_rfor`) is trained on labeled NDVI/EVI
   time series. Because the model's features are positional (one per
   composite), the cube and training series must share the same 23-step
   crop-year timeline. If the crop year is still in progress and the last
   composites aren't published yet, every training series is trimmed to match
   (down to a minimum of 18 steps); the window start must stay on the Sep 14
   composite so the calendars align.
3. **Classify** — `sits_classify` produces a per-class probability cube.
4. **Smooth** — `sits_smooth` applies Bayesian smoothing, using each pixel's
   neighborhood to remove salt-and-pepper noise.
5. **Label** — `sits_label_classification` takes the argmax, yielding the
   final class map.
6. **Finalize** — the run directory gets a legend (`run_labels.json`), a
   review CSV comparing predictions against user labels, per-class areas, and
   a rendered preview PNG.

### Training samples: three sources

The model is only as good as its training series. Three sources exist, from
quickest to most accurate:

1. **Canned samples** (default) — the public `samples_matogrosso_mod13q1` set
   from the `sitsdata` R package: ~1.8k points, 7 classes (Cerrado, Forest,
   Pasture, and four soy rotations), collected 2006–2016 in Mato Grosso, the
   neighbor state. Good for a working baseline; see
   [Known limitations](#known-limitations).
2. **User labels** (`data/labels/labels.csv`) — hand-curated ground-truth
   points you add via the `labels` commands. When present they are merged
   into training automatically and can introduce new classes (e.g.
   Sugarcane). See [Ground truth labels](#ground-truth-labels).
3. **Era-calibrated MapBiomas samples** (`make-samples` + `run
   --samples-csv`) — the recommended path for recent crop years. Terra's
   orbital drift makes recent MODIS years look different from the 2006–2016
   canned signatures, so the model should be trained on series from the era
   being classified. `make-samples` derives training points from [MapBiomas
   Collection 10](https://brasil.mapbiomas.org) (Brazil-wide 30 m annual
   land-cover maps on public cloud storage, read by windowed range-requests —
   no full download):
   - a point qualifies when its MapBiomas class is **stable** (identical in
     2023 and 2024) and **homogeneous** (a full 8×8 block of 30 m pixels —
     one 250 m MODIS footprint — is single-class);
   - random windows are sampled across the state with a per-window quota so
     no single region dominates (classes too concentrated to satisfy it fall
     back with a warning);
   - mosaic/ambiguous MapBiomas classes are excluded on purpose, and classes
     with fewer than 30 usable points are dropped.

   The resulting CSV (longitude, latitude, dates, label) is automatically
   split into spatially disjoint train/holdout sets, and `run --samples-csv
   <train.csv>` then extracts each point's actual MODIS series from the run's
   cube and trains **only** on those (replacing sources 1–2). Extraction is
   the expensive, network-bound step, so results are cached as `.rds` files
   next to the CSV.

### Validation: model accuracy vs map accuracy

Two different questions, two different tools:

- **`validate`** — 5-fold cross-validation of the *model* on its own training
  samples (confusion matrix, kappa). This measures how separable the classes
  are, **not** how accurate the map is for the target state.
- **`validate-map`** — samples an already-classified map at independent
  reference points and reports overall accuracy, kappa, and per-class
  producer/user accuracy. For an honest estimate, use the spatially disjoint
  holdout set that `make-samples`/`split-samples` produced and make sure the
  map was trained without those points. The written `accuracy.json` carries a
  caveat: MapBiomas-derived points are weak labels, not field truth.
- **`areas --compare`** — a sanity check at the aggregate level: compares the
  map's per-class area percentages against a published benchmark (SIGA-MS
  land-use tables for MS).

### Module map

| Module (`src/agro_predictor/`) | Role |
|---|---|
| `__main__.py` | CLI; imports pysits lazily so `check` and pure-Python commands work before R is installed |
| `config.py` | Run presets (`smoke`, `full_state`), crop-year window, class registry with display names/colors |
| `pipeline.py` | The sits workflow: cube → train → classify → smooth → label; review CSV export; cross-validation |
| `samples.py` | Loads the canned `sitsdata` samples; merges sample sets |
| `labels.py` | The user label table: validation, dedupe, merging, spatial train/holdout split, extraction cache keys |
| `mapbiomas.py` | Era-calibrated sample generation from MapBiomas COGs |
| `prefetch.py` | Resumable parallel download of BDC COGs into the local cache, with integrity verification |
| `roi.py` | State boundaries from the IBGE malhas API, cached as GeoJSON in `data/roi/` |
| `areas.py` | Mosaic/clip classified tiles, per-class km² and %, preview PNG, benchmark comparison |
| `validation.py` | R-free map validation: raster point sampling, confusion matrix, accuracy metrics |
| `checkup.py` | The `check` command's dependency-ordered health checks |

pysits bridges Python to the R sits package via rpy2, which is why R is a
hard dependency — everything raster/tabular around the core workflow
(`areas.py`, `validation.py`, `mapbiomas.py`, `prefetch.py`) is pure Python
(rasterio/geopandas/pandas).

## Prerequisites

- macOS (Apple Silicon) with [Homebrew](https://brew.sh) and [uv](https://docs.astral.sh/uv/)
- Xcode Command Line Tools (`xcode-select --install`)

No BDC account or access token is required: the imagery lives on INPE's
open-data portal (<https://data.inpe.br/dados/>) and is served openly. sits
discovers it via the BDC STAC API (`https://data.inpe.br/bdc/stac/v1`). If a
`BDC_ACCESS_KEY` is set in the environment, sits uses it; it is optional.

## Setup

```sh
./scripts/bootstrap_macos.sh     # installs R (CRAN cask), sits/arrow/sitsdata, uv sync
uv run agro-predictor check      # verifies the whole chain — must be 5/5 PASS
```

`check` verifies, in dependency order: R, the R packages, the pysits/rpy2
bridge, BDC STAC reachability, and that a small test cube can be built.

## Quick start

```sh
uv run agro-predictor run --preset smoke     # ~55x50 km Dourados soy belt, ~10-30 min
open output/smoke/classified_preview.png
```

Then a full state with era-calibrated training (the recommended recipe):

```sh
uv run agro-predictor make-samples --state MS --start 2025-09-14 --end 2026-08-31
uv run agro-predictor prefetch --preset full --state MS --start 2025-09-14 --end 2026-08-31
uv run agro-predictor run --preset full --state MS --start 2025-09-14 --end 2026-08-31 \
    --local-data --samples-csv data/labels/mapbiomas_ms_2025-09-14_2026-08-31_train.csv
uv run agro-predictor validate-map --dir output/ms-2025-2026 --state MS \
    --points data/labels/mapbiomas_ms_2025-09-14_2026-08-31_holdout.csv
uv run agro-predictor areas --dir output/ms-2025-2026 --state MS --compare siga-ms-2024-25
```

## Command reference

| Command | What it does |
|---|---|
| `check` | Verify the R/sits/pysits/BDC setup (must be 5/5 PASS) |
| `run` | Run the classification pipeline (see options below) |
| `prefetch` | Download a run's BDC imagery into the local cache |
| `make-samples` | Build era-calibrated training points from MapBiomas, auto-split train/holdout |
| `split-samples <csv>` | Spatially disjoint train/holdout split of any sits-schema CSV |
| `validate` | 5-fold cross-validation of the model (confusion matrix, kappa) |
| `validate-map` | Validate a classified run against reference points |
| `areas` | Per-class km²/% for a run, preview PNG, optional benchmark comparison |
| `labels …` | Manage the ground-truth label table (see below) |
| `fetch-roi` | Download and cache a state boundary (IBGE) |
| `samples-info` | Describe the canned training sample set |

`run` options:

- `--preset smoke|full` (default `smoke`) — smoke always uses the Dourados
  bbox; `full` classifies the whole state given by `--state` (default `MS`)
  and names the run `<uf>-<startyear>-<endyear>` under `output/`.
- `--start/--end` — crop-year window (default `2023-09-14`..`2024-08-31`).
  Keep `--start` on a Sep 14 composite: that yields exactly 23 MOD13Q1
  instances and keeps the training calendar aligned.
- `--samples-csv <train.csv>` — train only on era samples extracted at these
  points (from `make-samples`).
- `--no-labels` — ignore `labels.csv`, train on canned samples only.
- `--local-data` — classify from the prefetched cache instead of streaming;
  errors out with the exact `prefetch` command if files are missing.
- `--name`, `--memsize` (GB), `--multicores` — run-name and hardware
  overrides (full-state default: 12 GB, 6 cores).

`validate` accepts `--samples-csv` to cross-validate the cached era samples
(requires the `.rds` extraction cache a prior `run --samples-csv` created),
`--no-labels`, and `--multicores`; the summary is saved under
`output/validation/`.

`split-samples` accepts `--holdout-fraction` (0.2), `--cell-deg` (0.5) and
`--seed` (42): points are grouped into grid cells and whole cells are assigned
to the holdout, so train and holdout are spatially disjoint. A class that is
irreducibly co-located falls back to a point-level split with a loud warning.

## Ground truth labels

The correction loop: add or import labels → run → inspect
`output/<run>/review.csv` (prediction vs. your label at each point,
mismatches first) → correct or extend the labels → re-run.

The git-tracked table is `data/labels/labels.csv`:

| Columns | Content |
|---|---|
| `longitude, latitude, start_date, end_date, label` | sits-native sample fields |
| `source, note, added_on` | provenance |

Omitted dates default to the crop-year window. Labels may use the built-in
classes or add new ones such as `Sugarcane`; new classes appear in the map
legend and receive automatic preview colors (the full registry with display
names and colors is `config.CLASSES`).

```sh
uv run agro-predictor labels add --lon -55.3 --lat -21.5 --label Sugarcane
uv run agro-predictor labels import points.csv
uv run agro-predictor labels list
uv run agro-predictor labels summary
uv run agro-predictor labels extract          # warm the extraction cache (ROI = the labels' own bbox)
uv run agro-predictor labels review --run smoke   # rebuild review.csv for a past run
```

- When `labels.csv` has entries, `run` and `validate` use them automatically
  (unless `--samples-csv` or `--no-labels` is given).
- A class needs on the order of dozens of points to influence the random
  forest; the pipeline warns below 20 samples. Classes with fewer samples
  than the five folds are excluded from `validate`.
- A handful of mismatches in `review.csv` is expected; that is the correction
  loop working.
- `data/labels/cache/` holds extracted time series keyed by a content hash of
  the labels + window + bands. It is gitignored and safe to delete;
  re-extraction is automatic. Similarly, editing an era-sample CSV
  invalidates its `.rds` caches and the next `run`/`validate` re-extracts.

## Outputs

Each run writes to `output/<run-name>/`:

| File | Content |
|---|---|
| `*_probs_v1.tif` | per-class probability cube |
| `*_bayes_v1.tif` | Bayesian-smoothed probabilities |
| `*_class_v1.tif` | final labeled map (open in QGIS) |
| `*_class_mosaic.tif` / `*_clipped.tif` | merged (and boundary-clipped) final map for multi-tile runs |
| `classified_preview.png` | rendered map with legend and per-class percentages |
| `run_labels.json` | map legend + run window; needed by `labels review`, `areas`, `validate-map` |
| `areas.csv` / `areas.json` | per-class pixels, km², percentages |
| `training_labels.csv` | labels table at training start |
| `review.csv` | predicted vs. user label at each labeled point |
| `validation/` | `validate-map` artifacts: sampled points, confusion matrix, `accuracy.json` |

### Areas and benchmarks

`areas` mosaics a run's classified tiles if needed, clips to the state
boundary (skip with `--no-clip`), computes per-class km² and percentages,
regenerates the preview, and checks the clipped total against the IBGE state
area. `--compare <benchmark>` additionally writes `benchmark_delta.csv`
against a published land-use table (currently `siga-ms-2023-24` and
`siga-ms-2024-25`), aggregating model classes into benchmark categories
(e.g. all soy rotations → "soybean"; Cerrado/Forest/Wetland/Grassland →
"native").

```sh
uv run agro-predictor areas --dir output/ms-2025-2026 --state MS --compare siga-ms-2024-25
```

## Local data cache (flaky networks)

On networks where COG streaming is unreliable, download first, then classify
offline:

```sh
uv run agro-predictor prefetch --preset full --state MS
uv run agro-predictor run --preset full --state MS --local-data
```

`prefetch` lists the run's assets via the BDC STAC API and downloads them
with 4 parallel workers into `data/cache/mod13q1/` (shared across runs;
gitignored). Downloads resume via HTTP range requests, sizes are verified
against the STAC metadata, every file is opened with rasterio to catch
corruption, and corrupt files are re-downloaded once before failing. The
streaming path is also hardened: GDAL HTTP retry/timeout defaults are set so
a dropped connection retries instead of stalling the classification (any
`GDAL_HTTP_*` variables you set yourself win).

## Known limitations

- **Sample transfer** (canned samples): collected in Mato Grosso, the
  neighbor state, in 2006–2016. Without local labels or era samples, land
  cover absent from that set — Pantanal wetlands, open water, sugarcane,
  planted forest — is forced into the nearest available class, and Terra's
  orbital drift shifts recent-year signatures. Prefer `make-samples` +
  `--samples-csv` for recent crop years; treat canned-sample maps as a
  pipeline baseline, not a publishable product.
- **Weak reference labels**: MapBiomas-derived points are model output, not
  field truth; `validate-map` accuracy against them is an upper-bound-style
  consistency check, honest only when the holdout was excluded from training.
- Single crop year per run (change dates via `--start/--end`).

## Troubleshooting

- **`import pysits` fails / R_HOME errors** — `export R_HOME="$(R RHOME)"`; if
  R was upgraded, rebuild the bridge:
  `uv pip install --force-reinstall --no-deps --no-binary :all: rpy2-rinterface`.
- **Formula R vs cask R** — the bootstrap uses the Homebrew `r` formula, which
  compiles every CRAN package from source (~1 h first run; sf/terra need the
  gdal/proj/geos/udunits libraries the script installs). With admin rights,
  `brew install --cask r` + CRAN's prebuilt binaries is much faster — but
  never mix the two R installations.
- **pandas 3 incompatibility** — pysits 1.5.4 imports `pandas._typing.Self`,
  removed in pandas 3; this project pins `pandas<3` (already handled in
  pyproject.toml).
- **Timeline mismatch error** — the cube window must contain exactly 23
  MOD13Q1 composites; shift `--start` to the first composite on/after Sep 1
  (e.g. `2023-09-14`). A Sep 1 start pulls in the overlapping Aug 29
  composite and yields 24.
- **Crash mid-classification** — `--memsize` larger than available RAM kills
  the R session with an opaque rpy2 error; lower it.
- **Classification stalls forever** — usually COG streaming on a flaky
  network; use `prefetch` + `--local-data`.
- **`sitsdata` install is slow** — it is a large data package fetched from
  GitHub once.
