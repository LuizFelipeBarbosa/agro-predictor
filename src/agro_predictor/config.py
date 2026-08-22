"""Run configuration: what to classify, where, and with how much hardware."""

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT_ROOT / "output"
ROI_DIR = PROJECT_ROOT / "data" / "roi"
LABELS_DIR = PROJECT_ROOT / "data" / "labels"
LABELS_CSV_PATH = LABELS_DIR / "labels.csv"
LABELS_CACHE_DIR = LABELS_DIR / "cache"

# BDC data on data.inpe.br is open access (verified 2026-08: anonymous range
# requests succeed) and sits ships a default token, so no BDC_ACCESS_KEY is
# required; if the user sets one, sits picks it up from the environment.
BDC_STAC_URL = "https://data.inpe.br/bdc/stac/v1"

# MOD13Q1 has 23 16-day composites per year. BDC's composite periods start
# Aug 29 / Sep 14 / ...; starting Sep 14 yields exactly 23 instances through
# Aug 28, matching the training samples' timeline (a Sep 1 start pulls in the
# overlapping Aug 29 composite and yields 24).
CROP_YEAR_START = "2023-09-14"
CROP_YEAR_END = "2024-08-31"

# ~55 x 50 km over the Maracaju/Dourados soy belt, chosen so that soy,
# pasture and cerrado all appear in a cheap smoke run.
SMOKE_BBOX = {
    "lon_min": -55.60,
    "lat_min": -21.80,
    "lon_max": -55.05,
    "lat_max": -21.35,
}

# Mainland Brazil, loose sanity bounds for label coordinates (catches swapped
# or positive coordinates, not state membership).
BRAZIL_BBOX = {
    "lon_min": -74.1,
    "lat_min": -33.8,
    "lon_max": -34.7,
    "lat_max": 5.3,
}


@dataclass(frozen=True)
class RunConfig:
    name: str
    start_date: str
    end_date: str
    roi: dict | Path
    bands: tuple[str, ...] = ("NDVI", "EVI")
    collection: str = "MOD13Q1-6.1"
    source: str = "BDC"
    memsize_gb: int = 8
    multicores: int = 4
    version: str = "v1"
    use_labels: bool = True
    # When set, train ONLY on series extracted at these points (era-calibrated
    # samples, e.g. from MapBiomas) instead of the canned sitsdata set.
    samples_csv: Path | None = None
    local_data_dir: Path | None = None

    @property
    def output_dir(self) -> Path:
        return OUTPUT_ROOT / self.name


def smoke() -> RunConfig:
    return RunConfig(
        name="smoke",
        start_date=CROP_YEAR_START,
        end_date=CROP_YEAR_END,
        roi=SMOKE_BBOX,
    )


def full_state(
    uf: str = "MS",
    start_date: str = CROP_YEAR_START,
    end_date: str = CROP_YEAR_END,
) -> RunConfig:
    uf = uf.lower()
    return RunConfig(
        name=f"{uf}-{start_date[:4]}-{end_date[:4]}",
        start_date=start_date,
        end_date=end_date,
        roi=ROI_DIR / f"{uf}_boundary.geojson",
        memsize_gb=12,
        multicores=6,
    )


@dataclass(frozen=True)
class ClassStyle:
    name: str
    display: str
    color: str


CLASSES: tuple[ClassStyle, ...] = (
    ClassStyle("Cerrado", "Cerrado (savanna)", "#a1d99b"),
    ClassStyle("Forest", "Forest", "#00441b"),
    ClassStyle("Pasture", "Pasture", "#fee391"),
    ClassStyle("Planted_Forest", "Eucalyptus / planted forest", "#238b45"),
    ClassStyle("Soybean", "Soybean (any rotation)", "#fe9929"),
    ClassStyle("Sugarcane", "Sugarcane", "#dd3497"),
    ClassStyle("Water", "Water", "#225ea8"),
    ClassStyle("Wetland", "Wetland", "#41b6c4"),
    ClassStyle("Grassland", "Grassland", "#d9f0a3"),
    ClassStyle("Soy_Corn", "Soy → corn", "#ec7014"),
    ClassStyle("Soy_Cotton", "Soy → cotton", "#807dba"),
    ClassStyle("Soy_Fallow", "Soy → fallow", "#fdd0a2"),
    ClassStyle("Soy_Millet", "Soy → millet", "#d94801"),
)

CLASS_NAMES: tuple[str, ...] = tuple(style.name for style in CLASSES)
CANNED_CLASSES: tuple[str, ...] = (
    "Cerrado",
    "Forest",
    "Pasture",
    "Soy_Corn",
    "Soy_Cotton",
    "Soy_Fallow",
    "Soy_Millet",
)

IBGE_STATE_AREAS_KM2 = {"MS": 357_142.01, "GO": 340_242.86}


def class_style(name: str) -> ClassStyle:
    """Return the registered style or a stable tab10 fallback."""
    import hashlib

    from matplotlib import colormaps
    from matplotlib import colors as mcolors

    styles = {style.name: style for style in CLASSES}
    if name in styles:
        return styles[name]

    color_index = hashlib.sha256(name.encode("utf-8")).digest()[0] % 10
    color = mcolors.to_hex(colormaps["tab10"](color_index))
    return ClassStyle(name=name, display=name, color=color)
