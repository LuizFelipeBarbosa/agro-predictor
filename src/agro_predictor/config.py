"""Run configuration: what to classify, where, and with how much hardware."""

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT_ROOT / "output"
ROI_DIR = PROJECT_ROOT / "data" / "roi"
MS_BOUNDARY_PATH = ROI_DIR / "ms_boundary.geojson"

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


def full_state() -> RunConfig:
    return RunConfig(
        name="ms-2023-2024",
        start_date=CROP_YEAR_START,
        end_date=CROP_YEAR_END,
        roi=MS_BOUNDARY_PATH,
        memsize_gb=12,
        multicores=6,
    )
