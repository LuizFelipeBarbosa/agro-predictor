"""Training samples: samples_matogrosso_mod13q1 from the sitsdata R package.

Known limitation: these ~1.8k samples were collected in Mato Grosso, the
neighbor state. Mato Grosso do Sul land-cover types absent from the training
set (Pantanal wetlands, open water, sugarcane, planted forest) will be forced
into the nearest available class. Good enough for a first working pipeline,
not for a publishable map.
"""

from pysits import load_samples, sits_bands, sits_select, sits_timeline
from pysits import summary as sits_summary

SAMPLES_NAME = "samples_matogrosso_mod13q1"
EXPECTED_TIMELINE_STEPS = 23


def load_training_samples(bands: tuple[str, ...]):
    samples = load_samples(SAMPLES_NAME, package="sitsdata")
    return sits_select(samples, bands=list(bands))


def combine_samples(canned, user):
    """Row-bind two sits sample sets into one training set."""
    from pysits.models.data.ts import SITSTimeSeriesModel
    from rpy2.robjects.packages import importr

    dplyr = importr("dplyr")
    # Deliberate private-attribute access is contained at the pysits/R boundary.
    combined = dplyr.bind_rows(canned._instance, user._instance)
    return SITSTimeSeriesModel(combined)


def describe_samples() -> None:
    samples = load_samples(SAMPLES_NAME, package="sitsdata")
    print(f"Dataset: {SAMPLES_NAME} (sitsdata)")
    print(f"Bands: {list(sits_bands(samples))}")
    timeline = sits_timeline(samples)
    print(f"Timeline: {len(timeline)} instances (expected {EXPECTED_TIMELINE_STEPS})")
    print("Label counts:")
    print(sits_summary(samples))
