"""R-free helpers for band-aware sample caches and validation variants."""

from pathlib import Path


def era_cache_paths(csv_path: Path, bands: tuple[str, ...]) -> tuple[Path, Path]:
    """Return band-qualified filtered/raw caches, migrating current legacy caches."""
    csv_path = Path(csv_path)
    band_suffix = "-".join(sorted(bands))
    rds_path = csv_path.with_suffix(f".{band_suffix}.rds")
    raw_path = csv_path.with_suffix(f".{band_suffix}.raw.rds")

    if set(bands) == {"NDVI", "EVI"}:
        legacy_paths = (csv_path.with_suffix(".rds"), csv_path.with_suffix(".raw.rds"))
        for legacy_path, current_path in zip(legacy_paths, (rds_path, raw_path), strict=True):
            if (
                not current_path.exists()
                and legacy_path.exists()
                and legacy_path.stat().st_mtime >= csv_path.stat().st_mtime
            ):
                legacy_path.rename(current_path)

    return rds_path, raw_path


def cache_is_current(cache_path: Path, csv_path: Path) -> bool:
    """Return whether a cache exists and is at least as new as its source CSV."""
    return cache_path.exists() and cache_path.stat().st_mtime >= Path(csv_path).stat().st_mtime


def variant_slug(
    bands: tuple[str, ...],
    num_trees: int,
    mtry: int | None,
    rebalance: tuple[int, int] | None,
) -> str:
    """Encode the model settings that distinguish validation outputs."""
    parts = [f"{len(bands)}b", f"nt{num_trees}"]
    if rebalance is not None:
        parts.append(f"rb{rebalance[0]}x{rebalance[1]}")
    if mtry is not None:
        parts.append(f"mtry{mtry}")
    return "-".join(parts)


def reject_rebalance_in_kfold(rebalance: tuple[int, int] | None) -> None:
    """Prevent synthetic neighbors from leaking across cross-validation folds."""
    if rebalance is not None:
        raise ValueError(
            "Rebalancing is not allowed in k-fold mode because applying it before folding "
            "can leak synthetic neighbors across folds. Use holdout mode instead."
        )
