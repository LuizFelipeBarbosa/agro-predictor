"""Area summaries and previews for classified rasters."""

import json
import traceback
from datetime import date
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask

from agro_predictor import config, roi
from agro_predictor.config import (
    CANNED_CLASSES,
    CLASS_NAMES,
    IBGE_STATE_AREAS_KM2,
    class_style,
)

SIGA_MS_2023_24 = {
    "pasture": 48.3,
    "native": 30.7,
    "soybean": 11.8,
    "eucalyptus": 4.1,
    "sugarcane": 2.5,
}
# SIGA-MS "Uso e Ocupacao do Solo", MS 1a safra 2024/2025 (Famasul Boletim
# SigaBov Ed. 74, Aug 2026; state total 35,714,492 ha). The full 2025/26
# table was not yet published as of Aug 2026 — for 2025/26 maps this is
# the freshest full benchmark; 2025/26 point references: SIGA-MS final
# soybean 4,620,446.8 ha (~12.9% of state), CONAB 11o levantamento
# soybean 4,372,500 ha.
SIGA_MS_2024_25 = {
    "pasture": 46.7,
    "native": 30.8,
    "soybean": 12.7,
    "eucalyptus": 4.8,
    "sugarcane": 2.5,
}

BENCHMARKS = {
    "siga-ms-2023-24": SIGA_MS_2023_24,
    "siga-ms-2024-25": SIGA_MS_2024_25,
}
BENCHMARK_CATEGORIES: dict[str, tuple[str, ...]] = {
    "soybean": (
        "Soybean",
        "Soy_Corn",
        "Soy_Cotton",
        "Soy_Fallow",
        "Soy_Millet",
        "Soy_Sorghum",
    ),
    "eucalyptus": ("Planted_Forest",),
    "native": ("Cerrado", "Forest", "Wetland", "Grassland"),
    "pasture": ("Pasture",),
    "sugarcane": ("Sugarcane",),
    "water": ("Water",),
}


def load_run_legend(run_dir: Path) -> list[dict]:
    """Load the code-to-class legend recorded for a run."""
    labels_path = run_dir / "run_labels.json"
    if labels_path.exists():
        labels = json.loads(labels_path.read_text(encoding="utf-8"))["labels"]
    else:
        labels = sorted(CANNED_CLASSES)

    legend = []
    for code, name in enumerate(labels, start=1):
        style = class_style(name)
        legend.append(
            {
                "code": code,
                "name": name,
                "display": style.display,
                "color": style.color,
            }
        )
    return legend


def pixel_area_km2(transform) -> float:
    """Return one raster pixel's area in square kilometres."""
    return abs(transform.a * transform.e) / 1e6


def merge_tiles(tifs: list[Path], dest: Path) -> None:
    import rasterio
    from rasterio.merge import merge

    sources = [rasterio.open(tif) for tif in tifs]
    try:
        mosaic, transform = merge(sources)
        meta = sources[0].meta | {
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": transform,
            "compress": "deflate",
        }
        with rasterio.open(dest, "w", **meta) as out:
            out.write(mosaic)
    finally:
        for src in sources:
            src.close()


def find_or_build_mosaic(run_dir: Path) -> Path:
    """Return the run's mosaic tif, merging per-tile classifications if needed."""
    run_dir = Path(run_dir)
    class_tifs = sorted(
        tif
        for tif in run_dir.glob("TERRA_MODIS_*_class_*.tif")
        if not tif.stem.endswith("_clipped")
    )

    mosaics = sorted(run_dir.glob("*_class_mosaic.tif"))
    mosaic_tif = run_dir / f"{run_dir.name}_class_mosaic.tif"
    if mosaics:
        existing_mosaic = (
            mosaic_tif
            if mosaic_tif in mosaics
            else max(mosaics, key=lambda tif: tif.stat().st_mtime)
        )
        if len(mosaics) > 1:
            print(f"Multiple mosaics found; using {existing_mosaic.name}")
        if not class_tifs or existing_mosaic.stat().st_mtime >= max(
            tif.stat().st_mtime for tif in class_tifs
        ):
            return existing_mosaic

    if not class_tifs:
        raise FileNotFoundError(f"No classified GeoTIFFs found in {run_dir}")
    if not mosaics and len(class_tifs) == 1:
        return class_tifs[0]

    merge_tiles(class_tifs, mosaic_tif)
    print(f"Mosaic saved: {mosaic_tif}")
    return mosaic_tif


def clip_to_boundary(class_tif: Path, boundary, dest: Path) -> Path:
    """Clip a classified raster to a boundary and return the destination."""
    with rasterio.open(class_tif) as src:
        raster_boundary = boundary.to_crs(src.crs)
        nodata = src.nodata if src.nodata is not None else 255
        clipped, transform = mask(
            src,
            raster_boundary.geometry,
            crop=True,
            all_touched=False,
            nodata=nodata,
        )
        meta = src.meta | {
            "height": clipped.shape[1],
            "width": clipped.shape[2],
            "transform": transform,
            "nodata": nodata,
            "compress": "deflate",
        }

    with rasterio.open(dest, "w", **meta) as out:
        out.write(clipped)
    return dest


def class_areas(class_tif: Path, legend: list[dict]) -> pd.DataFrame:
    """Compute classified pixel counts, areas, and percentages."""
    with rasterio.open(class_tif) as src:
        data = src.read(1)
        nodata = src.nodata if src.nodata is not None else 255
        area_per_pixel = pixel_area_km2(src.transform)

    valid_values = data[data != nodata]
    counts = np.bincount(valid_values.ravel()) if valid_values.size else np.array([])
    pixels_by_code = {
        entry["code"]: int(counts[entry["code"]]) if entry["code"] < len(counts) else 0
        for entry in legend
    }
    total_pixels = sum(pixels_by_code.values())

    rows = []
    for entry in legend:
        pixels = pixels_by_code[entry["code"]]
        rows.append(
            {
                "code": entry["code"],
                "name": entry["name"],
                "display": entry["display"],
                "pixels": pixels,
                "km2": pixels * area_per_pixel,
                "pct": pixels / total_pixels * 100 if total_pixels else 0.0,
            }
        )

    rows.append(
        {
            "code": pd.NA,
            "name": pd.NA,
            "display": pd.NA,
            "pixels": total_pixels,
            "km2": total_pixels * area_per_pixel,
            "pct": 100.0 if total_pixels else 0.0,
        }
    )
    return pd.DataFrame(rows)


def compare_to_benchmark(areas_df: pd.DataFrame, benchmark_key: str) -> pd.DataFrame:
    """Compare modeled class percentages with a named benchmark."""
    benchmark = BENCHMARKS[benchmark_key]
    class_rows = areas_df[areas_df["code"].notna()]
    rows = []

    for category, class_names in BENCHMARK_CATEGORIES.items():
        model_pct = class_rows.loc[class_rows["name"].isin(class_names), "pct"].sum()
        benchmark_pct = benchmark.get(category, np.nan)
        rows.append(
            {
                "category": category,
                "model_pct": model_pct,
                "benchmark_pct": benchmark_pct,
                "delta_pp": model_pct - benchmark_pct,
            }
        )

    covered_classes = {
        class_name
        for class_names in BENCHMARK_CATEGORIES.values()
        for class_name in class_names
    }
    for _, row in class_rows.loc[~class_rows["name"].isin(covered_classes)].iterrows():
        if isinstance(row["display"], str):
            display = row["display"]
        elif row["name"] in CLASS_NAMES:
            display = class_style(row["name"]).display
        else:
            display = row["name"]
        rows.append(
            {
                "category": display,
                "model_pct": row["pct"],
                "benchmark_pct": np.nan,
                "delta_pp": np.nan,
            }
        )

    return pd.DataFrame(
        rows,
        columns=("category", "model_pct", "benchmark_pct", "delta_pp"),
    )


def render_preview(
    class_tif: Path,
    legend: list[dict],
    path: Path,
    boundary=None,
    areas_df=None,
) -> None:
    """Render a classified GeoTIFF and its legend to a PNG."""
    from matplotlib import colors as mcolors
    from matplotlib import patches
    from matplotlib import pyplot as plt
    from rasterio.transform import array_bounds

    with rasterio.open(class_tif) as src:
        data = src.read(1)
        bounds = array_bounds(src.height, src.width, src.transform)
        raster_crs = src.crs

    ordered_legend = sorted(legend, key=lambda entry: entry["code"])
    colors = [entry["color"] for entry in ordered_legend]
    masked = np.ma.masked_outside(data, 1, len(ordered_legend))
    west, south, east, north = bounds

    fig, ax = plt.subplots(figsize=(10, 7), dpi=150)
    ax.imshow(
        masked,
        cmap=mcolors.ListedColormap(colors),
        vmin=1,
        vmax=len(ordered_legend),
        interpolation="nearest",
        extent=(west, east, south, north),
        origin="upper",
    )
    if boundary is not None:
        boundary.to_crs(raster_crs).plot(
            ax=ax,
            edgecolor="black",
            facecolor="none",
            linewidth=0.8,
        )
        ax.set_xlim(west, east)
        ax.set_ylim(south, north)

    pct_by_code = {}
    if areas_df is not None:
        pct_by_code = {
            int(row["code"]): row["pct"]
            for _, row in areas_df[areas_df["code"].notna()].iterrows()
        }
    handles = []
    for entry in ordered_legend:
        label = entry["display"]
        if entry["code"] in pct_by_code:
            label += f" — {pct_by_code[entry['code']]:.1f}%"
        handles.append(patches.Patch(color=entry["color"], label=label))

    ax.set_axis_off()
    ax.set_title(class_tif.stem, fontsize=9)
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Preview saved: {path}")


def compute_run_areas(
    run_dir: Path,
    state: str | None = "MS",
    clip: bool = True,
    benchmark: str | None = None,
) -> pd.DataFrame:
    """Build run-level area artifacts from existing classified rasters."""
    run_dir = Path(run_dir)
    mosaic_tif = find_or_build_mosaic(run_dir)

    boundary = None
    boundary_path = None
    source_tif = mosaic_tif
    normalized_state = state.upper() if state else None
    if clip:
        if normalized_state is None:
            raise ValueError("state is required when clipping to a boundary")
        uf = normalized_state.lower()
        boundary_path = config.ROI_DIR / f"{uf}_boundary.geojson"
        if not boundary_path.exists():
            roi.fetch_state_boundary(uf)
        boundary = gpd.read_file(boundary_path)
        if boundary.crs is None:
            boundary = roi.load_roi(boundary_path)
        clipped_tif = mosaic_tif.with_name(f"{mosaic_tif.stem}_clipped.tif")
        if not clipped_tif.exists() or clipped_tif.stat().st_mtime <= mosaic_tif.stat().st_mtime:
            clip_to_boundary(mosaic_tif, boundary, clipped_tif)
        source_tif = clipped_tif

    legend = load_run_legend(run_dir)
    areas_df = class_areas(source_tif, legend)
    areas_df.to_csv(run_dir / "areas.csv", index=False)

    with rasterio.open(source_tif) as src:
        source_pixel_area = pixel_area_km2(src.transform)
    payload = {
        "run": run_dir.name,
        "source": str(source_tif),
        "boundary": str(boundary_path) if boundary_path is not None else None,
        "state": normalized_state,
        "ibge_km2": IBGE_STATE_AREAS_KM2.get(normalized_state),
        "pixel_area_km2": source_pixel_area,
        "generated_on": date.today().isoformat(),  # noqa: DTZ011
        "rows": json.loads(areas_df.to_json(orient="records")),
    }
    (run_dir / "areas.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )

    try:
        render_preview(
            source_tif,
            legend,
            run_dir / "classified_preview.png",
            boundary=boundary,
            areas_df=areas_df,
        )
    except Exception:  # noqa: BLE001 — tabular area outputs remain useful without a preview
        traceback.print_exc()
        print("Mosaic/preview failed; open the tif in QGIS instead.")

    if benchmark:
        comparison = compare_to_benchmark(areas_df, benchmark)
        comparison.to_csv(run_dir / "benchmark_delta.csv", index=False)
        print(comparison.to_string(index=False))

    print(areas_df.to_string(index=False))
    ibge_km2 = IBGE_STATE_AREAS_KM2.get(normalized_state)
    if ibge_km2 is not None:
        total_km2 = areas_df.iloc[-1]["km2"]
        difference_pct = (total_km2 - ibge_km2) / ibge_km2 * 100
        print(
            f"TOTAL: {total_km2:,.2f} km2 vs IBGE reference {ibge_km2:,.2f} km2 "
            f"({difference_pct:+.2f}% difference)"
        )
    return areas_df
