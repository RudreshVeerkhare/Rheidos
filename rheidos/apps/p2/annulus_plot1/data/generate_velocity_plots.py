#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


EXPECTED_SAMPLES = 500
FILTER_TOL = 5.0e-3
POSITION_TOL = 2.0e-3
EPS = 1.0e-14
BASIS_LABELS = ("v_dr", "v_dtheta")
BASIS_NAMES = ("dr", "dtheta")


@dataclass
class VelocitySeries:
    filename: str
    theta_file: float
    theta: float
    rin: float
    rout: float
    radius: np.ndarray
    positions: np.ndarray
    analytical: np.ndarray
    discrete: np.ndarray
    analytical_basis: np.ndarray
    discrete_basis: np.ndarray
    error_basis: np.ndarray
    error: np.ndarray
    error_norm: np.ndarray
    interior: np.ndarray
    max_position_delta: float


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=(
            "Generate analytical vs discrete annulus vortex-core velocity plots "
            "and error summaries."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=script_dir,
        help="Folder containing analytical/ and discrete/ CSV directories.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output folder for plots and summary CSVs. Defaults to DATA_DIR/plots.",
    )
    parser.add_argument(
        "--h",
        type=float,
        default=0.2,
        help="Mesh edge length used to mark boundary h-bands as unreliable.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for raster plot outputs.",
    )
    parser.add_argument(
        "--formats",
        default="png,pdf",
        help="Comma-separated plot formats, for example png,pdf or png.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat non-fatal validation warnings as errors.",
    )
    return parser.parse_args()


def import_matplotlib():
    cache_dir = Path(tempfile.gettempdir()) / "rheidos_annulus_plot_mpl_cache"
    mpl_config_dir = cache_dir / "mplconfig"
    xdg_cache_dir = cache_dir / "xdg"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    xdg_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache_dir))

    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator

        return plt, MaxNLocator
    except ModuleNotFoundError as exc:
        if exc.name == "matplotlib":
            raise SystemExit(
                "Matplotlib is required to generate plots. Install with:\n"
                "  python -m pip install -r requirements-viz.txt"
            ) from exc
        raise


def parse_file_params(path: Path) -> Tuple[float, float, float]:
    parts = path.stem.split("_")
    if len(parts) != 3:
        raise ValueError(
            f"Expected filename '<theta>_<Rin>_<Rout>.csv', got {path.name!r}"
        )
    try:
        theta, rin, rout = (float(part) for part in parts)
    except ValueError as exc:
        raise ValueError(f"Could not parse numeric parameters from {path.name!r}") from exc
    return theta, rin, rout


def read_csv_rows(path: Path) -> Tuple[List[Dict[str, str]], Sequence[str]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = reader.fieldnames or []
    if not rows:
        raise ValueError(f"{path} has no data rows")
    return rows, fieldnames


def require_columns(path: Path, fieldnames: Sequence[str], columns: Iterable[str]) -> None:
    missing = sorted(set(columns) - set(fieldnames))
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")


def close_to(value: float, target: float, tol: float = FILTER_TOL) -> bool:
    return abs(value - target) <= tol


def filtered_probe_rows(
    path: Path,
    *,
    rin: float,
    rout: float,
    required_columns: Sequence[str],
    strict: bool,
    warnings: List[str],
) -> List[Dict[str, str]]:
    rows, fieldnames = read_csv_rows(path)
    require_columns(path, fieldnames, required_columns)
    kept = [
        row
        for row in rows
        if close_to(float(row["Rin"]), rin) and close_to(float(row["Rout"]), rout)
    ]
    if len(kept) != EXPECTED_SAMPLES:
        message = (
            f"{path.name}: expected {EXPECTED_SAMPLES} filtered probe rows for "
            f"Rin={rin:.2f}, Rout={rout:.2f}, found {len(kept)}"
        )
        if strict:
            raise ValueError(message)
        warnings.append(message)
    if not kept:
        raise ValueError(f"{path.name}: no rows matched Rin={rin:.2f}, Rout={rout:.2f}")
    return kept


def rows_to_array(rows: Sequence[Dict[str, str]], columns: Sequence[str]) -> np.ndarray:
    data = np.array([[float(row[column]) for column in columns] for row in rows])
    if not np.all(np.isfinite(data)):
        raise ValueError(f"Non-finite numeric values found in columns: {columns}")
    return data


def representative_theta(rows: Sequence[Dict[str, str]]) -> float:
    theta_values = rows_to_array(rows, ("theta",)).reshape(-1)
    return float(np.median(theta_values))


def project_to_annulus_basis(positions: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    radial_length = np.linalg.norm(positions[:, [0, 2]], axis=1)
    if np.any(radial_length <= EPS):
        raise ValueError("Cannot project velocity at a zero-radius position")

    e_dr = np.zeros_like(positions)
    e_dr[:, 0] = positions[:, 0] / radial_length
    e_dr[:, 2] = positions[:, 2] / radial_length

    e_dtheta = np.zeros_like(positions)
    e_dtheta[:, 0] = -positions[:, 2] / radial_length
    e_dtheta[:, 2] = positions[:, 0] / radial_length

    return np.column_stack(
        (
            np.einsum("ij,ij->i", vectors, e_dr),
            np.einsum("ij,ij->i", vectors, e_dtheta),
        )
    )


def load_velocity_pair(
    analytical_path: Path,
    discrete_path: Path,
    *,
    strict: bool,
    h: float,
    warnings: List[str],
) -> VelocitySeries:
    theta_file, rin, rout = parse_file_params(analytical_path)
    discrete_theta_file, discrete_rin, discrete_rout = parse_file_params(discrete_path)
    if (theta_file, rin, rout) != (discrete_theta_file, discrete_rin, discrete_rout):
        raise ValueError(
            f"Filename parameters do not match: {analytical_path.name} vs "
            f"{discrete_path.name}"
        )

    shared_columns = ("ptnum", "theta", "P_x", "P_y", "P_z", "Rin", "Rout")
    analytical_rows = filtered_probe_rows(
        analytical_path,
        rin=rin,
        rout=rout,
        required_columns=shared_columns
        + (
            "analytic_vortex_core_vel_x",
            "analytic_vortex_core_vel_y",
            "analytic_vortex_core_vel_z",
        ),
        strict=strict,
        warnings=warnings,
    )
    discrete_rows = filtered_probe_rows(
        discrete_path,
        rin=rin,
        rout=rout,
        required_columns=shared_columns
        + ("vortex_core_vel_x", "vortex_core_vel_y", "vortex_core_vel_z"),
        strict=strict,
        warnings=warnings,
    )

    if len(analytical_rows) != len(discrete_rows):
        raise ValueError(
            f"{analytical_path.name}: analytical/discrete filtered row counts differ "
            f"({len(analytical_rows)} vs {len(discrete_rows)})"
        )

    analytical_pos = rows_to_array(analytical_rows, ("P_x", "P_y", "P_z"))
    discrete_pos = rows_to_array(discrete_rows, ("P_x", "P_y", "P_z"))
    position_delta = np.linalg.norm(analytical_pos - discrete_pos, axis=1)
    max_position_delta = float(np.max(position_delta))
    if max_position_delta > POSITION_TOL:
        message = (
            f"{analytical_path.name}: max analytical/discrete probe position mismatch "
            f"is {max_position_delta:.3e}, above tolerance {POSITION_TOL:.1e}"
        )
        if strict:
            raise ValueError(message)
        warnings.append(message)

    analytical = rows_to_array(
        analytical_rows,
        (
            "analytic_vortex_core_vel_x",
            "analytic_vortex_core_vel_y",
            "analytic_vortex_core_vel_z",
        ),
    )
    discrete = rows_to_array(
        discrete_rows,
        ("vortex_core_vel_x", "vortex_core_vel_y", "vortex_core_vel_z"),
    )

    positions = 0.5 * (analytical_pos + discrete_pos)
    radius = np.linalg.norm(positions, axis=1)
    order = np.argsort(radius, kind="stable")
    theta = 0.5 * (representative_theta(analytical_rows) + representative_theta(discrete_rows))

    analytical = analytical[order]
    discrete = discrete[order]
    positions = positions[order]
    radius = radius[order]
    analytical_basis = project_to_annulus_basis(positions, analytical)
    discrete_basis = project_to_annulus_basis(positions, discrete)
    error_basis = discrete_basis - analytical_basis
    error = discrete - analytical
    error_norm = np.linalg.norm(error_basis, axis=1)
    interior = (radius >= rin + h) & (radius <= rout - h)

    return VelocitySeries(
        filename=analytical_path.name,
        theta_file=theta_file,
        theta=theta,
        rin=rin,
        rout=rout,
        radius=radius,
        positions=positions,
        analytical=analytical,
        discrete=discrete,
        analytical_basis=analytical_basis,
        discrete_basis=discrete_basis,
        error_basis=error_basis,
        error=error,
        error_norm=error_norm,
        interior=interior,
        max_position_delta=max_position_delta,
    )


def load_all_series(
    data_dir: Path, *, strict: bool, h: float
) -> Tuple[List[VelocitySeries], List[str]]:
    analytical_dir = data_dir / "analytical"
    discrete_dir = data_dir / "discrete"
    if not analytical_dir.is_dir():
        raise FileNotFoundError(f"Missing analytical CSV directory: {analytical_dir}")
    if not discrete_dir.is_dir():
        raise FileNotFoundError(f"Missing discrete CSV directory: {discrete_dir}")

    analytical_files = {path.name: path for path in analytical_dir.glob("*.csv")}
    discrete_files = {path.name: path for path in discrete_dir.glob("*.csv")}
    missing_discrete = sorted(set(analytical_files) - set(discrete_files))
    missing_analytical = sorted(set(discrete_files) - set(analytical_files))
    if missing_discrete or missing_analytical:
        parts = []
        if missing_discrete:
            parts.append("missing discrete files: " + ", ".join(missing_discrete[:8]))
        if missing_analytical:
            parts.append("missing analytical files: " + ", ".join(missing_analytical[:8]))
        raise ValueError("; ".join(parts))

    warnings: List[str] = []
    series = [
        load_velocity_pair(
            analytical_files[name],
            discrete_files[name],
            strict=strict,
            h=h,
            warnings=warnings,
        )
        for name in sorted(analytical_files)
    ]
    series.sort(key=lambda item: item.theta)
    return series, warnings


def rms(values: np.ndarray, axis=None) -> np.ndarray:
    return np.sqrt(np.mean(np.square(values), axis=axis))


def compute_theta_summary(series: Sequence[VelocitySeries]) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    for item in series:
        for scope, mask in (("all", np.ones_like(item.interior, dtype=bool)), ("interior", item.interior)):
            err = item.error_basis[mask]
            analytical = item.analytical_basis[mask]
            err_norm = item.error_norm[mask]
            row: Dict[str, float] = {
                "theta_file": item.theta_file,
                "theta": item.theta,
                "rin": item.rin,
                "rout": item.rout,
                "scope": scope,
                "sample_count": int(np.sum(mask)),
                "position_max_delta": item.max_position_delta,
                "vector_mean_error": float(np.mean(err_norm)),
                "vector_rmse": float(np.sqrt(np.mean(np.sum(err * err, axis=1)))),
                "vector_max_error": float(np.max(err_norm)),
                "relative_vector_rmse": float(
                    np.sqrt(np.mean(np.sum(err * err, axis=1)))
                    / max(float(np.sqrt(np.mean(np.sum(analytical * analytical, axis=1)))), EPS)
                ),
            }
            for index, suffix in enumerate(BASIS_NAMES):
                component = err[:, index]
                row[f"mae_{suffix}"] = float(np.mean(np.abs(component)))
                row[f"rmse_{suffix}"] = float(rms(component))
                row[f"max_abs_{suffix}"] = float(np.max(np.abs(component)))
            rows.append(row)
    return rows


def stack_field(series: Sequence[VelocitySeries], field: str) -> np.ndarray:
    return np.stack([getattr(item, field) for item in series], axis=0)


def compute_radius_theta_variation(series: Sequence[VelocitySeries]) -> List[Dict[str, float]]:
    radius_matrix = np.stack([item.radius for item in series], axis=0)
    radius = np.mean(radius_matrix, axis=0)
    analytical = stack_field(series, "analytical_basis")
    discrete = stack_field(series, "discrete_basis")
    interior = np.all(np.stack([item.interior for item in series], axis=0), axis=0)

    rows: List[Dict[str, float]] = []
    for radius_index, radius_value in enumerate(radius):
        row: Dict[str, float] = {
            "radius_index": radius_index,
            "radius": float(radius_value),
            "interior": int(interior[radius_index]),
            "theta_count": len(series),
        }
        for source_name, stack in (("analytical", analytical), ("discrete", discrete)):
            quantities = {
                "dr": stack[:, radius_index, 0],
                "dtheta": stack[:, radius_index, 1],
                "speed": np.linalg.norm(stack[:, radius_index, :], axis=1),
            }
            for quantity, values in quantities.items():
                prefix = f"{source_name}_{quantity}"
                row[f"{prefix}_theta_mean"] = float(np.mean(values))
                row[f"{prefix}_theta_std"] = float(np.std(values))
                row[f"{prefix}_theta_min"] = float(np.min(values))
                row[f"{prefix}_theta_max"] = float(np.max(values))
                row[f"{prefix}_theta_range"] = float(np.max(values) - np.min(values))
        rows.append(row)
    return rows


def write_rows_csv(path: Path, rows: Sequence[Dict[str, float]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write to {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def configure_plot_style(plt) -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "#fbfcfd",
            "axes.edgecolor": "#2f3742",
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": "#8a94a3",
            "grid.alpha": 0.25,
            "grid.linewidth": 0.6,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "lines.linewidth": 1.7,
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )


def save_figure(fig, out_dir: Path, stem: str, formats: Sequence[str], dpi: int) -> List[Path]:
    paths: List[Path] = []
    for fmt in formats:
        fmt = fmt.strip().lower().lstrip(".")
        if not fmt:
            continue
        path = out_dir / f"{stem}.{fmt}"
        save_kwargs = {"dpi": dpi} if fmt in {"png", "jpg", "jpeg", "tif", "tiff"} else {}
        fig.savefig(path, **save_kwargs)
        paths.append(path)
    return paths


def shade_boundary(ax, rin: float, rout: float, h: float) -> None:
    shade_color = "#b8c2cc"
    ax.axvspan(rin, min(rin + h, rout), color=shade_color, alpha=0.24, linewidth=0)
    ax.axvspan(max(rout - h, rin), rout, color=shade_color, alpha=0.24, linewidth=0)
    ax.axvline(rin + h, color="#657786", linestyle="--", linewidth=0.8, alpha=0.75)
    ax.axvline(rout - h, color="#657786", linestyle="--", linewidth=0.8, alpha=0.75)


def circular_distance(a: float, b: float) -> float:
    return abs(math.atan2(math.sin(a - b), math.cos(a - b)))


def selected_theta_series(series: Sequence[VelocitySeries]) -> List[VelocitySeries]:
    selected: List[VelocitySeries] = []
    selected_names = set()
    for target in (0.0, 0.5 * math.pi, math.pi, 1.5 * math.pi):
        candidate = min(series, key=lambda item: circular_distance(item.theta, target))
        if candidate.filename not in selected_names:
            selected.append(candidate)
            selected_names.add(candidate.filename)
    selected.sort(key=lambda item: item.theta)
    return selected


def plot_velocity_profiles(
    plt,
    series: Sequence[VelocitySeries],
    *,
    out_dir: Path,
    formats: Sequence[str],
    dpi: int,
    h: float,
) -> List[Path]:
    chosen = selected_theta_series(series)
    fig, axes = plt.subplots(
        nrows=2,
        ncols=len(chosen),
        figsize=(3.7 * len(chosen), 5.8),
        sharex="col",
        squeeze=False,
    )
    analytical_color = "#1f5a93"
    discrete_color = "#d65f1f"
    for col, item in enumerate(chosen):
        for row, label in enumerate(BASIS_LABELS):
            ax = axes[row, col]
            shade_boundary(ax, item.rin, item.rout, h)
            ax.plot(
                item.radius,
                item.analytical_basis[:, row],
                color=analytical_color,
                label="analytical" if row == 0 and col == 0 else None,
            )
            ax.plot(
                item.radius,
                item.discrete_basis[:, row],
                color=discrete_color,
                alpha=0.82,
                label="discrete" if row == 0 and col == 0 else None,
            )
            ax.scatter(
                item.radius[::8],
                item.discrete_basis[::8, row],
                color=discrete_color,
                s=8,
                alpha=0.55,
                linewidth=0,
            )
            if col == 0:
                ax.set_ylabel(label)
            if row == 1:
                ax.set_xlabel("radius R")
            if row == 0:
                ax.set_title(f"theta = {item.theta:.2f} rad")
            ax.margins(x=0.01)
    axes[0, 0].legend(loc="best", frameon=True, framealpha=0.92)
    fig.suptitle("Analytical and discrete velocity in annulus polar directions", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    paths = save_figure(fig, out_dir, "velocity_profiles_selected", formats, dpi)
    plt.close(fig)
    return paths


def plot_residual_vs_radius(
    plt,
    series: Sequence[VelocitySeries],
    *,
    out_dir: Path,
    formats: Sequence[str],
    dpi: int,
    h: float,
) -> List[Path]:
    chosen = selected_theta_series(series)
    radius = np.mean(np.stack([item.radius for item in series], axis=0), axis=0)
    residual_stack = np.stack([item.error_norm for item in series], axis=0)
    residual_median = np.median(residual_stack, axis=0)
    residual_p10 = np.percentile(residual_stack, 10, axis=0)
    residual_p90 = np.percentile(residual_stack, 90, axis=0)

    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    for item in series:
        ax.plot(
            item.radius,
            np.maximum(item.error_norm, EPS),
            color="#7d8791",
            linewidth=0.65,
            alpha=0.22,
        )
    ax.fill_between(
        radius,
        np.maximum(residual_p10, EPS),
        np.maximum(residual_p90, EPS),
        color="#2f3742",
        alpha=0.12,
        linewidth=0,
        label="10-90% theta band",
    )
    ax.plot(
        radius,
        np.maximum(residual_median, EPS),
        color="#2f3742",
        linewidth=2.0,
        label="median over theta",
    )
    colors = ("#1f5a93", "#d65f1f", "#12805c", "#6b4fbb")
    for color, item in zip(colors, chosen):
        ax.plot(
            item.radius,
            np.maximum(item.error_norm, EPS),
            color=color,
            label=f"theta = {item.theta:.2f} rad",
        )
    shade_boundary(ax, chosen[0].rin, chosen[0].rout, h)
    ax.set_yscale("log")
    ax.set_xlabel("radius R")
    ax.set_ylabel("residual ||v_discrete - v_analytical||")
    ax.set_title("Polar velocity residual vs radius")
    ax.legend(loc="best", frameon=True, framealpha=0.92)
    ax.margins(x=0.01)
    fig.tight_layout()
    paths = save_figure(fig, out_dir, "residual_vs_radius_selected", formats, dpi)
    plt.close(fig)
    return paths


def theta_tick_labels() -> Tuple[List[float], List[str]]:
    return (
        [0.0, 0.5 * math.pi, math.pi, 1.5 * math.pi, 2.0 * math.pi],
        ["0", "pi/2", "pi", "3pi/2", "2pi"],
    )


def plot_vector_error_heatmap(
    plt,
    series: Sequence[VelocitySeries],
    *,
    out_dir: Path,
    formats: Sequence[str],
    dpi: int,
    h: float,
) -> List[Path]:
    theta = np.array([item.theta for item in series])
    radius = np.mean(np.stack([item.radius for item in series], axis=0), axis=0)
    err = np.stack([item.error_norm for item in series], axis=0)
    plot_data = np.log10(np.maximum(err, EPS))
    finite = plot_data[np.isfinite(plot_data)]
    vmin = float(np.percentile(finite, 2.0))
    vmax = float(np.percentile(finite, 99.0))

    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    image = ax.imshow(
        plot_data,
        origin="lower",
        aspect="auto",
        extent=[radius[0], radius[-1], theta[0], theta[-1]],
        interpolation="nearest",
        cmap="magma",
        vmin=vmin,
        vmax=vmax,
    )
    first = series[0]
    shade_boundary(ax, first.rin, first.rout, h)
    ticks, labels = theta_tick_labels()
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels)
    ax.set_xlabel("radius R")
    ax.set_ylabel("theta")
    ax.set_title("Vector error heatmap: log10(||v_discrete - v_analytical||)")
    colorbar = fig.colorbar(image, ax=ax, pad=0.02)
    colorbar.set_label("log10 vector error")
    fig.tight_layout()
    paths = save_figure(fig, out_dir, "vector_error_heatmap", formats, dpi)
    plt.close(fig)
    return paths


def summary_metric_arrays(
    summary_rows: Sequence[Dict[str, float]], metric: str, scope: str
) -> Tuple[np.ndarray, np.ndarray]:
    rows = [row for row in summary_rows if row["scope"] == scope]
    rows.sort(key=lambda row: row["theta"])
    theta = np.array([float(row["theta"]) for row in rows])
    values = np.array([float(row[metric]) for row in rows])
    return theta, values


def plot_theta_error_summary(
    plt,
    max_n_locator,
    summary_rows: Sequence[Dict[str, float]],
    *,
    out_dir: Path,
    formats: Sequence[str],
    dpi: int,
) -> List[Path]:
    metrics = (
        ("vector_mean_error", "mean vector error"),
        ("vector_rmse", "vector RMSE"),
        ("vector_max_error", "max vector error"),
    )
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(8.8, 7.0), sharex=True)
    for ax, (metric, label) in zip(axes, metrics):
        theta_all, all_values = summary_metric_arrays(summary_rows, metric, "all")
        theta_interior, interior_values = summary_metric_arrays(summary_rows, metric, "interior")
        ax.plot(
            theta_all,
            np.maximum(all_values, EPS),
            color="#6b4fbb",
            marker="o",
            markersize=3,
            label="all samples",
        )
        ax.plot(
            theta_interior,
            np.maximum(interior_values, EPS),
            color="#12805c",
            marker="o",
            markersize=3,
            label="interior only",
        )
        ax.set_yscale("log")
        ax.set_ylabel(label)
        ax.yaxis.set_major_locator(max_n_locator(nbins=5))
    ticks, labels = theta_tick_labels()
    axes[-1].set_xticks(ticks)
    axes[-1].set_xticklabels(labels)
    axes[-1].set_xlabel("theta")
    axes[0].legend(loc="best", frameon=True, framealpha=0.92)
    fig.suptitle("Theta-wise discrete-vs-analytical velocity error", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    paths = save_figure(fig, out_dir, "theta_error_summary", formats, dpi)
    plt.close(fig)
    return paths


def plot_velocity_theta_variation(
    plt,
    variation_rows: Sequence[Dict[str, float]],
    series: Sequence[VelocitySeries],
    *,
    out_dir: Path,
    formats: Sequence[str],
    dpi: int,
    h: float,
) -> List[Path]:
    radius = np.array([float(row["radius"]) for row in variation_rows])
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(9.2, 6.6), sharex=True)
    quantities = (("dr", "v_dr"), ("dtheta", "v_dtheta"), ("speed", "|v|"))
    fig.delaxes(axes[1, 1])
    axes_flat = (axes[0, 0], axes[0, 1], axes[1, 0])
    for ax, (quantity, label) in zip(axes_flat, quantities):
        analytical_std = np.array(
            [float(row[f"analytical_{quantity}_theta_std"]) for row in variation_rows]
        )
        discrete_std = np.array(
            [float(row[f"discrete_{quantity}_theta_std"]) for row in variation_rows]
        )
        shade_boundary(ax, series[0].rin, series[0].rout, h)
        ax.plot(radius, analytical_std, color="#1f5a93", label="analytical")
        ax.plot(radius, discrete_std, color="#d65f1f", linestyle="--", label="discrete")
        ax.set_title(f"{label} variation across theta")
        ax.set_ylabel("std across theta")
        ax.margins(x=0.01)
    for ax in axes_flat:
        ax.set_xlabel("radius R")
    axes_flat[0].legend(loc="best", frameon=True, framealpha=0.92)
    fig.suptitle("Velocity variation at constant radius across theta", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    paths = save_figure(fig, out_dir, "velocity_theta_variation", formats, dpi)
    plt.close(fig)
    return paths


def validate_common_grids(series: Sequence[VelocitySeries], *, strict: bool) -> float:
    radius_matrix = np.stack([item.radius for item in series], axis=0)
    reference = np.mean(radius_matrix, axis=0)
    max_delta = float(np.max(np.abs(radius_matrix - reference)))
    if max_delta > POSITION_TOL and strict:
        raise ValueError(
            f"Radial grids differ across theta by up to {max_delta:.3e}, "
            f"above tolerance {POSITION_TOL:.1e}"
        )
    return max_delta


def parse_formats(raw: str) -> List[str]:
    formats = [item.strip().lower().lstrip(".") for item in raw.split(",") if item.strip()]
    if not formats:
        raise ValueError("--formats must include at least one output format")
    return formats


def print_report(
    *,
    series: Sequence[VelocitySeries],
    warnings: Sequence[str],
    radial_grid_delta: float,
    outputs: Sequence[Path],
    data_dir: Path,
    out_dir: Path,
) -> None:
    interior_counts = sorted({int(np.sum(item.interior)) for item in series})
    max_position_delta = max(item.max_position_delta for item in series)
    print("Annulus velocity plot generation complete")
    print(f"  data directory: {data_dir}")
    print(f"  output directory: {out_dir}")
    print(f"  matched theta files: {len(series)}")
    print(f"  filtered samples per theta: {len(series[0].radius)}")
    print(f"  interior samples per theta: {interior_counts}")
    print(f"  max analytical/discrete position mismatch: {max_position_delta:.3e}")
    print(f"  max radial-grid mismatch across theta: {radial_grid_delta:.3e}")
    if warnings:
        print("  validation warnings:")
        for warning in warnings:
            print(f"    - {warning}")
    print("  outputs:")
    for path in outputs:
        print(f"    {path}")


def main() -> int:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    out_dir = (args.out_dir.resolve() if args.out_dir else data_dir / "plots")
    out_dir.mkdir(parents=True, exist_ok=True)
    formats = parse_formats(args.formats)

    plt, max_n_locator = import_matplotlib()
    configure_plot_style(plt)

    series, warnings = load_all_series(data_dir, strict=args.strict, h=args.h)
    if not series:
        raise ValueError(f"No matched CSV files found under {data_dir}")
    radial_grid_delta = validate_common_grids(series, strict=args.strict)

    outputs: List[Path] = []
    theta_summary = compute_theta_summary(series)
    theta_summary_path = out_dir / "theta_error_summary.csv"
    write_rows_csv(theta_summary_path, theta_summary)
    outputs.append(theta_summary_path)

    variation = compute_radius_theta_variation(series)
    variation_path = out_dir / "radius_theta_variation.csv"
    write_rows_csv(variation_path, variation)
    outputs.append(variation_path)

    outputs.extend(
        plot_velocity_profiles(
            plt, series, out_dir=out_dir, formats=formats, dpi=args.dpi, h=args.h
        )
    )
    outputs.extend(
        plot_residual_vs_radius(
            plt, series, out_dir=out_dir, formats=formats, dpi=args.dpi, h=args.h
        )
    )
    outputs.extend(
        plot_vector_error_heatmap(
            plt, series, out_dir=out_dir, formats=formats, dpi=args.dpi, h=args.h
        )
    )
    outputs.extend(
        plot_theta_error_summary(
            plt,
            max_n_locator,
            theta_summary,
            out_dir=out_dir,
            formats=formats,
            dpi=args.dpi,
        )
    )
    outputs.extend(
        plot_velocity_theta_variation(
            plt,
            variation,
            series,
            out_dir=out_dir,
            formats=formats,
            dpi=args.dpi,
            h=args.h,
        )
    )

    print_report(
        series=series,
        warnings=warnings,
        radial_grid_delta=radial_grid_delta,
        outputs=outputs,
        data_dir=data_dir,
        out_dir=out_dir,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
