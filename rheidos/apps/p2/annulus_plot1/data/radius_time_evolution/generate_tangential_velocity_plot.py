#!/usr/bin/env python3
"""Compare analytical and discrete tangential velocity components over time."""

from __future__ import annotations

import argparse
import csv
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ANALYTICAL_CSV = "analytical_Rin_1_Rout_2.csv"
DEFAULT_DISCRETE_CSV = "discrete_Rin_1_Rout_2.csv"
DEFAULT_OUTPUT_STEM = "tangential_velocity_comparison_Rin_1_Rout_2"
POSITION_COLUMNS = ("P_x", "P_y", "P_z")
ANALYTICAL_VELOCITY_COLUMNS = (
    "analytic_vortex_core_vel_x",
    "analytic_vortex_core_vel_y",
    "analytic_vortex_core_vel_z",
)
COEXACT_VELOCITY_COLUMNS = ("coexact_vel_x", "coexact_vel_y", "coexact_vel_z")
HARMONIC_VELOCITY_COLUMNS = ("harmonic_vel_x", "harmonic_vel_y", "harmonic_vel_z")
TOTAL_VELOCITY_COLUMNS = ("velocity_x", "velocity_y", "velocity_z")
ANALYTICAL_REQUIRED_COLUMNS = {"ptnum", *POSITION_COLUMNS, *ANALYTICAL_VELOCITY_COLUMNS}
DISCRETE_REQUIRED_COLUMNS = {
    "ptnum",
    *POSITION_COLUMNS,
    *COEXACT_VELOCITY_COLUMNS,
    *HARMONIC_VELOCITY_COLUMNS,
    *TOTAL_VELOCITY_COLUMNS,
}

Vector = tuple[float, float, float]


@dataclass(frozen=True)
class VelocitySeries:
    label: str
    path: Path
    source_steps: tuple[int, ...]
    steps: tuple[int, ...]
    tangential_velocities: tuple[float, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot tangential velocity components for the analytical solution, "
            "discrete coexact velocity, harmonic velocity, and total velocity."
        )
    )
    parser.add_argument(
        "--analytical-csv",
        type=Path,
        default=Path(DEFAULT_ANALYTICAL_CSV),
        help=f"Analytical evolution CSV. Default: {DEFAULT_ANALYTICAL_CSV}",
    )
    parser.add_argument(
        "--discrete-csv",
        type=Path,
        default=Path(DEFAULT_DISCRETE_CSV),
        help=f"Discrete evolution CSV. Default: {DEFAULT_DISCRETE_CSV}",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.01,
        help="Time step size used to convert plotted timestep index to physical time.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="PNG output resolution.",
    )
    parser.add_argument(
        "--output-stem",
        default=DEFAULT_OUTPUT_STEM,
        help="Output filename stem; PNG and PDF are written beside this script.",
    )
    parser.add_argument(
        "--input-order",
        choices=("reverse", "forward"),
        default="reverse",
        help=(
            "Chronological order of rows in the exported CSVs. Default: reverse, "
            "because the current exports write the physical initial condition last."
        ),
    )
    return parser.parse_args()


def resolve_input_path(path: Path) -> Path:
    if path.is_absolute():
        return path

    cwd_path = Path.cwd() / path
    if cwd_path.exists():
        return cwd_path.resolve()

    return (SCRIPT_DIR / path).resolve()


def require_columns(
    path: Path,
    fieldnames: Optional[Iterable[str]],
    required_columns: set[str],
) -> None:
    actual = set(fieldnames or ())
    missing = sorted(required_columns - actual)
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")


def parse_timestep(value: str, path: Path, row_number: int) -> int:
    try:
        step = float(value)
    except ValueError as exc:
        raise ValueError(
            f"{path}: row {row_number} has non-numeric ptnum {value!r}"
        ) from exc

    if not step.is_integer():
        raise ValueError(f"{path}: row {row_number} has non-integer ptnum {value!r}")
    return int(step)


def row_vector(
    row: dict[str, str],
    columns: tuple[str, str, str],
    path: Path,
    row_number: int,
) -> Vector:
    try:
        return tuple(float(row[column]) for column in columns)  # type: ignore[return-value]
    except ValueError as exc:
        raise ValueError(
            f"{path}: row {row_number} has non-numeric values in {', '.join(columns)}"
        ) from exc


def tangential_component(position: Vector, velocity: Vector) -> float:
    x, _, z = position
    radius = math.hypot(x, z)
    if radius == 0.0:
        raise ValueError("Cannot compute tangential component at radius 0")

    e_theta = (-z / radius, 0.0, x / radius)
    return sum(velocity_i * tangent_i for velocity_i, tangent_i in zip(velocity, e_theta))


def read_velocity_series(
    path: Path,
    label: str,
    velocity_columns: tuple[str, str, str],
    required_columns: set[str],
    reverse_input_order: bool,
) -> VelocitySeries:
    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found: {path}")

    rows: list[tuple[int, float]] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        require_columns(path, reader.fieldnames, required_columns)
        for row_number, row in enumerate(reader, start=2):
            step = parse_timestep(row["ptnum"], path, row_number)
            position = row_vector(row, POSITION_COLUMNS, path, row_number)
            velocity = row_vector(row, velocity_columns, path, row_number)
            rows.append((step, tangential_component(position, velocity)))

    if not rows:
        raise ValueError(f"{path} has no data rows")

    if reverse_input_order:
        rows.reverse()

    return VelocitySeries(
        label=label,
        path=path,
        source_steps=tuple(step for step, _ in rows),
        steps=tuple(range(len(rows))),
        tangential_velocities=tuple(value for _, value in rows),
    )


def validate_inputs(
    analytical: VelocitySeries,
    discrete_series: tuple[VelocitySeries, ...],
    dt: float,
    dpi: int,
) -> None:
    if dt <= 0.0:
        raise ValueError(f"--dt must be positive, got {dt}")
    if dpi <= 0:
        raise ValueError(f"--dpi must be positive, got {dpi}")
    for series in discrete_series:
        if analytical.source_steps != series.source_steps:
            raise ValueError(
                "Analytical and discrete CSVs must contain the same timestep indices; "
                f"got {analytical.path.name} source range "
                f"{analytical.source_steps[0]}..{analytical.source_steps[-1]} "
                f"and {series.label} source range "
                f"{series.source_steps[0]}..{series.source_steps[-1]}"
            )
        if analytical.steps != series.steps:
            raise ValueError(
                f"Analytical and {series.label} plotted timestep arrays differ"
            )


def import_matplotlib():
    cache_dir = Path(tempfile.gettempdir()) / "rheidos_tangential_velocity_mpl_cache"
    mpl_config_dir = cache_dir / "mplconfig"
    xdg_cache_dir = cache_dir / "xdg"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    xdg_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache_dir))

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#2f3742",
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": "#8a94a3",
            "grid.alpha": 0.25,
            "grid.linewidth": 0.6,
            "font.size": 9,
            "axes.labelsize": 10,
            "legend.fontsize": 8.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "lines.linewidth": 2.0,
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )
    return plt, FuncFormatter


def timestep_ticks(steps: tuple[int, ...], target_intervals: int = 4) -> list[int]:
    start = steps[0]
    end = steps[-1]
    if start == end:
        return [start]

    rough_interval = max((end - start) / target_intervals, 1.0)
    magnitude = 10 ** math.floor(math.log10(rough_interval))
    interval = magnitude
    for multiple in (1.0, 2.0, 2.5, 5.0, 10.0):
        candidate = multiple * magnitude
        if candidate >= rough_interval:
            interval = candidate
            break
    interval = max(1, int(math.ceil(interval)))

    ticks = [start]
    next_tick = math.ceil((start + 1) / interval) * interval
    while next_tick < end:
        ticks.append(int(next_tick))
        next_tick += interval
    if ticks[-1] != end:
        ticks.append(end)
    return ticks


def apply_time_axes(ax, steps: tuple[int, ...], dt: float, func_formatter) -> None:
    times = [step * dt for step in steps]
    ax.set_xlim(times[0], times[-1])
    step_ticks = timestep_ticks(steps)
    ax.set_xticks([step * dt for step in step_ticks])
    ax.xaxis.set_major_formatter(func_formatter(lambda value, _: f"{value:g}"))
    ax.set_xlabel("time")

    top_ax = ax.secondary_xaxis(
        "top",
        functions=(lambda time: time / dt, lambda step: step * dt),
    )
    top_ax.set_xlabel("timestep")
    top_ax.set_xticks(step_ticks)
    top_ax.xaxis.set_major_formatter(func_formatter(lambda value, _: f"{value:.0f}"))


def series_stats(values: tuple[float, ...]) -> dict[str, float]:
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return {
        "min": min(values),
        "max": max(values),
        "mean": mean,
        "std": math.sqrt(variance),
    }


def plot_tangential_velocity(
    analytical: VelocitySeries,
    coexact: VelocitySeries,
    harmonic: VelocitySeries,
    total: VelocitySeries,
    dt: float,
    output_stem: str,
    dpi: int,
) -> tuple[Path, Path, dict[str, float]]:
    plt, func_formatter = import_matplotlib()
    steps = analytical.steps
    times = [step * dt for step in steps]
    differences = [
        coexact_value - analytical_value
        for analytical_value, coexact_value in zip(
            analytical.tangential_velocities,
            coexact.tangential_velocities,
        )
    ]

    fig, ax = plt.subplots(figsize=(7.2, 4.35))
    ax.plot(
        times,
        analytical.tangential_velocities,
        color="#1f5a93",
        label=r"analytical $u_\theta$",
    )
    ax.plot(
        times,
        coexact.tangential_velocities,
        color="#d65f1f",
        linestyle="--",
        label=r"coexact $u_\theta$",
    )
    ax.plot(
        times,
        harmonic.tangential_velocities,
        color="#2b7a4b",
        linestyle="-.",
        label=r"harmonic $u_\theta$",
    )
    ax.plot(
        times,
        total.tangential_velocities,
        color="#7c3aed",
        linestyle=":",
        label=r"total $u_\theta$",
    )
    apply_time_axes(ax, steps, dt, func_formatter)
    ax.yaxis.set_major_formatter(func_formatter(lambda value, _: f"{value:g}"))
    ax.set_ylabel(r"tangential velocity $u_\theta$")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True)
    fig.tight_layout()

    png_path = SCRIPT_DIR / f"{output_stem}.png"
    pdf_path = SCRIPT_DIR / f"{output_stem}.pdf"
    fig.savefig(png_path, dpi=dpi)
    fig.savefig(pdf_path)
    plt.close(fig)

    analytical_stats = series_stats(analytical.tangential_velocities)
    coexact_stats = series_stats(coexact.tangential_velocities)
    harmonic_stats = series_stats(harmonic.tangential_velocities)
    total_stats = series_stats(total.tangential_velocities)
    rms_difference = math.sqrt(sum(value * value for value in differences) / len(differences))
    stats = {
        "rows": float(len(steps)),
        "source_first_step": float(analytical.source_steps[0]),
        "source_last_step": float(analytical.source_steps[-1]),
        "first_step": float(steps[0]),
        "last_step": float(steps[-1]),
        "first_time": times[0],
        "last_time": times[-1],
        "analytical_mean": analytical_stats["mean"],
        "analytical_std": analytical_stats["std"],
        "analytical_min": analytical_stats["min"],
        "analytical_max": analytical_stats["max"],
        "coexact_mean": coexact_stats["mean"],
        "coexact_std": coexact_stats["std"],
        "coexact_min": coexact_stats["min"],
        "coexact_max": coexact_stats["max"],
        "harmonic_mean": harmonic_stats["mean"],
        "harmonic_std": harmonic_stats["std"],
        "harmonic_min": harmonic_stats["min"],
        "harmonic_max": harmonic_stats["max"],
        "total_mean": total_stats["mean"],
        "total_std": total_stats["std"],
        "total_min": total_stats["min"],
        "total_max": total_stats["max"],
        "mean_difference": coexact_stats["mean"] - analytical_stats["mean"],
        "max_abs_difference": max(abs(value) for value in differences),
        "rms_difference": rms_difference,
    }
    return png_path, pdf_path, stats


def main() -> None:
    args = parse_args()
    reverse_input_order = args.input_order == "reverse"
    analytical = read_velocity_series(
        resolve_input_path(args.analytical_csv),
        "analytical",
        ANALYTICAL_VELOCITY_COLUMNS,
        ANALYTICAL_REQUIRED_COLUMNS,
        reverse_input_order,
    )
    coexact = read_velocity_series(
        resolve_input_path(args.discrete_csv),
        "coexact",
        COEXACT_VELOCITY_COLUMNS,
        DISCRETE_REQUIRED_COLUMNS,
        reverse_input_order,
    )
    harmonic = read_velocity_series(
        resolve_input_path(args.discrete_csv),
        "harmonic",
        HARMONIC_VELOCITY_COLUMNS,
        DISCRETE_REQUIRED_COLUMNS,
        reverse_input_order,
    )
    total = read_velocity_series(
        resolve_input_path(args.discrete_csv),
        "total",
        TOTAL_VELOCITY_COLUMNS,
        DISCRETE_REQUIRED_COLUMNS,
        reverse_input_order,
    )
    validate_inputs(analytical, (coexact, harmonic, total), args.dt, args.dpi)

    png_path, pdf_path, stats = plot_tangential_velocity(
        analytical=analytical,
        coexact=coexact,
        harmonic=harmonic,
        total=total,
        dt=args.dt,
        output_stem=args.output_stem,
        dpi=args.dpi,
    )

    print(
        "Loaded "
        f"{int(stats['rows'])} rows, input order {args.input_order}, "
        f"source ptnums {int(stats['source_first_step'])}..{int(stats['source_last_step'])}, "
        f"plotted timesteps {int(stats['first_step'])}..{int(stats['last_step'])}, "
        f"time {stats['first_time']:.2f}..{stats['last_time']:.2f}"
    )
    print(
        "Analytical tangential velocity: "
        f"mean {stats['analytical_mean']:.9f}, std {stats['analytical_std']:.9f}, "
        f"range {stats['analytical_min']:.9f}..{stats['analytical_max']:.9f}"
    )
    print(
        "Coexact tangential velocity: "
        f"mean {stats['coexact_mean']:.9f}, std {stats['coexact_std']:.9f}, "
        f"range {stats['coexact_min']:.9f}..{stats['coexact_max']:.9f}"
    )
    print(
        "Harmonic tangential velocity: "
        f"mean {stats['harmonic_mean']:.9f}, std {stats['harmonic_std']:.9f}, "
        f"range {stats['harmonic_min']:.9f}..{stats['harmonic_max']:.9f}"
    )
    print(
        "Total tangential velocity: "
        f"mean {stats['total_mean']:.9f}, std {stats['total_std']:.9f}, "
        f"range {stats['total_min']:.9f}..{stats['total_max']:.9f}"
    )
    print(f"Mean coexact - analytical difference: {stats['mean_difference']:.9f}")
    print(f"RMS coexact - analytical difference: {stats['rms_difference']:.9f}")
    print(f"Max abs coexact - analytical difference: {stats['max_abs_difference']:.9f}")
    print(f"Wrote {png_path}")
    print(f"Wrote {pdf_path}")


if __name__ == "__main__":
    main()
