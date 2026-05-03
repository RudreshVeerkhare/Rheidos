#!/usr/bin/env python3
"""Generate a radius-vs-time comparison plot for annulus vortex evolution."""

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
DEFAULT_OUTPUT_STEM = "radius_time_evolution_Rin_1_Rout_2"
DEFAULT_VALIDATION_OUTPUT_STEM = "travelled_distance_harmonic_validation_Rin_1_Rout_2"
POSITION_COLUMNS = ("P_x", "P_y", "P_z")
HARMONIC_COLUMNS = ("harmonic_vel_x", "harmonic_vel_y", "harmonic_vel_z")
ANALYTICAL_REQUIRED_COLUMNS = {"ptnum", *POSITION_COLUMNS}
DISCRETE_REQUIRED_COLUMNS = {"ptnum", *POSITION_COLUMNS, *HARMONIC_COLUMNS}

Vector = tuple[float, float, float]


@dataclass(frozen=True)
class EvolutionSeries:
    label: str
    path: Path
    source_steps: tuple[int, ...]
    steps: tuple[int, ...]
    positions: tuple[Vector, ...]
    radii: tuple[float, ...]
    harmonic_velocities: Optional[tuple[Vector, ...]] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot point-vortex radial distance over time for analytical and "
            "discrete annulus simulations."
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
        help=f"Discrete Poisson evolution CSV. Default: {DEFAULT_DISCRETE_CSV}",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.01,
        help="Time step size used to convert timestep index to physical time.",
    )
    parser.add_argument(
        "--rin",
        type=float,
        default=1.0,
        help="Inner annulus radius to mark on the plot.",
    )
    parser.add_argument(
        "--rout",
        type=float,
        default=2.0,
        help="Outer annulus radius to mark on the plot.",
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
        "--validation-output-stem",
        default=DEFAULT_VALIDATION_OUTPUT_STEM,
        help=(
            "Hypothesis-validation output filename stem; PNG and PDF are "
            "written beside this script."
        ),
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
    path: Path, fieldnames: Optional[Iterable[str]], required_columns: set[str]
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


def row_vector(row: dict[str, str], columns: tuple[str, str, str], path: Path, row_number: int) -> Vector:
    try:
        return tuple(float(row[column]) for column in columns)  # type: ignore[return-value]
    except ValueError as exc:
        raise ValueError(
            f"{path}: row {row_number} has non-numeric values in {', '.join(columns)}"
        ) from exc


def read_evolution_series(
    path: Path,
    label: str,
    required_columns: set[str],
    include_harmonic: bool = False,
    reverse_input_order: bool = True,
) -> EvolutionSeries:
    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found: {path}")

    rows: list[tuple[int, Vector, Optional[Vector]]] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        require_columns(path, reader.fieldnames, required_columns)
        for row_number, row in enumerate(reader, start=2):
            step = parse_timestep(row["ptnum"], path, row_number)
            position = row_vector(row, POSITION_COLUMNS, path, row_number)
            harmonic_velocity = (
                row_vector(row, HARMONIC_COLUMNS, path, row_number)
                if include_harmonic
                else None
            )
            rows.append((step, position, harmonic_velocity))

    if not rows:
        raise ValueError(f"{path} has no data rows")

    if reverse_input_order:
        rows.reverse()

    source_steps = tuple(step for step, _, _ in rows)
    steps = tuple(range(len(rows)))
    positions = tuple(position for _, position, _ in rows)
    radii = tuple(math.hypot(position[0], position[2]) for position in positions)
    harmonic_velocities = None
    if include_harmonic:
        harmonic_velocities = tuple(
            harmonic_velocity
            for _, _, harmonic_velocity in rows
            if harmonic_velocity is not None
        )
        if len(harmonic_velocities) != len(rows):
            raise ValueError(f"{path}: missing harmonic velocities in some rows")

    return EvolutionSeries(
        label=label,
        path=path,
        source_steps=source_steps,
        steps=steps,
        positions=positions,
        radii=radii,
        harmonic_velocities=harmonic_velocities,
    )


def validate_inputs(
    analytical: EvolutionSeries,
    discrete: EvolutionSeries,
    dt: float,
    rin: float,
    rout: float,
    dpi: int,
) -> None:
    if dt <= 0.0:
        raise ValueError(f"--dt must be positive, got {dt}")
    if rin >= rout:
        raise ValueError(f"--rin must be smaller than --rout, got {rin} >= {rout}")
    if dpi <= 0:
        raise ValueError(f"--dpi must be positive, got {dpi}")
    if analytical.source_steps != discrete.source_steps:
        raise ValueError(
            "Analytical and discrete CSVs must contain the same timestep indices; "
            f"got {analytical.path.name} source range "
            f"{analytical.source_steps[0]}..{analytical.source_steps[-1]} "
            f"and {discrete.path.name} source range "
            f"{discrete.source_steps[0]}..{discrete.source_steps[-1]}"
        )
    if analytical.steps != discrete.steps:
        raise ValueError("Analytical and discrete plotted timestep arrays differ")
    if discrete.harmonic_velocities is None:
        raise ValueError(f"{discrete.path} does not include harmonic velocity data")


def import_matplotlib():
    cache_dir = Path(tempfile.gettempdir()) / "rheidos_radius_time_mpl_cache"
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


def y_limits_with_boundaries(
    radii: Iterable[float],
    rin: float,
    rout: float,
    padding_fraction: float = 0.06,
) -> tuple[float, float]:
    values = [*radii, rin, rout]
    lower = min(values)
    upper = max(values)
    span = upper - lower
    if span <= 0.0:
        span = max(abs(upper), 1.0)
    padding = span * padding_fraction
    return lower - padding, upper + padding


def y_ticks_with_boundaries(
    radii: Iterable[float],
    rin: float,
    rout: float,
) -> list[float]:
    values = [*radii, rin, rout]
    lower = min(values)
    upper = max(values)
    base_ticks = [lower + (upper - lower) * i / 4.0 for i in range(5)]
    ticks = [*base_ticks, rin, rout]
    return sorted({round(tick, 6) for tick in ticks})


def vector_sub(lhs: Vector, rhs: Vector) -> Vector:
    return tuple(a - b for a, b in zip(lhs, rhs))  # type: ignore[return-value]


def vector_dot(lhs: Vector, rhs: Vector) -> float:
    return sum(a * b for a, b in zip(lhs, rhs))


def vector_norm(vector: Vector) -> float:
    return math.sqrt(vector_dot(vector, vector))


def vector_scale(vector: Vector, scale: float) -> Vector:
    return tuple(component * scale for component in vector)  # type: ignore[return-value]


def cumulative_path_lengths(positions: tuple[Vector, ...]) -> list[float]:
    distances = [0.0]
    total = 0.0
    for index in range(1, len(positions)):
        total += vector_norm(vector_sub(positions[index], positions[index - 1]))
        distances.append(total)
    return distances


def harmonic_accumulations(
    positions: tuple[Vector, ...],
    harmonic_velocities: tuple[Vector, ...],
    dt: float,
) -> tuple[list[float], list[float]]:
    magnitude_distances = [0.0]
    signed_distances = [0.0]

    for index in range(1, len(positions)):
        increment = vector_sub(positions[index], positions[index - 1])
        increment_length = vector_norm(increment)
        travel_dir = (
            vector_scale(increment, 1.0 / increment_length)
            if increment_length > 0.0
            else (0.0, 0.0, 0.0)
        )
        harmonic_velocity = harmonic_velocities[index - 1]
        magnitude_distances.append(
            magnitude_distances[-1] + vector_norm(harmonic_velocity) * dt
        )
        signed_distances.append(
            signed_distances[-1] + vector_dot(harmonic_velocity, travel_dir) * dt
        )

    return magnitude_distances, signed_distances


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


def plot_radius_time(
    analytical: EvolutionSeries,
    discrete: EvolutionSeries,
    dt: float,
    rin: float,
    rout: float,
    output_stem: str,
    dpi: int,
) -> tuple[Path, Path]:
    plt, func_formatter = import_matplotlib()

    steps = analytical.steps
    times = [step * dt for step in steps]
    all_radii = [*analytical.radii, *discrete.radii]

    fig, ax = plt.subplots(figsize=(7.2, 4.35))
    ax.plot(
        times,
        analytical.radii,
        color="#1f5a93",
        label="analytical",
    )
    ax.plot(
        times,
        discrete.radii,
        color="#d65f1f",
        linestyle="--",
        label="discrete Poisson",
    )

    ax.axhline(
        rin,
        color="#2f3742",
        linestyle=":",
        linewidth=1.0,
        alpha=0.9,
        label=rf"$R_{{\mathrm{{in}}}} = {rin:g}$",
    )
    ax.axhline(
        rout,
        color="#2f3742",
        linestyle="-.",
        linewidth=1.0,
        alpha=0.9,
        label=rf"$R_{{\mathrm{{out}}}} = {rout:g}$",
    )

    ax.set_ylim(*y_limits_with_boundaries(all_radii, rin, rout))
    ax.set_yticks(y_ticks_with_boundaries(all_radii, rin, rout))
    ax.yaxis.set_major_formatter(func_formatter(lambda value, _: f"{value:g}"))
    apply_time_axes(ax, steps, dt, func_formatter)
    ax.set_ylabel("radial distance")

    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True)
    fig.tight_layout()

    png_path = SCRIPT_DIR / f"{output_stem}.png"
    pdf_path = SCRIPT_DIR / f"{output_stem}.pdf"
    fig.savefig(png_path, dpi=dpi)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path, pdf_path


def plot_harmonic_validation(
    analytical: EvolutionSeries,
    discrete: EvolutionSeries,
    dt: float,
    output_stem: str,
    dpi: int,
) -> tuple[Path, Path, dict[str, float]]:
    if discrete.harmonic_velocities is None:
        raise ValueError("Discrete series has no harmonic velocity data")

    plt, func_formatter = import_matplotlib()
    steps = analytical.steps
    times = [step * dt for step in steps]
    analytical_distance = cumulative_path_lengths(analytical.positions)
    discrete_distance = cumulative_path_lengths(discrete.positions)
    travelled_distance_delta = [
        discrete_value - analytical_value
        for analytical_value, discrete_value in zip(analytical_distance, discrete_distance)
    ]
    harmonic_magnitude, harmonic_signed = harmonic_accumulations(
        discrete.positions,
        discrete.harmonic_velocities,
        dt,
    )

    fig, ax = plt.subplots(figsize=(7.2, 4.35))
    ax.axhline(0.0, color="#2f3742", linewidth=0.8, alpha=0.65)
    ax.plot(
        times,
        travelled_distance_delta,
        color="#1f5a93",
        label=r"$s_{\mathrm{discrete}} - s_{\mathrm{analytical}}$",
    )
    ax.plot(
        times,
        harmonic_magnitude,
        color="#d65f1f",
        linestyle="--",
        label=r"$\sum ||u_h||\Delta t$",
    )
    ax.plot(
        times,
        harmonic_signed,
        color="#2b7a4b",
        linestyle="-.",
        label=r"$\sum u_h \cdot \hat{t}_{\mathrm{discrete}}\Delta t$",
    )
    apply_time_axes(ax, steps, dt, func_formatter)
    ax.yaxis.set_major_formatter(func_formatter(lambda value, _: f"{value:g}"))
    ax.set_ylabel("travelled distance difference")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True)
    fig.tight_layout()

    png_path = SCRIPT_DIR / f"{output_stem}.png"
    pdf_path = SCRIPT_DIR / f"{output_stem}.pdf"
    fig.savefig(png_path, dpi=dpi)
    fig.savefig(pdf_path)
    plt.close(fig)

    stats = {
        "rows": float(len(steps)),
        "first_step": float(steps[0]),
        "last_step": float(steps[-1]),
        "source_first_step": float(analytical.source_steps[0]),
        "source_last_step": float(analytical.source_steps[-1]),
        "first_time": times[0],
        "last_time": times[-1],
        "initial_position_gap": vector_norm(
            vector_sub(analytical.positions[0], discrete.positions[0])
        ),
        "analytical_distance": analytical_distance[-1],
        "discrete_distance": discrete_distance[-1],
        "distance_delta": travelled_distance_delta[-1],
        "harmonic_magnitude": harmonic_magnitude[-1],
        "harmonic_signed": harmonic_signed[-1],
    }
    return png_path, pdf_path, stats


def main() -> None:
    args = parse_args()
    reverse_input_order = args.input_order == "reverse"
    analytical = read_evolution_series(
        resolve_input_path(args.analytical_csv),
        "analytical",
        ANALYTICAL_REQUIRED_COLUMNS,
        reverse_input_order=reverse_input_order,
    )
    discrete = read_evolution_series(
        resolve_input_path(args.discrete_csv),
        "discrete Poisson",
        DISCRETE_REQUIRED_COLUMNS,
        include_harmonic=True,
        reverse_input_order=reverse_input_order,
    )
    validate_inputs(analytical, discrete, args.dt, args.rin, args.rout, args.dpi)

    radius_png_path, radius_pdf_path = plot_radius_time(
        analytical=analytical,
        discrete=discrete,
        dt=args.dt,
        rin=args.rin,
        rout=args.rout,
        output_stem=args.output_stem,
        dpi=args.dpi,
    )
    validation_png_path, validation_pdf_path, stats = plot_harmonic_validation(
        analytical=analytical,
        discrete=discrete,
        dt=args.dt,
        output_stem=args.validation_output_stem,
        dpi=args.dpi,
    )

    print(
        "Loaded "
        f"{int(stats['rows'])} rows, input order {args.input_order}, "
        f"source ptnums {int(stats['source_first_step'])}..{int(stats['source_last_step'])}, "
        f"plotted timesteps {int(stats['first_step'])}..{int(stats['last_step'])}, "
        f"time {stats['first_time']:.2f}..{stats['last_time']:.2f}"
    )
    print(f"Initial analytical-discrete position gap: {stats['initial_position_gap']:.9f}")
    print(f"Final analytical travelled distance: {stats['analytical_distance']:.9f}")
    print(f"Final discrete travelled distance: {stats['discrete_distance']:.9f}")
    print(
        "Final discrete - analytical travelled distance: "
        f"{stats['distance_delta']:.9f}"
    )
    print(f"Final harmonic magnitude accumulation: {stats['harmonic_magnitude']:.9f}")
    print(f"Final harmonic signed projection accumulation: {stats['harmonic_signed']:.9f}")
    print(f"Wrote {radius_png_path}")
    print(f"Wrote {radius_pdf_path}")
    print(f"Wrote {validation_png_path}")
    print(f"Wrote {validation_pdf_path}")


if __name__ == "__main__":
    main()
