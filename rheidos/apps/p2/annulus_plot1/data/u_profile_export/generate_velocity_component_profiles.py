#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import os
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


SIDELEN = 0.1
EPS = 1.0e-14

OUTPUT_COLUMNS = (
    "sidelen",
    "component",
    "theta_file",
    "theta",
    "Rin",
    "Rout",
    "r_min_plot",
    "r_max_plot",
    "in_plot_window",
    "ptnum",
    "r",
    "P_x",
    "P_y",
    "P_z",
    "u_exact",
    "u_numerical",
    "u_error",
    "exact_vx",
    "exact_vy",
    "exact_vz",
    "numerical_vx",
    "numerical_vy",
    "numerical_vz",
    "position_delta",
)


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Export and plot an annulus velocity component profile u(r)."
    )
    parser.add_argument(
        "--analytical",
        type=Path,
        default=script_dir / "analytical" / "0.00_1.00_2.00.csv",
        help="Analytical velocity CSV.",
    )
    parser.add_argument(
        "--discrete",
        type=Path,
        default=script_dir / "discrete" / "0.00_1.00_2.00.csv",
        help="Discrete velocity CSV.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=script_dir / "plots",
        help="Output directory for the clean CSV and plots.",
    )
    parser.add_argument(
        "--sidelen",
        type=float,
        default=SIDELEN,
        help="Boundary side length to exclude from the plotted profile.",
    )
    parser.add_argument("--dpi", type=int, default=300, help="PNG output DPI.")
    parser.add_argument(
        "--component",
        choices=("theta", "r"),
        default="theta",
        help="Annulus velocity component to project and plot.",
    )
    return parser.parse_args()


def configure_matplotlib():
    cache_dir = Path(tempfile.gettempdir()) / "rheidos_u_theta_profile_mpl_cache"
    mpl_config_dir = cache_dir / "mplconfig"
    xdg_cache_dir = cache_dir / "xdg"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    xdg_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache_dir))

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.patches import Wedge

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
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 8.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "lines.linewidth": 2.0,
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )
    return plt, Wedge


def parse_file_params(path: Path) -> Tuple[float, float, float]:
    parts = path.stem.split("_")
    if len(parts) != 3:
        raise ValueError(f"Expected '<theta>_<Rin>_<Rout>.csv', got {path.name!r}")
    return tuple(float(part) for part in parts)  # type: ignore[return-value]


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


Vector = Tuple[float, float, float]


def row_vector(row: Dict[str, str], columns: Sequence[str]) -> Vector:
    return tuple(float(row[column]) for column in columns)  # type: ignore[return-value]


def vector_dot(left: Vector, right: Vector) -> float:
    return sum(left_value * right_value for left_value, right_value in zip(left, right))


def vector_norm(vector: Vector) -> float:
    return math.sqrt(sum(value * value for value in vector))


def component_unit_vector(position: Vector, component: str) -> Vector:
    radial_length = math.hypot(float(position[0]), float(position[2]))
    if radial_length <= EPS:
        raise ValueError("Cannot project velocity at zero annulus radius")
    if component == "r":
        return (position[0] / radial_length, 0.0, position[2] / radial_length)
    if component == "theta":
        return (-position[2] / radial_length, 0.0, position[0] / radial_length)
    raise ValueError(f"Unknown component {component!r}")


def project_component(position: Vector, vector: Vector, component: str) -> float:
    return vector_dot(vector, component_unit_vector(position, component))


def build_rows(
    analytical_path: Path, discrete_path: Path, sidelen: float, component: str
) -> Tuple[List[Dict[str, float | int | str]], Dict[str, float | str]]:
    theta_file, rin, rout = parse_file_params(analytical_path)
    discrete_params = parse_file_params(discrete_path)
    if (theta_file, rin, rout) != discrete_params:
        raise ValueError(
            f"Filename parameters do not match: {analytical_path.name} vs "
            f"{discrete_path.name}"
        )

    analytical_rows, analytical_fields = read_csv_rows(analytical_path)
    discrete_rows, discrete_fields = read_csv_rows(discrete_path)
    shared_columns = ("ptnum", "theta", "P_x", "P_y", "P_z", "Rin", "Rout")
    require_columns(
        analytical_path,
        analytical_fields,
        shared_columns
        + (
            "analytic_vortex_core_vel_x",
            "analytic_vortex_core_vel_y",
            "analytic_vortex_core_vel_z",
        ),
    )
    require_columns(
        discrete_path,
        discrete_fields,
        shared_columns + ("vortex_core_vel_x", "vortex_core_vel_y", "vortex_core_vel_z"),
    )
    if len(analytical_rows) != len(discrete_rows):
        raise ValueError(
            "Analytical/discrete row counts differ: "
            f"{len(analytical_rows)} vs {len(discrete_rows)}"
        )

    r_min_plot = rin + sidelen
    r_max_plot = rout - sidelen
    output_rows: List[Dict[str, float | int | str]] = []

    for analytical_row, discrete_row in zip(analytical_rows, discrete_rows):
        ptnum = int(float(analytical_row["ptnum"]))
        discrete_ptnum = int(float(discrete_row["ptnum"]))
        if ptnum != discrete_ptnum:
            raise ValueError(f"Mismatched ptnum: {ptnum} vs {discrete_ptnum}")

        analytical_position = row_vector(analytical_row, ("P_x", "P_y", "P_z"))
        discrete_position = row_vector(discrete_row, ("P_x", "P_y", "P_z"))
        position = tuple(
            0.5 * (analytical_value + discrete_value)
            for analytical_value, discrete_value in zip(analytical_position, discrete_position)
        )
        position_delta = vector_norm(
            tuple(
                analytical_value - discrete_value
                for analytical_value, discrete_value in zip(
                    analytical_position, discrete_position
                )
            )
        )
        radius = vector_norm(position)

        exact_vector = row_vector(
            analytical_row,
            (
                "analytic_vortex_core_vel_x",
                "analytic_vortex_core_vel_y",
                "analytic_vortex_core_vel_z",
            ),
        )
        numerical_vector = row_vector(
            discrete_row,
            ("vortex_core_vel_x", "vortex_core_vel_y", "vortex_core_vel_z"),
        )
        u_exact = project_component(position, exact_vector, component)
        u_numerical = project_component(position, numerical_vector, component)

        output_rows.append(
            {
                "sidelen": sidelen,
                "component": component,
                "theta_file": theta_file,
                "theta": 0.5
                * (float(analytical_row["theta"]) + float(discrete_row["theta"])),
                "Rin": rin,
                "Rout": rout,
                "r_min_plot": r_min_plot,
                "r_max_plot": r_max_plot,
                "in_plot_window": int(r_min_plot <= radius <= r_max_plot),
                "ptnum": ptnum,
                "r": radius,
                "P_x": float(position[0]),
                "P_y": float(position[1]),
                "P_z": float(position[2]),
                "u_exact": u_exact,
                "u_numerical": u_numerical,
                "u_error": u_numerical - u_exact,
                "exact_vx": float(exact_vector[0]),
                "exact_vy": float(exact_vector[1]),
                "exact_vz": float(exact_vector[2]),
                "numerical_vx": float(numerical_vector[0]),
                "numerical_vy": float(numerical_vector[1]),
                "numerical_vz": float(numerical_vector[2]),
                "position_delta": position_delta,
            }
        )

    output_rows.sort(key=lambda row: (float(row["r"]), int(row["ptnum"])))
    metadata = {
        "theta_file": theta_file,
        "rin": rin,
        "rout": rout,
        "r_min_plot": r_min_plot,
        "r_max_plot": r_max_plot,
        "sidelen": sidelen,
        "component": component,
    }
    return output_rows, metadata


def write_clean_csv(path: Path, rows: Sequence[Dict[str, float | int | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def profile_y_limits(rows: Sequence[Dict[str, float | int | str]]) -> Tuple[float, float]:
    plot_rows = [row for row in rows if int(row["in_plot_window"]) == 1]
    if not plot_rows:
        raise ValueError("No rows are inside the requested plot window")
    y_values = (
        [float(row["u_exact"]) for row in plot_rows]
        + [float(row["u_numerical"]) for row in plot_rows]
    )
    y_pad = 0.08 * max(max(y_values) - min(y_values), EPS)
    return min(y_values) - y_pad, max(y_values) + y_pad


def add_annulus_inset(fig, Wedge, metadata: Dict[str, float | str]) -> None:
    inset = fig.add_axes([0.805, 0.54, 0.155, 0.28])
    rin = float(metadata["rin"])
    rout = float(metadata["rout"])
    r_min_plot = float(metadata["r_min_plot"])
    r_max_plot = float(metadata["r_max_plot"])

    inset.add_patch(
        Wedge(
            (0.0, 0.0),
            rout,
            0.0,
            360.0,
            width=rout - rin,
            facecolor="#d9dde1",
            edgecolor="#7d8791",
            linewidth=0.8,
            alpha=0.92,
        )
    )
    inset.plot([rin, rout], [0.0, 0.0], color="#9aa4ae", linewidth=1.2)
    inset.plot([r_min_plot, r_max_plot], [0.0, 0.0], color="#d65f1f", linewidth=4.0)
    inset.scatter(
        [r_min_plot, r_max_plot],
        [0.0, 0.0],
        color="#d65f1f",
        s=18,
        zorder=3,
    )
    inset.set_xlim(-0.12 * rout, 1.08 * rout)
    inset.set_ylim(-0.56 * rout, 0.56 * rout)
    inset.set_aspect("equal", adjustable="box")
    inset.set_xticks([])
    inset.set_yticks([])
    inset.set_title("plotted band", pad=2, fontsize=8)
    for spine in inset.spines.values():
        spine.set_color("#7d8791")
        spine.set_linewidth(0.8)


def plot_profile(
    rows: Sequence[Dict[str, float | int | str]],
    metadata: Dict[str, float | str],
    out_dir: Path,
    dpi: int,
    y_limits: Tuple[float, float] | None = None,
) -> Tuple[Path, Path]:
    plt, Wedge = configure_matplotlib()
    plot_rows = [row for row in rows if int(row["in_plot_window"]) == 1]
    if not plot_rows:
        raise ValueError("No rows are inside the requested plot window")

    radius = [float(row["r"]) for row in plot_rows]
    u_exact = [float(row["u_exact"]) for row in plot_rows]
    u_numerical = [float(row["u_numerical"]) for row in plot_rows]

    fig, ax = plt.subplots(figsize=(8.6, 4.65))
    ax.axhline(0.0, color="#2f3742", linewidth=0.75, alpha=0.68)
    ax.plot(radius, u_exact, color="#1f5a93", label=r"$u_{\mathrm{exact}}(r)$")
    ax.plot(
        radius,
        u_numerical,
        color="#d65f1f",
        linestyle="--",
        marker="o",
        markevery=18,
        markersize=3.2,
        label=r"$u_{\mathrm{numerical}}(r)$",
    )
    component = str(metadata["component"])
    component_math = r"\theta" if component == "theta" else "r"
    component_title = "angular" if component == "theta" else "radial"
    component_stem = "u_theta" if component == "theta" else "u_r"

    ax.set_xlim(float(metadata["r_min_plot"]), float(metadata["r_max_plot"]))
    ax.set_ylim(*(y_limits or profile_y_limits(rows)))
    ax.set_xlabel("radius r")
    ax.set_ylabel(rf"{component_title} velocity $u_{{{component_math}}}(r)$")
    ax.set_title(rf"Annulus {component_title} velocity profile, $\theta = 0$")
    ax.text(
        0.02,
        0.965,
        (
            f"Rin = {float(metadata['rin']):.1f}, Rout = {float(metadata['rout']):.1f}, "
            f"sidelen = {float(metadata['sidelen']):.1f}"
        ),
        transform=ax.transAxes,
        fontsize=8.5,
        color="#4b5563",
        va="top",
    )
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=2,
        frameon=True,
        framealpha=0.92,
    )
    ax.margins(x=0.0)
    fig.tight_layout(rect=(0.0, 0.08, 0.78, 1.0))
    add_annulus_inset(fig, Wedge, metadata)

    png_path = out_dir / f"{component_stem}_profile_0.00_1.00_2.00.png"
    pdf_path = out_dir / f"{component_stem}_profile_0.00_1.00_2.00.pdf"
    fig.savefig(png_path, dpi=dpi)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path, pdf_path


def validate_rows(
    rows: Sequence[Dict[str, float | int | str]], metadata: Dict[str, float | str]
) -> None:
    if len(rows) != 500:
        raise ValueError(f"Expected 500 exported rows, got {len(rows)}")
    r_min_plot = float(metadata["r_min_plot"])
    r_max_plot = float(metadata["r_max_plot"])
    plot_count = 0
    for index, row in enumerate(rows):
        for column in OUTPUT_COLUMNS:
            value = row[column]
            if isinstance(value, str):
                continue
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"Non-finite value in row {index}, column {column}")
        if float(row["sidelen"]) != float(metadata["sidelen"]):
            raise ValueError(f"Unexpected sidelen in row {index}: {row['sidelen']}")
        if str(row["component"]) != str(metadata["component"]):
            raise ValueError(f"Unexpected component in row {index}: {row['component']}")
        expected = int(r_min_plot <= float(row["r"]) <= r_max_plot)
        if int(row["in_plot_window"]) != expected:
            raise ValueError(f"Bad in_plot_window flag in row {index}")
        plot_count += expected
    if plot_count != 400:
        raise ValueError(f"Expected 400 plotted rows, got {plot_count}")


def main() -> int:
    args = parse_args()
    analytical_path = args.analytical.resolve()
    discrete_path = args.discrete.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, metadata = build_rows(
        analytical_path, discrete_path, args.sidelen, args.component
    )
    validate_rows(rows, metadata)

    component_stem = "u_theta" if args.component == "theta" else "u_r"
    csv_path = out_dir / f"{component_stem}_profile_0.00_1.00_2.00.csv"
    write_clean_csv(csv_path, rows)
    y_limits = None
    if args.component == "r":
        theta_rows, theta_metadata = build_rows(
            analytical_path, discrete_path, args.sidelen, "theta"
        )
        validate_rows(theta_rows, theta_metadata)
        y_limits = profile_y_limits(theta_rows)
    png_path, pdf_path = plot_profile(rows, metadata, out_dir, args.dpi, y_limits)

    max_position_delta = max(float(row["position_delta"]) for row in rows)
    print(f"Annulus {component_stem} profile generation complete")
    print(f"  exported rows: {len(rows)}")
    print(f"  plotted rows: {sum(int(row['in_plot_window']) for row in rows)}")
    print(
        "  radial plot window: "
        f"[{float(metadata['r_min_plot']):.6g}, {float(metadata['r_max_plot']):.6g}]"
    )
    print(f"  max analytical/discrete position delta: {max_position_delta:.3e}")
    print(f"  outputs:\n    {csv_path}\n    {png_path}\n    {pdf_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
