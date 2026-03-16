from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping
import json


@dataclass(frozen=True)
class MeshConfig:
    kind: str = "icosphere"
    subdivisions: int = 2
    radius: float = 1.0


@dataclass(frozen=True)
class SolverConfig:
    backend: str = "auto"  # auto|taichi|scipy|scipy_constrained
    pin_index: int = 0


@dataclass(frozen=True)
class TimeConfig:
    dt: float = 0.01
    substeps: int = 1
    max_hops: int = 32


@dataclass(frozen=True)
class VortexConfig:
    preset: str = "ring"  # ring|dipole|random
    n_vortices: int = 12
    gamma_scale: float = 1.0


@dataclass(frozen=True)
class RenderConfig:
    width: int = 1280
    height: int = 720
    vsync: bool = True
    show_glyphs: bool = True
    show_stream_tint: bool = True
    glyph_scale: float = 0.1
    vortex_radius: float = 0.01


@dataclass(frozen=True)
class SimulationConfig:
    mesh: MeshConfig = field(default_factory=MeshConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)
    time: TimeConfig = field(default_factory=TimeConfig)
    vortex: VortexConfig = field(default_factory=VortexConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    seed: int = 1234

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _dict_get(d: Mapping[str, Any], key: str, default: Any) -> Any:
    val = d.get(key, default)
    return default if val is None else val


def config_from_mapping(data: Mapping[str, Any]) -> SimulationConfig:
    mesh_d = _dict_get(data, "mesh", {})
    solver_d = _dict_get(data, "solver", {})
    time_d = _dict_get(data, "time", {})
    vortex_d = _dict_get(data, "vortex", {})
    render_d = _dict_get(data, "render", {})

    return SimulationConfig(
        mesh=MeshConfig(
            kind=str(_dict_get(mesh_d, "kind", "icosphere")),
            subdivisions=int(_dict_get(mesh_d, "subdivisions", 2)),
            radius=float(_dict_get(mesh_d, "radius", 1.0)),
        ),
        solver=SolverConfig(
            backend=str(_dict_get(solver_d, "backend", "auto")),
            pin_index=int(_dict_get(solver_d, "pin_index", 0)),
        ),
        time=TimeConfig(
            dt=float(_dict_get(time_d, "dt", 0.01)),
            substeps=int(_dict_get(time_d, "substeps", 1)),
            max_hops=int(_dict_get(time_d, "max_hops", 32)),
        ),
        vortex=VortexConfig(
            preset=str(_dict_get(vortex_d, "preset", "ring")),
            n_vortices=int(_dict_get(vortex_d, "n_vortices", 12)),
            gamma_scale=float(_dict_get(vortex_d, "gamma_scale", 1.0)),
        ),
        render=RenderConfig(
            width=int(_dict_get(render_d, "width", 1280)),
            height=int(_dict_get(render_d, "height", 720)),
            vsync=bool(_dict_get(render_d, "vsync", True)),
            show_glyphs=bool(_dict_get(render_d, "show_glyphs", True)),
            show_stream_tint=bool(_dict_get(render_d, "show_stream_tint", True)),
            glyph_scale=float(_dict_get(render_d, "glyph_scale", 0.1)),
            vortex_radius=float(_dict_get(render_d, "vortex_radius", 0.01)),
        ),
        seed=int(_dict_get(data, "seed", 1234)),
    )


def load_config(path: str | Path | None) -> SimulationConfig:
    if path is None:
        return SimulationConfig()

    p = Path(path)
    text = p.read_text(encoding="utf-8")
    suffix = p.suffix.lower()

    if suffix == ".json":
        data = json.loads(text)
        if not isinstance(data, Mapping):
            raise ValueError(f"Config JSON must be an object: {p}")
        return config_from_mapping(data)

    if suffix in {".toml", ".tml"}:
        try:
            import tomllib  # py311+
        except Exception:  # pragma: no cover
            import tomli as tomllib  # type: ignore[no-redef]
        data = tomllib.loads(text)
        if not isinstance(data, Mapping):
            raise ValueError(f"Config TOML must be a table: {p}")
        return config_from_mapping(data)

    raise ValueError(f"Unsupported config extension '{suffix}'. Use .json or .toml")


def apply_overrides(
    cfg: SimulationConfig,
    *,
    steps_dt: float | None = None,
    seed: int | None = None,
    solver_backend: str | None = None,
    no_gui: bool | None = None,
) -> SimulationConfig:
    out = cfg
    if steps_dt is not None:
        out = replace(out, time=replace(out.time, dt=float(steps_dt)))
    if seed is not None:
        out = replace(out, seed=int(seed))
    if solver_backend is not None:
        out = replace(out, solver=replace(out.solver, backend=str(solver_backend)))
    if no_gui is True:
        out = replace(out, render=replace(out.render, vsync=False))
    return out
