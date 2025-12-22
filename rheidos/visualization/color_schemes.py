from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Protocol, Sequence

import numpy as np

RGBA = tuple[float, float, float, float]


@dataclass(frozen=True)
class LegendTick:
    value: float
    label: str


@dataclass(frozen=True)
class ColorStop:
    position: float  # in [0, 1]
    color: RGBA

    def __post_init__(self) -> None:
        if not 0.0 <= self.position <= 1.0:
            raise ValueError(f"ColorStop position must be in [0,1], got {self.position}")
        if len(self.color) != 4:
            raise ValueError("ColorStop color must be RGBA length 4")
        if any((c < 0.0 or c > 1.0) for c in self.color):
            raise ValueError("ColorStop color components must be within [0,1]")


@dataclass(frozen=True)
class ColorLegend:
    title: str
    ticks: Sequence[LegendTick]
    stops: Sequence[ColorStop]
    units: Optional[str] = None
    description: Optional[str] = None

    def __post_init__(self) -> None:
        if len(self.stops) < 2:
            raise ValueError("Legend requires at least 2 color stops")
        if sorted(s.position for s in self.stops) != [s.position for s in self.stops]:
            raise ValueError("Legend color stops must be sorted by position")


class ColorScheme(Protocol):
    name: str

    def apply(self, values: np.ndarray) -> np.ndarray:
        ...

    def legend(self) -> ColorLegend:
        ...


def _to_array(values: Any, *, dtype: np.dtype[Any] = np.float32) -> np.ndarray:
    arr = np.asarray(values, dtype=dtype)
    if arr.size == 0:
        return arr.reshape((-1,))
    return arr


def _lerp(a: float, b: float, t: np.ndarray) -> np.ndarray:
    return a + (b - a) * t


def _lerp_colors(c0: RGBA, c1: RGBA, t: np.ndarray) -> np.ndarray:
    c0_arr = np.asarray(c0, dtype=np.float32)
    c1_arr = np.asarray(c1, dtype=np.float32)
    return _lerp(c0_arr[None, :], c1_arr[None, :], t[:, None])


class DivergingColorScheme:
    """
    Blue-white-red diverging map for signed fields.
    """

    def __init__(self, name: str = "diverging", max_abs: float = 1.0) -> None:
        self.name = name
        self.max_abs = max(1e-8, float(max_abs))
        self._legend = ColorLegend(
            title="Signed magnitude",
            ticks=(
                LegendTick(value=-self.max_abs, label=f"-{self.max_abs:g}"),
                LegendTick(value=0.0, label="0"),
                LegendTick(value=self.max_abs, label=f"{self.max_abs:g}"),
            ),
            stops=(
                ColorStop(0.0, (0.231, 0.298, 0.753, 1.0)),
                ColorStop(0.5, (0.865, 0.865, 0.865, 1.0)),
                ColorStop(1.0, (0.706, 0.016, 0.150, 1.0)),
            ),
            description="Blue/white/red diverging map centered at 0.",
        )

    def apply(self, values: np.ndarray) -> np.ndarray:
        arr = _to_array(values, dtype=np.float32)
        flat = arr.reshape(-1)
        norm = 0.5 + 0.5 * np.clip(flat / self.max_abs, -1.0, 1.0)

        colors = np.empty((flat.shape[0], 4), dtype=np.float32)
        mid_mask = norm <= 0.5
        if np.any(mid_mask):
            t = norm[mid_mask] / 0.5
            colors[mid_mask] = _lerp_colors(self._legend.stops[0].color, self._legend.stops[1].color, t)
        high_mask = ~mid_mask
        if np.any(high_mask):
            t = (norm[high_mask] - 0.5) / 0.5
            colors[high_mask] = _lerp_colors(self._legend.stops[1].color, self._legend.stops[2].color, t)
        return colors.reshape(arr.shape + (4,))

    def legend(self) -> ColorLegend:
        return self._legend


class SequentialColorScheme:
    """
    Dark-to-bright sequential map for non-negative magnitudes.
    """

    def __init__(self, name: str = "sequential", max_value: float = 1.0) -> None:
        self.name = name
        self.max_value = max(1e-8, float(max_value))
        self._legend = ColorLegend(
            title="Magnitude",
            ticks=(
                LegendTick(value=0.0, label="0"),
                LegendTick(value=self.max_value * 0.5, label=f"{0.5 * self.max_value:g}"),
                LegendTick(value=self.max_value, label=f"{self.max_value:g}"),
            ),
            stops=(
                ColorStop(0.0, (0.062, 0.066, 0.109, 1.0)),
                ColorStop(1.0, (0.992, 0.815, 0.274, 1.0)),
            ),
            description="Sequential map from dark navy to warm yellow.",
        )

    def apply(self, values: np.ndarray) -> np.ndarray:
        arr = _to_array(values, dtype=np.float32)
        flat = arr.reshape(-1)
        norm = np.clip(flat / self.max_value, 0.0, 1.0)
        colors = _lerp_colors(self._legend.stops[0].color, self._legend.stops[1].color, norm)
        return colors.reshape(arr.shape + (4,))

    def legend(self) -> ColorLegend:
        return self._legend


class CategoricalColorScheme:
    """
    Palette for discrete labels or categorical fields.
    """

    def __init__(
        self,
        name: str = "categorical",
        palette: Optional[Sequence[RGBA]] = None,
        title: str = "Category",
    ) -> None:
        self.name = name
        self._palette = tuple(
            palette
            or (
                (0.894, 0.102, 0.110, 1.0),
                (0.215, 0.494, 0.721, 1.0),
                (0.302, 0.686, 0.290, 1.0),
                (0.596, 0.306, 0.639, 1.0),
                (1.0, 0.498, 0.0, 1.0),
                (1.0, 1.0, 0.2, 1.0),
            )
        )
        self._legend = ColorLegend(
            title=title,
            ticks=tuple(LegendTick(value=float(i), label=str(i)) for i in range(len(self._palette))),
            stops=tuple(ColorStop(i / max(1, len(self._palette) - 1), c) for i, c in enumerate(self._palette)),
            description="Categorical palette cycling over labels.",
        )

    def apply(self, values: np.ndarray) -> np.ndarray:
        arr = _to_array(values, dtype=np.int32)
        if np.any(arr < 0):
            raise ValueError("CategoricalColorScheme expects non-negative integer labels")
        flat = arr.reshape(-1)
        colors = np.empty((flat.shape[0], 4), dtype=np.float32)
        palette = np.asarray(self._palette, dtype=np.float32)
        for idx, label in enumerate(flat):
            colors[idx] = palette[int(label) % len(palette)]
        return colors.reshape(arr.shape + (4,))

    def legend(self) -> ColorLegend:
        return self._legend


_COLOR_SCHEMES: Dict[str, ColorScheme] = {}


def register_color_scheme(name: str, scheme: ColorScheme, *, overwrite: bool = False) -> None:
    if not overwrite and name in _COLOR_SCHEMES:
        raise ValueError(f"Color scheme '{name}' already registered")
    _COLOR_SCHEMES[name] = scheme


def create_color_scheme(name: str) -> ColorScheme:
    try:
        return _COLOR_SCHEMES[name]
    except KeyError as exc:
        raise KeyError(f"Unknown color scheme '{name}'") from exc


def list_color_schemes() -> Sequence[str]:
    return sorted(_COLOR_SCHEMES.keys())


def _register_builtins() -> None:
    register_color_scheme("diverging", DivergingColorScheme(), overwrite=True)
    register_color_scheme("sequential", SequentialColorScheme(), overwrite=True)
    register_color_scheme("categorical", CategoricalColorScheme(), overwrite=True)


_register_builtins()

__all__ = [
    "ColorLegend",
    "ColorScheme",
    "ColorStop",
    "LegendTick",
    "DivergingColorScheme",
    "SequentialColorScheme",
    "CategoricalColorScheme",
    "register_color_scheme",
    "create_color_scheme",
    "list_color_schemes",
]
