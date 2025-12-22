from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Generic, Mapping, Optional, Protocol, Sequence, TypeVar

import numpy as np

ArrayLike = Any
SimulationFactory = Callable[..., "Simulation"]
SampleT = TypeVar("SampleT", bound="FieldSample")
Provider = Callable[[], Optional[SampleT]]


@dataclass
class ArraySpec:
    """
    Runtime shape/dtype contract for arrays exposed by simulations.

    Shapes support dynamic dimensions via None. Validation raises ValueError on
    mismatch; dtype is enforced via astype(copy=False) to keep consumers honest.
    """

    name: str
    shape: Sequence[Optional[int]]
    dtype: np.dtype[Any]
    allow_empty: bool = True

    def validate(self, array: ArrayLike) -> np.ndarray:
        arr = np.asarray(array, dtype=self.dtype)
        expected_rank = len(self.shape)
        if arr.ndim != expected_rank:
            raise ValueError(f"{self.name}: expected rank {expected_rank}, got {arr.ndim}")
        if arr.size == 0 and self.allow_empty:
            return arr
        for idx, expected in enumerate(self.shape):
            if expected is None:
                continue
            if arr.shape[idx] != expected:
                raise ValueError(
                    f"{self.name}: expected dim {idx} == {expected}, got {arr.shape[idx]}"
                )
        return arr


@dataclass
class VectorFieldSample:
    """
    Uniform contract for vector field data consumed by generic views.
    """

    positions: np.ndarray  # (N, 3) float32
    vectors: np.ndarray  # (N, 3) float32
    magnitudes: Optional[np.ndarray] = None  # (N,) float32, optional precomputed norms
    dirty: bool = True

    def validate(self) -> None:
        pos_spec = ArraySpec("positions", (None, 3), np.float32, allow_empty=True)
        vec_spec = ArraySpec("vectors", (None, 3), np.float32, allow_empty=True)
        mag_spec = ArraySpec("magnitudes", (None,), np.float32, allow_empty=True)

        self.positions = pos_spec.validate(self.positions)
        self.vectors = vec_spec.validate(self.vectors)
        if self.positions.shape[0] != self.vectors.shape[0]:
            raise ValueError(
                f"positions/vectors length mismatch: {self.positions.shape[0]} vs {self.vectors.shape[0]}"
            )
        if self.magnitudes is not None:
            self.magnitudes = mag_spec.validate(self.magnitudes)
            if self.magnitudes.shape[0] != self.positions.shape[0]:
                raise ValueError(
                    f"positions/magnitudes length mismatch: {self.positions.shape[0]} vs {self.magnitudes.shape[0]}"
                )


@dataclass
class ScalarFieldSample:
    """
    Contract for scalar grids or flattened samples to be textured/colored.
    """

    values: np.ndarray  # (...,) float32
    uvs: Optional[np.ndarray] = None  # (N, 2) float32 if sampling irregularly
    dirty: bool = True

    def validate(self) -> None:
        val_spec = ArraySpec("values", tuple([None] * np.asarray(self.values).ndim), np.float32)
        self.values = val_spec.validate(self.values)
        if self.uvs is not None:
            uv_spec = ArraySpec("uvs", (None, 2), np.float32, allow_empty=True)
            self.uvs = uv_spec.validate(self.uvs)
            flat_values = self.values.reshape(-1)
            if self.uvs.shape[0] != flat_values.shape[0]:
                raise ValueError(
                    f"uvs length {self.uvs.shape[0]} does not match scalar sample size {flat_values.shape[0]}"
                )


@dataclass
class SimulationState:
    """
    Shared state container for simulations.

    Stores immutable config, runtime buffers, per-buffer dirty flags, and arbitrary
    metadata. Buffers can be validated against ArraySpec at set-time.
    """

    config: Dict[str, Any] = field(default_factory=dict)
    buffers: Dict[str, np.ndarray] = field(default_factory=dict)
    dirty: Dict[str, bool] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def set_buffer(self, key: str, value: ArrayLike, spec: Optional[ArraySpec] = None) -> None:
        arr = np.asarray(value) if spec is None else spec.validate(value)
        self.buffers[key] = arr
        self.dirty[key] = True

    def get_buffer(self, key: str) -> Optional[np.ndarray]:
        return self.buffers.get(key)

    def mark_dirty(self, key: str, dirty: bool = True) -> None:
        if key not in self.buffers:
            raise KeyError(f"Unknown buffer '{key}'")
        self.dirty[key] = dirty

    def mark_all_clean(self) -> None:
        for k in list(self.dirty.keys()):
            self.dirty[k] = False

    def is_dirty(self, key: str) -> bool:
        return bool(self.dirty.get(key, False))

    def dirty_buffers(self) -> Dict[str, np.ndarray]:
        return {k: v for k, v in self.buffers.items() if self.dirty.get(k, False)}


@dataclass
class FieldMeta:
    """Metadata describing a vector/scalar field for UI/legend selection."""

    field_id: str
    label: str
    units: Optional[str] = None
    description: Optional[str] = None


class FieldSample(Protocol):
    def validate(self) -> None:
        ...


@dataclass
class FieldInfo(Generic[SampleT]):
    """Pair of metadata and provider callable for a simulation field."""

    meta: FieldMeta
    provider: Provider[SampleT]

    def fetch(self) -> Optional[SampleT]:
        return self.provider()


class FieldRegistry(Generic[SampleT]):
    """
    Registry for vector/scalar fields on a simulation.

    Provides metadata for enumeration and provider access for rendering.
    """

    def __init__(self) -> None:
        self._fields: Dict[str, FieldInfo[SampleT]] = {}

    def register(self, meta: FieldMeta, provider: Provider[SampleT], *, overwrite: bool = False) -> None:
        if not overwrite and meta.field_id in self._fields:
            raise ValueError(f"Field '{meta.field_id}' already registered")
        self._fields[meta.field_id] = FieldInfo(meta=meta, provider=provider)

    def get(self, field_id: str) -> Optional[FieldInfo[SampleT]]:
        return self._fields.get(field_id)

    def items(self) -> Mapping[str, FieldInfo[SampleT]]:
        return dict(self._fields)

    def __len__(self) -> int:
        return len(self._fields)

    def __iter__(self):
        return iter(self._fields.items())


class Simulation(Protocol):
    """
    Minimal simulation lifecycle + data access contract.

    Implementations should populate a SimulationState and expose read-only views
    of positions/vectors/scalars for visualization without leaking internal
    sim-specific types.
    """

    name: str

    def configure(self, cfg: Optional[Mapping[str, Any]] = None) -> None:
        ...

    def reset(self, seed: Optional[int] = None) -> None:
        ...

    def step(self, dt: float) -> None:
        ...

    def get_state(self) -> SimulationState:
        ...

    def get_positions_view(self) -> Optional[np.ndarray]:
        ...

    def get_vector_fields(self) -> Mapping[str, FieldInfo[VectorFieldSample]]:
        ...

    def get_scalar_fields(self) -> Mapping[str, FieldInfo[ScalarFieldSample]]:
        ...

    def get_metadata(self) -> Mapping[str, Any]:
        ...


_SIMULATION_REGISTRY: Dict[str, SimulationFactory] = {}


def register_simulation(name: str, factory: SimulationFactory, *, overwrite: bool = False) -> None:
    """Register a simulation factory under a string key."""
    if not overwrite and name in _SIMULATION_REGISTRY:
        raise ValueError(f"Simulation '{name}' already registered")
    _SIMULATION_REGISTRY[name] = factory


def create_simulation(name: str, **kwargs: Any) -> Simulation:
    """Instantiate a simulation by registry name."""
    try:
        factory = _SIMULATION_REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"Unknown simulation '{name}'") from exc
    return factory(**kwargs)


def list_simulations() -> Sequence[str]:
    return sorted(_SIMULATION_REGISTRY.keys())


__all__ = [
    "ArraySpec",
    "Simulation",
    "SimulationFactory",
    "SimulationState",
    "VectorFieldSample",
    "ScalarFieldSample",
    "FieldMeta",
    "FieldInfo",
    "FieldRegistry",
    "register_simulation",
    "create_simulation",
    "list_simulations",
]
