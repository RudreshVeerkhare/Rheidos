from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import time
from typing import Any, Callable, Deque, Dict, Generic, Optional, TypeVar

from ..abc.controller import Controller
from ..abc.observer import Observer
from ..abc.view import View
from ..compute.resource import ResourceRef

T = TypeVar("T")
P = TypeVar("P")


@dataclass(frozen=True)
class Semantic:
    domain: str
    meaning: str
    topology: Optional[str] = None
    frame: Optional[str] = None
    units: Optional[str] = None


@dataclass(frozen=True)
class Snapshot(Generic[T]):
    version: int
    payload: T
    timestamp: float
    topology: Optional[str] = None


class Signal(Generic[T]):
    def __init__(
        self,
        ref: ResourceRef[Any],
        *,
        semantic: Optional[Semantic] = None,
        reader: Optional[Callable[[Any], T]] = None,
    ) -> None:
        self.ref = ref
        self.semantic = semantic
        self._reader = reader or (lambda value: value)

    def version(self, *, ensure: bool = False) -> int:
        if ensure:
            self.ref.ensure()
        return self.ref.version()

    def read_snapshot(self, *, ensure: bool = False) -> Snapshot[T]:
        if ensure:
            self.ref.ensure()
        payload = self._reader(self.ref.get(ensure=False))
        version = self.ref.version()
        topology = self.semantic.topology if self.semantic else None
        return Snapshot(version=version, payload=payload, timestamp=time.time(), topology=topology)


class Action(Generic[P]):
    def __init__(
        self,
        payload_type: type[P] | tuple[type[P], ...],
        handler: Callable[[P], None],
    ) -> None:
        self.payload_type = payload_type
        self._handler = handler

    def invoke(self, payload: P) -> None:
        if self.payload_type is not object and not isinstance(payload, self.payload_type):
            expected = self.payload_type
            raise TypeError(f"Action payload {type(payload)} does not match {expected}")
        self._handler(payload)


class Adapter:
    def __init__(self, *, name: str) -> None:
        self.name = name
        self.signals: Dict[str, Signal[Any]] = {}
        self.actions: Dict[str, Action[Any]] = {}
        self.topologies: Dict[str, Any] = {}

    def register_signal(self, name: str, signal: Signal[Any]) -> None:
        if name in self.signals:
            raise KeyError(f"Signal '{name}' already registered.")
        self.signals[name] = signal

    def register_action(self, name: str, action: Action[Any]) -> None:
        if name in self.actions:
            raise KeyError(f"Action '{name}' already registered.")
        self.actions[name] = action

    def compute(self) -> None:
        pass


class ComputeScheduler(Observer):
    def __init__(
        self,
        adapter: Adapter,
        *,
        name: Optional[str] = None,
        sort: int = -20,
        always_tick: bool = False,
    ) -> None:
        super().__init__(name=name or f"{adapter.name}-scheduler", sort=sort)
        self._adapter = adapter
        self._always_tick = bool(always_tick)
        self._pending: Deque[tuple[str, Any]] = deque()
        self._dirty = False

    def enqueue(self, action_name: str, payload: Any) -> None:
        if action_name not in self._adapter.actions:
            raise KeyError(f"Unknown action '{action_name}'")
        self._pending.append((action_name, payload))

    def update(self, dt: float) -> None:
        self.flush()

    def flush(self) -> None:
        while self._pending:
            name, payload = self._pending.popleft()
            action = self._adapter.actions[name]
            action.invoke(payload)
            self._dirty = True

        if self._dirty or self._always_tick:
            self._adapter.compute()
            self._dirty = False


@dataclass(frozen=True)
class BindingRule:
    domain: str
    meaning: str
    factory: Callable[[Signal[Any], Any, Adapter], Any]

    def matches(self, semantic: Semantic) -> bool:
        return semantic.domain == self.domain and semantic.meaning == self.meaning


class AdapterBinder:
    def __init__(self, engine, *, rules: Optional[list[BindingRule]] = None) -> None:
        self._engine = engine
        self._rules = list(rules or [])

    def add_rule(self, rule: BindingRule) -> None:
        self._rules.append(rule)

    def bind(self, adapter: Adapter, *, topologies: Optional[Dict[str, Any]] = None) -> list[Any]:
        topo_map = dict(topologies or {})
        adapter.topologies = topo_map
        created: list[Any] = []

        for signal in adapter.signals.values():
            semantic = signal.semantic
            if semantic is None:
                continue
            for rule in self._rules:
                if not rule.matches(semantic):
                    continue
                topo = topo_map.get(semantic.topology) if semantic.topology else None
                binding = rule.factory(signal, topo, adapter)
                self._attach(binding)
                created.append(binding)
                break

        return created

    def _attach(self, binding: Any) -> None:
        if isinstance(binding, Observer):
            self._engine.add_observer(binding)
            return
        if isinstance(binding, View):
            self._engine.add_view(binding)
            return
        if isinstance(binding, Controller):
            self._engine.add_controller(binding)
            return
        raise TypeError(f"Unsupported binding type: {type(binding)}")
