from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from rheidos.sim.interaction import Action, Adapter, Semantic, Signal
from rheidos.utils.taichi_bridge import field_to_numpy

from .system import Charge, PoissonSystem


@dataclass(frozen=True)
class ChargeBatch:
    charges: Sequence[Charge]


class PoissonAdapter(Adapter):
    def __init__(self, system: PoissonSystem, *, instance: str = "domain") -> None:
        super().__init__(name=f"poisson:{instance}")
        self._system = system
        self._instance = instance
        self._dirty = False

        u_signal = Signal(
            system.poisson.u,
            semantic=Semantic(domain="vertex", meaning="scalar", topology=instance, frame="object"),
            reader=field_to_numpy,
        )
        self.register_signal("u", u_signal)

        self.register_action("constraints", Action(ChargeBatch, self._apply_constraints))

    def _apply_constraints(self, batch: ChargeBatch) -> None:
        self._system.apply_charges(batch.charges)
        self._dirty = True

    def compute(self) -> None:
        if not self._dirty:
            return
        self._system.solve()
        self._dirty = False

