from rheidos.compute import ModuleBase, World, ResourceSpec, shape_of
from ..stream_func import StreamFunctionModule
from .coexact import CoexactHamiltonianProducer

import taichi as ti


class HamiltonianModule(ModuleBase):
    NAME = "HamiltonianModule"

    def __init__(self, world: World, *, scope: str = "") -> None:
        super().__init__(world, scope=scope)

        self.stream_func = world.require(StreamFunctionModule)

        self.H = self.resource(
            "H",
            spec=ResourceSpec(
                kind="taichi_field", dtype=ti.f32, shape=(), allow_none=True
            ),
            doc="Hamiltonian of the point vortex system.",
        )

        coexact_hamiltonian_producer = CoexactHamiltonianProducer(
            psi=self.stream_func.psi,
            omega=self.stream_func.omega,
            H=self.H,
        )

        self.declare_resource(
            self.H,
            deps=(self.stream_func.psi, self.stream_func.omega),
            producer=coexact_hamiltonian_producer,
        )
