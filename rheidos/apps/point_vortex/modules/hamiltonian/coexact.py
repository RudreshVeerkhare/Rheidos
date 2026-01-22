from dataclasses import dataclass
from typing import Any

import taichi as ti
from rheidos.compute import WiredProducer, ResourceRef, out_field
from rheidos.compute.registry import Registry


@dataclass
class CoexactHamiltonianProducerIO:
    psi: ResourceRef[ti.Field]
    omega: ResourceRef[ti.Field]

    H: ResourceRef[ti.Field] = out_field()


@ti.data_oriented
class CoexactHamiltonianProducer(WiredProducer[CoexactHamiltonianProducerIO]):

    def __init__(
        self,
        psi: ResourceRef[ti.Field],
        omega: ResourceRef[ti.Field],
        H: ResourceRef[ti.Field],
    ) -> None:
        super().__init__(CoexactHamiltonianProducerIO(psi, omega, H))

    @ti.kernel
    def _hamitonian(self, psi: ti.template(), omega: ti.template(), H: ti.template()):
        H[None] = 0.0
        for i in psi:
            ti.atomic_add(H[None], psi[i] * omega[i])
        H[None] = 0.5 * H[None]

    def compute(self, reg: Registry) -> None:
        omega = self.io.omega.get()
        psi = self.io.psi.get()

        H = self.io.H.peek()

        if H is None:
            H = ti.field(dtype=ti.f32, shape=())
            self.io.H.set_buffer(H, bump=False)

        self._hamitonian(psi, omega, H)  # $$H = \frac{1}{2} \psi^T \omega$$
        self.io.H.commit()
