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
    @ti.kernel
    def _hamitonian(self, psi: ti.template(), omega: ti.template(), H: ti.template()):
        H[None] = 0.0
        for i in psi:
            ti.atomic_add(H[None], psi[i] * omega[i])
        H[None] = 0.5 * H[None]

    def compute(self, reg: Registry) -> None:
        inputs = self.require_inputs()
        omega = inputs["omega"].get()
        psi = inputs["psi"].get()

        H = self.ensure_outputs(reg)["H"].peek()

        self._hamitonian(psi, omega, H)  # $$H = \frac{1}{2} \psi^T \omega$$
        self.io.H.commit()
