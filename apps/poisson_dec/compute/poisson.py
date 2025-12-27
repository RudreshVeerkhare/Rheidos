from rheidos.compute.registry import Registry
from rheidos.compute.resource import ResourceSpec, ResourceRef, ShapeFn, Shape
from rheidos.compute.world import World, ModuleBase
from rheidos.compute.wiring import out_field, WiredProducer
import taichi as ti

from .mesh import MeshModule
from .dec import DECModule
from dataclasses import dataclass
from typing import Any, Optional

@dataclass
class SolvePoissonDirichletIO:
    E_verts: ResourceRef[Any]
    w: ResourceRef[Any]
    mask: ResourceRef[Any]
    value: ResourceRef[Any]
    u: ResourceRef[Any] = out_field()

    @classmethod
    def from_modules(cls, mesh: "MeshModule", dec: "DECModule", poisson: "PoissonSolverModule") -> "SolvePoissonDirichletIO":
        return cls(
            E_verts=mesh.E_verts,
            w=dec.star1,
            mask=poisson.constraint_mask,
            value=poisson.constraint_value,
            u=poisson.u,
        )

@ti.data_oriented
class SolvePoissonDirichlet(WiredProducer[SolvePoissonDirichletIO]):
    """
    Harmonic interpolation with Dirichlet constraints using cotan stiffness.

    Solve K u = 0 on free vertices, u fixed on constrained vertices, implemented by
    turning constrained rows into identity rows: (A u)_i = u_i, b_i = val_i.

    Implementation: CG on free DOFs, with masked dot products.
    """

    def __init__(self, io: SolvePoissonDirichletIO) -> None:
        super().__init__(io)
        self._dot_out = ti.field(dtype=ti.f32, shape=())
        # cached scratch
        self._nV: int = 0
        self._b: Optional[Any] = None
        self._r: Optional[Any] = None
        self._p: Optional[Any] = None
        self._Ap: Optional[Any] = None
        self._Ax: Optional[Any] = None

    def _ensure_scratch(self, nV: int) -> None:
        if nV == self._nV and self._b is not None:
            return
        self._nV = nV
        self._b = ti.field(dtype=ti.f32, shape=(nV,))
        self._r = ti.field(dtype=ti.f32, shape=(nV,))
        self._p = ti.field(dtype=ti.f32, shape=(nV,))
        self._Ap = ti.field(dtype=ti.f32, shape=(nV,))
        self._Ax = ti.field(dtype=ti.f32, shape=(nV,))

    @ti.kernel
    def _set_initial(self, x: ti.template(), mask: ti.template(), val: ti.template()):
        for i in x:
            x[i] = 0.0
        for i in x:
            if mask[i] == 1:
                x[i] = val[i]

    @ti.kernel
    def _compute_b(self, b: ti.template(), mask: ti.template(), val: ti.template()):
        for i in b:
            b[i] = 0.0
        for i in b:
            if mask[i] == 1:
                b[i] = val[i]

    @ti.kernel
    def _apply_A(self, x: ti.template(), y: ti.template(), E_verts: ti.template(), w: ti.template(), mask: ti.template()):
        # y = Kx, but constrained rows become identity: y_i = x_i
        for i in y:
            y[i] = 0.0

        for e in E_verts:
            i = E_verts[e][0]
            j = E_verts[e][1]
            we = w[e]
            ti.atomic_add(y[i], we * (x[i] - x[j]))
            ti.atomic_add(y[j], we * (x[j] - x[i]))

        for i in y:
            if mask[i] == 1:
                y[i] = x[i]

    @ti.kernel
    def _compute_r(self, b: ti.template(), Ax: ti.template(), r: ti.template(), mask: ti.template()):
        for i in r:
            if mask[i] == 0:
                r[i] = b[i] - Ax[i]
            else:
                r[i] = 0.0

    @ti.kernel
    def _copy(self, src: ti.template(), dst: ti.template()):
        for i in dst:
            dst[i] = src[i]

    @ti.kernel
    def _x_plus_alpha_p_free(self, x: ti.template(), p: ti.template(), alpha: ti.f32, mask: ti.template(), val: ti.template()):
        for i in x:
            if mask[i] == 0:
                x[i] += alpha * p[i]
            else:
                x[i] = val[i]

    @ti.kernel
    def _r_minus_alpha_Ap_free(self, r: ti.template(), Ap: ti.template(), alpha: ti.f32, mask: ti.template()):
        for i in r:
            if mask[i] == 0:
                r[i] -= alpha * Ap[i]
            else:
                r[i] = 0.0

    @ti.kernel
    def _p_update_free(self, p: ti.template(), r: ti.template(), beta: ti.f32, mask: ti.template()):
        for i in p:
            if mask[i] == 0:
                p[i] = r[i] + beta * p[i]
            else:
                p[i] = 0.0

    @ti.kernel
    def _dot_free(self, a: ti.template(), b: ti.template(), mask: ti.template(), out: ti.template()):
        out[None] = 0.0
        for i in a:
            if mask[i] == 0:
                ti.atomic_add(out[None], a[i] * b[i])

    def compute(self, reg: Registry) -> None:
        io = self.io
        E = io.E_verts.get(ensure=False)
        w = io.w.get(ensure=False)
        mask = io.mask.get(ensure=False)
        val = io.value.get(ensure=False)

        if E is None or w is None or mask is None or val is None:
            raise RuntimeError("Missing inputs for Poisson solve.")

        nV = int(mask.shape[0])

        u = io.u.get(ensure=False)
        if u is None or u.shape != (nV,):
            u = ti.field(dtype=ti.f32, shape=(nV,))
            io.u.set_buffer(u, bump=False)

        self._ensure_scratch(nV)
        # Assert-narrowing instead of cast(Any, ...)
        assert self._b is not None and self._r is not None and self._p is not None
        assert self._Ap is not None and self._Ax is not None
        b = self._b
        r = self._r
        p = self._p
        Ap = self._Ap
        Ax = self._Ax

        x = u

        self._set_initial(x, mask, val)
        self._compute_b(b, mask, val)

        self._apply_A(x, Ax, E, w, mask)
        self._compute_r(b, Ax, r, mask)
        self._copy(r, p)

        self._dot_free(r, r, mask, self._dot_out)
        rr = float(self._dot_out[None])
        rr0 = rr
        if rr0 < 1e-30:
            io.u.commit()
            return

        max_iter = 800
        tol = 1e-6

        for _ in range(max_iter):
            self._apply_A(p, Ap, E, w, mask)

            self._dot_free(p, Ap, mask, self._dot_out)
            pAp = float(self._dot_out[None])
            if abs(pAp) < 1e-30:
                break

            alpha = rr / pAp
            self._x_plus_alpha_p_free(x, p, alpha, mask, val)
            self._r_minus_alpha_Ap_free(r, Ap, alpha, mask)

            self._dot_free(r, r, mask, self._dot_out)
            rr_new = float(self._dot_out[None])

            if rr_new <= (tol * tol) * rr0:
                rr = rr_new
                break

            beta = rr_new / rr
            self._p_update_free(p, r, beta, mask)
            rr = rr_new

        io.u.commit()

def _shape_of(ref: ResourceRef[Any]) -> ShapeFn:
    def fn(reg: Registry) -> Optional[Shape]:
        buf = reg.read(ref.name, ensure=False)
        if buf is None or not hasattr(buf, "shape"):
            return None
        return tuple(buf.shape)
    return fn


class PoissonSolverModule(ModuleBase):
    NAME = "poisson"

    def __init__(self, world: World, *, scope: str = "") -> None:
        super().__init__(world, scope=scope)
        mesh = self.require(MeshModule)
        dec = self.require(DECModule)

        self.constraint_mask = self.resource(
            "constraint_mask",
            spec=ResourceSpec(kind="taichi_field", dtype=ti.i32, shape_fn=_shape_of(mesh.V_pos), allow_none=True),
            doc="i32 mask (1=Dirichlet)",
            declare=True,
            buffer=None,
            description="Dirichlet mask",
        )
        self.constraint_value = self.resource(
            "constraint_value",
            spec=ResourceSpec(kind="taichi_field", dtype=ti.f32, shape_fn=_shape_of(mesh.V_pos), allow_none=True),
            doc="f32 values for Dirichlet verts",
            declare=True,
            buffer=None,
            description="Dirichlet values",
        )

        self.u = self.resource(
            "u",
            spec=ResourceSpec(kind="taichi_field", dtype=ti.f32, shape_fn=_shape_of(mesh.V_pos), allow_none=True),
            doc="solution scalar u",
            declare=False,
        )

        solver = SolvePoissonDirichlet(SolvePoissonDirichletIO.from_modules(mesh=mesh, dec=dec, poisson=self))
        deps = (mesh.E_verts.name, dec.star1.name, self.constraint_mask.name, self.constraint_value.name)

        self.declare_resource(self.u, buffer=None, deps=deps, producer=solver, description="Poisson solution")
