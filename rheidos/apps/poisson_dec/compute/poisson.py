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
class SolvePoissonDirichlet(WiredProducer["SolvePoissonDirichletIO"]):
    """
    Improved Poisson/Dirichlet (harmonic interpolation) solver.

    Key changes vs your version:
      - Validates shapes early (prevents silent OOB).
      - Builds and caches vertex-adjacency (CSR-ish) once per (nV, nE) to avoid atomics in SpMV.
      - Solves the *reduced free-DOF* system explicitly (CG assumptions stay sane).
      - Avoids per-iteration host sync: alpha/beta are computed on device scalars.
      - Warm-starts: does NOT zero free vertices each cook; only enforces constraints.
    """

    def __init__(self, io: "SolvePoissonDirichletIO") -> None:
        super().__init__(io)

        # Cached sizes
        self._nV: int = 0
        self._nE: int = 0
        self._nAdj: int = 0  # 2*nE

        # --- Adjacency (CSR-like) cache ---
        self._deg: Optional[Any] = None        # i32 [nV]
        self._offsets: Optional[Any] = None    # i32 [nV+1]
        self._cursor: Optional[Any] = None     # i32 [nV] (fill cursor)
        self._nbrs: Optional[Any] = None       # i32 [nAdj]
        self._w_adj: Optional[Any] = None      # f32 [nAdj]

        # --- Scratch vectors (f32 [nV]) ---
        self._r: Optional[Any] = None
        self._p: Optional[Any] = None
        self._Ap: Optional[Any] = None
        self._Ax: Optional[Any] = None

        # --- Device scalars for CG (shape=()) ---
        self._rr = ti.field(dtype=ti.f32, shape=())
        self._rr0 = ti.field(dtype=ti.f32, shape=())
        self._pAp = ti.field(dtype=ti.f32, shape=())
        self._rr_new = ti.field(dtype=ti.f32, shape=())
        self._alpha = ti.field(dtype=ti.f32, shape=())
        self._beta = ti.field(dtype=ti.f32, shape=())
        self._stop = ti.field(dtype=ti.i32, shape=())   # 0=run, 1=converged, 2=max_iter, 3=breakdown
        self._it = ti.field(dtype=ti.i32, shape=())     # iteration counter

    # -------------------------------------------------------------------------
    # Allocation / caching
    # -------------------------------------------------------------------------

    def _ensure_cache(self, nV: int, nE: int) -> None:
        """Ensure scratch + adjacency arrays exist for these sizes."""
        nAdj = 2 * nE

        need_realloc = (
            nV != self._nV
            or nE != self._nE
            or self._deg is None
            or self._nbrs is None
            or self._r is None
        )
        if not need_realloc:
            return

        self._nV = nV
        self._nE = nE
        self._nAdj = nAdj

        # adjacency
        self._deg = ti.field(dtype=ti.i32, shape=(nV,))
        self._offsets = ti.field(dtype=ti.i32, shape=(nV + 1,))
        self._cursor = ti.field(dtype=ti.i32, shape=(nV,))
        self._nbrs = ti.field(dtype=ti.i32, shape=(nAdj,))
        self._w_adj = ti.field(dtype=ti.f32, shape=(nAdj,))

        # scratch
        self._r = ti.field(dtype=ti.f32, shape=(nV,))
        self._p = ti.field(dtype=ti.f32, shape=(nV,))
        self._Ap = ti.field(dtype=ti.f32, shape=(nV,))
        self._Ax = ti.field(dtype=ti.f32, shape=(nV,))

    # -------------------------------------------------------------------------
    # Adjacency build (CSR-ish)
    # -------------------------------------------------------------------------

    @ti.kernel
    def _clear_i32(self, a: ti.template()):
        for i in a:
            a[i] = 0

    @ti.kernel
    def _count_degrees(self, E_verts: ti.template(), deg: ti.template()):
        # deg[v] = number of incident half-edges
        for e in E_verts:
            i = E_verts[e][0]
            j = E_verts[e][1]
            ti.atomic_add(deg[i], 1)
            ti.atomic_add(deg[j], 1)

    @ti.kernel
    def _prefix_sum_offsets(self, deg: ti.template(), offsets: ti.template(), cursor: ti.template()):
        offsets[0] = 0
        ti.loop_config(serialize=True)
        for i in range(deg.shape[0]):
            offsets[i + 1] = offsets[i] + deg[i]
        for i in range(deg.shape[0]):
            cursor[i] = offsets[i]
    
    @ti.kernel
    def _fill_adjacency(
        self,
        E_verts: ti.template(),
        w: ti.template(),
        cursor: ti.template(),
        nbrs: ti.template(),
        w_adj: ti.template(),
    ):
        # For each undirected edge (i,j), add two directed entries:
        # i -> j with weight we, and j -> i with weight we
        for e in E_verts:
            i = E_verts[e][0]
            j = E_verts[e][1]
            we = w[e]

            pi = ti.atomic_add(cursor[i], 1)
            nbrs[pi] = j
            w_adj[pi] = we

            pj = ti.atomic_add(cursor[j], 1)
            nbrs[pj] = i
            w_adj[pj] = we

    # -------------------------------------------------------------------------
    # Core ops for reduced free-DOF system
    # -------------------------------------------------------------------------

    @ti.kernel
    def _enforce_dirichlet(self, x: ti.template(), mask: ti.template(), val: ti.template()):
        # Warm-start friendly: do not touch free vertices, only enforce constrained values.
        for i in x:
            if mask[i] == 1:
                x[i] = val[i]

    @ti.kernel
    def _apply_K_free(
        self,
        x: ti.template(),
        y: ti.template(),
        offsets: ti.template(),
        nbrs: ti.template(),
        w_adj: ti.template(),
        mask: ti.template(),
    ):
        # y[i] = (Kx)[i] for FREE rows only; constrained rows set to 0.
        # K is cotan stiffness assembled as sum_{j} w_ij (x_i - x_j).
        for i in y:
            if mask[i] == 0:
                s = 0.0
                start = offsets[i]
                end = offsets[i + 1]
                xi = x[i]
                for k in range(start, end):
                    j = nbrs[k]
                    wij = w_adj[k]
                    s += wij * (xi - x[j])
                y[i] = s
            else:
                y[i] = 0.0

    @ti.kernel
    def _init_r_p_from_x(
        self,
        x: ti.template(),
        Ax: ti.template(),
        r: ti.template(),
        p: ti.template(),
        mask: ti.template(),
    ):
        # For harmonic interpolation: RHS on free vertices is 0.
        # Solve K_ff u_f = -K_fc u_c, which is equivalent to: r = -K(x) on free rows when x_c is enforced.
        for i in r:
            if mask[i] == 0:
                ri = -Ax[i]
                r[i] = ri
                p[i] = ri
            else:
                r[i] = 0.0
                p[i] = 0.0

    @ti.kernel
    def _dot_free_to_scalar(self, a: ti.template(), b: ti.template(), mask: ti.template(), out: ti.template()):
        out[None] = 0.0
        for i in a:
            if mask[i] == 0:
                ti.atomic_add(out[None], a[i] * b[i])

    @ti.kernel
    def _compute_alpha_and_stop(self, rr: ti.template(), pAp: ti.template(), alpha: ti.template(), stop: ti.template()):
        # stop = 3 on breakdown (nonpositive/near-zero curvature)
        eps = 1e-20
        if pAp[None] <= eps:
            stop[None] = 3
            alpha[None] = 0.0
        else:
            alpha[None] = rr[None] / pAp[None]

    @ti.kernel
    def _x_r_update_free(
        self,
        x: ti.template(),
        p: ti.template(),
        r: ti.template(),
        Ap: ti.template(),
        alpha: ti.template(),
        mask: ti.template(),
        val: ti.template(),
    ):
        a = alpha[None]
        for i in x:
            if mask[i] == 0:
                x[i] = x[i] + a * p[i]
                r[i] = r[i] - a * Ap[i]
            else:
                x[i] = val[i]
                r[i] = 0.0



    # --- Fix 2: no 'return' in non-static control flow ---
    @ti.kernel
    def _update_beta_check_stop(
        self,
        rr: ti.template(),
        rr0: ti.template(),
        rr_new: ti.template(),
        beta: ti.template(),
        stop: ti.template(),
        it: ti.template(),
        max_iter: ti.i32,
        tol2: ti.f32,
    ):
        s = stop[None]

        if s == 0:
            it_val = it[None]

            if it_val >= max_iter:
                stop[None] = 2  # max_iter
            else:
                if rr_new[None] <= tol2 * rr0[None]:
                    rr[None] = rr_new[None]
                    stop[None] = 1  # converged
                else:
                    beta[None] = rr_new[None] / rr[None]
                    rr[None] = rr_new[None]
                    it[None] = it_val + 1


    @ti.kernel
    def _p_update_free(self, p: ti.template(), r: ti.template(), beta: ti.template(), mask: ti.template()):
        b = beta[None]
        for i in p:
            if mask[i] == 0:
                p[i] = r[i] + b * p[i]
            else:
                p[i] = 0.0

    # -------------------------------------------------------------------------
    # Public compute
    # -------------------------------------------------------------------------

    def compute(self, reg: "Registry") -> None:
        io = self.io

        E = io.E_verts.get(ensure=False)
        w = io.w.get(ensure=False)
        mask = io.mask.get(ensure=False)
        val = io.value.get(ensure=False)

        if E is None or w is None or mask is None or val is None:
            raise RuntimeError("Missing inputs for Poisson solve.")

        # ---- Early validation (prevents silent OOB/garbage) ----
        if len(E.shape) != 1:
            raise RuntimeError(f"E_verts must be 1D over edges; got shape={E.shape}")
        if len(w.shape) != 1:
            raise RuntimeError(f"w must be 1D over edges; got shape={w.shape}")
        if w.shape[0] != E.shape[0]:
            raise RuntimeError(f"w and E_verts disagree: w.shape={w.shape}, E.shape={E.shape}")

        if len(mask.shape) != 1 or len(val.shape) != 1:
            raise RuntimeError(f"mask/value must be 1D; got mask.shape={mask.shape}, value.shape={val.shape}")
        if mask.shape[0] != val.shape[0]:
            raise RuntimeError(f"mask and value disagree: mask.shape={mask.shape}, value.shape={val.shape}")

        nE = int(E.shape[0])
        nV = int(mask.shape[0])

        # ---- Ensure output u exists and matches nV ----
        u = io.u.get(ensure=False)
        if u is None or u.shape != (nV,):
            u = ti.field(dtype=ti.f32, shape=(nV,))
            io.u.set_buffer(u, bump=False)

        # ---- Ensure cached memory, build adjacency when sizes changed ----
        self._ensure_cache(nV, nE)
        assert self._deg is not None and self._offsets is not None and self._cursor is not None
        assert self._nbrs is not None and self._w_adj is not None
        assert self._r is not None and self._p is not None and self._Ap is not None and self._Ax is not None

        # Rebuild adjacency whenever we reallocated (size-change trigger).
        # If you have resource versioning in your framework, key this off E/w "dirty" instead.
        self._clear_i32(self._deg)
        self._count_degrees(E, self._deg)
        self._prefix_sum_offsets(self._deg, self._offsets, self._cursor)
        # (Optional sanity: offsets[nV] should equal 2*nE, but we won't host-sync here.)
        self._fill_adjacency(E, w, self._cursor, self._nbrs, self._w_adj)

        # ---- CG solve on free vertices ----
        x = u
        self._enforce_dirichlet(x, mask, val)
        self._apply_K_free(x, self._Ax, self._offsets, self._nbrs, self._w_adj, mask)
        self._init_r_p_from_x(x, self._Ax, self._r, self._p, mask)

        self._dot_free_to_scalar(self._r, self._r, mask, self._rr)
        # Initialize rr0, stop, iter
        self._rr0[None] = self._rr[None]
        self._it[None] = 0
        self._stop[None] = 0

        # Early out if already solved
        # (One host sync here is fine; it avoids a lot of launches.)
        rr0 = float(self._rr0[None])
        if rr0 <= 1e-20:
            io.u.commit()
            return

        max_iter = 800
        tol = 1e-6
        tol2 = tol * tol

        # Blocked polling: we only read stop every so often (host sync is now rare).
        block = 25
        blocks = (max_iter + block - 1) // block

        for _ in range(blocks):
            # Run up to `block` iterations worth of kernel launches;
            # we don't host-sync between iterations.
            for __ in range(block):
                # Ap = K p (free rows)
                self._apply_K_free(self._p, self._Ap, self._offsets, self._nbrs, self._w_adj, mask)

                # pAp
                self._dot_free_to_scalar(self._p, self._Ap, mask, self._pAp)

                # alpha (and breakdown detection)
                self._compute_alpha_and_stop(self._rr, self._pAp, self._alpha, self._stop)

                # x, r update
                self._x_r_update_free(x, self._p, self._r, self._Ap, self._alpha, mask, val)

                # rr_new
                self._dot_free_to_scalar(self._r, self._r, mask, self._rr_new)

                # beta + convergence/max_iter
                self._update_beta_check_stop(self._rr, self._rr0, self._rr_new, self._beta, self._stop, self._it, max_iter, tol2)

                # p update
                self._p_update_free(self._p, self._r, self._beta, mask)

            stop = int(self._stop[None])  # host sync, but only once per block
            if stop != 0:
                break

        # Note: stop==3 means breakdown (non-SPD / degeneracy / bad weights). You may want to raise.
        # For now we still commit the best effort x.
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
