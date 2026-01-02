from dataclasses import dataclass
from typing import Any, Optional

import taichi as ti

from rheidos.compute import WiredProducer, Registry, ResourceRef, out_field


# -----------------------------------------------------------------------------
# IO
# -----------------------------------------------------------------------------


@dataclass
class SolvePoissonDirichletIO:

    # topology + weights
    E_verts: ResourceRef[Any]  # (nE, vec2i) or similar; indexable as E[e][0], E[e][1]
    w: ResourceRef[Any]  # (nE,) edge weights (cotan / star1 etc.)

    # Dirichlet constraints
    mask: ResourceRef[Any]  # (nV,) i32, 1=constrained
    value: ResourceRef[Any]  # (nV,) f32, value on constrained verts

    # output
    u: ResourceRef[Any] = out_field()  # (nV,) f32

    # Optional RHS: solves K u = rhs on free verts (with Dirichlet). If None -> harmonic interp.
    rhs: Optional[ResourceRef[Any]] = None  # (nV,) f32 or None


# -----------------------------------------------------------------------------
# Producer
# -----------------------------------------------------------------------------


@ti.data_oriented
class SolvePoissonDirichlet(WiredProducer[SolvePoissonDirichletIO]):
    """
    Optimized Poisson/Dirichlet solver (PCG over free vertices).

    Main performance wins vs the earlier version:
      1) CSR TOPOLOGY (offsets/nbrs) built only when topology changes (nV/nE/E field identity).
         Then per-cook we only update weights via an edge->halfedge slot map (no atomics).
      2) PCG iteration fused to TWO kernels per iter (Step A + Step B). No separate dot kernels.
      3) Optional Jacobi preconditioner (default ON) usually cuts iterations a lot.
      4) Warm-start: does not zero u; only enforces Dirichlet constraints.

    Notes:
      - CG/PCG assumes SPD. If your cotan weights go non-Delaunay (negative cotans),
        the system may become indefinite -> breakdown (stop==3).
      - This solves vertex-based scalar Poisson K u = rhs with Dirichlet constraints.
        If rhs is None, rhs=0 -> harmonic interpolation.
    """

    def __init__(
        self,
        E_verts: ResourceRef[Any],
        w: ResourceRef[Any],
        mask: ResourceRef[Any],
        value: ResourceRef[Any],
        rhs: Optional[ResourceRef[Any]],
        u: ResourceRef[Any],
        *,
        max_iter: int = 800,
        tol: float = 1e-6,
        poll_block: int = 25,
        use_jacobi: bool = True,
        always_rebuild_topology: bool = False,
        block_dim: int = 256,
    ) -> None:
        io = SolvePoissonDirichletIO(E_verts, w, mask, value, u, rhs)
        super().__init__(io)

        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.poll_block = int(poll_block)
        self.use_jacobi = bool(use_jacobi)
        self.always_rebuild_topology = bool(always_rebuild_topology)
        self.block_dim = int(block_dim)

        # cached sizes + topology identity
        self._nV: int = 0
        self._nE: int = 0
        self._topology_valid: bool = False
        self._E_id: int = 0  # python id(E_field) snapshot

        # CSR-ish topology
        self._deg: Optional[Any] = None  # i32 [nV]
        self._offsets: Optional[Any] = None  # i32 [nV+1]
        self._cursor: Optional[Any] = None  # i32 [nV] (build cursor)
        self._nbrs: Optional[Any] = None  # i32 [2*nE]

        # weights (per halfedge slot) + edge->slot mapping
        self._w_adj: Optional[Any] = None  # f32 [2*nE]
        self._e2h0: Optional[Any] = None  # i32 [nE]
        self._e2h1: Optional[Any] = None  # i32 [nE]

        # preconditioner diag (optional)
        self._diag: Optional[Any] = None  # f32 [nV]

        # scratch
        self._r: Optional[Any] = None  # f32 [nV]
        self._p: Optional[Any] = None  # f32 [nV]
        self._Ap: Optional[Any] = None  # f32 [nV]
        self._Ax: Optional[Any] = None  # f32 [nV]
        self._z: Optional[Any] = None  # f32 [nV] (preconditioned residual)

        # device scalars
        self._rz = ti.field(dtype=ti.f32, shape=())  # current r·z
        self._rz0 = ti.field(dtype=ti.f32, shape=())  # initial r·z
        self._pAp = ti.field(dtype=ti.f32, shape=())  # p·Ap
        self._rz_new = ti.field(dtype=ti.f32, shape=())  # new r·z
        self._alpha = ti.field(dtype=ti.f32, shape=())
        self._beta = ti.field(dtype=ti.f32, shape=())
        self._stop = ti.field(
            dtype=ti.i32, shape=()
        )  # 0=run, 1=converged, 2=max_iter, 3=breakdown
        self._it = ti.field(dtype=ti.i32, shape=())

    # -------------------------------------------------------------------------
    # Allocation / caching
    # -------------------------------------------------------------------------

    def _ensure_cache(self, nV: int, nE: int) -> bool:
        """Ensure arrays exist for (nV,nE). Returns True if reallocated."""
        need = (
            nV != self._nV
            or nE != self._nE
            or self._deg is None
            or self._nbrs is None
            or self._w_adj is None
            or self._r is None
            or self._z is None
        )
        if not need:
            return False

        self._nV = nV
        self._nE = nE
        nAdj = 2 * nE

        self._deg = ti.field(dtype=ti.i32, shape=(nV,))
        self._offsets = ti.field(dtype=ti.i32, shape=(nV + 1,))
        self._cursor = ti.field(dtype=ti.i32, shape=(nV,))
        self._nbrs = ti.field(dtype=ti.i32, shape=(nAdj,))

        self._w_adj = ti.field(dtype=ti.f32, shape=(nAdj,))
        self._e2h0 = ti.field(dtype=ti.i32, shape=(nE,))
        self._e2h1 = ti.field(dtype=ti.i32, shape=(nE,))

        self._diag = ti.field(dtype=ti.f32, shape=(nV,))

        self._r = ti.field(dtype=ti.f32, shape=(nV,))
        self._p = ti.field(dtype=ti.f32, shape=(nV,))
        self._Ap = ti.field(dtype=ti.f32, shape=(nV,))
        self._Ax = ti.field(dtype=ti.f32, shape=(nV,))
        self._z = ti.field(dtype=ti.f32, shape=(nV,))

        self._topology_valid = False
        self._E_id = 0
        return True

    # -------------------------------------------------------------------------
    # Topology build (CSR-ish): offsets + nbrs + edge->halfedge slots
    # -------------------------------------------------------------------------

    @ti.kernel
    def _deg_clear_and_count(self, E_verts: ti.template(), deg: ti.template()):
        ti.loop_config(block_dim=256)
        for i in deg:
            deg[i] = 0
        ti.loop_config(block_dim=256)
        for e in E_verts:
            i = E_verts[e][0]
            j = E_verts[e][1]
            ti.atomic_add(deg[i], 1)
            ti.atomic_add(deg[j], 1)

    @ti.kernel
    def _prefix_sum_offsets_and_cursor(
        self, deg: ti.template(), offsets: ti.template(), cursor: ti.template()
    ):
        offsets[0] = 0
        # Topology rebuild is rare; keep this simple + deterministic.
        ti.loop_config(serialize=True)
        for i in range(deg.shape[0]):
            offsets[i + 1] = offsets[i] + deg[i]
        ti.loop_config(block_dim=256)
        for i in cursor:
            cursor[i] = offsets[i]

    @ti.kernel
    def _fill_adjacency_topology(
        self,
        E_verts: ti.template(),
        cursor: ti.template(),
        nbrs: ti.template(),
        e2h0: ti.template(),
        e2h1: ti.template(),
    ):
        # For each undirected edge (i,j), add two directed slots:
        #   slot pi stores neighbor j for vertex i  (i -> j)
        #   slot pj stores neighbor i for vertex j  (j -> i)
        ti.loop_config(block_dim=256)
        for e in E_verts:
            i = E_verts[e][0]
            j = E_verts[e][1]

            pi = ti.atomic_add(cursor[i], 1)
            nbrs[pi] = j
            e2h0[e] = pi

            pj = ti.atomic_add(cursor[j], 1)
            nbrs[pj] = i
            e2h1[e] = pj

    # -------------------------------------------------------------------------
    # Weight update (fast): w_adj[slot] = w[e] via e2h maps (no atomics)
    # -------------------------------------------------------------------------

    @ti.kernel
    def _update_weights_from_edges(
        self,
        w: ti.template(),
        e2h0: ti.template(),
        e2h1: ti.template(),
        w_adj: ti.template(),
    ):
        ti.loop_config(block_dim=256)
        for e in w:
            we = w[e]
            w_adj[e2h0[e]] = we
            w_adj[e2h1[e]] = we

    # -------------------------------------------------------------------------
    # Jacobi preconditioner diag: diag[i] = sum_j w_ij
    # -------------------------------------------------------------------------

    @ti.kernel
    def _build_diag(
        self,
        offsets: ti.template(),
        w_adj: ti.template(),
        mask: ti.template(),
        diag: ti.template(),
    ):
        ti.loop_config(block_dim=256)
        for i in diag:
            if mask[i] == 0:
                s = 0.0
                start = offsets[i]
                end = offsets[i + 1]
                for k in range(start, end):
                    s += w_adj[k]
                diag[i] = s
            else:
                diag[i] = 0.0

    # -------------------------------------------------------------------------
    # Apply operator on free rows: y = K x (free only), constrained rows -> 0
    # K row i: sum_j w_ij (x_i - x_j)
    # -------------------------------------------------------------------------

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
        ti.loop_config(block_dim=256)
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

    # -------------------------------------------------------------------------
    # PCG init (fused):
    #  - enforce Dirichlet in x (only constrained verts touched)
    #  - Ax = Kx (free rows)
    #  - r  = rhs - Ax on free, else 0
    #  - z  = M^{-1} r (Jacobi) or z=r (no precond)
    #  - p  = z
    #  - rz = r·z, rz0=rz, stop=0, it=0
    # -------------------------------------------------------------------------

    @ti.kernel
    def _pcg_init(
        self,
        x: ti.template(),
        rhs: ti.template(),
        has_rhs: ti.i32,
        offsets: ti.template(),
        nbrs: ti.template(),
        w_adj: ti.template(),
        mask: ti.template(),
        val: ti.template(),
        diag: ti.template(),
        use_jacobi: ti.i32,
        Ax: ti.template(),
        r: ti.template(),
        z: ti.template(),
        p: ti.template(),
        rz: ti.template(),
        rz0: ti.template(),
        stop: ti.template(),
        it: ti.template(),
    ):
        # enforce constraints (warm-start friendly)
        ti.loop_config(block_dim=256)
        for i in x:
            if mask[i] == 1:
                x[i] = val[i]

        # Ax = Kx (free rows only)
        ti.loop_config(block_dim=256)
        for i in Ax:
            if mask[i] == 0:
                s = 0.0
                start = offsets[i]
                end = offsets[i + 1]
                xi = x[i]
                for k in range(start, end):
                    j = nbrs[k]
                    wij = w_adj[k]
                    s += wij * (xi - x[j])
                Ax[i] = s
            else:
                Ax[i] = 0.0

        # r/z/p and rz reduction
        rz[None] = 0.0
        ti.loop_config(block_dim=256)
        for i in r:
            if mask[i] == 0:
                bi = 0.0
                if has_rhs == 1:
                    bi = rhs[i]
                ri = bi - Ax[i]
                r[i] = ri

                zi = ri
                if use_jacobi == 1:
                    d = diag[i]
                    # guard: avoid NaNs when diag is tiny
                    zi = ri / (d + 1e-20)
                z[i] = zi
                p[i] = zi
                ti.atomic_add(rz[None], ri * zi)
            else:
                r[i] = 0.0
                z[i] = 0.0
                p[i] = 0.0

        rz0[None] = rz[None]
        stop[None] = 0
        it[None] = 0

    # -------------------------------------------------------------------------
    # PCG iteration fused to 2 kernels/iter
    #
    # Step A:
    #  - Ap = K p  (free rows)
    #  - pAp = p·Ap
    #  - alpha = rz / pAp (or breakdown)
    # -------------------------------------------------------------------------

    @ti.kernel
    def _pcg_step_A(
        self,
        p: ti.template(),
        Ap: ti.template(),
        offsets: ti.template(),
        nbrs: ti.template(),
        w_adj: ti.template(),
        mask: ti.template(),
        rz: ti.template(),
        pAp: ti.template(),
        alpha: ti.template(),
        stop: ti.template(),
    ):
        pAp[None] = 0.0

        ti.loop_config(block_dim=256)
        for i in Ap:
            if mask[i] == 0:
                s = 0.0
                start = offsets[i]
                end = offsets[i + 1]
                pi = p[i]
                for k in range(start, end):
                    j = nbrs[k]
                    wij = w_adj[k]
                    s += wij * (pi - p[j])
                Ap[i] = s
                ti.atomic_add(pAp[None], pi * s)
            else:
                Ap[i] = 0.0

        # scalar tail
        eps = 1e-20
        if pAp[None] <= eps:
            stop[None] = 3
            alpha[None] = 0.0
        else:
            alpha[None] = rz[None] / pAp[None]

    # -------------------------------------------------------------------------
    # Step B:
    #  - x,r update (free only, enforce constraints)
    #  - z = M^{-1} r (or z=r)
    #  - rz_new = r·z
    #  - beta / stop / it
    #  - p = z + beta p
    # -------------------------------------------------------------------------

    @ti.kernel
    def _pcg_step_B(
        self,
        x: ti.template(),
        p: ti.template(),
        r: ti.template(),
        Ap: ti.template(),
        mask: ti.template(),
        val: ti.template(),
        diag: ti.template(),
        use_jacobi: ti.i32,
        alpha: ti.template(),
        rz: ti.template(),
        rz0: ti.template(),
        rz_new: ti.template(),
        beta: ti.template(),
        stop: ti.template(),
        it: ti.template(),
        max_iter: ti.i32,
        tol2: ti.f32,
    ):
        rz_new[None] = 0.0
        a = alpha[None]

        ti.loop_config(block_dim=256)
        for i in x:
            if mask[i] == 0:
                xi = x[i] + a * p[i]
                ri = r[i] - a * Ap[i]
                x[i] = xi
                r[i] = ri

                zi = ri
                if use_jacobi == 1:
                    zi = ri / (diag[i] + 1e-20)

                # stash z back into p temporarily? we keep p separate; compute rz_new + later p update.
                # We'll compute rz_new here and update p in the final loop.
                ti.atomic_add(rz_new[None], ri * zi)

                # store z into Ap as a temp? NO: Ap needed. Use r's spare? We'll just recompute z in p-update loop.
                # (Recomputing z is cheap vs extra memory traffic; also keeps code simple.)
            else:
                x[i] = val[i]
                r[i] = 0.0

        # scalar tail: convergence/max_iter/beta
        if stop[None] == 0:
            it_val = it[None]
            if it_val >= max_iter:
                stop[None] = 2
            else:
                if rz_new[None] <= tol2 * rz0[None]:
                    rz[None] = rz_new[None]
                    stop[None] = 1
                else:
                    beta[None] = rz_new[None] / (rz[None] + 1e-30)
                    rz[None] = rz_new[None]
                    it[None] = it_val + 1

        # p update: p = z + beta p  (free only)
        b = beta[None]
        ti.loop_config(block_dim=256)
        for i in p:
            if mask[i] == 0:
                zi = r[i]
                if use_jacobi == 1:
                    zi = r[i] / (diag[i] + 1e-20)
                p[i] = zi + b * p[i]
            else:
                p[i] = 0.0

    # -------------------------------------------------------------------------
    # Public compute
    # -------------------------------------------------------------------------

    def compute(self, reg: Registry) -> None:
        io = self.io

        E = io.E_verts.peek()
        w = io.w.peek()
        mask = io.mask.peek()
        val = io.value.peek()

        rhs_ref = io.rhs
        rhs = rhs_ref.peek() if rhs_ref is not None else None

        if E is None or w is None or mask is None or val is None:
            raise RuntimeError("Missing inputs for Poisson solve.")

        # ---- validation (cheap + prevents silent garbage) ----
        if len(E.shape) != 1:
            raise RuntimeError(f"E_verts must be 1D over edges; got shape={E.shape}")
        if len(w.shape) != 1 or w.shape[0] != E.shape[0]:
            raise RuntimeError(
                f"w must be (nE,). got w.shape={w.shape}, E.shape={E.shape}"
            )
        if len(mask.shape) != 1 or len(val.shape) != 1 or mask.shape[0] != val.shape[0]:
            raise RuntimeError(
                f"mask/value must be (nV,) and match. got mask={mask.shape}, value={val.shape}"
            )
        if rhs is not None:
            if len(rhs.shape) != 1 or rhs.shape[0] != mask.shape[0]:
                raise RuntimeError(
                    f"rhs must be (nV,) matching mask. got rhs={rhs.shape}, mask={mask.shape}"
                )

        nE = int(E.shape[0])
        nV = int(mask.shape[0])

        # ---- ensure output u exists and matches nV ----
        u = io.u.peek()
        if u is None or u.shape != (nV,):
            u = ti.field(dtype=ti.f32, shape=(nV,))
            io.u.set_buffer(u, bump=False)

        # ---- allocate caches ----
        realloc = self._ensure_cache(nV, nE)
        assert self._deg is not None
        assert self._offsets is not None
        assert self._cursor is not None
        assert self._nbrs is not None
        assert self._w_adj is not None
        assert self._e2h0 is not None and self._e2h1 is not None
        assert self._diag is not None
        assert (
            self._r is not None
            and self._p is not None
            and self._Ap is not None
            and self._Ax is not None
            and self._z is not None
        )

        # ---- topology rebuild heuristic ----
        E_id = id(E)
        topology_dirty = (
            self.always_rebuild_topology
            or (not self._topology_valid)
            or realloc
            or (E_id != self._E_id)
        )

        if topology_dirty:
            self._deg_clear_and_count(E, self._deg)
            self._prefix_sum_offsets_and_cursor(self._deg, self._offsets, self._cursor)
            self._fill_adjacency_topology(
                E, self._cursor, self._nbrs, self._e2h0, self._e2h1
            )
            self._topology_valid = True
            self._E_id = E_id

        # ---- per-cook: update weights quickly ----
        self._update_weights_from_edges(w, self._e2h0, self._e2h1, self._w_adj)

        # ---- per-cook: preconditioner diag (if enabled) ----
        use_jacobi_i32 = 1 if self.use_jacobi else 0
        if self.use_jacobi:
            self._build_diag(self._offsets, self._w_adj, mask, self._diag)

        # ---- PCG init ----
        has_rhs = 1 if rhs is not None else 0
        rhs_buf = rhs if rhs is not None else val  # dummy buffer; ignored if has_rhs==0

        self._pcg_init(
            u,
            rhs_buf,
            has_rhs,
            self._offsets,
            self._nbrs,
            self._w_adj,
            mask,
            val,
            self._diag,
            use_jacobi_i32,
            self._Ax,
            self._r,
            self._z,
            self._p,
            self._rz,
            self._rz0,
            self._stop,
            self._it,
        )

        # early out (one host sync to skip useless work)
        rz0 = float(self._rz0[None])
        if rz0 <= 1e-20:
            io.u.commit()
            return

        max_iter = self.max_iter
        tol2 = float(self.tol * self.tol)

        block = max(1, self.poll_block)
        blocks = (max_iter + block - 1) // block

        for _ in range(blocks):
            for __ in range(block):
                # A: Ap + pAp + alpha
                self._pcg_step_A(
                    self._p,
                    self._Ap,
                    self._offsets,
                    self._nbrs,
                    self._w_adj,
                    mask,
                    self._rz,
                    self._pAp,
                    self._alpha,
                    self._stop,
                )

                # B: x/r/z + rz_new + beta/stop + p update
                self._pcg_step_B(
                    u,
                    self._p,
                    self._r,
                    self._Ap,
                    mask,
                    val,
                    self._diag,
                    use_jacobi_i32,
                    self._alpha,
                    self._rz,
                    self._rz0,
                    self._rz_new,
                    self._beta,
                    self._stop,
                    self._it,
                    max_iter,
                    tol2,
                )

            stop = int(self._stop[None])  # host sync only once per block
            if stop != 0:
                break

        # stop==3 means breakdown (likely indefinite K). Still commit best-effort u.
        io.u.commit()
