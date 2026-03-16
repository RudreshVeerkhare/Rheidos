from __future__ import annotations

import argparse
import json
from typing import Sequence

from .config import apply_overrides, load_config
from .demo import run_demo
from .sim import run_headless


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Pure Taichi P2 point-vortex demo")
    p.add_argument("--config", type=str, default=None, help="Path to .json/.toml config")
    p.add_argument("--steps", type=int, default=None, help="Number of simulation steps")
    p.add_argument("--seed", type=int, default=None, help="Random seed override")
    p.add_argument(
        "--solver-backend",
        type=str,
        default=None,
        choices=["auto", "taichi", "scipy", "scipy_constrained"],
        help="Solver backend override",
    )
    p.add_argument("--dt", type=float, default=None, help="Timestep override")
    p.add_argument("--no-gui", action="store_true", help="Run deterministic headless mode")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    cfg = load_config(args.config)
    cfg = apply_overrides(
        cfg,
        steps_dt=args.dt,
        seed=args.seed,
        solver_backend=args.solver_backend,
        no_gui=args.no_gui,
    )

    if args.no_gui:
        out = run_headless(cfg, steps=int(args.steps or 100), seed=args.seed)
        diag = out["diagnostics"]
        summary = {
            "solver_backend": diag.solver_backend,
            "residual_l2": diag.residual_l2,
            "rhs_circulation": diag.rhs_circulation,
            "hops_total": diag.hops_total,
            "hops_max": diag.hops_max,
            "positions_shape": list(out["positions"].shape),
        }
        print(json.dumps(summary, indent=2))
        return 0

    run_demo(cfg, steps=args.steps, seed=args.seed, no_gui=False)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
