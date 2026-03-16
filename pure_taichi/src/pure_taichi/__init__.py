"""Standalone pure Taichi P2 point-vortex simulator."""

from .config import SimulationConfig, load_config
from .demo import run_demo
from .sim import P2PointVortexSim, run_headless

__all__ = [
    "SimulationConfig",
    "load_config",
    "P2PointVortexSim",
    "run_demo",
    "run_headless",
]
