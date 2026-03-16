# pure_taichi

Standalone P2 FEEC point-vortex simulator with Taichi-first sparse solve and GGUI visualization.

## Quick start

```bash
cd pure_taichi
python -m pip install -e .
pure-taichi-p2 --no-gui --steps 50
pure-taichi-p2
```

## Features

- Closed-surface triangle meshes
- P2 scalar space (vertex + edge-midpoint DOFs)
- Taichi-first sparse solve with SciPy fallback
- Midpoint advection with edge walking in barycentric coordinates
- Interactive Taichi GGUI demo and deterministic headless runs
