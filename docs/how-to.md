# How-to Guides

How-to guides are goal-oriented recipes for the preserved compute and Houdini runtime.

## Compute: Validate buffers with `ResourceSpec`

Goal:
- reject wrong dtype or shape early

```python
import numpy as np
from rheidos.compute import ModuleBase, ResourceSpec, World


class DemoModule(ModuleBase):
    NAME = "demo"

    def __init__(self, world: World, *, scope: str = "") -> None:
        super().__init__(world, scope=scope)
        self.positions = self.resource(
            "positions",
            declare=True,
            spec=ResourceSpec(kind="numpy", dtype=np.float32, shape=(4, 3)),
        )


world = World()
mod = world.require(DemoModule)
mod.positions.set(np.zeros((4, 3), dtype=np.float32))
```

If you set a `(4, 2)` array or the wrong dtype, validation raises `TypeError`.

## Compute: Allocate outputs with `producer_output(..., alloc=...)`

Goal:
- allocate dynamic outputs from inside a decorator producer

```python
import numpy as np
from rheidos.compute import ModuleBase, World, producer, producer_output


def alloc_out(_reg, ctx):
    return np.zeros_like(ctx.inputs.a.peek())


class DemoModule(ModuleBase):
    NAME = "demo"

    def __init__(self, world: World, *, scope: str = "") -> None:
        super().__init__(world, scope=scope)
        self.a = self.resource("a", declare=True, buffer=np.array([2.0]))
        self.out = self.resource("out")
        self.bind_producers()

    @producer(inputs=("a",), outputs=(producer_output("out", alloc=alloc_out),))
    def run(self, ctx) -> None:
        out = ctx.ensure_outputs(require_shape=False)["out"].peek()
        out[:] = ctx.inputs.a.get() + 3.0
        ctx.outputs.out.commit()
```

## Compute: Debug staleness with `Registry.explain`

Goal:
- see why a resource is stale and which dependencies are involved

```python
print(world.reg.explain(module.output.name, depth=4))
```

Use this when a resource recomputes unexpectedly or fails to update when you expect it to.

## Houdini: Mirror an input and call app logic

Goal:
- use the supported session-entrypoint pattern from `rheidos/apps/p2/cook_sop.py`

```python
from rheidos.houdini.runtime import session


@session("demo", debugger=True)
def node1(ctx) -> None:
    src_io = ctx.input_io(0)
    out_io = ctx.output_io()
    out_io.geo_out.clear()
    out_io.geo_out.merge(src_io.geo_in)

    # Your cook logic here.
```

This is the pattern the active P2 app uses before dispatching into its module graph.

## Houdini: Use `run_solver` with `setup(ctx)` / `step(ctx)`

Goal:
- run a stateful solver without writing your own session/cache plumbing

```python
from rheidos.houdini.runtime import run_solver


def setup(ctx) -> None:
    ctx.session.stats["solver_started"] = True


def step(ctx) -> None:
    P = ctx.P().copy()
    P[:, 1] += 0.1 * float(ctx.dt)
    ctx.set_P(P)
```

Point `run_solver(...)` at a module that exposes `setup(ctx)` and `step(ctx)`. The driver handles session reuse, profiling, publishing, and repeated-step suppression.

## Log simulation scalars to TensorBoard

Goal:
- write scalar simulation values with the shared logger API

```python
from rheidos import logger

logger.configure(logdir="/tmp/rheidos_tb", run_name="demo")
logger.log("energy", 123.0)
```

Inside Houdini cooks or solvers you can use the same import. If no explicit
`logdir` is configured, the runtime defaults to `<hip_dir or cwd>/_tb_logs/<hip_name>`.

## Houdini: Attach a debugger with `debugpy`

Goal:
- attach VS Code or another debugpy client to a running cook

1. Install `debugpy` into Houdini's Python:

```bash
<hython> -m pip install --user debugpy
```

2. Add or expose node parameters such as:
- `debug_enable`
- `debug_port`
- `debug_port_strategy`
- `debug_take_ownership`
- `debug_break_next`
- `debug_allow_remote`

3. Attach from VS Code:

```json
{
  "type": "python",
  "request": "attach",
  "host": "127.0.0.1",
  "port": 5678
}
```

4. Press `debug_break_next` and recook.

The full debugger behavior is documented in `docs/vscode-debugger.md`.
