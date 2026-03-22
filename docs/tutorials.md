# Tutorials

Tutorials are guided learning exercises. Follow them in order.

## Tutorial: Build a tiny decorator-based compute graph

Audience:
- Python developers new to `rheidos.compute`

Steps:
1. Create `demo_compute.py`:

```python
from rheidos.compute import ModuleBase, ResourceSpec, World, producer


class MathModule(ModuleBase):
    NAME = "math"

    def __init__(self, world: World, *, scope: str = "") -> None:
        super().__init__(world, scope=scope)
        self.x = self.resource(
            "x",
            declare=True,
            spec=ResourceSpec(kind="python", dtype=int),
            doc="Input scalar",
        )
        self.y = self.resource(
            "y",
            spec=ResourceSpec(kind="python", dtype=int),
            doc="Squared output",
        )
        self.bind_producers()

    @producer(inputs=("x",), outputs=("y",))
    def square(self, ctx) -> None:
        ctx.commit(y=int(ctx.inputs.x.get()) ** 2)


def main() -> None:
    world = World()
    mod = world.require(MathModule)
    mod.x.set(6)
    print("y =", mod.y.get())
    print(world.reg.explain(mod.y.name, depth=2))


if __name__ == "__main__":
    main()
```

2. Run it:

```bash
python demo_compute.py
```

Result:
- `y = 36`
- a short dependency tree from `Registry.explain`

## Tutorial: Cook geometry in a Houdini Python SOP with `@session`

Audience:
- Houdini users who want the smallest supported entrypoint pattern

Steps:
1. Create a Box SOP, then a Triangulate SOP, then a Python SOP.
2. Paste this into the Python SOP:

```python
from rheidos.houdini.runtime import session


@session("demo")
def node1(ctx) -> None:
    out_io = ctx.output_io()
    src_io = ctx.input_io(0)
    out_io.geo_out.clear()
    out_io.geo_out.merge(src_io.geo_in)

    P = ctx.P().copy()
    P[:, 1] += 0.1
    ctx.set_P(P)
```

3. Cook the node.

Result:
- the output geometry mirrors input 0
- points are offset by `0.1` in Y

Notes:
- The active app in this repo uses the same session-entrypoint pattern in `rheidos/apps/p2/cook_sop.py`.
- For multi-input workflows, read secondary geometry with `ctx.input_io(1)`, `ctx.input_io(2)`, and so on.
