# How-to Guides

How-to guides are goal oriented and assume you already know the basics.

## Compute: Validate buffers with ResourceSpec

Goal: Ensure buffers match expected type, dtype, and shape.

Steps:
1) Declare the resource with a spec.
   ```python
   import numpy as np
   from rheidos.compute import ModuleBase, ResourceSpec, World

   class MyModule(ModuleBase):
       NAME = "demo"
       def __init__(self, world: World, *, scope: str = "") -> None:
           super().__init__(world, scope=scope)
           self.positions = self.resource(
               "positions",
               declare=True,
               spec=ResourceSpec(kind="numpy", dtype=np.float32, shape=(4, 3)),
               doc="(4,3) float32 positions",
           )

   world = World()
   mod = world.require(MyModule)
   mod.positions.set(np.zeros((4, 3), dtype=np.float32))
   ```
2) Try an invalid buffer to see validation fail.
   ```python
   mod.positions.set(np.zeros((4, 2), dtype=np.float32))
   ```

Result: A `TypeError` is raised when the buffer does not match the spec.

## Compute: Allocate-before-fill for Taichi fields

Goal: Allocate a Taichi field only when size changes, then commit after fill.

Steps:
1) Use `set_buffer(..., bump=False)` before filling.
   ```python
   import taichi as ti

   field = ref.peek()
   if field is None or tuple(field.shape) != (n,):
       field = ti.field(dtype=ti.f32, shape=(n,))
       ref.set_buffer(field, bump=False)

   # Fill the field here...
   ref.commit()
   ```

Result: The resource becomes fresh only after the data is filled.

## Compute: Debug staleness with Registry.explain

Goal: See why a resource is stale and which producer owns it.

Steps:
1) Call `explain` on the registry.
   ```python
   print(world.reg.explain(mod.output.name, depth=4))
   ```

Result: A tree shows each dependency, its version, and whether it is stale.

## Houdini: Use the Solver SOP template with setup/step

Goal: Run a stateful solver using the template from `rheidos/apps/point_vortex/solver_sop.py`.

Steps:
1) Paste the template into a Solver SOP Python script.
2) Update the import to your own solver module.
   ```python
   # from rheidos.apps.point_vortex.app import setup, step
   from my_solver import setup, step
   ```
3) Create `my_solver.py` with optional `setup(ctx)` and required `step(ctx)`.
   ```python
   import numpy as np

   def setup(ctx) -> None:
       ctx.session.stats["solver_started"] = True

   def step(ctx) -> None:
       P = ctx.P().copy()
       P[:, 1] += 0.1 * float(ctx.dt)
       ctx.set_P(P)
   ```
4) (Optional) Add an integer parameter named `substep` to the Solver SOP.

Result: The solver moves points upward by `0.1 * dt` each cook.

## Houdini: Read and write attributes with GeometryIO

Goal: Read existing attributes and write new ones in a Python SOP.

Steps:
1) Construct a `GeometryIO` and read an attribute.
   ```python
   import hou
   from rheidos.houdini.geo import GeometryIO, OWNER_POINT

   node = hou.pwd()
   geo_in = node.inputs()[0].geometry()
   io = GeometryIO(geo_in, geo_in)
   P = io.read(OWNER_POINT, "P", components=3)
   ```
2) Write a new attribute to the output geometry.
   ```python
   geo_out = node.geometry()
   geo_out.clear()
   geo_out.merge(geo_in)

   io = GeometryIO(geo_in, geo_out)
   io.write(OWNER_POINT, "u", P[:, :1], create=True)
   ```

Result: The `u` attribute is created on points.

## Houdini: Record diagnostics from CookContext

Goal: Log events into the session for later inspection.

Steps:
1) Call `ctx.log` in your cook or step function.
   ```python
   def cook(ctx) -> None:
       ctx.log("my_event", value=123)
   ```
2) Inspect `session.log_entries` in a Houdini Python shell.

Result: The session contains structured log entries with timestamps.
