# Tutorials

Tutorials are guided learning experiences. Follow each step in order.

## Tutorial: Build a tiny compute graph

Audience: Python developers new to the compute module.

Prerequisites:
- The rheidos package is importable in your Python environment.

Steps:
1) Action: Create a file named demo_compute.py with the following contents.
   Snippet:
   ```python
   from dataclasses import dataclass

   from rheidos.compute import (
       ModuleBase,
       Registry,
       ResourceRef,
       WiredProducer,
       World,
       out_field,
   )


   @dataclass
   class SquareIO:
       x: ResourceRef[int]
       y: ResourceRef[int] = out_field()


   class SquareProducer(WiredProducer[SquareIO]):
       def compute(self, reg: Registry) -> None:
           x = self.io.x.get(ensure=False)
           self.io.y.set(x * x)


   class MathModule(ModuleBase):
       NAME = "math"

       def __init__(self, world: World, *, scope: str = "") -> None:
           super().__init__(world, scope=scope)
           self.x = self.resource("x", declare=True, doc="Input scalar")
           self.y = self.resource("y", doc="Output scalar")
           producer = SquareProducer(SquareIO(self.x, self.y))
           self.declare_resource(self.y, deps=(self.x,), producer=producer)


   def main() -> None:
       world = World()
       mod = world.require(MathModule)
       mod.x.set(6)
       print("y =", mod.y.get())
       print(world.reg.explain(mod.y.name, depth=2))


   if __name__ == "__main__":
       main()
   ```
   Result: You have a minimal compute graph with one input (x) and one derived output (y).
2) Action: Run the script.
   Command:
   ```bash
   python demo_compute.py
   ```
   Result: You should see `y = 36` and a short dependency tree printed from `Registry.explain`.

## Tutorial: Color geometry in a Python SOP with CookContext

Audience: Houdini users who want a first CookContext workflow.

Prerequisites:
- Houdini is available with Python.
- The rheidos package is importable in Houdini.

Steps:
1) Action: Create a Box SOP, then a Triangulate SOP, then a Python SOP.
   Result: The Python SOP receives triangulated geometry.
2) Action: Paste the template from `rheidos/apps/point_vortex/cook_sop.py` into the Python SOP.
   Result: The SOP runs the template at cook time.
3) Action: Update the import in the template to point at your cook script.
   Snippet:
   ```python
   # from rheidos.apps.point_vortex.app import cook
   from my_cook import cook
   ```
   Result: The SOP will call your own `cook(ctx)` implementation.
4) Action: Create `my_cook.py` on a directory that Houdini can import.
   Snippet:
   ```python
   import numpy as np
   from rheidos.houdini.geo import OWNER_POINT


   def cook(ctx) -> None:
       P = ctx.P()
       z = P[:, 2]
       z_min = float(z.min())
       z_ptp = float(z.max() - z_min) or 1.0
       t = (z - z_min) / z_ptp
       Cd = np.stack([t, 1.0 - t, 0.2], axis=1).astype(np.float32)
       ctx.write(OWNER_POINT, "Cd", Cd, create=True)
   ```
   Result: The script colors the mesh by height.
5) Action: Cook the node (or press Enter in the viewport).
   Result: The geometry displays a red-to-green gradient.
