Philosophy: Build Small Abstractions, Fast

This project is intentionally small. The goal isn’t to give you a giant engine; it’s to give you a few sharp tools that compose cleanly so you can sketch ideas quickly and keep layering your own abstractions as you learn what you need.

Principles
- Separate concerns:
  - Views render. They can be turned on/off without affecting simulation state.
  - Controllers handle input/state changes.
  - Observers do per‑frame, non‑render work (sim step, logging, metrics).
- Embrace data‑oriented transition points:
  - Simulation state lives in Taichi fields (or Numpy). Rendering reads Numpy.
  - Mesh/Texture helpers are one‑way gates: push arrays in; they end up on the GPU.
- Keep the main loop boring:
  - Engine orchestrates Panda3D tasks and time. Avoid cleverness here; put your cleverness in your Views/Observers.
- Prefer composition to configuration:
  - Add multiple small views instead of one mega‑view with flags.
  - Use `ToggleViewController` to swap modes without branching in your render code.

Create your own layers
- Once a pattern repeats, extract it:
  - A “SimStepper” observer that runs fixed‑timestep integration with a sub‑stepping accumulator
  - A “ScalarFieldView” that colors a mesh by a per‑vertex scalar with a palette
  - A “VectorFieldView” that draws glyphs/lines from a per‑vertex vector field
- Keep these in your project’s `views/` and `observers/` folders; this repo is meant to be imported and extended, not edited heavily.

Experimentation workflow
1) Notebook: bootstrap `Engine(interactive=True)`, attach FPV camera, drop in a primitive mesh
2) Write a tiny observer to mutate the mesh each frame; get feedback fast
3) When it’s interesting, extract it into a class/file, and build toggles to compare variants
4) When you hit performance issues, move the inner loops to Taichi kernels
5) Wrap up a demo script so others can run it without a notebook

Why Panda3D?
- Stable, battle‑tested scene graph and input system
- Simple to reason about: what you attach to `render` is what you see
- Python‑first, yet performant

Why Taichi?
- You get high‑level Python + low‑level performance for per‑element loops
- Easy to scale from CPU to GPU without rewriting your math

Trade‑offs
- Copying data between Taichi fields and Panda3D buffers is explicit — that clarity helps you control performance
- The framework avoids heavy “entity/component/system” plumbing on purpose; you add structure only when it starts paying rent

Mantra
- Make it render, then make it right, then make it fast — in that order.

