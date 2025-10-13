FAQ / Troubleshooting

The window doesn’t show.
- Ensure Panda3D is installed: `python -c "import panda3d.core as p; print('ok')"`
- On macOS/Linux, make sure a display is available (no headless session)
- In notebooks, avoid competing event loops (don’t start your own `asyncio.run()` while the engine is in interactive mode)

My keys/mouse don’t do anything.
- Click the window once to focus it, then hold left mouse to enable mouselook
- If using FPV controller, press and hold left mouse to capture the cursor

Mesh loads, but looks black.
- Add lights and a material or call `setShaderAuto()` — the default surface view already does that, but custom nodes may render unlit

Taichi complains it’s missing.
- Install it: `pip install taichi`. If running on GPU, ensure CUDA/Metal/Vulkan support per Taichi’s docs

Performance tanks with big meshes.
- You’re probably copying large arrays each frame. Try updating at a lower rate, downsampled geometry, or move more logic into Taichi and only push the final result

The camera flips or behaves oddly.
- The provided FPV controller clamps pitch to avoid gimbal issues; if you write your own, normalize and clamp your direction vectors and use a stable up vector (Z+ in Panda3D)

Where do I put my own building blocks?
- Create your own `views/`, `controllers/`, and `observers/` modules in your project; import the framework and compose

