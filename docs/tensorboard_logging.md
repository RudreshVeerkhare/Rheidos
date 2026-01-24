# TensorBoard Logging (Metrics)

This project uses a lightweight TensorBoard logger (`TBLogger`) that wraps
`tensorboardX.SummaryWriter`. It is **not** tied to profiling and does **not**
spawn subprocesses. It writes event files in-process; you view them with the
TensorBoard CLI.

---

## Quick Start (Houdini)

In Houdini cook/solver code you already receive a `CookContext`:

```python
def step(ctx: CookContext) -> None:
    tb = getattr(ctx.session, "tb", None)
    if tb is None:
        return

    value = 1.23
    tb.add_scalar("my_metric", value, tb.next_step())
```

This logs a scalar series named `my_metric`.

---

## Where Logs Go (Default)

If you do not set `profile_logdir` on the node, logs go to:

```
<hip_dir or cwd>/_tb_logs/<hip_name>/<node_path_with_slashes_replaced>/<session_id>
```

Notes:
- If the HIP file is unsaved, `<hip_name>` is `untitled`.
- If Houdini is not available, it falls back to `os.getcwd()`.
- `<session_id>` is unique per Houdini session (PID + timestamp).

To override, set the node parameter `profile_logdir`.

---

## Dependencies

The logger uses `tensorboardX`. Install it for Houdini's Python:

```bash
hython -m pip install --user tensorboardX
```

To view results:

```bash
tensorboard --logdir <your logdir>
```

Open `http://localhost:6006`.

---

## Step Management

`TBLogger` keeps its own step counter:

```python
tb.next_step()      # increments and returns step
tb.step = 100       # set explicitly
step = tb.step      # read current step
```

If you want deterministic steps per frame:

```python
tb.step = int(ctx.frame)  # or int(ctx.frame * 1000 + ctx.substep)
tb.add_scalar("energy", value, tb.step)
```

---

## Common Patterns

### Scalar (Time Series)

```python
tb.add_scalar("hamiltonian", h_value, tb.next_step())
```

### Histogram

```python
tb.add_histogram("gamma", gammas, tb.next_step())
```

### Image

```python
tb.add_image("debug/field", image_np, tb.next_step(), dataformats="HWC")
```

### Text

```python
tb.add_text("notes", "solver converged", tb.next_step())
```

### Multiple Scalars (Single Step)

```python
tb.add_scalars(
    "losses",
    {"data": data_loss, "prior": prior_loss},
    tb.next_step(),
)
```

---

## Houdini Example: Hamiltonian Plot

From `rheidos/apps/point_vortex/solver_sop.py`:

```python
hamiltonian = world.require(HamiltonianModule)
h_field = hamiltonian.H.get()
h_value = None if h_field is None else float(h_field[None])

tb = getattr(ctx.session, "tb", None)
if tb is not None and h_value is not None:
    tb.add_scalar("hamiltonian", h_value, tb.next_step())
```

---

## Custom Logging Helpers

`TBLogger` supports custom functions that are exposed like built-ins.

```python
@tb.register()
def add_vector_norm(tb_logger, writer, tag, vec, step):
    norm = float(np.linalg.norm(vec))
    return writer.add_scalar(tag, norm, step)

tb.add_vector_norm("vel/norm", v, tb.next_step())
```

Rules:
- `@tb.register()` uses the function name.
- `tb.register("custom_name")(fn)` sets a specific name.
- Custom functions receive `(tb_logger, writer, *args, **kwargs)`.

---

## Non-Houdini Usage

If you need a standalone logger:

```python
from rheidos.compute.profiler.tb import TBLogger, TBConfig

tb = TBLogger(TBConfig(logdir="/tmp/rheidos_tb"))
tb.add_scalar("demo", 1.0, tb.next_step())
tb.flush()
```

---

## Performance Tips

- Log every N frames if the values are high-frequency.
- Avoid large images/histograms every frame.
- Prefer `float(...)` values for scalars to avoid extra conversions.

---

## Troubleshooting

**No events found**
- Ensure `tensorboardX` is installed for Houdini's Python.
- Make sure the node cooks at least once.
- Confirm the logdir path is correct (see "Where Logs Go").

**TensorBoard errors on launch**
- Check that the logdir exists and is readable.
- Delete stale event files if needed and recook.
