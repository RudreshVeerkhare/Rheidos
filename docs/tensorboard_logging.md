# TensorBoard Logging

Rheidos includes a lightweight TensorBoard logger, `TBLogger`, backed by `tensorboardX`. It is independent from the profiler UI and writes event files in-process.

## Quick start

Inside a cook or solver entrypoint:

```python
def cook(ctx) -> None:
    tb = getattr(ctx.session, "tb", None)
    if tb is None:
        return

    value = 1.23
    tb.add_scalar("demo/value", value, tb.next_step())
```

## Default log path

If the node does not override `profile_logdir`, logs are written to:

```text
<hip_dir or cwd>/_tb_logs/<hip_name>/<node_path>/<session_id>
```

Where:
- `<hip_name>` is `untitled` for unsaved HIP files
- `<session_id>` is unique per Houdini session

## Dependencies

Install `tensorboardX` into Houdini's Python:

```bash
hython -m pip install --user tensorboardX
```

Then view logs with:

```bash
tensorboard --logdir <your logdir>
```

## Step management

`TBLogger` keeps its own step counter:

```python
tb.next_step()
tb.step = 100
current = tb.step
```

For deterministic frame-based logging:

```python
tb.step = int(ctx.frame)
tb.add_scalar("energy", value, tb.step)
```

## Common patterns

Scalar:

```python
tb.add_scalar("solver/residual", residual, tb.next_step())
```

Histogram:

```python
tb.add_histogram("gamma", gammas, tb.next_step())
```

Image:

```python
tb.add_image("debug/field", image_np, tb.next_step(), dataformats="HWC")
```

Text:

```python
tb.add_text("notes", "solver converged", tb.next_step())
```

Grouped scalars:

```python
tb.add_scalars("losses", {"data": data_loss, "prior": prior_loss}, tb.next_step())
```

## Custom helpers

You can register custom logging helpers:

```python
@tb.register()
def add_vector_norm(tb_logger, writer, tag, vec, step):
    return writer.add_scalar(tag, float(np.linalg.norm(vec)), step)
```

## Standalone usage

```python
from rheidos.compute.profiler.tb import TBConfig, TBLogger

tb = TBLogger(TBConfig(logdir="/tmp/rheidos_tb"))
tb.add_scalar("demo", 1.0, tb.next_step())
tb.flush()
```

## Troubleshooting

No events found:
- confirm the node cooked
- confirm `tensorboardX` is installed for Houdini's Python
- confirm the log directory exists

TensorBoard launch errors:
- check the log path
- delete stale event files if necessary and recook
