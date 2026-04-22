# TensorBoard Logging

Rheidos exposes scalar TensorBoard logging through the top-level `rheidos.logger`
module. The same import works in plain Python, `rheidos.compute` producers, and
Houdini runtime code.

## Quick start

Standalone or compute-only code:

```python
from rheidos import logger

logger.configure(logdir="/tmp/rheidos_tb", run_name="demo")
logger.log("energy", 1.23)
```

Inside a Houdini cook or solver entrypoint:

```python
from rheidos import logger


def cook(ctx) -> None:
    logger.configure(run_name="annulus")
    logger.log("harmonic_coefficient", 1.23, category="p1_annulus")
```

## Default log path

- Standalone mode writes under the configured `logdir`.
- Houdini mode defaults to:

```text
<hip_dir or cwd>/_tb_logs/<hip_name>
```

Each run gets a human-readable directory name:

```text
run-my-label-0007__2026-04-20_16-54-59
run-0008__2026-04-20_17-01-12
```

Rheidos also writes:
- `<base>/latest-run.json`
- `<run_dir>/run.json`

## Step policy

`logger.log(...)` resolves the TensorBoard step in this order:
- explicit `step=...`
- ambient runtime hint, such as Houdini frame number for `substep == 0`
- logger-local monotonic counter

## Tag naming

- `logger.log("residual", value, category="solver")` writes `solver/residual`
- `logger.log("solver/residual", value)` uses the tag as-is

## Dependencies

Install `tensorboardX` into the Python environment that runs the simulation:

```bash
python -m pip install tensorboardX
```

Then view logs with:

```bash
tensorboard --logdir <your logdir>
```

## Troubleshooting

No events found:
- confirm the code path actually called `logger.log(...)`
- confirm `tensorboardX` is installed in the active Python
- confirm the configured or runtime-provided log root exists

TensorBoard launch errors:
- check the resolved log path
- delete stale event files if necessary and rerun
