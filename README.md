# Rheidos

Rheidos is a small compute layer plus Houdini runtime used by the active P2 application in [`rheidos/apps/p2`](rheidos/apps/p2).

The repo now keeps one app-level surface:
- `rheidos/apps/p2`: current application entrypoints and modules
- `rheidos/compute`: resource graph, decorator producers, module/world helpers
- `rheidos/houdini`: CookContext, session runtime, debugger, profiler, reset helpers

Legacy app stacks, legacy visualization integrations, and the old standalone simulation package have been removed from the supported surface.

## Tests

Run the preserved test matrix with:

```bash
venv/bin/python -m pytest -q tests/apps/p2
venv/bin/python -m pytest -q tests/houdini
venv/bin/python -m pytest -q tests/test_profiler_trace.py tests/test_profiler_ui.py tests/test_summary_server.py tests/test_summary_store.py
```
