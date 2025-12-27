# Binding Explanation: Why Signals, Actions, and Binders

This binding layer is a deliberate compromise: it keeps compute decoupled from rendering while still allowing fast "plug-and-play" wiring for demos and experiments.

## Signals vs actions
Compute outputs are modeled as `Signal`s because they represent data that can be sampled (often with a version). Inputs are modeled as `Action`s because they are commands: they may touch multiple resources and can trigger a solve. Treating both as "ports" created a leaky abstraction and pushed semantics into the compute layer.

## One contract surface
Signals wrap compute `ResourceRef`s directly, so the validation contract stays in one place (`ResourceSpec`). This avoids duplicating schema logic and reduces the chance that two specs disagree.

## Explicit scheduling
The scheduler keeps compute and render decoupled:
- Render does not call compute directly.
- Inputs are queued, then compute runs in a controlled tick.
This makes headless and async compute possible without dragging Panda3D into the pipeline.

## Semantics and topologies
`Semantic` tags capture intent (`domain`, `meaning`, `topology`). The binder uses them to auto-wire signals to render adapters. The adapter does not care about Panda3D objects; it only labels signals. The binder does not care about compute; it only binds based on semantics and available topologies.

## Tradeoffs
- Auto-binding is only as good as the semantics you provide. If you omit tags, nothing binds.
- The binder is intentionally simple: it picks the first matching rule. If you need more complex logic, register a more specific rule or perform binding manually.
- The scheduler does not enforce latency or performance budgets; it simply coalesces work to a tick. If you need fixed-step or multi-threaded compute, add a specialized scheduler.

## Why not reuse the old interaction ports?
The old interaction layer duplicated schema logic and made output reads non-atomic. It also encouraged "writer calls compute immediately" which re-coupled UI events to compute. The new structure keeps "data" and "commands" distinct and keeps compute ownership inside the adapter.
