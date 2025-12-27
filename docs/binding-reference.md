# Binding Reference (Adapter/Binder API)

Module: `rheidos/sim/interaction.py`

## `Semantic`
Fields:
- `domain: str`
- `meaning: str`
- `topology: Optional[str]`
- `frame: Optional[str]`
- `units: Optional[str]`

## `Snapshot[T]`
Fields:
- `version: int`
- `payload: T`
- `timestamp: float`
- `topology: Optional[str]`

## `Signal[T]`
Constructor:
```python
Signal(ref, semantic=None, reader=None)
```
Methods:
- `version(ensure: bool = False) -> int`
- `read_snapshot(ensure: bool = False) -> Snapshot[T]`

## `Action[P]`
Constructor:
```python
Action(payload_type, handler)
```
Methods:
- `invoke(payload: P) -> None`

## `Adapter`
Constructor:
```python
Adapter(name: str)
```
Fields:
- `signals: dict[str, Signal[Any]]`
- `actions: dict[str, Action[Any]]`
- `topologies: dict[str, Any]`

Methods:
- `register_signal(name: str, signal: Signal[Any]) -> None`
- `register_action(name: str, action: Action[Any]) -> None`
- `compute() -> None` (override in subclasses)

## `ComputeScheduler`
Constructor:
```python
ComputeScheduler(adapter, name=None, sort=-20, always_tick=False)
```
Methods:
- `enqueue(action_name: str, payload: Any) -> None`
- `flush() -> None`
- `update(dt: float) -> None`

## `BindingRule`
Fields:
- `domain: str`
- `meaning: str`
- `factory: Callable[[Signal[Any], Any, Adapter], Any]`

## `AdapterBinder`
Constructor:
```python
AdapterBinder(engine, rules=None)
```
Methods:
- `add_rule(rule: BindingRule) -> None`
- `bind(adapter: Adapter, topologies: Optional[dict[str, Any]] = None) -> list[Any]`

Binding behavior:
- `bind()` looks at each `Signal.semantic`, matches the first rule with the same `(domain, meaning)`, and calls its factory.
- The factory receives the `Signal`, the resolved topology object (if any), and the adapter.
- The returned object must be an `Observer`, `View`, or `Controller`.
