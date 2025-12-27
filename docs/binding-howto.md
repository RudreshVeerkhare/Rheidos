# Binding How-to Guides

These guides assume you already use the Adapter/Binder API and want to accomplish specific tasks.

## Bind a signal to a custom observer
Goal: connect a `Signal` to your own observer instead of the built-in deformer.

```python
from rheidos.abc.observer import Observer
from rheidos.sim.interaction import BindingRule, AdapterBinder

class MyObserver(Observer):
    def __init__(self, signal):
        super().__init__(name="MyObserver", sort=-5)
        self._signal = signal
        self._last = -1

    def update(self, dt: float) -> None:
        snap = self._signal.read_snapshot()
        if snap.version == self._last:
            return
        # do something with snap.payload
        self._last = snap.version

def _bind_vertex_scalar(signal, topology, adapter):
    return MyObserver(signal)

binder = AdapterBinder(eng)
binder.add_rule(BindingRule(domain="vertex", meaning="scalar", factory=_bind_vertex_scalar))
binder.bind(adapter, topologies={"domain": mesh})
```

## Bind multiple topologies in one adapter
Goal: reuse one adapter for multiple meshes (by topology id).

```python
adapter = MyAdapter(system, instance="multi")
mesh_map = {
    "domain_a": mesh_a,
    "domain_b": mesh_b,
}
binder.bind(adapter, topologies=mesh_map)
```

Ensure your `Signal.semantic.topology` matches keys in `mesh_map`.

## Queue actions from UI or controllers
Goal: send inputs to compute without calling it directly.

```python
scheduler.enqueue("constraints", ChargeBatch(charges))
```

If you need to batch multiple inputs before compute runs, enqueue them all before the next frame; the scheduler will coalesce the compute tick.

## Force compute every frame
Goal: run compute continuously (e.g., animation) instead of on demand.

```python
scheduler = ComputeScheduler(adapter, always_tick=True)
eng.add_observer(scheduler)
```

## Bind to a view or controller instead of an observer
Goal: attach a binding that is a `View` or `Controller`.

```python
def _bind_selection(signal, topology, adapter):
    view = MySelectionView(...)
    controller = MyController(...)
    eng.add_view(view)
    return controller
```

Note: if you return a `View` or `Controller`, the binder will attach it automatically.
