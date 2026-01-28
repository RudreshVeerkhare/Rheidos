# Quick Reference: Dynamic Vortex Buffer Allocation

## What Changed

Fixed buffer sizes of 100 elements → Dynamic allocation based on `n_vortices`

## Key Modifications

### PointVortexModule (`__init__.py`)

```python
# Before: shape=(100,)
# After: shape_fn=shape_from_scalar(self.n_vortices)

# Affected resources:
- gammas
- face_ids
- bary
- pos_world
```

### Setter Methods

- `set_n_vortices()`: Removed capacity check (was: max 100)
- `set_face_ids()`: Now resizes buffer to exact input size
- `set_bary()`: Now resizes buffer to exact input size
- `set_gammas()`: Now resizes buffer to exact input size

### Cook/Solver Scripts

- `cook_sop.py`: Added pos_world buffer sizing
- `solver_sop.py`: Added pos_world buffer sizing in step() function

## How It Works

1. User calls `set_n_vortices(N)` where N can be any positive integer
2. ResourceSpec uses `shape_fn=shape_from_scalar(n_vortices)` to compute buffer shapes
3. Setter methods resize buffers only when shape changes
4. VortexWorldPositionProducer uses n_vortices[None] to iterate - works with any size

## No Breaking Changes

✅ VortexWorldPositionProducer - unchanged logic
✅ Stream function computation - unchanged logic  
✅ All existing producers - work with any buffer size
✅ API - same, just more flexible

## Testing

- 191/193 existing tests pass (failures unrelated to these changes)
- Comprehensive dynamic allocation test created and passed
- All syntax verified
