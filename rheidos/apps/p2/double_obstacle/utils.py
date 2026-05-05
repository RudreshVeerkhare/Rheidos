from rheidos.houdini.runtime.cook_context import CookContext
import numpy as np


def read_detail_vector_or_zero(ctx: CookContext, name: str, size: int) -> np.ndarray:
    try:
        value = ctx.input_io(0).read_detail(name, dtype=np.float64)
    except KeyError:
        return np.zeros((size,), dtype=np.float64)

    value = np.asarray(value, dtype=np.float64).reshape(-1)
    if value.shape != (size,):
        raise ValueError(f"Detail attribute {name!r} must have shape ({size},)")
    return value


def write_detail_vector(ctx: CookContext, name: str, value: np.ndarray) -> None:
    value = np.asarray(value, dtype=np.float64).reshape(-1)
    if value.size == 0:
        # Houdini detail attributes cannot represent a zero-tuple cleanly.
        # Missing state is equivalent to the empty harmonic state on K=0 meshes.
        return
    ctx.write_detail(name, value, create=True)
