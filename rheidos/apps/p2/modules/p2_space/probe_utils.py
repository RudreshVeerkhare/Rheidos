import numpy as np


def probe_arrays(probes):
    if isinstance(probes, tuple) and len(probes) == 2:
        faceids = np.asarray(probes[0], dtype=np.int64)
        bary = np.asarray(probes[1], dtype=np.float64)
        if (
            faceids.ndim == 1
            and bary.ndim == 2
            and bary.shape[1] == 3
            and faceids.shape[0] == bary.shape[0]
        ):
            return faceids, bary

    if len(probes) == 0:
        return (
            np.empty((0,), dtype=np.int64),
            np.empty((0, 3), dtype=np.float64),
        )

    faceids, bary = zip(*probes)
    return np.asarray(faceids, dtype=np.int64), np.asarray(bary, dtype=np.float64)
