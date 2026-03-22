import numpy as np


def probe_arrays(probes):
    if len(probes) == 0:
        return (
            np.empty((0,), dtype=np.int64),
            np.empty((0, 3), dtype=np.float64),
        )

    faceids, bary = zip(*probes)
    return np.asarray(faceids, dtype=np.int64), np.asarray(bary, dtype=np.float64)
