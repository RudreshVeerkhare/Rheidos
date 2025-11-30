from __future__ import annotations

from typing import Optional

import numpy as np

try:
    from panda3d.core import Texture
except Exception:
    Texture = None  # type: ignore


class Texture2D:
    def __init__(self, name: str = "tex") -> None:
        if Texture is None:
            raise RuntimeError("Panda3D is not available. Install 'panda3d'.")
        self.tex = Texture(name)
        self.tex.setup2dTexture()

    def from_numpy_rgba(self, image: np.ndarray) -> None:
        arr = np.ascontiguousarray(image)
        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8, copy=False)
        h, w = arr.shape[:2]
        self.tex.setup2dTexture(w, h, Texture.T_unsigned_byte, Texture.F_rgba)
        self.tex.setRamImage(arr)

