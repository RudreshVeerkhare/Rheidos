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
        if arr.ndim != 3 or arr.shape[2] != 4:
            raise ValueError(
                f"Expected image with shape (H, W, 4) for RGBA data, got {arr.shape}"
            )
        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8, copy=False)

        h, w = arr.shape[:2]
        needs_resize = self.tex.getXSize() != w or self.tex.getYSize() != h

        # Only recreate the texture when the resolution changes; otherwise update in-place.
        if needs_resize or self.tex.getComponentType() != Texture.T_unsigned_byte:
            self.tex.setup2dTexture(w, h, Texture.T_unsigned_byte, Texture.F_rgba)

        try:
            # setRamImageAs avoids realloc when size matches; fall back to setRamImage otherwise.
            self.tex.setRamImageAs(arr, "RGBA")
        except Exception:
            self.tex.setRamImage(arr)
