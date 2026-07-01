from __future__ import annotations


class TaichiProbe:
    def __init__(self, enabled: bool, sync_on_sample: bool):
        self.enabled = enabled
        self.sync_on_sample = sync_on_sample

    def _get_taichi(self):
        if not self.enabled:
            return None
        try:
            import taichi as ti
        except Exception:
            self.enabled = False
            return None
        return ti

    def clear(self) -> None:
        ti = self._get_taichi()
        if ti is None:
            return
        ti.profiler.clear_kernel_profiler_info()

    def sync(self) -> None:
        if not self.sync_on_sample:
            return
        ti = self._get_taichi()
        if ti is None:
            return
        ti.sync()

    def kernel_total_ms(self) -> float:
        ti = self._get_taichi()
        if ti is None:
            return 0.0
        return float(ti.profiler.get_kernel_profiler_total_time()) * 1000.0
