from __future__ import annotations

import pytest

from pure_taichi.cli import main


def test_cli_no_gui_smoke() -> None:
    pytest.importorskip("scipy")
    code = main(["--no-gui", "--steps", "2", "--solver-backend", "scipy"])
    assert code == 0
