import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
UI_DIR = ROOT / "rheidos" / "compute" / "profiler" / "ui" / "dist"


class ProfilerUiSmokeTest(unittest.TestCase):
    def test_index_has_root_and_assets(self) -> None:
        html = (UI_DIR / "index.html").read_text(encoding="utf-8")
        self.assertIn('id="root"', html)
        self.assertIn("/assets/", html)

    def test_assets_exist(self) -> None:
        assets_dir = UI_DIR / "assets"
        self.assertTrue(assets_dir.is_dir())
        js_files = list(assets_dir.glob("*.js"))
        css_files = list(assets_dir.glob("*.css"))
        self.assertTrue(js_files)
        self.assertTrue(css_files)


if __name__ == "__main__":
    unittest.main()
