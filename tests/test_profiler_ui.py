import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
UI_DIR = ROOT / "rheidos" / "compute" / "profiler" / "ui"


class ProfilerUiSmokeTest(unittest.TestCase):
    def test_index_has_core_sections(self) -> None:
        html = (UI_DIR / "index.html").read_text(encoding="utf-8")
        self.assertIn('href="/style.css"', html)
        self.assertIn('src="/app.js"', html)
        required_ids = [
            "page-dag",
            "page-tables",
            "dag-container",
            "node-details",
            "dag-mode",
            "dag-search",
            "fit-view",
            "ui-hz",
            "live-toggle",
            "producer-head",
            "producer-body",
            "category-panels",
        ]
        for element_id in required_ids:
            self.assertIn(f'id="{element_id}"', html)

    def test_dag_layout_is_vertical(self) -> None:
        code = (UI_DIR / "app.js").read_text(encoding="utf-8")
        self.assertGreaterEqual(code.count('rankDir: "TB"'), 2)

    def test_dag_selection_is_single(self) -> None:
        code = (UI_DIR / "app.js").read_text(encoding="utf-8")
        self.assertIn('selectionType: "single"', code)
        self.assertIn("boxSelectionEnabled: false", code)
        self.assertIn("autounselectify: true", code)

    def test_tables_page_scrolls(self) -> None:
        import re

        css = (UI_DIR / "style.css").read_text(encoding="utf-8")
        self.assertRegex(css, r"\.page-tables\s*\{[^}]*overflow: auto;")


if __name__ == "__main__":
    unittest.main()
