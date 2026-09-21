from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import build_site  # noqa: E402


class BuildSiteTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.out = Path(self._tmp.name) / "site"
        build_site.build(self.out)
        self.manifest = json.loads((self.out / "app" / "manifest.json").read_text(encoding="utf-8"))

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_every_module_and_runtime_file_is_published(self) -> None:
        expected = {"entrypoint.py", "app.py", ".streamlit/config.toml", build_site.DEMO_WORKBOOK}
        expected |= {f"modules/{path.name}" for path in (ROOT / "modules").glob("*.py")}
        self.assertEqual(set(self.manifest["files"]), expected)
        for url in self.manifest["files"].values():
            self.assertTrue((self.out / url.split("?", 1)[0]).is_file(), url)

    def test_no_hidden_paths_are_served(self) -> None:
        """upload-pages-artifact drops dot-files, so served URLs must not contain any."""
        for url in self.manifest["files"].values():
            path = url.split("?", 1)[0]
            self.assertFalse(any(part.startswith(".") for part in path.split("/")), path)

    def test_host_page_and_requirements(self) -> None:
        for name in build_site.STATIC_FILES:
            self.assertTrue((self.out / name).is_file(), name)
        self.assertEqual(self.manifest["entrypoint"], "entrypoint.py")
        self.assertIn("xgboost", self.manifest["requirements"])
        self.assertNotIn("shap", self.manifest["requirements"])  # needs numba; unavailable in Pyodide

    def test_refuses_to_overwrite_a_non_build_directory(self) -> None:
        target = Path(self._tmp.name) / "not-a-build"
        target.mkdir()
        (target / "keep.txt").write_text("user data", encoding="utf-8")
        with self.assertRaises(SystemExit):
            build_site.build(target)
        self.assertTrue((target / "keep.txt").is_file())


if __name__ == "__main__":
    unittest.main()
