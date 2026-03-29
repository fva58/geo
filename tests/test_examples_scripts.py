"""Smoke tests for runnable example scripts."""

import runpy
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = ROOT / "examples"


class TestExampleScripts(unittest.TestCase):
    """Run the end-to-end example scripts in temporary output directories."""

    def _load_main(self, filename: str):
        namespace = runpy.run_path(
            str(EXAMPLES / filename),
            run_name="geo_example_test",
        )
        return namespace["main"]

    def test_space_object_mesh_pipeline(self):
        """The space/object/mesh example should produce a summary."""
        main = self._load_main("05_space_object_mesh_pipeline.py")
        with tempfile.TemporaryDirectory() as tempdir:
            result = main(tempdir)
            self.assertGreater(result["sphere_cap_vertices"], 0)
            self.assertGreater(result["torus_patch_cells"], 0)
            self.assertTrue(
                (Path(tempdir) / "space_object_mesh_summary.txt").exists()
            )

    def test_metric_object_zoo_pipeline(self):
        """The broader metric-zoo example should produce a summary."""
        main = self._load_main("06_metric_object_zoo_pipeline.py")
        with tempfile.TemporaryDirectory() as tempdir:
            result = main(tempdir)
            self.assertGreater(result["line_samples"], 0)
            self.assertGreater(result["arc_cells"], 0)
            self.assertTrue(
                (Path(tempdir) / "metric_object_zoo_summary.txt").exists()
            )

    def test_plot_and_export_pipeline(self):
        """The plot/export example should write mesh artifacts."""
        main = self._load_main("07_plot_and_export_pipeline.py")
        with tempfile.TemporaryDirectory() as tempdir:
            result = main(tempdir)
            self.assertTrue(result["png_exists"])
            self.assertTrue(result["obj_exists"])
            self.assertTrue(result["ply_exists"])
            self.assertTrue(result["gltf_exists"])
            self.assertTrue(
                (Path(tempdir) / "plot_and_export_summary.txt").exists()
            )
