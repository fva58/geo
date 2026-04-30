"""Structural checks for example notebooks."""

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = ROOT / "examples"


class TestExampleNotebooks(unittest.TestCase):
    """Check that example notebooks match the current API naming."""

    def _load_notebook(self, name: str) -> dict[str, object]:
        return json.loads((EXAMPLES / name).read_text(encoding="utf-8"))

    def test_updated_metric_notebook_uses_current_names(self):
        """The higher-level workflow notebook should use metric terminology."""
        notebook = self._load_notebook(
            "03_riemannian_objects_and_projections.ipynb"
        )
        text = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
        )
        self.assertIn("Metric Objects", text)
        self.assertIn("MetricGeometricObject", text)
        self.assertNotIn("RiemannianGeometricObject", text)

    def test_set_operations_notebook_uses_metric_wrapper(self):
        """The 2D set-operations notebook should use metric object wrappers."""
        notebook = self._load_notebook("04_set_operations_2d.ipynb")
        text = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
        )
        self.assertIn("MetricGeometricObject", text)
        self.assertNotIn("RiemannianGeometricObject", text)
