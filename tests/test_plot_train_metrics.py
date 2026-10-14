"""Standard-library checks for persisted training metric plots."""

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from xml.etree import ElementTree


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "plot_train_metrics.py"
spec = importlib.util.spec_from_file_location("plot_train_metrics", SCRIPT)
plotter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plotter)


class PlotTrainMetricsTest(unittest.TestCase):
    def test_resume_duplicate_step_and_interrupted_final_write(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.jsonl"
            records = [
                {"step": 1, "data": {"actor/distillation/loss": 0.2}},
                {"step": 2, "data": {"actor/distillation/loss": 0.1, "actor/grad_norm": 1.5}},
                {"step": 2, "data": {"actor/distillation/loss": 0.08, "actor/grad_norm": 1.2}},
            ]
            path.write_text("".join(json.dumps(record) + "\n" for record in records) + '{"step":3,')
            rows = plotter.read_metrics(path)
            self.assertEqual([row["step"] for row in rows], [1, 2])
            self.assertEqual(rows[-1]["actor/distillation/loss"], 0.08)
            svg = plotter.render_svg(rows)
            ElementTree.fromstring(svg)
            self.assertIn("Training metrics", svg)
            self.assertIn("<polyline", svg)

    def test_empty_and_missing_metrics_render(self):
        rows = [{"step": 4, "actor/loss": 0.4}]
        svg = plotter.render_svg(rows)
        self.assertIn("No data", svg)
        self.assertIn("<circle", svg)

    def test_cli_writes_svg_and_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.jsonl"
            path.write_text(json.dumps({"step": 1, "data": {"actor/loss": 0.25}}) + "\n")
            output = Path(directory) / "curves.svg"
            subprocess.run([sys.executable, str(SCRIPT), str(path), "--output", str(output)], check=True)
            self.assertTrue(output.read_text().startswith("<svg"))
            self.assertIn("actor/loss", output.with_suffix(".csv").read_text())


if __name__ == "__main__":
    unittest.main()
