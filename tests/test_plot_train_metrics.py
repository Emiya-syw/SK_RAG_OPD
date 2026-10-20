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
    def test_auto_filter_requires_isolated_common_scale_dip(self):
        def row(step, scale=1.0):
            return {
                "step": step,
                "actor/distillation/teacher_mass": 0.98 * scale,
                "actor/distillation/student_mass": 0.97 * scale,
                "actor/distillation/teacher_topk_coverage": 0.90 * scale,
                "actor/distillation/loss": 0.02,
            }
        isolated = [row(1), row(2, 11 / 12), row(3)]
        self.assertEqual([f["step"] for f in plotter.detect_padding_mean_dips(isolated)], [2])
        sustained = [row(1), row(2, 11 / 12), row(3, 11 / 12), row(4)]
        self.assertEqual(plotter.detect_padding_mean_dips(sustained), [])
        isolated[1]["actor/distillation/teacher_topk_coverage"] = 0.90
        self.assertEqual(plotter.detect_padding_mean_dips(isolated), [])
        self.assertEqual(plotter.detect_padding_mean_dips([
            {"step": 1, "actor/distillation/loss": 0.01},
            {"step": 2, "actor/distillation/loss": 0.50},
            {"step": 3, "actor/distillation/loss": 0.01},
        ]), [])

    def test_padding_exclusion_preserves_loss_and_leaves_gap(self):
        key = "actor/distillation/student_mass"
        rows = [
            {"step": step, key: mass, "actor/distillation/loss": 0.02,
             "response_length/mean": 400, "actor/entropy": 1.0}
            for step, mass in [(1, 0.98), (2, 0.89), (3, 0.98)]
        ]
        masked = plotter.exclude_padding_means(rows, {2})
        self.assertEqual(rows[1][key], 0.89)
        self.assertNotIn(key, masked[1])
        self.assertEqual(masked[1]["actor/distillation/loss"], 0.02)
        self.assertEqual(masked[1]["response_length/mean"], 400)
        self.assertEqual(masked[1]["actor/entropy"], 1.0)
        svg = plotter.render_svg(
            [{"step": row["step"], **({key: row[key]} if key in row else {})}
             for row in masked], {2}
        )
        root = ElementTree.fromstring(svg)
        ns = {"s": "http://www.w3.org/2000/svg"}
        self.assertEqual(len(root.findall("s:circle", ns)), 2)
        self.assertEqual(len(root.findall("s:polyline", ns)), 0)
        self.assertIn("no interpolation", svg)

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
