import ast
import csv
import json
import subprocess
import sys
import tempfile
import unittest
import re
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class PublicArtifactsTest(unittest.TestCase):
    def test_submission_deck_is_exactly_five_slides(self):
        pptx = ROOT / "slides" / "ME344_RiceChem_Option2_5_Slides.pptx"
        pdf = ROOT / "slides" / "ME344_RiceChem_Option2_5_Slides.pdf"
        self.assertTrue(pptx.exists())
        self.assertTrue(pdf.read_bytes().startswith(b"%PDF-"))
        with zipfile.ZipFile(pptx) as archive:
            slide_xml = [
                name for name in archive.namelist()
                if re.fullmatch(r"ppt/slides/slide\d+\.xml", name)
            ]
        self.assertEqual(len(slide_xml), 5)

    def test_container_and_serving_images_are_digest_pinned(self):
        docker_text = (ROOT / "Dockerfile").read_text() + (ROOT / "Dockerfile.tpu").read_text()
        for line in docker_text.splitlines():
            if line.startswith("FROM "):
                self.assertIn("@sha256:", line)
        for manifest_name in ("serve-gpu-4b.yaml", "serve-tpu-4b.yaml"):
            manifest = (ROOT / "k8s" / manifest_name).read_text()
            literal_images = [
                line.strip() for line in manifest.splitlines()
                if line.strip().startswith("image:") and "${" not in line
            ]
            self.assertTrue(literal_images)
            self.assertTrue(all("@sha256:" in line for line in literal_images))

    def test_profiling_notebook_is_complete_and_syntactically_valid(self):
        notebook = json.loads((ROOT / "notebooks" / "hardware_profile.ipynb").read_text())
        code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
        self.assertGreaterEqual(len(code_cells), 6)
        joined = "\n".join("".join(cell["source"]) for cell in code_cells)
        self.assertIn("results_report.json", joined)
        self.assertIn("training_timing_", joined)
        self.assertIn("hardware_comparison.csv", joined)
        self.assertTrue((ROOT / "results" / "hardware_comparison.json").exists())
        self.assertTrue((ROOT / "results" / "hardware_comparison.csv").exists())
        self.assertIn("'checkpoint'", joined)
        self.assertNotIn("raise FileNotFoundError", joined)
        self.assertLess(joined.index("hardware_path ="), joined.index("c1 = completed_hardware"))
        execution_counts = [cell.get("execution_count") for cell in code_cells]
        self.assertTrue(all(isinstance(count, int) for count in execution_counts))
        self.assertEqual(execution_counts, sorted(execution_counts))
        self.assertTrue(any(cell.get("outputs") for cell in code_cells))
        errors = [
            output
            for cell in code_cells
            for output in cell.get("outputs", [])
            if output.get("output_type") == "error"
        ]
        self.assertEqual(errors, [])
        for cell in code_cells:
            ast.parse("".join(cell["source"]))

    def test_result_receipts_verify(self):
        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "verify_results.py")],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )

    def test_hardware_template_has_the_declared_schema(self):
        path = ROOT / "results" / "hardware_comparison_TEMPLATE.csv"
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 6)
        self.assertTrue(all(None not in row for row in rows))
        self.assertTrue(all(row["cost_usd"] == "" for row in rows))
        self.assertTrue(all(row["telemetry_source"] for row in rows))
        self.assertTrue(all(row["status"].startswith("pending") for row in rows))

    def test_figure_generation_is_deterministic(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            command = [
                sys.executable,
                str(ROOT / "scripts" / "generate_figures.py"),
                "--output-dir",
                str(output),
            ]
            subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
            first = {path.name: path.read_bytes() for path in output.glob("*.svg")}
            subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
            second = {path.name: path.read_bytes() for path in output.glob("*.svg")}
            self.assertEqual(first, second)
            self.assertEqual(set(first), {"accuracy_cost.svg", "throughput.svg", "controlled_hardware.svg"})
            self.assertIn(b"83.3% agreement", first["accuracy_cost.svg"])
            self.assertIn(b"73/s", first["throughput.svg"])
            self.assertIn(b"40.7", first["controlled_hardware.svg"])


if __name__ == "__main__":
    unittest.main()
