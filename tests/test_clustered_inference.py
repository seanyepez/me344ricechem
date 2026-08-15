import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from clustered_inference import analyze_paired, main  # noqa: E402


def fixture_rows(predictions):
    labels = [1, 0, 1, 0]
    responses = ["r1", "r1", "r2", "r2"]
    return [
        {
            "response_id": responses[index],
            "example_id": f"e{index}",
            "qid": "q1" if index < 2 else "q2",
            "item_idx": index % 2,
            "label": labels[index],
            "pred": predictions[index],
        }
        for index in range(4)
    ]


class ClusteredInferenceTest(unittest.TestCase):
    def test_clustered_analysis_is_deterministic(self):
        rows_a = fixture_rows([1, 0, 1, 0])
        rows_b = fixture_rows([0, 0, 0, 0])
        first = analyze_paired(
            rows_a, rows_b, "fine-tuned", "base", 200, 200, seed=7
        )
        second = analyze_paired(
            rows_a, rows_b, "fine-tuned", "base", 200, 200, seed=7
        )
        self.assertEqual(first, second)
        self.assertEqual(first["n_decisions"], 4)
        self.assertEqual(first["n_responses"], 2)
        self.assertEqual(first["difference_percentage_points"], 50.0)
        self.assertGreater(first["response_level_permutation_p"], 0.0)

    def test_clustered_analysis_requires_response_ids(self):
        rows_a = fixture_rows([1, 0, 1, 0])
        rows_b = fixture_rows([0, 0, 0, 0])
        rows_b[0].pop("response_id")
        with self.assertRaisesRegex(ValueError, "response_id"):
            analyze_paired(rows_a, rows_b, "a", "b", 10, 10)

    def test_cli_accepts_explicit_paths_and_labels(self):
        rows_a = fixture_rows([1, 0, 1, 0])
        rows_b = fixture_rows([0, 0, 0, 0])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path_a = root / "a.jsonl"
            path_b = root / "b.jsonl"
            output = root / "analysis.json"
            for path, label, rows in (
                (path_a, "receipt-a", rows_a),
                (path_b, "receipt-b", rows_b),
            ):
                payload = [{"_header": True, "label": label}, *rows]
                path.write_text("\n".join(json.dumps(row) for row in payload) + "\n")
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                code = main(
                    [
                        "--path-a", str(path_a),
                        "--path-b", str(path_b),
                        "--label-a", "fine-tuned",
                        "--label-b", "base",
                        "--bootstrap-reps", "20",
                        "--permutation-reps", "20",
                        "--seed", "9",
                        "--output", str(output),
                    ]
                )
            self.assertEqual(code, 0)
            result = json.loads(output.read_text())
            self.assertEqual(result["pair"], "fine-tuned vs base")
            self.assertEqual(result["source_a"]["file"], "a.jsonl")
            self.assertNotIn(str(root), output.read_text())


if __name__ == "__main__":
    unittest.main()
