import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluate_endpoint import make_result_row, validate_manifest_rows  # noqa: E402


class EvaluateEndpointTest(unittest.TestCase):
    def setUp(self):
        self.row = {
            "response_id": "response_abc",
            "example_id": "example_def",
            "qid": "q1",
            "item_idx": 2,
            "label": 1,
        }

    def test_result_preserves_manifest_identity(self):
        result = make_result_row(self.row, 1, "TRUE and ignored trailing text", 12.34)
        self.assertEqual(result["response_id"], self.row["response_id"])
        self.assertEqual(result["example_id"], self.row["example_id"])
        self.assertEqual(result["label"], 1)
        self.assertEqual(result["pred"], 1)
        self.assertEqual(result["latency_ms"], 12.3)

    def test_manifest_preflight_requires_unique_stable_ids(self):
        validate_manifest_rows([self.row])
        missing = dict(self.row)
        missing.pop("response_id")
        with self.assertRaisesRegex(ValueError, "missing stable identity"):
            validate_manifest_rows([missing])
        with self.assertRaisesRegex(ValueError, "duplicate example_id"):
            validate_manifest_rows([self.row, dict(self.row)])


if __name__ == "__main__":
    unittest.main()
