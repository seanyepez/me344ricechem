import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from metrics import cell_metrics, mcnemar_exact  # noqa: E402


class MetricsTest(unittest.TestCase):
    def setUp(self):
        self.header = {"label": "fixture", "wall_secs": 4.0, "throughput_per_sec": 1.0}
        self.rows = [
            {"response_id": "r1", "example_id": "e1", "qid": "q1", "item_idx": 0, "label": 1, "pred": 1},
            {"response_id": "r1", "example_id": "e2", "qid": "q1", "item_idx": 1, "label": 0, "pred": 0},
            {"response_id": "r2", "example_id": "e3", "qid": "q2", "item_idx": 0, "label": 1, "pred": 0},
            {"response_id": "r2", "example_id": "e4", "qid": "q2", "item_idx": 1, "label": 0, "pred": -1},
        ]

    def test_cell_metrics(self):
        result = cell_metrics(self.header, self.rows)
        self.assertEqual(result["n"], 4)
        self.assertEqual(result["acc_minus1"], 0.5)
        self.assertEqual(result["acc_skip"], 0.6667)
        self.assertEqual(result["abstain_rate"], 0.25)
        self.assertEqual(result["per_question"]["q1"]["acc"], 1.0)

    def test_paired_comparison_rejects_manifest_drift(self):
        drifted = [dict(row) for row in reversed(self.rows)]
        with self.assertRaisesRegex(ValueError, "example_id/order mismatch"):
            mcnemar_exact(self.rows, drifted)

    def test_paired_comparison_requires_example_ids(self):
        missing = [dict(row) for row in self.rows]
        missing[0].pop("example_id")
        with self.assertRaisesRegex(ValueError, "missing required field"):
            mcnemar_exact(self.rows, missing)

    def test_paired_comparison_rejects_label_drift(self):
        drifted = [dict(row) for row in self.rows]
        drifted[0]["label"] = 0
        with self.assertRaisesRegex(ValueError, "label mismatch"):
            mcnemar_exact(self.rows, drifted)

    def test_small_exact_p_is_not_rounded_to_zero(self):
        rows_a, rows_b = [], []
        for index in range(40):
            common = {
                "response_id": f"r{index}",
                "example_id": f"e{index}",
                "qid": "q1",
                "item_idx": 0,
                "label": 1,
            }
            rows_a.append({**common, "pred": 1})
            rows_b.append({**common, "pred": 0})
        result = mcnemar_exact(rows_a, rows_b)
        self.assertGreater(result["p"], 0.0)
        self.assertLess(result["p"], 0.00001)


if __name__ == "__main__":
    unittest.main()
