import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from prepare_ricechem import make_example_ids, sha16, verify_sha16  # noqa: E402


class PrepareRiceChemTest(unittest.TestCase):
    def test_ids_are_stable_and_response_groups_rubric_items(self):
        response_a, example_a = make_example_ids(
            "q2", "  same answer\r\n", 0, "first item"
        )
        response_b, example_b = make_example_ids(
            "q2", "same answer\n", 1, "second item"
        )
        self.assertEqual(response_a, response_b)
        self.assertNotEqual(example_a, example_b)
        self.assertEqual(
            (response_a, example_a),
            make_example_ids("q2", "same answer", 0, "first item"),
        )
        self.assertTrue(response_a.startswith("response_"))
        self.assertTrue(example_a.startswith("example_"))
        self.assertNotIn("same answer", response_a)

    def test_example_identity_changes_with_every_manifest_component(self):
        baseline = make_example_ids("q1", "answer", 2, "rubric")[1]
        variants = [
            make_example_ids("q2", "answer", 2, "rubric")[1],
            make_example_ids("q1", "different", 2, "rubric")[1],
            make_example_ids("q1", "answer", 3, "rubric")[1],
            make_example_ids("q1", "answer", 2, "different")[1],
        ]
        self.assertTrue(all(value != baseline for value in variants))

    def test_hash_gate_accepts_match_and_rejects_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "question_rubrics.json"
            path.write_text("frozen")
            verify_sha16(path, sha16(path))
            path.write_text("drifted")
            with self.assertRaisesRegex(ValueError, "!= frozen"):
                verify_sha16(path, "0000000000000000")


if __name__ == "__main__":
    unittest.main()
