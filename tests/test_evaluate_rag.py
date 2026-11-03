import json
import tempfile
import unittest
from pathlib import Path

from scripts.evaluate_rag import write_results


class EvaluateRagOutputTests(unittest.TestCase):
    def test_write_results_saves_cases_and_mean_scores(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "nested" / "results.json"
            cases = [
                {"question": "First?", "response": "Answer one", "scores": {"faithfulness": 0.8}},
                {"question": "Second?", "response": "Answer two", "scores": {"faithfulness": 1.0}},
            ]

            report = write_results(output_path, Path("cases.jsonl"), "test-model", 4, cases)

            saved = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(saved, report)
            self.assertEqual(saved["summary"]["faithfulness"], 0.9)
            self.assertEqual(len(saved["results"]), 2)


if __name__ == "__main__":
    unittest.main()