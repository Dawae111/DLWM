import unittest

from evaluation.evaluation import mmvp_pair_accuracy, score_prediction


class DLWMEvaluationTest(unittest.TestCase):
    def test_vqa_soft_uses_official_leave_one_out(self):
        answers = ["yes"] * 3 + ["no"] * 7
        self.assertAlmostEqual(
            score_prediction("yes", answers, "vqa_soft"),
            0.9,
        )

    def test_gqa_accuracy_is_standard_exact_match(self):
        self.assertEqual(score_prediction("Red", ["red"], "gqa_accuracy"), 1.0)
        self.assertEqual(score_prediction("the red", ["red"], "gqa_accuracy"), 0.0)

    def test_mmvp_requires_both_questions_in_pair(self):
        rows = [
            {"id": "1", "score": 1.0},
            {"id": "2", "score": 0.0},
            {"id": "3", "score": 1.0},
            {"id": "4", "score": 1.0},
        ]
        self.assertEqual(mmvp_pair_accuracy(rows), 0.5)


if __name__ == "__main__":
    unittest.main()
