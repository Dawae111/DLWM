import unittest

from scripts.prepare_dlwm_data import validate_spec


class DLWMDataProtocolTest(unittest.TestCase):
    def test_mmvp_is_rejected_for_training(self):
        with self.assertRaisesRegex(ValueError, "zero-shot"):
            validate_spec({"name": "mmvp", "split": "train"}, "train")

    def test_evaluation_split_is_rejected_for_training(self):
        with self.assertRaisesRegex(ValueError, "not allowed"):
            validate_spec({"name": "gqa", "split": "testdev"}, "train")

    def test_official_training_splits_are_allowed(self):
        for spec in (
            {"name": "gqa", "split": "train_balanced"},
            {"name": "vqav2", "split": "train2014"},
            {"name": "scienceqa", "split": "train"},
        ):
            validate_spec(spec, "train")


if __name__ == "__main__":
    unittest.main()
