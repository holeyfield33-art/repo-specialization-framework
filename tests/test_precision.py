import unittest
from pathlib import Path
from unittest.mock import patch

from src.model_runtime import InferenceError, validate_compute_dtype
from src.evaluation import run_condition_group


class PrecisionTests(unittest.TestCase):
    def test_t4_explicit_fp16(self):
        self.assertEqual(validate_compute_dtype("float16", False), "float16")

    def test_no_silent_bf16_fallback(self):
        with self.assertRaises(InferenceError):
            validate_compute_dtype("bfloat16", False)
        self.assertEqual(validate_compute_dtype("bfloat16", True), "bfloat16")

    def test_invalid_precision_rejected(self):
        with self.assertRaises(InferenceError):
            validate_compute_dtype("float32", True)

    @patch("src.evaluation.evaluate_condition", return_value=[])
    @patch("src.evaluation.GitSnapshot")
    @patch("src.evaluation.HFGenerator")
    def test_precision_reaches_base_and_tuned_generators(self, generator, snapshot, evaluate):
        for conditions, adapter in ((["A", "B"], None), (["C", "D"], Path("adapter"))):
            run_condition_group([], conditions, Path("repo"), "model", adapter,
                                compute_dtype="float16")
            self.assertEqual(generator.call_args.kwargs["compute_dtype"], "float16")
            self.assertEqual(generator.call_args.kwargs["adapter_path"], adapter)
        self.assertEqual(generator.return_value.close.call_count, 2)

    @patch("src.evaluation.evaluate_condition", side_effect=RuntimeError("failed task"))
    @patch("src.evaluation.GitSnapshot")
    @patch("src.evaluation.HFGenerator")
    def test_generator_released_on_failure(self, generator, snapshot, evaluate):
        with self.assertRaises(RuntimeError):
            run_condition_group([], ["A"], Path("repo"), "model", compute_dtype="float16")
        generator.return_value.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
