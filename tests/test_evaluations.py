import json
import sys
import unittest
from pathlib import Path

import harness_tester as ht
from evaluations import basic_suite, evaluate_generation, validate_evaluation


class EvaluationTests(unittest.TestCase):
    def test_basic_suite_validates(self):
        for case in basic_suite():
            ht.validate_case(case)

    def test_exact_match(self):
        case = basic_suite()[0]
        for output, score in (("4\n", 1), ("The answer is 4", 0)):
            self.assertEqual(evaluate_generation({"output": output}, case)["score"], score)

    def test_strict_json_and_expected(self):
        case = basic_suite()[1]
        for output, valid, score in (
            ('{"answer":4}', 1, 1),
            ('{"answer":true}', 1, 0),
            ('{"other":4}', 1, 0),
            ("[4]", 1, 0),
            ('```json\n{"answer":4}\n```', 0, 0),
            ('{"answer":NaN}', 0, 0),
            ('{"answer":1e999}', 0, 0),
        ):
            with self.subTest(output=output):
                result = evaluate_generation({"output": output}, case)
                self.assertEqual(result["score"], score)
                self.assertEqual(result["generation_metrics"]["valid_json"], valid)

    def test_json_type_is_not_bool_as_number(self):
        case = {"evaluation": {"metric": "json", "json_type": "number"}}
        self.assertEqual(evaluate_generation({"output": "true"}, case)["score"], 0)
        self.assertEqual(evaluate_generation({"output": "4"}, case)["score"], 1)

    def test_invalid_config_and_response(self):
        for evaluation in (
            [],
            {"metric": "unknown"},
            {"metric": "exact_match"},
            {"metric": "json", "json_type": "unknown"},
            {"metric": "json", "required_keys": ["a"]},
            {"metric": "json", "json_type": "object", "required_keys": [1]},
        ):
            with self.subTest(evaluation=evaluation), self.assertRaises(ValueError):
                validate_evaluation({"evaluation": evaluation})
        with self.assertRaises(ValueError):
            evaluate_generation({"output": 4}, basic_suite()[0])

    def test_real_command_output_is_evaluated(self):
        case = basic_suite()[1]
        command = [sys.executable, "-c", 'print(\'{"output":"{\\\\"answer\\\\":4}"}\')']
        result = ht.execute_trial(command, {"case": case}, Path.cwd(), 5)
        self.assertEqual(result["status"], "succeeded")
        self.assertEqual(result["response"]["score"], 1)


if __name__ == "__main__":
    unittest.main()
