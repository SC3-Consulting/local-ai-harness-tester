"""Deterministic generation checks; model format requests and evaluation are separate."""

import json


def validate_evaluation(case):
    """
    Validate the evaluation configuration for a given case.

    Args:
        case (dict): The test case containing the evaluation configuration.

    Raises:
        ValueError: If the evaluation configuration is invalid.
    """
    evaluation = case.get("evaluation")
    if evaluation is None:
        return
    if not isinstance(evaluation, dict):
        raise ValueError("Generation evaluation must be an object")
    metric = evaluation.get("metric")
    if metric == "exact_match":
        if not isinstance(case.get("expected"), str):
            raise ValueError("exact_match requires a string expected answer")
    elif metric == "json":
        kind = evaluation.get("json_type")
        if kind is not None and kind not in (
            "object",
            "array",
            "string",
            "number",
            "boolean",
            "null",
        ):
            raise ValueError("Unknown evaluation json_type")
        keys = evaluation.get("required_keys", [])
        if (
            not isinstance(keys, list)
            or any(not isinstance(key, str) for key in keys)
            or len(set(keys)) != len(keys)
        ):
            raise ValueError("required_keys must be an array of unique strings")
        if keys and kind != "object":
            raise ValueError("required_keys requires json_type 'object'")
    else:
        raise ValueError(f"Unsupported generation metric: {metric}")


def reject_constant(value):
    """
    Reject non-finite JSON values during parsing.

    Args:
        value: The JSON value being parsed.

    Raises:
        ValueError: Always raised to reject non-finite JSON values.
    """
    raise ValueError(f"Non-finite JSON value: {value}")


def evaluate_generation(response, case):
    """
    Evaluate the generation output of a model against the expected case.

    Args:
        response (dict): The model's response containing the output.
        case (dict): The test case containing the evaluation configuration.

    Returns:
        dict: The response augmented with evaluation scores and metrics.

    Raises:
        ValueError: If the evaluation configuration is invalid or the output is not as expected.
    """
    evaluation = case.get("evaluation")
    if evaluation is None:
        return response
    output = response.get("output")
    if not isinstance(output, str):
        raise ValueError("Generation evaluation requires a string output")
    if evaluation["metric"] == "exact_match":
        matched = output.strip() == case["expected"].strip()
        return {
            **response,
            "score": float(matched),
            "generation_metrics": {"exact_match": float(matched)},
        }
    try:
        parsed = json.loads(output, parse_constant=reject_constant)
        # JSON exponents may overflow floats without invoking parse_constant.
        json.dumps(parsed, allow_nan=False)
    except (ValueError, OverflowError):
        return {
            **response,
            "score": 0.0,
            "generation_metrics": {"valid_json": 0.0, "json_requirements_met": 0.0},
        }
    checks = {
        "object": isinstance(parsed, dict),
        "array": isinstance(parsed, list),
        "string": isinstance(parsed, str),
        "number": type(parsed) in (int, float),
        "boolean": type(parsed) is bool,
        "null": parsed is None,
    }
    matched = checks.get(evaluation.get("json_type"), True)
    keys = evaluation.get("required_keys", [])
    if keys:
        matched = matched and isinstance(parsed, dict) and all(key in parsed for key in keys)
    if "expected" in case:
        # Canonical JSON keeps booleans distinct from numbers.
        matched = matched and json.dumps(parsed, sort_keys=True) == json.dumps(
            case["expected"], sort_keys=True
        )
    return {
        **response,
        "score": float(matched),
        "generation_metrics": {"valid_json": 1.0, "json_requirements_met": float(matched)},
    }


def basic_suite():
    """
    Return a basic suite of test cases for evaluating model outputs.

    Returns:
        list: Test cases containing IDs, inputs, expected outputs,
            and evaluation configuration.
    """
    return [
        {
            "id": "arithmetic",
            "input": "What is 2 + 2? Reply with only the number.",
            "expected": "4",
            "evaluation": {"metric": "exact_match"},
        },
        {
            "id": "json-object",
            "input": 'Return only a JSON object with key "answer" and numeric value 4.',
            "expected": {"answer": 4},
            "evaluation": {"metric": "json", "json_type": "object", "required_keys": ["answer"]},
        },
        {
            "id": "json-array",
            "input": "Return only the JSON array of the integers 1, 2, 3, in that order.",
            "expected": [1, 2, 3],
            "evaluation": {"metric": "json", "json_type": "array"},
        },
    ]
