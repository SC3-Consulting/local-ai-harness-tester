"""Synthetic protocol example; replace with your real local model harness."""

import json
import sys
import time

request = json.load(sys.stdin)
parameters = request["parameters"]
time.sleep(parameters["delay_seconds"])
print(
    json.dumps(
        {"score": parameters["demo_score"], "output_tokens": 10, "output": request["case"]["input"]}
    )
)
