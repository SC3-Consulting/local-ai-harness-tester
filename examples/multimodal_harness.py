"""Synthetic media protocol example; confirms files are accessible."""

import json
import sys
from pathlib import Path

request = json.load(sys.stdin)
parts = request["case"]["input"]["parts"]
media = [
    {"type": part["type"], "bytes": Path(part["path"]).stat().st_size}
    for part in parts
    if part["type"] != "text"
]
print(json.dumps({"score": 1.0, "media": media}))
