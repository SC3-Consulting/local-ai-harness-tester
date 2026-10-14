"""Synthetic embedding protocol example, not a real embedding model."""

import json
import sys

request = json.load(sys.stdin)
case = request["case"]
if request["parameters"]["variant"] == "aligned":
    embedding = [1.0, 0.0]
else:
    embedding = [0.0, 1.0]
response = {"embedding": embedding}
if case["evaluation"]["metric"] == "retrieval":
    response["document_embeddings"] = {
        document["id"]: [1.0, 0.0] if document["id"] == "relevant" else [0.0, 1.0]
        for document in case["documents"]
    }
print(json.dumps(response))
