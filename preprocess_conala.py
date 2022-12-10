from sys import stdin
import json

dataset = json.load(stdin)
for row in dataset:
    if row["rewritten_intent"] is not None:
        preprocessed = {
            "nl": row["rewritten_intent"],
            "code": row["snippet"]
        }
        print(json.dumps(preprocessed))