"""Reject ambiguous duplicate keys in model-produced JSON."""

import json


def loads(text):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    return json.loads(text, object_pairs_hook=pairs)
