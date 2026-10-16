"""Effect receipts for stock AgentDojo tools.

Receipts describe affected records, not risk scores or benchmark ground truth.
One rule covers every tool: diff the cloned simulator state record by record and
export only records the call changed. Secret fields are redacted and run-time
timestamps dropped. Unchanged and unrelated records, and evaluation metadata,
are never exported.
"""

from __future__ import annotations

from copy import deepcopy

from pydantic import BaseModel
from pydantic_core import to_jsonable_python

SECRET_FIELDS = {"password", "id_number", "passport_number", "bank_account_number",
                 "credit_card_number"}
# Set from the wall clock by the simulator, so they would make identical calls look different.
VOLATILE_FIELDS = {"timestamp", "last_modified"}


def effect_snapshot(suite: str, env, call) -> dict:
    records: dict[str, dict] = {}
    _flatten(env, "", records)
    return {"resource": "environment",
            "scope": "Records changed by this call only; secret fields redacted, timestamps omitted.",
            "records": records}


def _json(value):
    if isinstance(value, (set, frozenset)):
        return sorted(to_jsonable_python(value), key=str)
    return to_jsonable_python(value)


def _plain(value) -> bool:
    if isinstance(value, (BaseModel, dict)):
        return False
    if isinstance(value, (list, tuple, set, frozenset)):
        return all(_plain(v) for v in value)
    return True


def _flatten(obj, path: str, out: dict) -> None:
    """One record per model object; dict entries and list items of models get their own path."""
    if isinstance(obj, BaseModel):
        fields = {}
        for name in type(obj).model_fields:
            if name.lower() in VOLATILE_FIELDS:
                continue
            value = getattr(obj, name)
            if _plain(value):
                fields[name] = _json(value)
            else:
                _flatten(value, f"{path}.{name}" if path else name, out)
        if fields:
            out[path or "environment"] = fields
        return
    items = obj.items() if isinstance(obj, dict) else enumerate(obj)
    for key, value in items:
        child = f"{path}[{key}]"
        if _plain(value):
            out[child] = {"value": _json(value)}
        else:
            _flatten(value, child, out)


def _redact(record: dict | None, other: dict | None) -> dict | None:
    if record is None:
        return None
    return {k: ("[redacted, changed]" if other is not None and other.get(k) != v else "[redacted]")
            if k.lower() in SECRET_FIELDS else deepcopy(v)
            for k, v in record.items()}


def effect_receipt(before: dict, after: dict, call) -> dict:
    records = []
    for key in sorted(before["records"].keys() | after["records"].keys()):
        old, new = before["records"].get(key), after["records"].get(key)
        if old == new:
            continue
        change = "created" if old is None else "deleted" if new is None else "updated"
        records.append({"resource": before["resource"], "id": key, "change": change,
                        "before": _redact(old, new), "after": _redact(new, old),
                        # Same-name fields of a changed record only. These are
                        # observations, not authorization scores or mismatch vetoes.
                        "argument_observations": [
                            {"field": field, "submitted": "[redacted]", "observed_after": "[redacted]"}
                            if field.lower() in SECRET_FIELDS else
                            {"field": field, "submitted": deepcopy(call.args[field]),
                             "observed_after": deepcopy(new[field])}
                            for field in sorted(call.args.keys() & (new or {}).keys())
                        ]})
    return {"status": "complete", "scope": before["scope"], "records": records,
            "notice": "Disposable preview only. Effects establish behavior, not authorization."}
