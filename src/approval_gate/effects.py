"""Audited effect receipts for a small subset of stock AgentDojo tools.

Receipts describe affected records, not risk scores or benchmark ground truth.
Only these adapters may inspect the cloned simulator state. They never export
the whole environment, passwords, unrelated records, or evaluation metadata.
"""

from __future__ import annotations

from copy import deepcopy

TRANSACTION_FIELDS = ("id", "sender", "recipient", "amount", "subject", "date", "recurring")
EMAIL_FIELDS = ("id_", "sender", "recipients", "cc", "bcc", "subject", "body", "attachments", "status")
BANK_READS = {"get_iban", "get_balance", "get_most_recent_transactions",
              "get_scheduled_transactions", "read_file"}


def effect_snapshot(suite: str, env, call) -> dict:
    """Capture only the collections/target that an audited stock tool can affect."""
    tool = call.function
    if suite == "banking" and tool in BANK_READS:
        return {"resource": "none", "scope": "Audited stock read; no simulator writes.",
                "records": {}}
    if suite == "banking" and tool in {
        "send_money", "schedule_transaction", "update_scheduled_transaction"
    }:
        scheduled = tool != "send_money"
        records = (env.bank_account.scheduled_transactions if scheduled
                   else env.bank_account.transactions)
        if tool == "update_scheduled_transaction":
            records = [r for r in records if r.id == call.args["id"]]
        return {
            "resource": "scheduled_transaction" if scheduled else "transaction",
            "scope": "Affected transaction records; fields also exposed by the account's read tools.",
            "records": {str(r.id): {k: r.model_dump(mode="json")[k] for k in TRANSACTION_FIELDS}
                        for r in records},
        }
    if suite == "workspace" and tool == "send_email":
        records = env.inbox.emails.values()
        return {
            "resource": "email",
            "scope": "Affected email records; timestamps and unrelated mailbox contents omitted.",
            "records": {str(r.id_): {k: r.model_dump(mode="json")[k] for k in EMAIL_FIELDS}
                        for r in records},
        }
    raise ValueError(f"No audited effect adapter for {suite}/{tool}")


def effect_receipt(before: dict, after: dict, call) -> dict:
    """Retain unchanged target fields too: an ignored argument can be consequential."""
    records = []
    include_unchanged = call.function == "update_scheduled_transaction"
    for key in sorted(before["records"].keys() | after["records"].keys()):
        old, new = before["records"].get(key), after["records"].get(key)
        if old == new and not include_unchanged:
            continue
        change = ("created" if old is None else "deleted" if new is None
                  else "unchanged" if old == new else "updated")
        records.append({"resource": before["resource"], "id": key, "change": change,
                        "before": deepcopy(old), "after": deepcopy(new),
                        # Same-name fields of an audited record only. These are
                        # observations, not authorization scores or mismatch vetoes.
                        "argument_observations": [
                            {"field": field, "submitted": deepcopy(call.args[field]),
                             "observed_after": deepcopy(new[field])}
                            for field in sorted(call.args.keys() & (new or {}).keys())
                        ]})
    return {"status": "complete", "scope": before["scope"], "records": records,
            "notice": "Disposable preview only. Effects establish behavior, not authorization."}
