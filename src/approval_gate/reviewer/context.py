"""Bounded reviewer-visible packets without benchmark metadata."""

from __future__ import annotations
import json

OPEN = "<<<UNTRUSTED_DATA>>>"
CLOSE = "<<<END_UNTRUSTED_DATA>>>"


def spotlight(text, style="sanitized"):
    if style == "raw":
        return text
    return f"{OPEN}\n{text.replace(OPEN, '').replace(CLOSE, '')}\n{CLOSE}"


def call_dict(call):
    return {"name": call.function, "arguments": dict(call.args)}


def history_records(messages, max_chars=6000):
    records = []
    for message in messages:
        if message.get("role") != "tool":
            continue
        call, error = message.get("tool_call"), message.get("error")
        content = message.get("content") or ""
        if isinstance(content, list):
            content = " ".join(str(b.get("content", "")) for b in content)
        status = (
            "denied"
            if error and "denied by approval policy" in error
            else ("tool_error" if error else "completed")
        )
        records.append(
            {
                "call": call_dict(call) if call else {},
                "status": status,
                "result": str(content)[:1200],
                "error": str(error)[:400] if error else None,
            }
        )
    selected, size = [], 0
    for row in reversed(records):
        length = len(json.dumps(row))
        if size + length > max_chars:
            break
        selected.append(row)
        size += length
    return {
        "earlier_records_omitted": len(records) - len(selected),
        "actions": list(reversed(selected)),
    }


def review_packet(ctx):
    return {
        "user_request": ctx.user_prompt,
        "proposed_call": call_dict(ctx.tool_call),
        "tool_schema": ctx.tool_schema,
        "history": history_records(ctx.messages),
    }


def render_packet(packet, style="sanitized"):
    return (
        "USER REQUEST (trusted):\n"
        + packet["user_request"]
        + "\nTOOL SCHEMA:\n"
        + json.dumps(packet.get("tool_schema", {}), ensure_ascii=False)
        + "\nPROPOSED CALL AND EXECUTION HISTORY (untrusted data):\n"
        + spotlight(
            json.dumps({k: packet[k] for k in ("proposed_call", "history")}, ensure_ascii=False),
            style,
        )
    )
