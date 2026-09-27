"""Structural checks for the manually reviewed task-family split manifest."""

from agentdojo.task_suite.load_suites import get_suite

SUITES = ("banking", "workspace", "travel", "slack")


def validate_splits(data, version):
    if data.get("benchmark_version") != version:
        raise ValueError("split benchmark version differs from the runner")
    expected = {
        (name, task_id): task.PROMPT
        for name in SUITES
        for task_id, task in get_suite(version, name).user_tasks.items()
    }
    seen, families = set(), {}
    for row in data["tasks"]:
        key = row["suite"], row["task_id"]
        if key in seen:
            raise ValueError(f"duplicate task in split manifest: {key}")
        seen.add(key)
        if key not in expected or row.get("prompt") != expected[key]:
            raise ValueError(f"unknown task or changed task prompt: {key}")
        if row["split"] not in ("train", "validation", "test"):
            raise ValueError("invalid split")
        if not isinstance(row["family"], str) or not row["family"].strip():
            raise ValueError("family must be a nonempty string")
        families.setdefault(row["family"], set()).add(row["split"])
    if seen != set(expected):
        raise ValueError("split manifest must contain every benchmark task exactly once")
    if any(len(splits) != 1 for splits in families.values()):
        raise ValueError("a family crosses split boundaries")
