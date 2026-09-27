"""Best-effort Telegram progress messages, separate from model requests and approval budgets."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import time

import httpx

from approval_gate.telemetry.events import fingerprint

LOGGER = logging.getLogger(__name__)


def add_notification_options(parser):
    parser.add_argument(
        "--notify",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="force Telegram alerts (or --no-notify); default: jobs estimated at >=1 hour",
    )


def duration(seconds):
    if seconds is None:
        return "unknown"
    seconds = max(0, round(seconds))
    if seconds < 60:
        return f"{seconds}s"
    minutes = seconds // 60
    return f"{minutes}m" if minutes < 60 else f"{minutes // 60}h {minutes % 60:02d}m"


class TelegramSender:
    """Use the same bot/token settings as the reference notify.py; never log credentials."""

    def __init__(self, token=None, chat_id=None, *, post=None):
        self._token = token or os.environ.get("TELEGRAM_BOT_TOKEN")
        self._chat_id = chat_id or os.environ.get("TELEGRAM_BOT_CHAT_ID")
        self._post = post or httpx.post
        self._warned = False

    @property
    def configured(self):
        return bool(self._token and self._chat_id)

    def __call__(self, message):
        if not self.configured:
            if not self._warned:
                LOGGER.warning("Telegram disabled: set TELEGRAM_BOT_TOKEN and TELEGRAM_BOT_CHAT_ID")
                self._warned = True
            return False
        try:
            response = self._post(
                f"https://api.telegram.org/bot{self._token}/sendMessage",
                json={"chat_id": self._chat_id, "text": message[:4000]},
                timeout=httpx.Timeout(5.0, connect=2.0),
                follow_redirects=False,
            )
            if response.status_code != 200 or response.json().get("ok") is not True:
                raise ValueError("Telegram rejected the message")
            return True
        except Exception:
            # HTTP exceptions may contain the bot token in their URL. Do not print them.
            if not self._warned:
                LOGGER.warning("Telegram notification failed; processing will continue")
                self._warned = True
            return False


class ProgressNotifier:
    """Emit start, crossed eighths, and one terminal message; ETA excludes cached work."""

    def __init__(
        self,
        label,
        total,
        *,
        estimated_seconds_per_item=180,
        completed=0,
        failed=0,
        enabled=None,
        output=None,
        sender=None,
        clock=time.monotonic,
    ):
        if total < 1 or not 0 <= completed <= total or not 0 <= failed <= completed:
            raise ValueError("invalid progress counts")
        self.label, self.total, self.completed, self.failed = label, total, completed, failed
        self.initial = completed
        self.estimate = estimated_seconds_per_item
        self.output = str(output) if output else None
        mode = os.environ.get("TELEGRAM_NOTIFICATIONS", "auto").lower()
        if enabled is None:
            enabled = mode in ("on", "true", "1") or (
                mode == "auto" and total * estimated_seconds_per_item >= 3600
            )
        self.enabled = enabled
        self.sender = sender if sender is not None else TelegramSender()
        self.clock = clock
        self.started_at = None
        self.bucket = completed * 8 // total
        self.closed = False

    def _emit(self, status, *, terminal=False):
        if not self.enabled:
            return
        elapsed = self.clock() - self.started_at if self.started_at is not None else 0
        done_now = self.completed - self.initial
        eta = ((elapsed / done_now) if done_now else self.estimate) * (self.total - self.completed)
        message = (
            f"[escalation-budget-gate] {self.label}\n{status}\n"
            f"Progress: {self.completed}/{self.total} ({100 * self.completed / self.total:.1f}%)\n"
            f"Errors: {self.failed} | Elapsed: {duration(elapsed)}"
        )
        if not terminal:
            message += f" | ETA: ~{duration(eta)}"
        if self.initial:
            message += f"\nAlready finished when attached: {self.initial}"
        if self.output:
            message += f"\nOutput: {self.output}"
        try:
            self.sender(message)
        except Exception:
            # A custom sender must not abort the workload either.
            LOGGER.warning("Telegram notification failed; processing will continue")

    def __enter__(self):
        self.started_at = self.clock()
        self._emit("Resumed / monitoring" if self.initial else "Started")
        return self

    def update(self, completed, *, failed=0):
        if (
            self.closed
            or not self.completed <= completed <= self.total
            or not 0 <= failed <= completed
        ):
            raise ValueError("progress must be monotonic and within the workload")
        self.completed, self.failed = completed, failed
        bucket = completed * 8 // self.total
        if bucket > self.bucket:
            self.bucket = bucket
            # Completion is sent only after output files/metrics have been saved.
            if bucket < 8:
                self._emit(f"Milestone: {bucket * 12.5:g}%")

    def __exit__(self, exc_type, exc, tb):
        if self.closed:
            return False
        if exc_type is not None:
            status = "Interrupted" if issubclass(exc_type, KeyboardInterrupt) else "Failed"
            status += f" ({exc_type.__name__}); completed outputs retained"
        elif self.completed != self.total:
            status = "Stopped before completion; resume the command to continue"
        elif self.failed:
            status = "Finished with errors; inspect outputs before continuing"
        else:
            status = "Completed"
        self._emit(status, terminal=True)
        self.closed = True
        return False


def cached_episode(root, config, case):
    """Only selected attempts are reused by run_episode; failed attempts need another run."""
    case_dir = Path(root) / fingerprint(config) / "cases" / fingerprint(case)
    pointer = case_dir / "selected.json"
    if not pointer.exists():
        return False
    directory = case_dir / json.loads(pointer.read_text())["attempt"]
    return (directory / "outcome.json").is_file()


def schedule_progress(root, schedule):
    """Read terminal outcomes for a frozen schedule without changing a running experiment."""
    completed = failed = 0
    for cell in schedule:
        directory = Path(root) / cell["experiment_id"] / "cases" / cell["case_id"]
        pointer = directory / "selected.json"
        if pointer.exists():
            path = directory / json.loads(pointer.read_text())["attempt"] / "outcome.json"
        else:
            paths = list(directory.glob("*/outcome.json"))
            if not paths:
                continue
            path = max(paths, key=lambda p: p.stat().st_mtime_ns)
        outcome = json.loads(path.read_text())
        if outcome["status"] not in ("completed", "limit", "error"):
            continue
        completed += 1
        failed += outcome["status"] == "error"
    return completed, failed
