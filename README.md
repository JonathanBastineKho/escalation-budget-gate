# escalation-budget-gate

Approval policies for AgentDojo, evaluated under a per-task review budget with a fallible local LLM reviewer.

- [Experiment commands and findings template](EXPERIMENT_FINDINGS.md): start here for the staged Mac/Ollama experiments.
- [Experiment design](EXPERIMENT_PLAN.md): research hypotheses, comparisons, metrics, and limitations.
- [Environment setup](SETUP.md): existing conda and Ollama setup notes. The experiment guide contains the current training dependencies and runner commands.

Implemented policies: full approval, full escalation, tool-name cascade, prompted task alignment, calibrated learned risk, and learned review value with an authorization-risk guard. An optional local MELON adaptation and budgeted adaptive attack search support additional comparisons.

The pilot, validation, and test matrices include both required baselines. `full_approve` auto-approves every tool call with zero reviewer interventions. `full_escalate` requests review for every proposed call; **k=∞** (`null` in YAML) is the full-review baseline. At finite k, further escalation requests are denied after the budget is consumed. The current reviewer is the simulated Qwen LLM, so full review is not a guarantee of maximum safety or actual human review.

The current runner writes immutable configuration manifests and per-attempt records under `results/`. Reports distinguish the AgentDojo success/attack proxy from strict safe completion based on independent authorization annotations. Missing annotations stay unknown. The annotation workflow uses local `qwen3.8:27b` for 200 training and 100 validation packets, exports a separate blind validation JSON, and reconciles your 100 independent reviews later. Training labels retain their LLM provenance; calibration and strict safety reports require human-verified labels. See section 5 of the experiment guide for the commands.

```bash
conda activate escalation-budget-gate
python -m unittest discover -s tests -v
python scripts/run_benchmark.py --model qwen3.8:27b --suite banking --policy full_approve --user-tasks user_task_0 --dry-run
```

The dry run performs no inference. Follow the experiment guide to create reviewed task-family splits before collecting research results. Older flat result directories remain readable by the legacy TCR plotter; new HITL/cost reports require the new per-episode logs.

Generate charts from saved results and open the local gallery on the Mac:

```bash
python scripts/analyze/experiment_report.py --runs results/pilot --output results/pilot_report
open results/pilot_report/index.html
```

Every report generates TCR@k curves, API attempts by role, simulated HITL load and budget exhaustion, completion-versus-cost plots, episode runtime, and run-status/authorization-audit coverage. All figures have PNG and PDF exports, grouped by comparison ID, suite and clean/attacked cases. The gallery also links the CSV tables. Report generation makes no LLM calls. For strict TCR, supply the human-verified authorization file with `--annotations`; otherwise the charts retain explicit unresolved bounds. See section 9 of the experiment guide for chart interpretation and held-out reporting commands.

Long-running commands support Telegram start, 12.5% progress milestones, completion, and failure alerts. Fill `TELEGRAM_BOT_TOKEN` and `TELEGRAM_BOT_CHAT_ID` in `.env`; `TELEGRAM_NOTIFICATIONS=auto` enables jobs estimated at least one hour. Use `--notify` / `--no-notify` to override. To monitor a pilot that is already running, use `python scripts/watch_progress.py --runs results/pilot --notify` in a second terminal. This watcher only reads progress files. Configuration and notification behavior are documented in section 4 of the experiment guide.
