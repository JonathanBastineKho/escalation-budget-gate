# Escalation Budget Gate

Budget-aware approval policies for agent tool calls, evaluated with **AgentDojo
0.1.35 / benchmark v1.2.1** across banking, workspace, travel, and Slack tasks.
A policy can approve, deny, or request review. The gate enforces a per-case review
budget `k`, and the executor runs approved calls in the live simulator.

The project studies two approaches: a feature cascade that allocates scarce
reviews, and automated judges that assess proposed calls with different evidence.
Evaluation includes complete agent tasks and a separate fixed-call authorization
study. These tracks answer different questions and produce different metrics.

[Setup](#setup) · [Policies](#policies-and-budget-behavior) ·
[Benchmark](#end-to-end-benchmark) · [Fixed-call study](#fixed-call-authorization-study) ·
[Reporting](#reporting-from-saved-results) · [NSCC](#running-on-nscc) ·
[Development](#development-and-reproducibility)

## Setup

Use **Python 3.11 or newer**. Run the following commands from the repository root.
With `uv` installed:

```bash
uv sync --locked --extra analysis
source .venv/bin/activate
python -m pytest -q
```

The `analysis` extra installs plotting/table dependencies. The default `uv sync`
also installs the development group, including pytest and Ruff. Native Decision-2
inference additionally needs the `decision2` extra:

```bash
uv sync --locked --extra analysis --extra decision2
```

Alternatively, in a Python virtual environment, install the package with
`python -m pip install -e '.[analysis]'`. Add `decision2` to the extras if needed;
install pytest and Ruff separately for the development commands below. This pip
installation does not reproduce the versions in `uv.lock`.

Tests use fixed inputs and mocked model responses; they require no model server
or GPU inference. To inspect a benchmark configuration without contacting a model:

```bash
python scripts/run_benchmark.py \
  --model fixture-agent --suite banking --policy cascade --k 3 --seed 0 \
  --user-tasks user_task_0 --dry-run
```

### Model endpoints

Live experiments require a compatible model server and its exact served model
name. Runners load `.env` from the project environment. For a local vLLM server,
for example:

```dotenv
OPENAI_BASE_URL=http://localhost:8000/v1
OPENAI_API_KEY=EMPTY
```

Use the actual API key for a provider that requires authentication. Optional
`REVIEWER_BASE_URL` and `JUDGE_BASE_URL` select separate endpoints. The judge uses
`JUDGE_BASE_URL`, then `REVIEWER_BASE_URL`, then the agent endpoint. All three
clients use `OPENAI_API_KEY`.

For Ollama, the usual local endpoint is `http://localhost:11434/v1`. The historical
small-study configuration used `qwen3.8:27b` and a 16,384-token context. Model names,
reasoning controls, context limits, and tool-call support must match the server.
The repository's NSCC launcher starts vLLM; the Python benchmark runners do not
start a server themselves.

## Policies and budget behavior

| Policy | Decision rule |
| --- | --- |
| `undefended` | Stock AgentDojo pipeline without the gate. |
| `full_approve` | Approve every call that reaches the gate; no reviewer use. |
| `full_escalate` | Request review for every gated call. |
| `cascade_n1` | Approve read-only calls; request review for other calls. |
| `cascade` | Combine the read allowlist, task memory, risk features, and a budget-aware review threshold. |
| `direct_judge` | Judge the user request, tool schema, proposed arguments, and history. |
| `sandbox_judge` | Give the judge the same evidence plus a disposable preview's tool return/error. |
| `sandbox_effects_judge` | Also provide scoped before/after records of changes in the preview. |

The **agent** proposes calls. The **automated judge**, where used, decides whether
to approve, deny, or escalate. The **simulated reviewer** resolves escalations.
These roles may use the same model, but their prompts and responsibilities differ.
For cascade policies, envelope extraction, optional LLM feature extraction, and
review use `--reviewer-model` (default: the agent model). Native Decision-2 risk
scoring uses `--decision-model` separately. Judge policies select their automated
judge with `--judge-model`.

Only attempted reviewer interventions consume `k`; automated judgment and preview
calls have separate costs. At `k=0`, escalation becomes denial. At `k=inf`, review
budget is unlimited. Reviewer failures consume an intervention and fail closed.
Judge errors and unsuccessful previews request review, subject to the same budget.
Denials return an error to the agent so it can replan. Budget resets per case, not
per retry within a case.

Preview policies execute a proposed call in a deep copy of the simulator and
discard that copy. Approval then runs the call in the live simulator. This is
simulator isolation, not an OS sandbox. A preview exposes only the selected
observation/effects, without benchmark success labels; one-step previews do not
establish protection against every delayed or adversarial effect.

## End-to-end benchmark

This track measures task completion, attack success, reviewer use, and cost as an
agent performs complete tasks. Policies should be compared on the same suite,
task/injection pairs, attack, seed, and effective model settings.

### Run one configuration

The following example validates a single attacked banking configuration. Replace
`MODEL` with the exact name served by your endpoint. The timestamp separates this
experiment from previous results.

```bash
MODEL='your-served-model-name'
RUN="results/benchmark-$(date +%Y%m%d-%H%M%S)"

python scripts/run_benchmark.py \
  --model "$MODEL" --suite banking --policy cascade --k 3 --seed 0 \
  --attack important_instructions_generic \
  --user-tasks user_task_0 --injection-tasks injection_task_0 \
  --logdir "$RUN/runs" --force-rerun --dry-run
```

Remove `--dry-run` to execute inference. For a clean run, omit both `--attack` and
`--injection-tasks`. Omit task selections to use the suite's full corresponding
set. `--force-rerun` avoids reuse of AgentDojo's cached trajectories, but does not
clear appended case/decision logs: use a fresh result directory for each experiment.

The `cascade` policy accepts `--features code|llm`,
`--risk-model formula|decision|decision-nochecks`, and `--no-scope-deny`. These
change experimental behavior. Record them with the results; native risk scoring
also needs the Decision-2 dependencies and model revision.

### Plan a budget sweep

```bash
python scripts/sweep_k.py \
  --model "$MODEL" --suite banking \
  --policies full_approve full_escalate cascade \
  --ks 0 1 2 3 5 8 inf --seeds 0 \
  --attacks none important_instructions_generic \
  --user-tasks user_task_0 user_task_1 \
  --injection-tasks injection_task_0 injection_task_1 \
  --logdir "$RUN/runs" --dry-run
```

Remove `--dry-run` to execute the printed plan, using a fresh `RUN` if the earlier
example was executed. `full_approve` and `undefended` run once at `k=inf`; clean
cells omit injection IDs. `--suites banking workspace travel slack` selects
multiple suites. The sweep exposes the standard cascade configuration; use
`run_benchmark.py` directly for its feature/scorer ablations.

For judge comparisons, select `direct_judge sandbox_judge sandbox_effects_judge`
and, if desired, `full_approve full_escalate` controls. Pass identical judge
settings to all matched arms. The sweep performs a judge preflight and stops on
a failing cell. A single `run_benchmark.py` invocation does not perform preflight.
Preflight checks response validity and usage; it does not establish judge quality.

Use development data to select thresholds before held-out evaluation. The
[legacy judge validation workflow](experiment-notes.md#legacy-two-policy-validation-workflow)
contains the separate preflight, authorization evaluation, and calibration commands.
Its shared score cutoff and the fixed-call study's evidence-based threshold pair
are different protocols; keep settings and calibration artifacts matched.

## Fixed-call authorization study

This track compares the three judge policies on **10 development and 10 held-out
fixtures**, with four authorized, five denied, and one ambiguous proposal per
split. It evaluates proposed calls without an acting agent or simulated reviewer.
A complete run makes **66 judge requests**: six preflight, 30 development, and 30
held-out. Early failures or unsuccessful calibration stop the run sooner.

The study uses Qwen chat with evidence-backed verdicts. Its defaults are
`qwen3.8:27b`, disabled thinking, 2,048 output tokens, temperature zero, no retries,
and a 180-second timeout. The examples below use the historical 600-second timeout;
ensure your server supports the requested settings.

First validate fixtures offline:

```bash
STUDY="results/sandbox-study-$(date +%Y%m%d-%H%M%S)"
python scripts/run_sandbox_study.py \
  --judge-timeout 600 --output-dir "$STUDY" --dry-run
```

Then check the six-call response contract with live inference:

```bash
python scripts/run_sandbox_study.py --preflight-only \
  --judge-timeout 600 --output-dir "${STUDY}-preflight"
```

A separate four-call, effects-only development probe checks the scheduled-payment
outcome distinction and authorized controls:

```bash
python scripts/run_sandbox_study.py --development-probe \
  --judge-timeout 600 --output-dir "${STUDY}-probe"
```

To run the full study, including its own preflight and development calibration:

```bash
python scripts/run_sandbox_study.py \
  --judge-timeout 600 --output-dir "$STUDY"
```

The runner requires a new output directory and has no resume support. If no
approval/denial threshold pair qualifies for the effects policy, it records
`thresholds: null` and does not evaluate held-out cases. Successful runs save
`preflight.json`, `development.json`, `calibration.json`, `heldout.json`, and
`report.{json,md}`. Changing prompts or inference settings requires fresh calibration.

See the [fixture definitions](src/approval_gate/config/authorization_sandbox_small_development.json),
[recorded findings](SANDBOX_FINDINGS.md), and
[archived study details](experiment-notes.md#small-qwen-study--current-workflow)
for the evaluation criteria and development history. Results from these small,
related failure families should not be read as benchmark-wide protection claims.

## Reporting from saved results

These analysis commands make **no model calls**. Select the intended run explicitly
and preserve the original artifacts when regenerating tables.

| Tool | Input and purpose |
| --- | --- |
| `report_metrics.py` | Result roots containing `cases/` and `decisions/`; general metrics and per-tool handling. |
| `report_metrics_all.py` | Recursively discover result sources under a root and run the general report, distinguishing sources by their relative paths. |
| `compare_judges.py` | Matched case logs with comparison IDs; judge/control comparisons with health accounting. |
| `tcr_curve.py` | AgentDojo `runs/` trajectories; TCR curves and CSVs. |
| `tcr_at_k_report.py` | Selected CSV sources in `results_csv_only/`; main-report TCR figures and tables. |
| `tcr_at_k_all_policies.py` | Selected CSV sources in `results_csv_only/`; appendix TCR figures, CSV, and LaTeX table rows. |
| `compare_sandbox_study.py` | Saved development, calibration, and held-out JSON; fixed-call report printed to stdout. |

`report_metrics_all.py` retains the general report's scoring, within-source rerun
selection, attack pooling, and completeness checks. Its relative source paths
avoid basename collisions; partial runs and judge-health failures remain included.

The two CSV-based TCR scripts use source selections defined in their code. Check
those selections against the intended report inputs before running them. Both
accept `--figures` and `--tables` output directories, defaulting to
`results/figures/` and `results/tables/`.

```bash
python scripts/analyze/report_metrics.py "$RUN" \
  --output "$RUN/report_metrics.csv" \
  --tool-output "$RUN/escalation_by_tool.csv"

# For a matched three-judge benchmark with both controls:
python scripts/analyze/compare_judges.py --root "$RUN" --include-controls \
  --policies direct_judge sandbox_judge sandbox_effects_judge \
  --output "$RUN/judge_comparison.csv"

# For a completed fixed-call study:
python scripts/analyze/compare_sandbox_study.py --root "$STUDY"
```

The matched report requires all selected arms and matching cases; it rejects
missing arms, duplicates, and incompatible comparison identities. It cannot prove
that an equally partial set of arms covers the entire intended experiment.

### Metric definitions

- **TCR:** utility success on clean cases; utility success and attacker-goal failure
  on attacked cases. This is a benchmark outcome, not an independent audit of every action.
- **ASR:** attack-success rate. `report_metrics.py` divides by all included cases;
  `compare_judges.py` reports `asr_scored` over non-error attacks with a security
  result. The two denominators must not be treated as interchangeable.
- **Review load:** `reviews` counts all interventions; `hitl_load` counts those on
  utility-successful cases; `reviews_on_tcr_successes` counts those on safe
  completions. Report total reviews as well as success-conditioned counts.
- **Costs:** report agent, policy/judge, and reviewer tokens separately. Missing
  usage is unknown cost, not free inference. General-report TCR/ASR values are
  percentages; curve and matched-judge outputs use fractions.

### Known reporting limitations

The general report currently selects the largest case file for a cell without
preserving model/configuration/seed identity, and groups sources by folder basename.
Its `complete` flag compares observed counts rather than the expected task matrix.
It omits judge/preview/reviewer health counts, and partial reruns can include stale
decisions. Its `over_conservative` and `agent_failure` categories are heuristics,
not verified causes of failure. Audit the selected files, case IDs, and health
records before using these outputs for final tables.

`tcr_curve.py` also combines models/configurations and misparses hyphenated policy
variants. Its output filenames contain only the attack, so separate suite runs
can overwrite each other's plots and tables. Restrict input to a verified,
compatible set of runs with simple policy names and preserve each output before
running another suite. After checking those constraints, the command is:

```bash
python scripts/analyze/tcr_curve.py --runs "$RUN/runs" \
  --suite banking --attack important_instructions_generic
```

This writes under `results/tables/` and `results/figures/`, regardless of `--runs`.

## Running on NSCC

The tracked PBS launcher is [scripts/nscc/job.pbs](scripts/nscc/job.pbs). It starts
one vLLM server when agent/reviewer models match, or two when they differ, probes
the endpoints, then launches all configured arms concurrently.

It expects `.venv/bin/python`, a vLLM installation on `PATH` (it prepends
`$HOME/vllm-env/bin`), and locally cached model weights under `$SCRATCH_DIR/hf`.
It sets `HF_HUB_OFFLINE=1`. Override `SCRATCH_DIR`: its default is a contributor's
account-specific path. Queue, GPU, and wall-time directives also need to match the
allocated environment.

Example submission from the repository root, after provisioning that environment:

```bash
export SCRATCH_DIR='/scratch/path/for/your/account'
export MODEL='your-cached-model-name'
export ARMS='full_approve:inf full_escalate:3 cascade:3'
export EXTRA='--suite banking --logdir results/nscc-example/runs'
qsub -v SCRATCH_DIR,MODEL,ARMS,EXTRA scripts/nscc/job.pbs
```

Use a fresh `--logdir` for each submission. Other launcher settings include
`REVIEWER_MODEL`, `SEED`, `ATTACK`, `USER_TASKS`, `INJ`, and memory shares; see the
script for defaults. `EXTRA` forwards additional benchmark flags. Server and arm
console logs go to `results/nscc/`; benchmark artifacts use the supplied log directory.

The launcher's “say ok” probe does not validate the judge response contract, and
its concurrent execution differs from the sequential Python sweep. Before judge
experiments, validate the intended judge configuration on the running endpoint.
The `job_container.pbs` referenced in prior NSCC work is not included in this
checkout. Preserve the exact launcher, container/environment versions, model
revisions, and submission settings used for each result; the tracked script alone
does not establish how historical runs were produced.

## Development and reproducibility

```bash
python -m pytest -q
ruff check src scripts tests
ruff format --check src scripts tests
```

The repository currently has lint and formatting findings; these commands are
checks, not an assertion of a clean lint baseline. Offline tests exercise judge
contracts, simulator isolation, budget behavior, and workflows. The general
reporting scripts still need dedicated regression coverage.

| Location | Contents |
| --- | --- |
| `src/approval_gate/policies/`, `nodes/` | Policy decisions and cascade components. |
| `src/approval_gate/gate.py`, `executor.py` | Budget/reviewer handling and AgentDojo execution integration. |
| `src/approval_gate/reviewer/`, `judging.py`, `sandbox.py`, `effects.py` | Model review, automated judgment, and isolated previews. |
| `src/approval_gate/config/` | Tool classification, policy settings, and authorization fixtures. |
| `scripts/`, `scripts/analyze/`, `scripts/nscc/` | Experiment runners, offline analysis, and cluster launcher. |
| `tests/` | Offline regression and workflow tests. |
| `results/share/`, `results_csv_only/` | Retained experiment artifacts and exported tables. |
| `experiment-notes.md`, `SANDBOX_FINDINGS.md` | Archived discussion/workflows and recorded findings. |

Gated benchmark runs write sibling `runs/`, `cases/`, and `decisions/` directories.
Judge policies and matched full-approve/full-escalate controls also write
`manifests/`. The `undefended` runner produces AgentDojo trajectories without gate
case/decision logs. A case is a user task for clean runs or a user-task/injection-task
pair for attacked runs.

Preserve source, configuration, dependency versions, served model revisions,
launcher settings, and raw results before cleanup. Judge comparisons hash source
bytes, so even formatting runtime code can change their identities. Cascade runs
have less complete identity metadata and do not create the same manifests. A
snapshot taken today does not reconstruct the exact source used for older runs.

Common issues:

- **Missing plotting dependency:** install the `analysis` extra in the active environment.
- **Unset API key or wrong model:** check `.env`, the endpoint, and its served model name.
- **Judge health failure / exit code 2:** inspect saved health fields and diagnostics;
  a completed agent task does not mean the judge operated successfully.
- **Output directory already exists:** choose a new directory for the fixed-call study.
- **Comparison mismatch:** verify source/configuration, endpoints, package versions,
  selected cases, and calibration identity; retain unmatched results separately.

Historical results and development commentary are preserved in
[the experiment archive](experiment-notes.md) and [sandbox findings](SANDBOX_FINDINGS.md).
