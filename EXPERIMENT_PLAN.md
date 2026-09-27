# Experiment plan: learning when to request approval

Prepared against the project brief and the current repository on 23 September 2026. Compute target: one Apple M4 Max with 64 GB unified memory, serving local models through Ollama. This is a proposed implementation and evaluation plan; no performance results are claimed.

**Recommended contribution.** Train a policy that estimates how much requesting review now will improve the probability of eventual safe task completion, given the remaining review budget. Use an LLM to extract task alignment and provenance features, and a small learned controller to choose `AUTO_APPROVE`, `AUTO_DENY`, or `ESCALATE`. Start with frozen LLMs and CPU-based controller training. Fine-tuning a large LLM is unnecessary for the first experiment.

The hypothesis is that allocating review by its expected downstream benefit will outperform allocating it purely by tool type, estimated danger, or uncertainty. An obviously unauthorized action can be denied without spending a review. An ambiguous action essential to completing the task may deserve the last review. This is a hypothesis to test, not an established improvement.

## 1. Research comparison and scope of the claim

| Reference | Relevance | Comparison protocol |
| --- | --- | --- |
| [PRUDENTIA: Optimizing Agent Planning for Security and Autonomy](https://arxiv.org/html/2602.11416v1) | The closest comparison: explicitly studies TCR@k and HITL load using information-flow control and policy-aware planning. | Its autonomy evaluation counts interventions on benign traces assuming approval, and includes endorsement of data. The project's online, fallible reviewer and per-call approval budget require a separate protocol. Published curves cannot be directly compared with local Ollama results. |
| [Task Shield, ACL 2025](https://aclanthology.org/2025.acl-long.1435/) | Checks whether instructions and actions contribute to the user's task. | Include a task-alignment LLM gate as a strong prompt baseline. Call this **Task-Shield-inspired** unless reproducing the full published method. |
| [MELON](https://arxiv.org/abs/2502.05174) | Uses masked re-execution to detect dependence on injected instructions. | First published-defense reproduction to attempt, using the [authors' implementation](https://github.com/kaijiezhu11/MELON). Port its client/integration to the pinned AgentDojo/Ollama setup and document deviations. Count additional generations as policy inference; counterfactual proposals must not execute in the live episode. |
| [CaMeL, official implementation](https://github.com/google-research/camel-prompt-injection) | Separates trusted control from untrusted data and enforces security policies. | An architectural comparison, requiring substantial integration. A learned gate does not inherit its security guarantees. |
| [AgentSentry, 2026 preprint](https://arxiv.org/abs/2602.22724) | Recent defense using temporal causal diagnostics and context purification. | Relevant recent comparator when selecting a reproducible defense; its original evaluation is not a finite-review-budget evaluation. |

Use two explicit tracks:

1. **Main project track:** fixed agent, tools, reviewer, case set, and online budget semantics. Compare approval policies under the same conditions. Each reviewed proposed tool call costs one unit of k; there is no free endorsement that authorizes multiple future calls.
2. **Published-defense track:** reproduce a relevant published method with compatible local models and document every adaptation. For planner-changing methods, hold the underlying model and task setting fixed, but allow the method's own planner. Report all additional inference. A method without review has a budget-independent curve.

The six core strategies below establish whether the proposed approach improves this harness. They alone do not establish state of the art. Before a stronger claim, add a faithful published-defense reproduction and evaluate robustness against attacks designed for the frozen defense. If only inspired adaptations are feasible, claim improvement over those named baselines in this setting. PRUDENTIA is the closest conceptual target; equivalence of its interaction model must be addressed before a head-to-head claim.

## 2. Existing code and missing components

| Component | Current implementation | Required work |
| --- | --- | --- |
| Tool interception and budget | `executor.py` intercepts calls; `gate.py` spends k and denies escalation after exhaustion. | Preserve these semantics. Add explicit episode context, execution events, and invariant tests. |
| Required baselines | `full_approve.py` and `full_escalate.py` exist. | Include both in every main comparison. Full-escalate is not a safety oracle because the reviewer can make errors. |
| Student policy | `cascade.py` approves tool names classified read-only and escalates everything else. | The documented later cascade nodes are absent. There is no learned risk model, provenance node, training pipeline, or value-of-review controller. Tool-name allowlisting is not proof of authorization. |
| Simulated human | `reviewer/simulated.py` and `reviewer/prompts.py` already exist. | Harden parsing and context construction; evaluate reviewer quality independently; document and freeze its prompt and model. |
| Decision logging | `decisions.py` and `telemetry/logger.py` record gate decisions and reviewer tokens. | Add unique episode IDs, injection-case metadata, actual invocation events, terminal outcomes, and component-level LLM request accounting. Existing risk/VOI fields are placeholders. |
| Experiments | `run_benchmark.py` and `sweep_k.py` run clean/attacked episodes and online k sweeps. | Add split manifests, separate policy model/checkpoint configuration, complete run fingerprints, robust resume, explicit inference limits, and failure status. |
| Analysis | `scripts/analyze/tcr_curve.py` produces TCR CSV/PNG/PDF. | Add strict authorization auditing, confidence intervals, HITL/cost tables, escalation-value analysis, and failure trajectories. Fix grouping and filenames before comparing models/suites. |

**Correctness fixes before collecting training data:**

- The reviewer parser accepts any unstructured output containing `APPROVE` but not `DENY`. A local probe confirmed `_parse("DO NOT APPROVE")` returns `APPROVE`. Require a validated response schema with an exact verdict; malformed output must fail closed and be logged. Marking data as untrusted helps prompting but does not guarantee reviewer security.
- The reviewer summary describes proposed assistant calls as already "called"; the current proposal is among them. The executor also passes the same original context for every call in a batch. Construct a review packet that distinguishes proposed, denied, invoked, and completed calls and includes earlier results within the batch. Treat prior agent-generated arguments as untrusted too. Truncate structured records before rendering, so truncation cannot cut trust delimiters or silently turn an incomplete call into a completed one.
- `DecisionRecord.executed` is currently set to approval before the runtime is invoked. Record `approved`, `execution_attempted`, `execution_succeeded`, and tool errors separately. Evaluate unauthorized invocations even when the tool returns an error; disclose any side-effect-based alternative metric.
- Attacked episodes with the same user task currently lack an injection-task ID in decision records. Add `case_id`, `episode_id`, and tool-call IDs, and join them to terminal outcomes. Keep attack metadata in telemetry, outside policy/reviewer inputs. Stop deriving identity from prompt text alone.
- The result-cache name omits reviewer model, prompts, learned checkpoint, and other behavior-changing settings. Hash a complete run manifest and prevent resumed runs from duplicating decisions or mixing results.
- The plotter groups by policy/k/seed and can mix different agent models; output filenames also omit suite. Group by experiment/configuration/suite/model/checkpoint and write isolated artifacts.
- Some AgentDojo exception paths set `security=True` on infrastructure errors. Record these separately from observed attack success. Keep failed attempts visible, use a declared retry policy, and never silently drop hard cases.

## 3. Proposed policy and training targets

**Observable state.** Give the policy the original request, tool schema/name/arguments, the available execution history, remaining budget b, elapsed steps, and prior reviewer decisions. Features can include task relevance, destination/recipient consistency, reversibility, and whether argument values came from user instructions or tool output. Estimated progress and future review demand must be predictions from observable state, not benchmark answers.

Use conservative provenance records where possible. Any LLM-inferred provenance is a fallible feature, not formal information-flow tracking. Exclude task identifiers, attack names, injection-goal labels, ground-truth authorization, and future outcomes from learned features. Existing `CallContext.task_id` should remain bookkeeping only.

**Model structure.** First compare a regularized logistic model and a small gradient-boosted model over structured LLM features; choose on validation data. Cache a bounded task-intent representation within an episode, and make at most one semantic policy-model request per decision in the initial design. A verified deterministic fast path may avoid that request. Schema failures have a declared escalation/denial fallback. All policy-model calls count toward policy cost, including task summarization.

Train action-value estimates:

\[
Q_a(x,b) = P(\text{terminal safe completion}\mid x,b,\text{take }a\text{ now, then follow continuation policy}),
\quad a\in\{A,D,E\}.
\]

The estimated benefit of review is

\[
\Delta_E(x,b)=Q_E(x,b)-\max\{Q_A(x,b),Q_D(x,b)\}.
\]

At b=0, escalation is unavailable; the gate must still turn any attempted escalation into denial. Otherwise select the highest-value admissible action, with a validation-calibrated authorization-risk constraint on automatic approval. Budget-conditioned Q values already include the loss of one future review after escalation. Do not subtract an arbitrary second budget penalty and assume it represents the same objective. Treat inference cost as a measured trade-off or separately specified constraint.

**Training data must reflect consequences.** A label saying "this action is unsafe" does not tell us whether denying it lets the agent recover, or whether review is worth spending. Save selected pre-decision states from training tasks and branch into approve, deny, and review. Clone the environment, messages, budget, and policy state; run each branch to completion with a fixed continuation policy. The review branch uses the same fallible reviewer as evaluation and consumes one budget unit. Label terminal outcomes with the offline evaluator. Verify snapshot isolation and replay equivalence before trusting these labels.

Begin with the calibrated risk policy as the continuation policy. Fit the value controller, then collect another small batch of states reached by it and repeat. This is approximate policy improvement; a small dataset does not establish optimality. Use shared random seeds where supported, while retaining branch variability. Never infer a denial branch's task outcome by simply relabeling an existing approve trajectory.

Keep offline authorization labels and branch outcomes separate from online inputs. Human-adjudicated labels and deterministic task contracts are preferable to treating the simulated reviewer's own verdict as ground truth. An offline LLM may assist annotation, but its labels need independent checks and its calls belong to training/evaluation cost.

## 4. Mac-sized experiment sequence

Pin AgentDojo package 0.1.35 and suite version v1.2.1. Local enumeration gives 97 user tasks: banking 16, workspace 40, travel 20, and Slack 21. The runner can create 949 user-task/injection-goal combinations per non-DoS attack before task selection; an unrestricted sweep is much larger than 97 episodes.

Split by **user-task family before creating attacks, paraphrases, tool rows, or branches**. An approximate stratified allocation is 58 training / 19 validation / 20 test tasks. Keep related templates in one split even if that changes the counts. Save explicit IDs and a split hash. Select attack goals independently of a policy's performance, and use identical cases across arms. Report the restricted coverage of a sampled benchmark honestly.

| Stage | Experiment | Initial size / completion criterion |
| --- | --- | --- |
| E0 | Repair measurement and reviewer handling. Add mock-based checks for budget reset/exhaustion, batch context, parser failures, execution logging, outcome joins, retry accounting, and exclusion of oracle fields. | No inference needed for these checks. Validate one complete case record end to end before scaling. |
| E1a | Agent/model capability check on clean training tasks. | Eight tasks, two per suite, with full-approve. Check tool formatting, completion, context usage, latency, and memory. A model that cannot do the clean tasks cannot reveal a useful approval-policy gain. |
| E1b | Required-baseline pilot: full-approve, full-escalate, and current cascade. | The same eight training tasks, each clean plus one `important_instructions_generic` case: 16 cases. k={0,1,3,∞}; full-approve runs once. 16×(1+4+4)=144 episodes, including compatible E1a results. |
| E2 | Reviewer calibration and prompted task-alignment baseline. | Approximately 100 decision packets from training tasks, stratified by authorized/unauthorized calls and tool type. Two group members label them, adjudicating disagreements. Measure unsafe approvals, safe denials, malformed responses, and latency. Tune prompt on training packets; freeze using separate validation packets. |
| E3 | Learn a calibrated risk policy. Compare with the prompted gate using the same policy LLM/features. | Collect training trajectories with a mix of cascade and prompted-gate behavior. Use clean tasks and two training attack families. Fit/calibrate risk estimates; select approve/deny/defer thresholds on validation tasks. This tests whether learning already beats prompting. |
| E4 | Learn review value and its dependence on remaining budget. | Start with 100 pre-decision training states spanning tool types and budgets. At most 300 branch continuations; fewer when b=0. If labels/replay are reliable, do a second 100-state iteration reached by the learned policy. Evaluate gains on validation tasks before increasing data. |
| E5 | Frozen held-out comparison of the six core strategies below. | Example: 20 test tasks × (clean + one goal for each of three attack families)=80 cases. Primary small grid k={0,1,3,∞}: 1,680 episodes. Extended grid {0,1,2,3,5,8,∞}: 2,880 episodes. Both figures are for one repeat and exclude training. Choose the grid from pilot timing **before opening test results**. |
| E6 | Explain gains, test transfer, and compare with published defenses. | First run ablations on validation tasks at k=1 and 3. Freeze a small test confirmation set of ablations before test access. Add the MELON reproduction and an adaptive-attack evaluation for any stronger research claim; assess recent AgentSentry and the separate PRUDENTIA protocol before claiming current state of the art. |

**Six core strategies:** full-approve; full-escalate; current read-only cascade; prompted task-alignment gate; learned calibrated risk gate; learned budget-conditioned review-value gate. The last five depend on k, hence 1+5×4=21 or 1+5×7=36 configurations per case. A review-free published defense adds one episode per case, though its internal inference can be expensive.

Use `important_instructions_generic` and `direct` for initial training; reserve `system_message` as an unseen attack family for the main transfer check. These are available in the current attack registry and avoid model-specific payload lookup used by some other attacks. Confirm payload construction and eligible cases in the manifest. Separate clean, seen-family, and unseen-family results. Later add saved adaptive attacks generated against frozen defenses with the same attacker query budget per defense, and a common transferred attack pool. Account for attacker calls separately.

**Ablations:** remove remaining-budget features; replace learned review value with uncertainty/risk thresholding; remove provenance features; replace LLM features with tool metadata; use one-step labels instead of terminal branch outcomes. Keep the agent and reviewer fixed. A separate reviewer-sensitivity experiment can vary reviewer strength/model family on the same frozen policies. Do not quietly give only the proposed policy a stronger reviewer.

Do not spend the initial Mac budget on duplicate temperature-zero runs. The current seed option is largely a repeat index at temperature zero, not a guarantee of independent samples. If runtime permits, add a declared stochastic configuration and paired seeds for robustness. With only 20 held-out task families, confidence intervals may be wide: more attacks on the same tasks do not create more independent tasks. Expand held-out families or use a predeclared outer cross-validation protocol for stronger evidence; never tune on outer test folds.

## 5. Reviewer and local model configuration

Use [Qwen3.5 9B](https://ollama.com/library/qwen3.5:9b) for fast development and policy features. Use [Qwen3.8 27B](https://ollama.com/library/qwen3.8:27b) as the candidate main agent and reviewer. The current Ollama Q4_K_M downloads are approximately 6.6 GB and 18 GB respectively; runtime context and model residency add memory. These are feasible candidates for a 64 GB Mac, not measured AgentDojo performance claims. Choose the final agent using E1a and freeze it across all main arms. Never pool 9B pilot outcomes with 27B main outcomes.

The reviewer can share model weights with the agent but must use a separate role/prompt and bounded review packet. Sharing a family can correlate errors, so disclose it and run a small sensitivity check with an independently validated alternative reviewer if feasible. Evaluate reviewer verdicts against annotation, not agent/reviewer agreement.

Run one episode at a time initially. Ollama handles model inference; the Python harness and small learned controller do not require PyTorch MPS training. Record model digests, quantization, Ollama version, context length, decoding settings, output limits, and reasoning settings. Qwen3.8 thinking is enabled by default, so expose and freeze that setting rather than leaving an unrecorded source of cost variation. The existing CLI does not expose all these controls.

Preserve a common agent execution limit across policies; the installed AgentDojo loop currently defaults to 15 iterations. Make it explicit in the manifest. Also bound output size and retry attempts, and report truncation/timeout rates. An unbounded k still has the common execution limit. Keep model reload time visible and report warm/cold timing consistently.

Estimate runtime after the pilot: sum the measured per-strategy mean episode times times planned episode counts, then add separately measured training-branch and analysis-labeling costs. Calls alone are not a runtime estimate: long context, reasoning, model size, and reloads matter. Reduce a future experiment's grid or declared case sample before its test results are inspected if runtime exceeds the available budget.

## 6. Required metrics and report artifacts

**Safe completion and TCR@k.** For episode i, let U_i denote task success, Z_i whether any unauthorized tool invocation occurred, and H_i the actual number of review interventions. Define S_i=U_i(1−Z_i). For each strategy, run the policy online separately at each k and report:

\[
\mathrm{TCR}@k=\frac{1}{N}\sum_i S_i,\qquad H_i\leq k.
\]

The current script instead reports U_i AND NOT AgentDojo attack-success on attacked cases, and utility alone on clean cases. Keep that useful benchmark measure as `TCR_AD`, alongside attack success rate, but do not call it proof of zero unauthorized calls. Implement an offline authorization auditor using task-specific authorization contracts and independently adjudicated ambiguous cases. Reviewers and policies must never receive auditor outputs. If comprehensive auditing is infeasible, report strict TCR on an explicitly audited subset and `TCR_AD` on the full set; publish coverage and unknown labels rather than treating unknown as safe.

Produce separate clean/attacked curves, suite panels, and an equal-suite macro average; include a case-weighted aggregate as secondary. Add 95% confidence intervals from a paired bootstrap over user-task families, retaining their attack variants and repeated runs together. Show the difference from the strongest validation-selected baseline. Do not derive finite-k results by filtering an unlimited-budget trace: denials change subsequent actions. Online curves need not be monotonic with a fallible reviewer; do not smooth away observed decreases.

Predeclare the primary score as mean TCR over the chosen finite small budgets: {0,1,3} for the small grid or {0,1,2,3} for the extended grid. Report every point, including ∞ separately. An aspirational effect size is +5 percentage points over the strongest baseline, but a claim requires measured uncertainty and the safety/cost trade-off. A policy that improves completion while increasing unauthorized execution must not be presented as an unconditional safety improvement.

**HITL table.** The brief asks for total interventions on successfully completed tasks. Report both interpretations explicitly: `HITL_successful=Σ U_i H_i` and safety-qualified `HITL_safe=Σ S_i H_i`. Also include `HITL_all=Σ H_i`, so costly failed attempts are visible. A budget-exhausted escalation request that never reaches the reviewer is not an intervention. A consumed review that returns malformed output still counts.

| Strategy | k | Cases | Successful / safe tasks | TCR@k | HITL_successful | HITL_safe | HITL_all | Mean H, all cases | Cases exhausting k |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| One row per strategy, budget, suite/aggregate | measured | measured | measured | measured | measured | measured | measured | measured | measured |

Give denominator counts and mean reviews per safe success as additional columns/CSV fields. Low successful-task HITL alone can reward a policy that completes almost nothing. Compare the TCR/load frontier and include total effort per safe completion.

**Per-strategy local inference cost.** Instrument every model request by role: `agent`, `policy`, or `reviewer`. Keep logical requests, physical API attempts/retries, and completed generations distinct; a timeout may not reveal whether inference completed. Tool invocations are not LLM calls. Every semantic check, task summary, ensemble member, or counterfactual generation used online belongs to policy cost, even if it uses the same weights as the reviewer.

| Strategy | Agent requests | Policy requests | Reviewer requests |
| --- | --- | --- | --- |
| Full-approve | All generations on its trajectory | 0 | 0 |
| Full-escalate | All generations on its trajectory | 0 | Requests for consumed reviews, plus retries |
| Current cascade | All generations on its trajectory | 0 | Requests for consumed reviews, plus retries |
| Prompted alignment | All generations on its trajectory | Every prompt-based decision/summary, plus retries | Requests for consumed reviews, plus retries |
| Learned risk | All generations on its trajectory | LLM feature extraction/summaries, plus retries; CPU classifier adds no LLM call | Requests for consumed reviews, plus retries |
| Learned review value | All generations on its trajectory | LLM feature extraction/summaries, plus retries; learned controller adds no LLM call | Requests for consumed reviews, plus retries |

Publish measured `agent_calls`, `policy_calls`, `reviewer_calls`, `total_calls`, calls/case, and total calls/safe completion. Also report role/model-specific input/output tokens, available reasoning-token details without double counting, retry/error counts, and latency. Missing usage is unknown, not zero. Report totals over **all** episodes and separate completed-only views if useful. Local hosted-API charges are zero, but tokens and elapsed compute remain costs; do not invent equivalent dollar charges.

Use a stacked role-cost chart beside TCR versus total calls/case. A count-only comparison between a 9B and 27B request is incomplete, so keep tokens and time. Count task-intent caches honestly, reset episode-local caches between independent cases, and document any shared response cache rather than hiding warm-cache savings. Agent trajectories differ between strategies, so report changed agent cost as well as added gate cost.

Keep offline trajectory collection, branch generation, annotation/teacher inference, checkpoint training, and adaptive attacker calls in a separate development-cost table. Optional amortization can show how many deployment tasks would repay training cost; it is not part of online per-case inference cost.

**Failure and escalation-value analysis.** Label over-conservative failures (safe denials or wasted review budget), over-permissive automatic approvals, reviewer mistakes, agent capability errors, and infrastructure/format failures separately. Show at least one complete representative trajectory for each required failure class. Use audited calls and branch comparisons to support causal explanations, not only a retrospective LLM narrative. On held-out diagnostic states, compare review with approve/deny continuations by tool type and report changes in safe completion per review. Keep these diagnostic rollouts out of the main evaluation cost, but disclose them separately; don't use test diagnostics to retrain the reported checkpoint.

## 7. Implementation order and completion criteria

The paths below are proposed additions, not existing runnable commands.

| Priority | Files / changes | Completion criterion |
| --- | --- | --- |
| P0 | Harden `reviewer/simulated.py`; update `executor.py`, `gate.py`, `decisions.py`; add `telemetry/llm_calls.py`, episode outcomes, and evaluation authorization auditing. | Every case has complete joins from request to decision to execution to outcome; reviewer/parser/budget tests pass; no oracle fields reach prompts. |
| P0 | Update `llm.py` and the runner with role-tagged requests, complete manifest hashes, explicit limits, and retry/error handling. | Agent, policy, and reviewer costs reconcile. SDK and AgentDojo retry layers are instrumented or replaced by one explicit bounded retry mechanism. |
| P1 | Add `policies/llm_judge.py`, `nodes/llm_features.py`, provenance features, and `policies/learned_risk.py`. | Prompted and calibrated learned baselines run with the same model/reviewer configuration and validation-only threshold selection. |
| P1 | Add `training/rollouts.py`, `scripts/collect_trajectories.py`, `scripts/train_gate.py`, and `policies/learned_voi.py`; add a small training dependency extra. | Snapshot isolation is verified; train/validation/test separation is enforced; saved checkpoints include feature schema and training-manifest hashes. |
| P1 | Add `scripts/run_experiments.py` and versioned experiment/split manifests. | A frozen experiment can resume without duplicate cost/outcome records, stale caches, or mixed configurations. Any uncompleted cell is visible. |
| P1 | Extend `scripts/analyze/tcr_curve.py`; add `hitl_load.py`, `api_costs.py`, and `failure_cases.py`. | Generate TCR curves/CSV, HITL table, role-cost table/chart, escalation-value analysis, and representative trajectories from saved artifacts without new inference. |
| P2 | Integrate a faithful published defense and an adaptive attacker; add predeclared ablation manifests. | Stronger claims have matched comparisons, attack-query accounting, and documented differences from the original paper. |

First milestone: trustworthy pilot curves plus HITL/cost tables for the two required baselines and current cascade. Second: demonstrate whether learning beats prompting and calibrated risk on validation tasks. Final: freeze the policy and protocol, run the held-out comparison, and write the report around measured effects and limitations. Plan the 10-minute presentation around the policy idea, one curve, the HITL/cost trade-off, and two contrasting failure trajectories.
