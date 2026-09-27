"""Fit risk or downstream-value heads using disjoint train/validation families."""

import argparse
import json
from pathlib import Path
from approval_gate.annotations import read_annotations, require_human, label_summary
from approval_gate.evaluation import selected_episodes, labels_from, authorization
from approval_gate.nodes.llm_features import FEATURES, vector
from approval_gate.policies.learned import predict
from approval_gate.telemetry.events import fingerprint, write_json
from approval_gate.training.models import fit


def risk_rows(path, split):
    rows = read_annotations(path)
    out = []
    for r in rows:
        if r["split"] != split:
            raise ValueError(f"{path}: expected only {split} rows")
        if split == "validation" or r.get("label_source") != "llm":
            require_human(r, f"risk {split}")
        elif not r.get("machine_annotation"):
            raise ValueError("LLM training labels need saved machine_annotation provenance")
        if r.get("authorized") is None or r.get("feature_parse_failed"):
            continue
        if type(r["authorized"]) is not bool:
            raise ValueError("authorization labels must be boolean")
        if set(r["features"]) != set(FEATURES):
            raise ValueError("collect semantic features with llm_judge before risk training")
        out.append({**r, "label": int(not r["authorized"]), "action": "risk"})
    return out


def value_rows(root, split, target, labels):
    rows = []
    for directory, outcome in selected_episodes(root):
        path = directory / "training_example.json"
        if not path.exists():
            continue
        r = json.loads(path.read_text())
        if r["split"] != split:
            raise ValueError(f"{path}: wrong split")
        if split == "train" and r["phase"] != "training":
            raise ValueError("diagnostic rollouts cannot train the policy")
        if r["status"] == "error" or r["parse_failed"]:
            continue
        label = (
            r["target_ad"]
            if target == "agentdojo"
            else authorization(directory, outcome, labels)["strict_success"]
        )
        if label is None:
            continue
        if set(r["features"]) != set(FEATURES):
            raise ValueError("snapshot lacks semantic features")
        rows.append({**r, "label": int(label)})
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kind", choices=["risk", "voi"], required=True)
    p.add_argument("--train", required=True)
    p.add_argument("--validation", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--target", choices=["strict", "agentdojo"], default="strict")
    p.add_argument(
        "--annotations", help="combined prefix and branch call annotations for strict VOI targets"
    )
    p.add_argument("--risk-checkpoint", help="required authorization guard for VOI")
    p.add_argument("--approve-threshold", type=float, default=0.1)
    p.add_argument("--deny-threshold", type=float, default=0.9)
    p.add_argument(
        "--ablation", choices=["none", "no_budget", "no_provenance", "tool_only"], default="none"
    )
    args = p.parse_args()
    if Path(args.output).exists():
        raise SystemExit("Choose a new checkpoint name; checkpoints are immutable")
    if not 0 <= args.approve_threshold < args.deny_threshold <= 1:
        raise ValueError("invalid risk thresholds")
    if args.kind == "risk":
        train, val = risk_rows(args.train, "train"), risk_rows(args.validation, "validation")
    else:
        if not args.risk_checkpoint:
            raise ValueError("--risk-checkpoint is required for the VOI auto-approval guard")
        labels = labels_from(args.annotations)
        train = value_rows(args.train, "train", args.target, labels)
        val = value_rows(args.validation, "validation", args.target, labels)
    if not train or not val:
        raise ValueError(
            "No usable labels; complete train labeling and human validation annotations/rollouts"
        )
    if {r["family"] for r in train} & {r["family"] for r in val}:
        raise ValueError("train/validation task-family leakage")
    models = {r["policy_model"] for r in train + val}
    if len(models) != 1:
        raise ValueError("feature-model mismatch across datasets")
    signatures = {r.get("feature_signature") for r in train + val}
    if len(signatures) != 1 or None in signatures:
        raise ValueError("mixed or missing feature prompt/model/inference signatures")
    heads, reports = {}, {}
    for action in ["risk"] if args.kind == "risk" else ["AUTO_APPROVE", "AUTO_DENY", "ESCALATE"]:
        tr, va = (
            [r for r in train if r["action"] == action],
            [r for r in val if r["action"] == action],
        )
        if len(tr) < 2 or len(va) < 2:
            raise ValueError(f"need at least two train/validation examples for {action}")
        candidates = []
        for l2 in (0.01, 0.1, 1.0):
            head = fit(
                [vector(r["features"], args.ablation) for r in tr], [r["label"] for r in tr], l2=l2
            )
            brier = sum(
                (predict(head, vector(r["features"], args.ablation)) - r["label"]) ** 2 for r in va
            ) / len(va)
            candidates.append((brier, head))
        score, heads[action] = min(candidates, key=lambda item: item[0])
        reports[action] = {
            "validation_brier": score,
            "train_n": len(tr),
            "validation_n": len(va),
            "train_class_counts": [sum(r["label"] == y for r in tr) for y in (0, 1)],
        }
    checkpoint = dict(
        kind="learned_" + args.kind,
        features=FEATURES,
        heads=heads,
        ablation=args.ablation,
        policy_model=next(iter(models)),
        feature_signature=next(iter(signatures)),
        approve_threshold=args.approve_threshold,
        deny_threshold=args.deny_threshold,
        target="authorization" if args.kind == "risk" else args.target,
        training_hash=fingerprint(train),
        validation_hash=fingerprint(val),
        training_families=sorted({r["family"] for r in train}),
        validation_families=sorted({r["family"] for r in val}),
        validation=reports,
    )
    if args.kind == "risk":
        checkpoint["annotation_provenance"] = {
            "train": label_summary(read_annotations(args.train)),
            "validation": label_summary(read_annotations(args.validation)),
            "training_uses_weak_labels": any(r.get("label_source") == "llm" for r in train),
        }
        if len({r["label"] for r in train}) < 2 or len({r["label"] for r in val}) < 2:
            raise ValueError(
                "risk training and calibration require both authorized and unauthorized labels"
            )
        # Held-out Platt-style probability calibration. Brier above is pre-calibration.
        predictions = [[predict(heads["risk"], vector(r["features"], args.ablation))] for r in val]
        checkpoint["calibrator"] = fit(predictions, [r["label"] for r in val], l2=0.1)
    else:
        guard = json.loads(Path(args.risk_checkpoint).read_text())
        if guard["kind"] != "learned_risk" or guard["policy_model"] != checkpoint["policy_model"]:
            raise ValueError("risk guard mismatch")
        if (
            guard["ablation"] != args.ablation
            or guard["feature_signature"] != checkpoint["feature_signature"]
        ):
            raise ValueError("train a matching risk guard for this feature/ablation configuration")
        checkpoint["risk_guard"] = guard
    write_json(args.output, checkpoint)
    print(json.dumps(checkpoint["validation"], indent=2))
    print(
        f"Wrote {args.output}; small or single-class action datasets need more data before research claims."
    )


if __name__ == "__main__":
    main()
