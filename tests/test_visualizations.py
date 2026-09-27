import copy
import tempfile
import unittest
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from approval_gate.visualizations import (
    _audit,
    _cost,
    _gallery,
    _hitl,
    _tcr,
    budget_label,
    matched_groups,
)


def tables(policy="full_escalate", k=3, comparison="fixture", attack="clean"):
    base = dict(
        comparison=comparison,
        suite="micro",
        attack=attack,
        strategy=policy,
        policy=policy,
        k=k,
        n=4,
    )
    metric = dict(
        base,
        tcr_ad=0.5,
        tcr_ad_ci_low=0.0,
        tcr_ad_ci_high=1.0,
        tcr_strict=None,
        tcr_strict_ci_low=None,
        tcr_strict_ci_high=None,
        strict_lower_bound=0.25,
        strict_upper_bound=0.75,
        completed_episodes=2,
        limited_episodes=1,
        execution_errors=1,
        executions=4,
        audited_executions=1,
    )
    cost = dict(
        base,
        agent_calls=8,
        policy_calls=4,
        reviewer_calls=6,
        calls_per_case=4.5,
        seconds_per_case=20,
    )
    hitl = dict(
        base,
        HITL_successful=2,
        HITL_all_selected=6,
        mean_reviews_all=1.5,
        budget_exhaustion_denied_cases=2,
    )
    return metric, cost, hitl


class VisualizationTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_comparison_and_attack_groups_do_not_mix_or_double_count_aliases(self):
        cells = [
            tables(comparison=c, attack=a)
            for c in ("a", "b")
            for a in ("clean", "none", "all_attacked")
        ]
        metrics, costs, hitl = map(list, zip(*cells))
        groups = matched_groups(metrics, costs, hitl)
        self.assertEqual(
            set(groups), {(c, "micro", a) for c in ("a", "b") for a in ("clean", "all_attacked")}
        )
        self.assertTrue(all(len(rows) == 1 for rows in groups.values()))
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            matched_groups(metrics, costs + costs[:1], hitl)
        with self.assertRaisesRegex(ValueError, "Missing matching"):
            matched_groups(metrics, costs[1:], hitl)
        hitl[0] = {**hitl[0], "n": 8}
        with self.assertRaisesRegex(ValueError, "denominators"):
            matched_groups(metrics, costs, hitl)

    def test_hitl_and_api_charts_use_all_cases_as_denominator(self):
        row = tables()
        colors = {"full_escalate": "red"}
        hitl = _hitl([row], colors)
        # Two reviews in successful episodes / four cases, not / successful cases.
        self.assertEqual([p.get_width() for p in hitl.axes[0].patches], [0.5, 1.0])
        self.assertEqual([p.get_width() for p in hitl.axes[1].patches], [0.5])
        costs = _cost([row], colors)
        self.assertEqual([p.get_width() for p in costs.axes[0].patches], [2.0, 1.0, 1.5])

    def test_audit_distinguishes_unknown_from_zero_executions_and_status_from_success(self):
        row = tables()
        empty = copy.deepcopy(row)
        empty[0].update(executions=0, audited_executions=0)
        fig = _audit([row, empty], {})
        status, audit = fig.axes
        np.testing.assert_allclose(
            [p.get_width() for p in status.patches], [0.5, 0.5, 0.25, 0.25, 0.25, 0.25]
        )
        np.testing.assert_allclose([p.get_width() for p in audit.patches], [0.25, 0, 0.75, 0])
        self.assertIn("N/A: no calls executed", [text.get_text() for text in audit.texts])
        self.assertIn("1/4 audited", [text.get_text() for text in audit.texts])

    def test_full_approve_repeats_reference_while_full_review_is_infinite_budget(self):
        rows = [tables("full_approve", "inf"), tables(k=0), tables(k=3), tables(k="inf")]
        colors = {"full_approve": "gray", "full_escalate": "red"}
        self.assertEqual(budget_label(rows[0][0]), "any")
        self.assertEqual(budget_label(rows[-1][0]), "∞")
        fig = _tcr(rows, colors)
        baseline = next(line for line in fig.axes[0].lines if line.get_label() == "Full-approve")
        np.testing.assert_array_equal(baseline.get_ydata(), [0.5, 0.5, 0.5])
        self.assertEqual(
            [tick.get_text() for tick in fig.axes[0].get_xticklabels()], ["0", "3", "∞"]
        )

    def test_missing_strict_labels_are_bounds_not_zero_success_estimates(self):
        metric, cost, hitl = tables()
        colors = {"full_escalate": "red"}
        fig = _tcr([(metric, cost, hitl)], colors)
        point_lines = [line for line in fig.axes[1].lines if line.get_marker() == "o"]
        self.assertTrue(point_lines)
        self.assertTrue(all(np.isnan(line.get_ydata(orig=False)).all() for line in point_lines))
        metric.update(
            tcr_strict=0.25, tcr_strict_ci_low=0.0, tcr_strict_ci_high=0.5, strict_upper_bound=0.25
        )
        audited = _tcr([(metric, cost, hitl)], colors)
        point = next(line for line in audited.axes[1].lines if line.get_marker() == "o")
        np.testing.assert_array_equal(point.get_ydata(), [0.25])

    def test_gallery_marks_diagnostics_and_escapes_labels(self):
        with tempfile.TemporaryDirectory() as temp:
            _gallery(
                Path(temp), [(("fixture", "micro", "<unsafe>"), [("tcr", "tcr_fixture")])], True
            )
            html = (Path(temp) / "index.html").read_text()
            self.assertIn("&lt;unsafe&gt;", html)
            self.assertNotIn("<unsafe>", html)
            self.assertIn("incomplete or unmatched", html)
            self.assertIn("tcr_fixture.pdf", html)
            self.assertNotIn("https://", html)


if __name__ == "__main__":
    unittest.main()
