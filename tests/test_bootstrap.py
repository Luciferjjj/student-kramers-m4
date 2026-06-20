"""Focused tests for extendable IOS and bootstrap checkpoints."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from student_kramers import config
from student_kramers.bootstrap import (
    run_contrast_bootstrap,
    run_exact_ios,
    run_parametric_bootstrap,
    summarize_ios,
)
from student_kramers.ios_analysis import (
    build_ios_pairwise_comparison,
    build_ios_regime_summary,
)


class BootstrapCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.params = config.REFERENCE_PARAMS_BY_MODEL["M2"]
        self.data = np.column_stack([np.linspace(-1.0, 1.0, 5), np.zeros(5)])

    def test_sampled_ios_can_be_extended_without_rerunning(self):
        calls = []

        def fake_fit(model_name, data, free_hat, k, maxiter, start_candidates):
            calls.append(k)
            return {
                "free_minus": free_hat.copy(),
                "heldout_nll_under_loo": 2.0,
                "optimizer_success": True,
                "optimizer_message": "ok",
                "nit": 1,
                "nfev": 1,
                "loo_valid": True,
                "seconds": 0.1,
            }

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "ios.csv"
            with (
                patch(
                    "student_kramers.bootstrap.partial_transition_nlls",
                    return_value=np.ones(4),
                ),
                patch("student_kramers.bootstrap.fit_leave_one_out", fake_fit),
            ):
                run_exact_ios(
                    "M2", self.data, self.params, output,
                    indices=[0], save_every=1, verbose=False,
                )
                table = run_exact_ios(
                    "M2", self.data, self.params, output,
                    indices=[0, 1, 2], save_every=1, verbose=False,
                )

        self.assertEqual(calls, [0, 1, 2])
        self.assertEqual(list(table["k"]), [0, 1, 2])

    def test_failed_ios_row_is_retried_and_excluded_from_formal_statistic(self):
        calls = []

        def fake_fit(model_name, data, free_hat, k, maxiter, start_candidates):
            calls.append(k)
            valid = calls.count(k) > 1
            return {
                "free_minus": free_hat.copy(),
                "heldout_nll_under_loo": 2.0 if valid else np.nan,
                "optimizer_success": valid,
                "optimizer_message": "ok" if valid else "failed",
                "nit": 1,
                "nfev": 1,
                "loo_valid": valid,
                "seconds": 0.1,
            }

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "ios.csv"
            with (
                patch(
                    "student_kramers.bootstrap.partial_transition_nlls",
                    return_value=np.ones(4),
                ),
                patch("student_kramers.bootstrap.fit_leave_one_out", fake_fit),
            ):
                first = run_exact_ios(
                    "M2", self.data, self.params, output,
                    indices=[0], save_every=1, verbose=False,
                )
                first_summary = summarize_ios(first, expected_n_transitions=4)
                second = run_exact_ios(
                    "M2", self.data, self.params, output,
                    indices=[0], save_every=1, verbose=False,
                )

        self.assertEqual(calls, [0, 0])
        self.assertFalse(first_summary["formal_T_N_complete"])
        self.assertTrue(np.isnan(first_summary["T_N"]))
        self.assertTrue(bool(second.loc[0, "loo_valid"]))
        self.assertEqual(int(second.loc[0, "attempt"]), 2)

    def test_numerically_usable_ios_row_does_not_require_success_exit_code(self):
        table = pd.DataFrame({
            "loo_valid": [True],
            "optimizer_success": [False],
            "heldout_nll_under_loo": [2.0],
            "ios_contribution": [1.0],
            "seconds": [0.1],
        })
        summary = summarize_ios(table, expected_n_transitions=1)

        self.assertTrue(summary["formal_T_N_complete"])
        self.assertEqual(summary["T_N"], 1.0)

    def test_parametric_bootstrap_can_be_extended_without_rerunning(self):
        calls = []

        def fake_simulation(*args, **kwargs):
            calls.append(kwargs["seed"])
            return self.data

        def fake_estimate(*args, **kwargs):
            return self.params.copy(), 10.0, 0

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "parametric.csv"
            with (
                patch("student_kramers.bootstrap.simulate_partial_data", fake_simulation),
                patch("student_kramers.bootstrap.estimate_model", fake_estimate),
            ):
                run_parametric_bootstrap(
                    "M2", self.params, self.data, output,
                    n_boot=1, verbose=False,
                )
                table = run_parametric_bootstrap(
                    "M2", self.params, self.data, output,
                    n_boot=3, verbose=False,
                )

        self.assertEqual(calls, [
            config.BOOTSTRAP_SEED,
            config.BOOTSTRAP_SEED + 1,
            config.BOOTSTRAP_SEED + 2,
        ])
        self.assertEqual(len(table), 3)

    def test_contrast_bootstrap_can_be_extended_without_rerunning(self):
        calls = []

        def fake_simulation(*args, **kwargs):
            calls.append(kwargs["seed"])
            return self.data

        def fake_estimate(model_name, *args, **kwargs):
            nll = 10.0 if model_name == "M2" else 9.0
            return self.params.copy(), nll, 0

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "contrast.csv"
            with (
                patch("student_kramers.bootstrap.simulate_partial_data", fake_simulation),
                patch("student_kramers.bootstrap.estimate_model", fake_estimate),
            ):
                run_contrast_bootstrap(
                    "M2", "M3", self.params, self.data, output,
                    n_boot=1, verbose=False,
                )
                table = run_contrast_bootstrap(
                    "M2", "M3", self.params, self.data, output,
                    n_boot=3, verbose=False,
                )

        self.assertEqual(calls, [
            config.BOOTSTRAP_SEED,
            config.BOOTSTRAP_SEED + 1,
            config.BOOTSTRAP_SEED + 2,
        ])
        self.assertEqual(len(table), 3)

    def test_ios_pairwise_comparison_reports_rank_agreement(self):
        transitions = pd.DataFrame({
            "m2_ios_contribution": [1.0, 2.0, 3.0],
            "m3_ios_contribution": [2.0, 4.0, 6.0],
            "m4_ios_contribution": [3.0, 2.0, 1.0],
            "regime_switch": [False, True, False],
        })
        result = build_ios_pairwise_comparison(transitions)
        m2_m3 = result.query("left_model == 'M2' and right_model == 'M3'").iloc[0]
        self.assertAlmostEqual(m2_m3["pearson_correlation"], 1.0)
        self.assertAlmostEqual(m2_m3["spearman_correlation"], 1.0)

    def test_ios_regime_summary_preserves_model_sums(self):
        transitions = pd.DataFrame({
            "m2_ios_contribution": [1.0, 2.0, 3.0],
            "m3_ios_contribution": [2.0, 4.0, 6.0],
            "m4_ios_contribution": [3.0, 2.0, 1.0],
            "regime_switch": [False, True, False],
        })
        result = build_ios_regime_summary(transitions)
        totals = result.groupby("model")["ios_sum"].sum()
        self.assertEqual(totals.to_dict(), {"M2": 6.0, "M3": 12.0, "M4": 6.0})


if __name__ == "__main__":
    unittest.main()
