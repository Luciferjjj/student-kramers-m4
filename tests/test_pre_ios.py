"""Focused tests for diagnostics required before IOS."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from student_kramers import config
from student_kramers.models import PARAM_NAMES, parameter_row
from greenland_application.pre_ios import (
    audit_diffusion_minima,
    nested_bootstrap_cumulative,
    run_nested_m3_m4_bootstrap,
    run_predictive_checks_checkpointed,
    summarize_nested_bootstrap,
    transition_improvement_table,
)


class PreIOSTests(unittest.TestCase):
    def test_q_audit_distinguishes_global_and_path_minima(self):
        params = config.REFERENCE_PARAMS_BY_MODEL["M4"]
        fits = pd.DataFrame([parameter_row("M4", params, 10.0)])
        data = np.column_stack([
            np.linspace(-1.0, 1.0, 20),
            np.linspace(-2.0, 2.0, 20),
        ])
        table = audit_diffusion_minima(fits, data)
        self.assertEqual(len(table), 1)
        self.assertGreaterEqual(
            table.loc[0, "q_path_min"], table.loc[0, "q_min_global"],
        )
        self.assertGreaterEqual(
            table.loc[0, "q_min_observed_rectangle"], table.loc[0, "q_min_global"],
        )

    def test_nested_bootstrap_uses_m3_boundary_for_m4(self):
        m3 = config.REFERENCE_PARAMS_BY_MODEL["M3"]
        m4 = config.REFERENCE_PARAMS_BY_MODEL["M4"]
        data = np.column_stack([np.linspace(-1.0, 1.0, 8), np.zeros(8)])
        calls = []

        def fake_simulate(*args, **kwargs):
            return data

        def fake_estimate(model_name, fitted_data, **kwargs):
            calls.append((model_name, np.asarray(kwargs["start"]).copy()))
            return (m3.copy(), 10.0, 0) if model_name == "M3" else (m4.copy(), 9.0, 0)

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "nested.csv"
            with patch("greenland_application.pre_ios.simulate_partial_data", fake_simulate), \
                    patch("greenland_application.pre_ios.estimate_model", fake_estimate):
                table = run_nested_m3_m4_bootstrap(
                    m3, m4, data, output, n_boot=1, verbose=False,
                )

        self.assertTrue(table.loc[0, "success"])
        self.assertEqual(table.loc[0, "contrast"], 2.0)
        self.assertIn("m3_eta", table)
        self.assertIn("m4_delta", table)
        self.assertIn("q_min_observed_m4", table)
        self.assertEqual([call[0] for call in calls], ["M3", "M4"])
        np.testing.assert_allclose(calls[1][1], m3)

    def test_nested_summary_uses_finite_sample_tail_probability(self):
        table = pd.DataFrame({
            "success": [True, True, True],
            "contrast": [0.0, 2.0, 4.0],
        })
        summary = summarize_nested_bootstrap(table, observed_contrast=3.0)
        self.assertEqual(summary.loc[0, "p_upper"], 0.5)
        self.assertEqual(summary.loc[0, "n_exceed"], 1)
        self.assertTrue(summary.loc[0, "extend_to_300"])

        cumulative = nested_bootstrap_cumulative(table, observed_contrast=3.0)
        np.testing.assert_allclose(cumulative["p_upper"], [0.5, 1/3, 0.5])
        self.assertEqual(cumulative["n_exceed"].tolist(), [0, 0, 1])

    def test_nested_bootstrap_can_be_extended_without_rerunning(self):
        m3 = config.REFERENCE_PARAMS_BY_MODEL["M3"]
        m4 = config.REFERENCE_PARAMS_BY_MODEL["M4"]
        data = np.column_stack([np.linspace(-1.0, 1.0, 8), np.zeros(8)])
        calls = []

        def fake_simulate(*args, **kwargs):
            calls.append(kwargs["seed"])
            return data

        def fake_estimate(model_name, fitted_data, **kwargs):
            return (m3.copy(), 10.0, 0) if model_name == "M3" else (m4.copy(), 9.0, 0)

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "nested.csv"
            with patch("greenland_application.pre_ios.simulate_partial_data", fake_simulate), \
                    patch("greenland_application.pre_ios.estimate_model", fake_estimate):
                run_nested_m3_m4_bootstrap(
                    m3, m4, data, output, n_boot=1, verbose=False,
                )
                table = run_nested_m3_m4_bootstrap(
                    m3, m4, data, output, n_boot=3, verbose=False,
                )

        self.assertEqual(len(table), 3)
        self.assertEqual(calls, [20260612, 20260613, 20260614])
        self.assertEqual(table["attempt"].tolist(), [1, 1, 1])

    def test_transition_improvement_sums_transition_gains(self):
        fits = pd.DataFrame([
            parameter_row("M3", config.REFERENCE_PARAMS_BY_MODEL["M3"], 10.0),
            parameter_row("M4", config.REFERENCE_PARAMS_BY_MODEL["M4"], 7.0),
        ])
        data = np.column_stack([
            np.linspace(-1.0, 1.0, 20),
            np.linspace(-2.0, 2.0, 20),
        ])
        with patch(
            "greenland_application.pre_ios.partial_transition_nlls",
            side_effect=[np.full(19, 2.0), np.full(19, 1.5)],
        ):
            table = transition_improvement_table(fits, data)
        self.assertAlmostEqual(table["gain_m4_over_m3"].sum(), 9.5)
        self.assertAlmostEqual(table["cumulative_gain"].iloc[-1], 9.5)

    def test_predictive_checks_can_be_extended_without_rerunning(self):
        fits = pd.DataFrame([
            parameter_row("M2", config.REFERENCE_PARAMS_BY_MODEL["M2"], 10.0),
        ])
        data = np.column_stack([np.linspace(-1.0, 1.0, 8), np.zeros(8)])
        calls = []

        def fake_simulation(params, n_obs, **kwargs):
            calls.append(kwargs["seed"])
            x = np.linspace(-1.0, 1.0, n_obs)
            return np.column_stack([x, np.zeros(n_obs)])

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {
                name: root / f"{name}.csv"
                for name in ("summary", "behavior", "waiting", "density_rep")
            }
            with patch("greenland_application.pre_ios.simulate_trajectory", fake_simulation):
                run_predictive_checks_checkpointed(
                    fits, data, paths, n_rep=1, verbose=False,
                )
                result = run_predictive_checks_checkpointed(
                    fits, data, paths, n_rep=3, verbose=False,
                )

        simulated = result["predictive_summary"]
        simulated = simulated[simulated["source"] == "simulated"]
        self.assertEqual(len(simulated), 3)
        self.assertEqual(calls, [20260612, 20260613, 20260614])


if __name__ == "__main__":
    unittest.main()
