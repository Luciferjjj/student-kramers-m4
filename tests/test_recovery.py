"""Focused tests for extendable complete/partial recovery studies."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from student_kramers import config
from student_kramers.recovery import run_recovery_study, summarize_recovery


class RecoveryTests(unittest.TestCase):
    def test_one_path_can_be_extended_without_rerunning_it(self):
        reference = config.REFERENCE_PARAMS_BY_MODEL["M4"]
        calls = []

        def fake_simulation(params, n_obs, h_obs, h_sim, init_state, seed):
            calls.append(seed)
            x = np.linspace(-1.0, 1.0, n_obs)
            return np.column_stack([x, np.ones(n_obs)])

        def fake_complete(model_name, data, h, **kwargs):
            return reference.copy(), 10.0, 0

        def fake_partial(model_name, data, h, **kwargs):
            return reference.copy(), 20.0, 0

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "recovery.csv"
            patches = (
                patch("student_kramers.recovery.simulate_trajectory", fake_simulation),
                patch("student_kramers.recovery.estimate_complete_model", fake_complete),
                patch("student_kramers.recovery.estimate_model", fake_partial),
            )
            with patches[0], patches[1], patches[2]:
                first = run_recovery_study(
                    "M4", reference, 1, 6, output, resume=True, verbose=False,
                )
                extended = run_recovery_study(
                    "M4", reference, 3, 6, output, resume=True, verbose=False,
                )

        self.assertEqual(len(first), 2)
        self.assertEqual(len(extended), 6)
        self.assertEqual(calls, [20260611, 20260612, 20260613])
        self.assertEqual(list(extended["traj"].unique()), [0, 1, 2])

        summary = summarize_recovery(extended, reference, "M4")
        self.assertTrue(np.allclose(summary["bias"], 0.0))
        self.assertTrue(np.allclose(summary["rmse"], 0.0))

    def test_partial_path_checkpoint_resumes_without_rerunning_complete_fit(self):
        reference = config.REFERENCE_PARAMS_BY_MODEL["M4"]
        complete_calls = []
        partial_calls = []

        def fake_simulation(params, n_obs, h_obs, h_sim, init_state, seed):
            x = np.linspace(-1.0, 1.0, n_obs)
            return np.column_stack([x, np.ones(n_obs)])

        def fake_complete(model_name, data, h, **kwargs):
            complete_calls.append(1)
            return reference.copy(), 10.0, 0

        def failing_partial(model_name, data, h, **kwargs):
            partial_calls.append("fail")
            raise KeyboardInterrupt

        def successful_partial(model_name, data, h, **kwargs):
            partial_calls.append("success")
            return reference.copy(), 20.0, 0

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "recovery.csv"
            with (
                patch("student_kramers.recovery.simulate_trajectory", fake_simulation),
                patch("student_kramers.recovery.estimate_complete_model", fake_complete),
                patch("student_kramers.recovery.estimate_model", failing_partial),
            ):
                with self.assertRaises(KeyboardInterrupt):
                    run_recovery_study(
                        "M4", reference, 1, 6, output, resume=True, verbose=False,
                    )

            with (
                patch("student_kramers.recovery.simulate_trajectory", fake_simulation),
                patch("student_kramers.recovery.estimate_complete_model", fake_complete),
                patch("student_kramers.recovery.estimate_model", successful_partial),
            ):
                resumed = run_recovery_study(
                    "M4", reference, 1, 6, output, resume=True, verbose=False,
                )

        self.assertEqual(len(resumed), 2)
        self.assertEqual(len(complete_calls), 1)
        self.assertEqual(partial_calls, ["fail", "success"])


if __name__ == "__main__":
    unittest.main()
