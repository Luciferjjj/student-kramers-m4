"""Smoke tests for the collaboration-facing package workflow."""
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from student_kramers import config
from student_kramers.data_loading import (
    load_model_fits,
    result_path,
    save_model_fits,
    save_table,
)
from student_kramers.models import MODELS, PARAM_NAMES, embed_params


class WorkflowTests(unittest.TestCase):
    def test_missing_run_returns_empty_fit_table(self):
        self.assertTrue(load_model_fits("test_run_that_does_not_exist").empty)

    def test_named_results_are_isolated(self):
        self.assertNotEqual(result_path("model_fits", run_name="a"), result_path("model_fits", run_name="b"))

    def test_model_fit_provenance_rejects_changed_data(self):
        params = embed_params(MODELS["M1"]["init"], "M1")
        table = pd.DataFrame([{"model": "M1", **dict(zip(PARAM_NAMES, params)), "nll": 1.0}])
        data = np.arange(20.0).reshape(10, 2)
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(config, "RUNS_DIR", Path(directory)):
                save_model_fits(table, data, "test")
                self.assertEqual(len(load_model_fits("test", data=data)), 1)
                with self.assertRaises(RuntimeError):
                    load_model_fits("test", data=data + 1.0)

    def test_legacy_fit_table_pads_new_m4_columns(self):
        old_names = PARAM_NAMES[:8]
        old_params = embed_params(MODELS["M1"]["init"], "M1")[:8]
        table = pd.DataFrame([{"model": "M1", **dict(zip(old_names, old_params))}])
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(config, "RUNS_DIR", Path(directory)):
                save_table(table, result_path("model_fits", run_name="legacy"))
                loaded = load_model_fits("legacy")
        np.testing.assert_array_equal(loaded[PARAM_NAMES[8:11]].to_numpy(), 0.0)

    def test_module_entry_points_load(self):
        for module in (
            "student_kramers.run_validation",
            "student_kramers.run_recovery",
            "student_kramers.run_discrimination",
            "greenland_application.run_single",
            "greenland_application.run_application",
            "greenland_application.run_bootstrap",
            "greenland_application.run_analysis",
            "greenland_application.run_pre_ios",
            "greenland_application.run_figures",
            "greenland_application.run_ios_analysis",
            "greenland_application.run_bootstrap_analysis",
            "greenland_application.run_modelwise_ios_bootstrap",
            "greenland_application.run_modelwise_ios_analysis",
        ):
            result = subprocess.run(
                [sys.executable, "-m", module, "--help"],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
