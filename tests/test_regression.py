"""Regression tests against small frozen numerical fixtures."""
import unittest
from pathlib import Path

import pandas as pd

from student_kramers.bootstrap import run_exact_ios
from student_kramers.data_loading import load_real_data
from student_kramers.likelihoods import partial_neg_log_lik, partial_transition_nlls
from student_kramers.models import PARAM_NAMES, extract_free_params


FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"


class RegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _, _, _, cls.data = load_real_data()
        cls.fits = pd.read_csv(FIXTURE_DIR / "model_fits.csv")

    def test_real_data_shape(self):
        self.assertEqual(self.data.shape, (2500, 2))

    def test_saved_nll_values(self):
        for _, row in self.fits.iterrows():
            model = row["model"]
            params = row[PARAM_NAMES].to_numpy(dtype=float)
            free = extract_free_params(params, model)
            nll = partial_neg_log_lik(free, self.data, model)
            self.assertAlmostEqual(nll, float(row["NLL"]), places=8)

    def test_m2_transition_sum(self):
        row = self.fits.loc[self.fits["model"] == "M2"].iloc[0]
        params = row[PARAM_NAMES].to_numpy(dtype=float)
        free = extract_free_params(params, "M2")
        values = partial_transition_nlls(free, self.data, "M2")
        self.assertAlmostEqual(values.sum(), float(row["NLL"]), places=8)

    def test_sampled_m2_ios_contributions(self):
        row = self.fits.loc[self.fits["model"] == "M2"].iloc[0]
        params = row[PARAM_NAMES].to_numpy(dtype=float)
        table = run_exact_ios(
            "M2", self.data, params,
            indices=[0, 100, 2498], resume=False, verbose=False,
        )
        reference = pd.read_csv(FIXTURE_DIR / "m2_ios_sample.csv").set_index("k")
        for _, result in table.iterrows():
            expected = reference.loc[int(result["k"]), "ios_contribution"]
            self.assertAlmostEqual(result["ios_contribution"], expected, places=12)


if __name__ == "__main__":
    unittest.main()
