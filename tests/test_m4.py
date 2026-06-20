"""M4 extension tests, including the required M4-to-M3 sanity checks."""
import unittest

import numpy as np

from student_kramers import config
from student_kramers.data_loading import build_partial_data
from student_kramers.likelihoods import (
    _A_mat,
    _branch_linear_moments,
    _branch_partial_moments,
    _precompute_matrices,
    _step_omega,
    _step_omega_batch,
    complete_neg_log_lik,
    partial_transition_nlls,
)
from student_kramers.models import (
    MODELS,
    diffusion_variance,
    embed_params,
    extract_free_params,
)
from student_kramers.simulation import simulate_trajectory


class M4Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m3_params = embed_params(MODELS["M3"]["init"], "M3")
        cls.m3_free = extract_free_params(cls.m3_params, "M3")
        cls.m4_boundary_free = extract_free_params(cls.m3_params, "M4")

    def test_m4_diffusion_reduces_to_m3(self):
        x = np.linspace(-3.0, 3.0, 11)
        v = np.linspace(-5.0, 5.0, 11)
        q3 = diffusion_variance(x, v, self.m3_params)
        q4 = diffusion_variance(x, v, embed_params(self.m4_boundary_free, "M4"))
        np.testing.assert_array_equal(q3, q4)

    def test_m4_simulation_reduces_to_m3(self):
        path_m3 = simulate_trajectory(
            self.m3_params, 30, h_obs=0.02, h_sim=0.001, seed=20260611,
        )
        path_m4 = simulate_trajectory(
            embed_params(self.m4_boundary_free, "M4"),
            30, h_obs=0.02, h_sim=0.001, seed=20260611,
        )
        np.testing.assert_array_equal(path_m3, path_m4)

    def test_m4_partial_likelihood_reduces_to_m3(self):
        path = simulate_trajectory(
            self.m3_params, 80, h_obs=0.02, h_sim=0.001, seed=20260612,
        )
        data = build_partial_data(path[:, 0])
        nlls_m3 = partial_transition_nlls(self.m3_free, data, "M3")
        nlls_m4 = partial_transition_nlls(self.m4_boundary_free, data, "M4")
        np.testing.assert_array_equal(nlls_m3, nlls_m4)

    def test_m4_complete_likelihood_reduces_to_m3(self):
        path = simulate_trajectory(
            self.m3_params, 80, h_obs=0.02, h_sim=0.001, seed=20260613,
        )
        nll_m3 = complete_neg_log_lik(self.m3_free, path, "M3", 0.02)
        nll_m4 = complete_neg_log_lik(self.m4_boundary_free, path, "M4", 0.02)
        self.assertEqual(nll_m3, nll_m4)

    def test_genuine_m4_has_finite_complete_and_partial_likelihoods(self):
        params = config.REFERENCE_PARAMS_BY_MODEL["M4"]
        free = extract_free_params(params, "M4")
        path = simulate_trajectory(
            params, 80, h_obs=0.02, h_sim=0.001, seed=20260614,
        )
        data = build_partial_data(path[:, 0])
        self.assertTrue(np.isfinite(complete_neg_log_lik(free, path, "M4", 0.02)))
        self.assertTrue(np.all(np.isfinite(partial_transition_nlls(free, data, "M4"))))

    def test_i1_to_i5_match_direct_moment_equations_for_all_models(self):
        Y = np.array([[1.2, -0.7], [-0.8, 1.1], [2.0, 0.2], [-1.5, -0.3]])
        kappa = np.array([0.4, -0.6, 0.4, -0.6])
        drift_const = -20.0
        for name, params in config.REFERENCE_PARAMS_BY_MODEL.items():
            with self.subTest(model=name):
                A = _A_mat(params, drift_const)
                mean_direct, cov_direct = _branch_linear_moments(
                    Y, 0.02, params, A, kappa,
                )
                mean_integrals, cov_integrals = _branch_partial_moments(
                    Y, 0.02, params, drift_const, kappa,
                )
                np.testing.assert_allclose(mean_integrals, mean_direct, atol=1e-13)
                np.testing.assert_allclose(cov_integrals, cov_direct, atol=1e-12)

    def test_vectorized_omega_matches_scalar_formula(self):
        Y = np.array([[1.2, -0.7], [-0.8, 1.1], [2.0, 0.2]])
        params = config.REFERENCE_PARAMS_BY_MODEL["M4"]
        branch = 0.4
        matrices = _precompute_matrices(params, -20.0, branch, 0.02)
        _, I1, I2, I3, I4, I5_G5 = matrices
        scalar = np.array([
            _step_omega(row, branch, I1, I2, I3, I4, I5_G5)
            for row in Y
        ])
        batch = _step_omega_batch(Y, branch, I1, I2, I3, I4, I5_G5)
        np.testing.assert_allclose(batch, scalar, rtol=1e-14, atol=1e-14)


if __name__ == "__main__":
    unittest.main()
