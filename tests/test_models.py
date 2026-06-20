"""Focused tests for model definitions and constraints."""
import unittest

import numpy as np

from student_kramers import config
from student_kramers.models import (
    MODELS,
    PARAM_NAMES,
    constraints_valid,
    diffusion_rectangle_bounds,
    diffusion_rectangle_minimum,
    diffusion_variance,
    diffusion_augmented_matrix,
    embed_params,
    extract_free_params,
    m4_from_cholesky_params,
    m4_to_cholesky_params,
    validate_model_registry,
)
from student_kramers.estimation import make_random_starts
from student_kramers.simulation import simulate_diffusion_check


class ModelTests(unittest.TestCase):
    def test_embed_extract_round_trip(self):
        for name, cfg in MODELS.items():
            full = embed_params(cfg["init"], name)
            np.testing.assert_allclose(extract_free_params(full, name), cfg["init"])

    def test_initial_values_satisfy_constraints(self):
        for name, cfg in MODELS.items():
            self.assertTrue(constraints_valid(cfg["init"], name), name)
            recovery = config.RECOVERY_INIT_PARAMS_BY_MODEL[name]
            self.assertTrue(constraints_valid(extract_free_params(recovery, name), name), name)

    def test_registry_is_consistent(self):
        validate_model_registry()

    def test_extra_warm_start_is_kept_after_boundary_start(self):
        data = np.array([[-2.0, -50.0], [-2.0, 50.0], [2.0, -50.0], [2.0, 50.0]])
        boundary = extract_free_params(embed_params(MODELS["M3"]["init"], "M3"), "M4")
        warm = boundary.copy()
        warm[8:11] = [3.0, 0.0, 0.0]
        starts = make_random_starts(
            "M4", boundary, n_starts=2, data=data, extra_starts=[warm],
        )
        np.testing.assert_allclose(starts[0], boundary)
        np.testing.assert_allclose(starts[1], warm)

    def test_simulated_diffusion_check_returns_one_row_per_path(self):
        params = config.REFERENCE_PARAMS_BY_MODEL["M3"]
        data = np.array([[-1.0, -10.0], [0.0, 0.0], [1.0, 10.0]])
        check = simulate_diffusion_check(
            params, data, n_rep=2, seed=7, h_obs=0.02, h_sim=0.01,
        )
        self.assertEqual(len(check), 2)
        self.assertTrue(check["success"].all())
        self.assertTrue((check["q_min_true"] > 0.0).all())

    def test_shared_parameter_vector_supports_m1_to_m4(self):
        self.assertEqual(len(PARAM_NAMES), 11)
        self.assertEqual(set(MODELS), {"M1", "M2", "M3", "M4"})
        for name in ("M1", "M2", "M3"):
            params = embed_params(MODELS[name]["init"], name)
            np.testing.assert_array_equal(params[8:11], 0.0)
        self.assertEqual(len(MODELS["M4"]["free_indices"]), 11)

    def test_global_positivity_allows_m3_boundary_but_rejects_linear_x_tail(self):
        m3 = embed_params(MODELS["M3"]["init"], "M3")
        self.assertTrue(constraints_valid(extract_free_params(m3, "M4"), "M4"))

        invalid = m3.copy()
        invalid[10] = 1.0
        self.assertGreater(diffusion_variance(0.0, 0.0, invalid), 0.0)
        self.assertFalse(constraints_valid(extract_free_params(invalid, "M4"), "M4"))

    def test_m4_regional_positivity_allows_non_global_quadratic(self):
        m3 = embed_params(MODELS["M3"]["init"], "M3")
        regional = m3.copy()
        regional[8] = -10.0
        regional[9] = -20.0
        data = np.array([[-2.0, -50.0], [-2.0, 50.0], [2.0, -50.0], [2.0, 50.0]])

        self.assertFalse(constraints_valid(extract_free_params(regional, "M4"), "M4"))
        self.assertFalse(constraints_valid(extract_free_params(regional, "M4"), "M4", data))
        lo, hi = diffusion_rectangle_bounds(data, margin=1.0)
        np.testing.assert_allclose(lo, [-6.0, -150.0])
        np.testing.assert_allclose(hi, [6.0, 150.0])
        _, q_min = diffusion_rectangle_minimum(regional, data, margin=1.0)
        self.assertGreater(q_min, 0.0)

    def test_m4_cholesky_round_trip_preserves_interior_parameters(self):
        params = config.REFERENCE_PARAMS_BY_MODEL["M4"]
        free = extract_free_params(params, "M4")
        restored = m4_from_cholesky_params(m4_to_cholesky_params(free))
        np.testing.assert_allclose(restored, free, rtol=1e-10, atol=1e-8)

    def test_m4_cholesky_coordinates_are_globally_valid(self):
        coordinates = np.array([
            30.0, -125.0, 40.0, 150.0, -20.0,
            2.0, -1.5, 1.2, -3.0, 4.0, 5.0,
        ])
        free = m4_from_cholesky_params(coordinates)
        self.assertTrue(constraints_valid(free, "M4"))
        augmented = diffusion_augmented_matrix(embed_params(free, "M4"))
        self.assertGreaterEqual(np.min(np.linalg.eigvalsh(augmented)), -1e-9)
        self.assertLess(free[5], 2.0*free[0])


if __name__ == "__main__":
    unittest.main()
