"""Focused tests for M3-versus-M4 discrimination scenarios."""
import unittest

import numpy as np

from student_kramers.discrimination import M4_EFFECT_SCALES, discrimination_truth
from student_kramers.models import constraints_valid, embed_params, extract_free_params


class DiscriminationTests(unittest.TestCase):
    def test_all_truth_scenarios_satisfy_generating_constraints(self):
        model, params = discrimination_truth("M3")
        self.assertEqual(model, "M3")
        self.assertTrue(constraints_valid(extract_free_params(params, model), model))

        for truth in M4_EFFECT_SCALES:
            with self.subTest(truth=truth):
                model, params = discrimination_truth(truth)
                self.assertEqual(model, "M4")
                self.assertTrue(constraints_valid(extract_free_params(params, model), model))

    def test_effect_strength_increases_position_terms(self):
        _, weak = discrimination_truth("weak")
        _, moderate = discrimination_truth("moderate")
        _, strong = discrimination_truth("strong")
        self.assertGreater(moderate[8], weak[8])
        self.assertGreater(strong[8], moderate[8])

    def test_m3_boundary_is_a_valid_m4_start(self):
        _, params = discrimination_truth("M3")
        free_m4 = extract_free_params(params, "M4")
        self.assertTrue(constraints_valid(free_m4, "M4"))
        np.testing.assert_array_equal(embed_params(free_m4, "M4"), params)


if __name__ == "__main__":
    unittest.main()
