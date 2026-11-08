"""Tests of the linear solve of the Tikhonov stream function optimisation."""

import numpy as np

from pyCoilGen.sub_functions.stream_function_optimization import _solve_normal_equations


def _pinv_solution(sensitivity, regularisation, target):
    return np.linalg.pinv(sensitivity.T @ sensitivity + regularisation.T @ regularisation) @ sensitivity.T @ target


def test_solve_matches_the_pseudo_inverse_for_a_well_conditioned_problem():
    rng = np.random.default_rng(0)
    sensitivity = rng.normal(size=(60, 20))
    regularisation = np.diag(rng.uniform(0.5, 2.0, size=20))
    target = rng.normal(size=60)
    np.testing.assert_allclose(_solve_normal_equations(sensitivity, regularisation, target),
                               _pinv_solution(sensitivity, regularisation, target), rtol=1e-8)


def test_solve_falls_back_to_the_pseudo_inverse_for_a_singular_problem():
    rng = np.random.default_rng(1)
    sensitivity = rng.normal(size=(30, 10))
    sensitivity[:, 9] = sensitivity[:, 8]       # Rank deficient
    regularisation = np.zeros((10, 10))         # and unregularised: A^T A is singular
    target = rng.normal(size=30)
    result = _solve_normal_equations(sensitivity, regularisation, target)
    assert np.all(np.isfinite(result))
    np.testing.assert_allclose(result, _pinv_solution(sensitivity, regularisation, target), rtol=1e-6, atol=1e-8)
