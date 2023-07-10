
import numpy as np
# Test support
from pytest import approx
# Code under test
from sub_functions.gauss_legendre_integration_points_triangle import gauss_legendre_integration_points_triangle, calc_weights_gauss


def test_calc_weights_gauss():
    n = 2
    eta, w = calc_weights_gauss(n)
    assert np.isnan(w).any() == False
    assert np.isnan(eta).any() == False
    assert w.shape == (2,)
    assert w[0] == approx(1.00)
    assert w[1] == approx(1.00)
    assert eta.shape == (2,)
    assert eta[0] == approx(-0.57735)
    assert eta[1] == approx(0.57735)


def test_gauss_legendre_integration_points_triangle():
    n = 2
    u, v, ck = gauss_legendre_integration_points_triangle(n)
    assert np.isnan(u).any() == False
    assert np.isnan(v).any() == False
    assert np.isnan(ck).any() == False

    assert u.shape == (4, 1)
    assert np.allclose(u, [[0.2113], [0.2113], [0.7887], [0.7887]], atol=0.0001)

    assert v.shape == (4, 1)
    assert np.allclose(v, [[0.1667], [0.622], [0.0447], [0.1667]], atol=0.0001)

    assert ck.shape == (4, 1)
    assert np.allclose(ck, [[0.1972], [0.1972], [0.0528], [0.0528]], atol=0.0001)
