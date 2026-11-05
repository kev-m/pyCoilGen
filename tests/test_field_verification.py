import numpy as np
import pytest

from pyCoilGen.helpers.field_verification import (MU_0, biot_savart_polyline, biot_savart_magpylib,
                                                  field_of_wire_paths, fit_linear_field, verify_field)


def circle(radius, num=720, z=0.0):
    angle = np.linspace(0, 2 * np.pi, num + 1)
    return np.array([radius * np.cos(angle), radius * np.sin(angle), np.full_like(angle, z)])


def test_loop_on_axis_matches_analytic_field():
    radius, current = 0.1, 3.0
    heights = np.array([0.0, 0.05, 0.2])
    points = np.array([np.zeros(3), np.zeros(3), heights])
    field = biot_savart_polyline(circle(radius), points, current)
    expected = MU_0 * current * radius ** 2 / (2 * (radius ** 2 + heights ** 2) ** 1.5)
    np.testing.assert_allclose(field[2], expected, rtol=1e-4)
    np.testing.assert_allclose(field[:2], 0, atol=1e-12)


def test_straight_wire_matches_analytic_field():
    # Long wire along z, field at distance d is mu0 I / (2 pi d), circulating around z (here: +y at +x)
    wire = np.array([np.zeros(2), np.zeros(2), [-500.0, 500.0]])
    field = biot_savart_polyline(wire, np.array([[0.02], [0.0], [0.0]]), 2.0)
    np.testing.assert_allclose(field[:, 0], [0, MU_0 * 2.0 / (2 * np.pi * 0.02), 0], atol=1e-9, rtol=1e-4)


def test_points_on_the_wire_are_finite():
    wire = np.array([[0.0, 1.0], [0.0, 0.0], [0.0, 0.0]])
    field = biot_savart_polyline(wire, np.array([[0.0, 0.5, 2.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]))
    assert np.all(np.isfinite(field)) and np.allclose(field, 0)


def test_chunking_does_not_change_the_result(monkeypatch):
    rng = np.random.default_rng(1)
    points = rng.uniform(-0.05, 0.05, (3, 50))
    reference = biot_savart_polyline(circle(0.2, 100), points)
    monkeypatch.setattr('pyCoilGen.helpers.field_verification._MAX_CHUNK_ELEMENTS', 300)
    np.testing.assert_allclose(biot_savart_polyline(circle(0.2, 100), points), reference, rtol=1e-12)


def test_magpylib_agrees_with_reference():
    pytest.importorskip('magpylib')
    rng = np.random.default_rng(2)
    points = rng.uniform(-0.05, 0.05, (3, 20))
    wire = circle(0.1, 60, z=0.03)
    np.testing.assert_allclose(biot_savart_magpylib(wire, points, 2.0), biot_savart_polyline(wire, points, 2.0),
                               rtol=1e-6, atol=1e-15)
    np.testing.assert_allclose(field_of_wire_paths([wire], points, 2.0, method='magpylib'),
                               field_of_wire_paths([wire], points, 2.0), rtol=1e-6, atol=1e-15)


def test_unknown_method():
    with pytest.raises(ValueError):
        field_of_wire_paths([circle(1)], np.zeros((3, 1)), method='nope')


def test_fit_linear_field_recovers_gradient():
    rng = np.random.default_rng(3)
    points = rng.uniform(-1, 1, (3, 40))
    values = 2.0 + 3.0 * points[0] - 1.0 * points[2]
    fit = fit_linear_field(points, values)
    np.testing.assert_allclose(fit['gradient'], [3, 0, -1], atol=1e-12)
    assert np.allclose(fit['residual'], 0)


def test_verify_field_reports_the_current_that_fits_the_target():
    # A Helmholtz-like pair, evaluated near the centre. The target is the field of 2.5 A, designed for 2 A.
    wires = [circle(0.1, 180, z=0.05), circle(0.1, 180, z=-0.05)]
    rng = np.random.default_rng(4)
    points = rng.uniform(-0.01, 0.01, (3, 30))
    target = field_of_wire_paths(wires, points, 2.5)
    result = verify_field(wires, points, target, design_current=2.0)
    assert result['best_current'] == pytest.approx(2.5)
    assert result['current_consistency'] == pytest.approx(1.25)
    assert result['rms_rel_error'] < 1e-9 and result['max_rel_error'] < 1e-9
    assert result['rms_rel_error_at_design_current'] > 1.0
    assert result['efficiency'] >= 0 and result['linearity_error'] >= 0
