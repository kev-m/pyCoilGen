"""
Compare the compiled (Numba) kernels with the numpy kernels, which are the reference.

The tests are skipped when Numba is not installed. The compiled kernels are forced on with PYCOILGEN_NUMBA=force, so that
small test problems exercise them too.
"""
# Libraries
import numpy as np
import pytest

# Local imports
from pyCoilGen.sub_functions import accel, wire_kernels
from pyCoilGen.sub_functions.sensitivity_kernels import integrate_over_node_triangles
from pyCoilGen.sub_functions.calculate_sensitivity_matrix import _biot_savart_kernel
from pyCoilGen.sub_functions.calculate_gradient_sensitivity_matrix import _gradient_kernel
from pyCoilGen.sub_functions.gauss_legendre_integration_points_triangle import gauss_legendre_integration_points_triangle

pytestmark = pytest.mark.skipif(not accel.is_available(), reason="Numba is not installed")

RTOL = 1e-11


def close(a, b):
    """Equal to rounding error, relative to the largest value."""
    a, b = np.asarray(a), np.asarray(b)
    return np.max(np.abs(a - b)) <= RTOL * max(np.max(np.abs(b)), 1e-300)


@pytest.fixture
def wire():
    rng = np.random.default_rng(7)
    seg = rng.normal(size=(300, 3)) * 0.3
    cur = rng.normal(size=(300, 3)) * 0.01
    tg = rng.normal(size=(40, 3)) * 0.05
    return seg, cur, tg


def test_wire_field_matches_numpy(wire, monkeypatch):
    monkeypatch.setenv('PYCOILGEN_NUMBA', 'force')
    assert close(wire_kernels.biot_savart_field(*wire), wire_kernels.biot_savart_field_numpy(*wire))


def test_wire_gradient_matches_numpy(wire, monkeypatch):
    monkeypatch.setenv('PYCOILGEN_NUMBA', 'force')
    fast = wire_kernels.biot_savart_gradient(*wire)
    reference = wire_kernels.biot_savart_gradient_numpy(*wire)
    for component_fast, component_ref in zip(fast, reference):
        assert component_fast.shape == component_ref.shape
        assert close(component_fast, component_ref)


def test_wire_kernels_accept_non_contiguous_input(wire, monkeypatch):
    monkeypatch.setenv('PYCOILGEN_NUMBA', 'force')
    seg, cur, tg = wire
    # The pipeline passes transposed (3, N) arrays
    result = wire_kernels.biot_savart_field(seg.T.copy().T, cur.T.copy().T, tg.T.copy().T)
    assert close(result, wire_kernels.biot_savart_field_numpy(seg, cur, tg))


def test_wire_kernels_empty_wire(monkeypatch):
    monkeypatch.setenv('PYCOILGEN_NUMBA', 'force')
    result = wire_kernels.biot_savart_field(np.zeros((0, 3)), np.zeros((0, 3)), np.ones((5, 3)))
    assert result.shape == (5, 3) and not result.any()


def make_node_triangles(rng, num_nodes=30):
    """Triangles ordered by node, with a different number per node, some nodes without triangles."""
    counts = rng.integers(0, 7, size=num_nodes)
    counts[3] = 0
    total = counts.sum()
    points = rng.normal(size=(total, 3, 3)) * 0.1
    current = rng.normal(size=(total, 3))
    area = rng.uniform(0.001, 0.01, size=total)
    return points, current, area, counts


@pytest.mark.parametrize('mode, kernel', [(0, _biot_savart_kernel), (1, _gradient_kernel)])
@pytest.mark.parametrize('gauss_order', [1, 2, 3])
def test_node_integral_matches_numpy(mode, kernel, gauss_order, monkeypatch):
    rng = np.random.default_rng(11)
    points, current, area, counts = make_node_triangles(rng)
    targets = rng.normal(size=(3, 25)) * 0.05 + np.array([[0], [0], [0.2]])
    u, v, w = gauss_legendre_integration_points_triangle(gauss_order)
    args = (points, current, area, counts, targets, u, v, w, kernel)

    monkeypatch.setenv('PYCOILGEN_NUMBA', 'off')
    reference = integrate_over_node_triangles(*args, fast_mode=mode)
    monkeypatch.setenv('PYCOILGEN_NUMBA', 'force')
    fast = integrate_over_node_triangles(*args, fast_mode=mode)

    assert fast.shape == reference.shape == (3, 25, len(counts))
    assert close(fast, reference)
    assert not fast[:, :, 3].any()  # the node without triangles


def test_off_uses_numpy(monkeypatch):
    monkeypatch.setenv('PYCOILGEN_NUMBA', 'off')
    assert accel.kernels_for(10 ** 12) is None


def test_auto_uses_numba_only_for_large_problems(monkeypatch):
    accel.reset_pairs_seen()
    monkeypatch.delenv('PYCOILGEN_NUMBA', raising=False)
    assert accel.kernels_for(accel.MIN_PAIRS - 1) is None
    assert accel.kernels_for(accel.MIN_PAIRS) is not None


def test_set_threads_is_limited_to_numba_threads():
    import numba
    assert accel.set_threads(1) == 1
    assert accel.set_threads(10 ** 6) == numba.config.NUMBA_NUM_THREADS
    accel.set_threads()


def test_nearest_point_distances_match_numpy_exactly(monkeypatch):
    from pyCoilGen.sub_functions.interconnect_among_groups import nearest_point_distances, nearest_point_distances_numpy
    rng = np.random.default_rng(3)
    points_a = rng.normal(size=(3, 120))
    points_b = rng.normal(size=(3, 75)) + 0.1
    monkeypatch.setenv('PYCOILGEN_NUMBA', 'force')
    fast_a, fast_b = nearest_point_distances(points_a, points_b)
    ref_a, ref_b = nearest_point_distances_numpy(points_a, points_b)
    assert np.array_equal(fast_a, ref_a) and np.array_equal(fast_b, ref_b)


@pytest.mark.parametrize('same_curve', [True, False])
def test_interx_matches_numpy_exactly(same_curve, monkeypatch):
    from pyCoilGen.sub_functions.shift_return_paths import InterX
    rng = np.random.default_rng(5)
    curve_1 = np.cumsum(rng.normal(size=(2, 150)), axis=1)
    # The mirror image of the first part of curve 1 about the diagonal, with some noise: it crosses curve 1
    curve_2 = curve_1[::-1, :90] + 0.05 * rng.normal(size=(2, 90))
    arguments = (curve_1,) if same_curve else (curve_1, curve_2)
    monkeypatch.setenv('PYCOILGEN_NUMBA', 'off')
    points_ref, edges_ref = InterX(*arguments)
    monkeypatch.setenv('PYCOILGEN_NUMBA', 'force')
    points_fast, edges_fast = InterX(*arguments)
    assert points_ref.shape[1] > 0  # the test curves do intersect
    assert np.array_equal(points_fast, points_ref) and np.array_equal(edges_fast, edges_ref)


def test_total_work_switches_to_numba(monkeypatch):
    monkeypatch.delenv('PYCOILGEN_NUMBA', raising=False)
    accel.reset_pairs_seen()
    try:
        small = accel.MIN_PAIRS // 10
        results = [accel.kernels_for(small) for _ in range(accel.MIN_TOTAL_PAIRS // small + 1)]
        assert results[0] is None and results[-1] is not None
    finally:
        accel.reset_pairs_seen()


def test_missing_numba_falls_back_to_numpy(wire, monkeypatch):
    # Simulate Numba not being installed
    monkeypatch.setattr(accel, '_kernels', None)
    monkeypatch.setattr(accel, '_tried', True)
    monkeypatch.setenv('PYCOILGEN_NUMBA', 'force')
    assert not accel.is_available()
    assert close(wire_kernels.biot_savart_field(*wire), wire_kernels.biot_savart_field_numpy(*wire))
