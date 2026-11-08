"""
Compare the blocked `InterX` (the segment intersections of a polyline) with the original, which built the full
(segments x segments) matrices and looped over their rows.
"""
import numpy as np
import pytest

from pyCoilGen.sub_functions import shift_return_paths as module
from pyCoilGen.sub_functions.shift_return_paths import InterX


def reference_diff(x, y):
    u = np.empty((x.shape[0], y.shape[0]))
    for i in range(u.shape[0]):
        diff_x = x[i, :-1] - y[i]
        diff_y = x[i, 1:] - y[i]
        u[i] = diff_x * diff_y
    return u


def reference_interx(L1, *varargin):
    """The original implementation."""
    if len(varargin) == 0:
        L2 = L1
        hF = np.less  # Avoid the inclusion of common points
    else:
        L2 = varargin[0]
        hF = np.less_equal

    x1, x2, y1, y2 = L1[0, :], L2[0, :], L1[1, :], L2[1, :]
    dx1, dy1, dx2, dy2 = np.diff(x1), np.diff(y1), np.diff(x2), np.diff(y2)
    S1 = dx1 * y1[:-1] - dy1 * x1[:-1]
    S2 = dx2 * y2[:-1] - dy2 * x2[:-1]

    P10 = np.multiply.outer(dx1, y2)
    P11 = np.multiply.outer(dy1, x2)
    P12 = reference_diff(P10 - P11, S1)
    P20 = np.multiply.outer(y1, dx2)
    P21 = np.multiply.outer(x1, dy2)
    P22 = reference_diff((P20 - P21).T, S2.T)

    C1 = hF(P12, 0).astype(int)
    C2 = hF(P22, 0).T.astype(int)
    j, i = np.where(C1 & C2)
    if len(i) == 0:
        return np.zeros((2, 0)), []

    L = dy2[j] * dx1[i] - dy1[i] * dx2[j]
    non_zero_indices = L.nonzero()
    i, j, L = i[non_zero_indices], j[non_zero_indices], L[non_zero_indices]
    sorted_edges = np.sort(np.vstack((i, j)).T, axis=1)
    intersect_edge_inds = np.unique(sorted_edges, axis=0)
    arr1 = (dx2[j] * S1[i] - dx1[i] * S2[j]) / L
    arr2 = (dy2[j] * S1[i] - dy1[i] * S2[j]) / L
    P = np.unique(np.vstack((arr1, arr2)).T, axis=0).T
    return P, intersect_edge_inds


def _random_polyline(rng, num_points, repeats=False):
    track = np.cumsum(rng.normal(size=(2, num_points)), axis=1) * 0.3
    if repeats:  # Superimposed points: zero length segments
        track[:, rng.integers(1, num_points, size=max(1, num_points // 10))] = track[:, 0:1]
    return track


def _loop(num_points):
    angle = np.linspace(0, 6 * np.pi, num_points)   # A spiral crosses itself less; a rose does
    radius = 1 + 0.5 * np.cos(5 * angle)
    return np.vstack((radius * np.cos(angle), radius * np.sin(angle)))


def _assert_same(actual, expected):
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(np.asarray(actual[1]), np.asarray(expected[1]))


@pytest.mark.parametrize('num_points', [2, 3, 10, 80, 400])
@pytest.mark.parametrize('seed', range(4))
def test_interx_matches_the_original(num_points, seed):
    track = _random_polyline(np.random.default_rng(seed), num_points)
    _assert_same(InterX(track), reference_interx(track))


@pytest.mark.parametrize('seed', range(3))
def test_interx_matches_the_original_with_repeated_points(seed):
    track = _random_polyline(np.random.default_rng(seed), 120, repeats=True)
    _assert_same(InterX(track), reference_interx(track))


def test_interx_matches_the_original_for_a_closed_self_crossing_loop():
    track = _loop(500)
    actual = InterX(track)
    assert actual[1].shape[0] > 0       # The test must have intersections to mean anything
    _assert_same(actual, reference_interx(track))


@pytest.mark.parametrize('seed', range(3))
def test_interx_matches_the_original_for_two_polylines(seed):
    rng = np.random.default_rng(seed)
    first, second = _random_polyline(rng, 90), _random_polyline(rng, 90)
    _assert_same(InterX(first, second), reference_interx(first, second))


@pytest.mark.parametrize('pairs_per_block', [1, 7, 1000])
def test_interx_does_not_depend_on_the_block_size(monkeypatch, pairs_per_block):
    track = _loop(300)
    expected = reference_interx(track)
    monkeypatch.setattr(module, '_PAIRS_PER_BLOCK', pairs_per_block)
    _assert_same(InterX(track), expected)


def test_interx_without_intersections():
    track = np.vstack((np.linspace(0, 1, 20), np.zeros(20)))
    points, edges = InterX(track)
    assert points.shape == (2, 0) and len(edges) == 0
