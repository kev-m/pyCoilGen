"""
Compare `nearest_point_distances` of `interconnect_among_groups` with the original per-point loop.
"""
import numpy as np
import pytest

from pyCoilGen.sub_functions import interconnect_among_groups as module
from pyCoilGen.sub_functions.interconnect_among_groups import nearest_point_distances


def reference_near_dist(group_a: np.ndarray, group_b: np.ndarray) -> np.ndarray:
    """The original: the distance of each point of `group_a` to the nearest point of `group_b`, point by point."""
    near_dist = np.zeros(group_a.shape[1])
    for point_ind in range(group_a.shape[1]):
        distances = np.sqrt((group_b[0, :] - group_a[0, point_ind])**2 +
                            (group_b[1, :] - group_a[1, point_ind])**2 +
                            (group_b[2, :] - group_a[2, point_ind])**2)
        near_dist[point_ind] = np.min(distances)
    return near_dist


@pytest.mark.parametrize('pairs_per_block', [1, 50, 100_000])
@pytest.mark.parametrize('num_a, num_b', [(1, 1), (1, 40), (40, 1), (33, 57), (300, 120)])
def test_nearest_point_distances_match_the_original(monkeypatch, pairs_per_block, num_a, num_b):
    monkeypatch.setattr(module, '_PAIRS_PER_BLOCK', pairs_per_block)
    rng = np.random.default_rng(num_a * 1000 + num_b)
    points_a = rng.normal(size=(3, num_a))
    points_b = rng.normal(size=(3, num_b))
    near_a, near_b = nearest_point_distances(points_a, points_b)
    np.testing.assert_array_equal(near_a, reference_near_dist(points_a, points_b))
    np.testing.assert_array_equal(near_b, reference_near_dist(points_b, points_a))   # Both directions, bit for bit


def test_nearest_point_distances_with_coincident_points():
    points_a = np.array([[0.0, 1.0, 2.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    points_b = np.array([[1.0, 5.0], [0.0, 0.0], [0.0, 0.0]])     # Shares the point (1, 0, 0)
    near_a, near_b = nearest_point_distances(points_a, points_b)
    np.testing.assert_array_equal(near_a, [1.0, 0.0, 1.0])
    np.testing.assert_array_equal(near_b, [0.0, 3.0])
