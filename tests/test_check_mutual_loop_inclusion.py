"""
Compare the vectorised `check_mutual_loop_inclusion` with the original, which looped over the test vertices.
"""
import numpy as np
import pytest

from pyCoilGen.sub_functions import check_mutual_loop_inclusion as module
from pyCoilGen.sub_functions.check_mutual_loop_inclusion import check_mutual_loop_inclusion


def reference_check_mutual_loop_inclusion(test_poly: np.ndarray, target_poly: np.ndarray) -> bool:
    """
    Check if the test polygon lies fully enclosed within the second polygon.

    This check is done with the winding number algorithm test for each vertex towards the second polygon.

    Args:
        test_poly (np.ndarray): Array representing the test polygon's 2D coordinates (shape: (2, num_vertices)).
        target_poly (np.ndarray): Array representing the target polygon's 2D coordinates (shape: (2, num_vertices)).

    Returns:
        bool: True if the test polygon is fully enclosed within the target polygon, False otherwise.
    """

    if len(test_poly.shape) == 1:
        test_poly = test_poly.reshape((2, 1))
    num_entries = test_poly.shape[1]
    winding_numbers = np.zeros(num_entries)
    for point_ind in range(num_entries):
        A = np.tile(test_poly[:, point_ind, np.newaxis], (1, target_poly.shape[1] - 1))
        B = target_poly[:, 1:]
        C = target_poly[:, :-1]

        vec1 = C - A
        vec2 = B - A

        angle = np.arctan2(vec1[0, :] * vec2[1, :] - vec1[1, :] * vec2[0, :],
                           vec1[0, :] * vec2[0, :] + vec1[1, :] * vec2[1, :])

        winding_numbers[point_ind] = round(abs(np.sum(angle) / (2 * np.pi)))

    inside_flag = np.all(winding_numbers == 1)

    return inside_flag




def _circle(radius, centre, num, closed=True):
    angle = np.linspace(0, 2 * np.pi, num, endpoint=closed)
    return np.vstack((centre[0] + radius * np.cos(angle), centre[1] + radius * np.sin(angle)))


@pytest.mark.parametrize('num_test, num_target', [(1, 5), (40, 60), (300, 800)])
def test_check_mutual_loop_inclusion_matches_the_original(num_test, num_target):
    outer = _circle(1.0, (0, 0), num_target)
    cases = [(_circle(0.5, (0.1, 0.0), num_test), outer),       # Inside
             (_circle(0.5, (0.8, 0.0), num_test), outer),       # Partly outside
             (_circle(0.2, (3.0, 3.0), num_test), outer),       # Outside
             (outer, _circle(0.5, (0.0, 0.0), num_target)),     # Enclosing instead
             (_circle(0.5, (0.0, 0.0), num_test), _circle(0.5, (0.0, 0.0), num_target))]  # On the contour
    for test_poly, target_poly in cases:
        assert check_mutual_loop_inclusion(test_poly, target_poly) == reference_check_mutual_loop_inclusion(test_poly, target_poly)


def test_check_mutual_loop_inclusion_single_point_and_blocks(monkeypatch):
    outer = _circle(1.0, (0, 0), 50)
    assert check_mutual_loop_inclusion(np.array([0.1, 0.1]), outer) == reference_check_mutual_loop_inclusion(np.array([0.1, 0.1]), outer)
    assert check_mutual_loop_inclusion(np.array([3.0, 0.1]), outer) == reference_check_mutual_loop_inclusion(np.array([3.0, 0.1]), outer)
    monkeypatch.setattr(module, '_PAIRS_PER_BLOCK', 1)
    inner = _circle(0.5, (0.1, 0.0), 30)
    assert check_mutual_loop_inclusion(inner, outer) is np.True_ or check_mutual_loop_inclusion(inner, outer)
