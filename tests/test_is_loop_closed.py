"""Tests for the closing of contour loops in process_raw_loops."""
import numpy as np

from pyCoilGen.sub_functions.process_raw_loops import is_loop_closed


def _open_loop(end):
    return np.array([[0.1, 0.5, 0.5, end[0]], [0.2, 0.2, 0.6, end[1]]])


def test_loop_that_ends_at_its_start_is_closed():
    assert is_loop_closed(_open_loop((0.1, 0.2)))
    # Rounding of the last bit does not matter
    assert is_loop_closed(_open_loop((0.1 + 1e-16, 0.2 - 1e-16)))


def test_open_loop_is_open_whether_one_or_both_coordinates_differ():
    assert not is_loop_closed(_open_loop((0.0, 0.0)))                 # Both differ
    # One coordinate equal to the start: the original code left this loop open, by accident of the exact comparison
    assert not is_loop_closed(_open_loop((0.0, 0.2)))
    assert not is_loop_closed(_open_loop((0.1, 0.0)))


def test_the_result_does_not_depend_on_the_last_bit():
    start = 0.47047423
    for y_end in (start, np.nextafter(start, 1), np.nextafter(start, 0)):
        loop = np.array([[-0.2795258, -0.28125, -0.28125, -0.2795258 + 1e-3], [start, start, 0.5, y_end]])
        assert not is_loop_closed(loop)                               # Always open
        loop[:, -1] = loop[:, 0] + [0.0, np.nextafter(start, 1) - start]
        assert is_loop_closed(loop)                                   # Always closed


def test_single_point_and_degenerate_loops():
    assert is_loop_closed(np.array([[0.3], [0.4]]))
