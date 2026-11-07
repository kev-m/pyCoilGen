"""Tests for the removal of loops that are smaller than the interconnection cut."""
import numpy as np

from pyCoilGen.sub_functions.process_raw_loops import is_loop_smaller_than


def _square(side, center=(0.0, 0.0, 0.0)):
    h = side / 2
    x, y, z = center
    return np.array([[x - h, x + h, x + h, x - h, x - h], [y - h, y - h, y + h, y + h, y - h], [z] * 5])


def test_loop_smaller_than_the_size_is_detected():
    assert is_loop_smaller_than(_square(0.004), 0.01)       # Diagonal 5.7 mm
    assert not is_loop_smaller_than(_square(0.02), 0.01)    # Diagonal 28 mm


def test_the_position_of_the_loop_does_not_matter():
    assert is_loop_smaller_than(_square(0.004, (-0.1, 0.0, 0.0113)), 0.01)
