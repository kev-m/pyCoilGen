# TODO: Remove
# Hack code
# Set up paths: Add the project root directory to the Python path
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import numpy as np
import json

# Test support
from helpers.visualisation import compare
from sub_functions.data_structures import Mesh
# Code under test
from sub_functions.calc_3d_rotation_matrix_by_vector import calc_3d_rotation_matrix_by_vector

def test_calc_3d_rotation_matrix_by_vector_basic():
    vec = [0,0,1]
    angle = 0
    rot_mat = calc_3d_rotation_matrix_by_vector(vec, angle)
    assert np.max(rot_mat) == 1.0
    assert np.min(rot_mat) == 0.0

    # No effect
    input = [1,1,1]
    rot = np.dot([input], rot_mat)
    assert rot[0].tolist() == input

# Rotate 90 degrees about Z changes [1,0,0] to [0,1,0]
def test_calc_3d_rotation_matrix_by_vector_about_z():
    vec = [0,0,1]
    angle = np.pi/2.0
    rot_mat = calc_3d_rotation_matrix_by_vector(vec, angle)
    assert np.max(rot_mat) == 1.0
    assert np.min(rot_mat) == -1.0

    input = [1,0,0]
    rot = np.dot([input, input, input], rot_mat)
    assert np.allclose(rot[0], [0,1,0])
    assert np.allclose(rot[1], [0,1,0])

    # No change
    input = vec
    rot = np.dot([input, input, input], rot_mat)
    assert np.allclose(rot[0], input)

# Rotate 90 degrees about X changes [0,0,1] to [0,-1,0]
def test_calc_3d_rotation_matrix_by_vector_about_x():
    vec = [1,0,0]
    angle = np.pi/2.0
    rot_mat = calc_3d_rotation_matrix_by_vector(vec, angle)
    assert np.max(rot_mat) == 1.0
    assert np.min(rot_mat) == -1.0

    input = [0,0,1]
    rot = np.dot([input, input, input], rot_mat)
    assert np.allclose(rot[0], [0,-1,0])
    assert np.allclose(rot[1], [0,-1,0])

    # No change
    input = vec
    rot = np.dot([input, input, input], rot_mat)
    assert np.allclose(rot[0], input)

# Rotate 90 degrees about Y changes [1,0,0] to [0,0,-1]
def test_calc_3d_rotation_matrix_by_vector_about_y():
    vec = [0,1,0]
    angle = np.pi/2.0
    rot_mat = calc_3d_rotation_matrix_by_vector(vec, angle)
    assert np.max(rot_mat) == 1.0
    assert np.min(rot_mat) == -1.0

    input = [1,0,0]
    rot = np.dot([input, input, input], rot_mat)
    assert np.allclose(rot[0], [0,0,-1])
    assert np.allclose(rot[1], [0,0,-1])

    # No change
    input = vec
    rot = np.dot([input, input, input], rot_mat)
    assert np.allclose(rot[0], input)

def cp():
    log.debug(" rot: %s", rot)
    log.debug(" rot_mat: %s", rot_mat)
    log.debug(" min(rot_mat): %s", np.min(rot_mat))


# TODO: Remove
if __name__ == "__main__":
    import logging
    # Set up logging
    log = logging.getLogger(__name__)
    logging.basicConfig(level=logging.DEBUG)

    test_calc_3d_rotation_matrix_by_vector_basic()
    test_calc_3d_rotation_matrix_by_vector_about_z()