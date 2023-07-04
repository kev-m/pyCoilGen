import numpy as np

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