import numpy as np

# Test support
# Code under test
from pyCoilGen.mesh_factory.build_planar_mesh import build_planar_mesh


def test_build_planar_mesh_basic():
    planar_height = 0.25
    planar_width = 0.4
    num_lateral_divisions = 3
    num_longitudinal_divisions = 4
    rotation_vector_x = 0
    rotation_vector_y = 0
    rotation_vector_z = 0
    rotation_angle = 0
    center_position_x = 0
    center_position_y = 0
    center_position_z = 0

    mesh = build_planar_mesh(planar_height, planar_width, num_lateral_divisions, num_longitudinal_divisions,
                             rotation_vector_x, rotation_vector_y, rotation_vector_z, rotation_angle,
                             center_position_x, center_position_y, center_position_z)

    # Min x is -0.5 width
    assert np.min(mesh.vertices[:, 0]) == -planar_width/2.0
    # Max x is 0.5 width
    assert np.max(mesh.vertices[:, 0]) == planar_width/2.0

    # Min y is -0.5 height
    assert np.min(mesh.vertices[:, 1]) == -planar_height/2.0
    # Max y is 0.5 height
    assert np.max(mesh.vertices[:, 1]) == planar_height/2.0

    # Min z is 0
    assert np.min(mesh.vertices[:, 2]) == 0.0
    # Max z is 0
    assert np.max(mesh.vertices[:, 2]) == 0.0

    # Test shape
    assert mesh.vertices.shape == ((num_lateral_divisions+1)*(num_longitudinal_divisions+1), 3)


def test_build_planar_mesh_rotate_Z():
    planar_height = 0.25
    planar_width = 0.4
    num_lateral_divisions = 3
    num_longitudinal_divisions = 4
    rotation_vector_x = 0
    rotation_vector_y = 0
    rotation_vector_z = 1
    rotation_angle = np.pi/2.0
    center_position_x = 0
    center_position_y = 0
    center_position_z = 0

    mesh = build_planar_mesh(planar_height, planar_width, num_lateral_divisions, num_longitudinal_divisions,
                             rotation_vector_x, rotation_vector_y, rotation_vector_z, rotation_angle,
                             center_position_x, center_position_y, center_position_z)

    # Min x is -0.5 height
    assert np.min(mesh.vertices[:, 0]) == -planar_height/2.0
    # Max x is 0.5 height
    assert np.max(mesh.vertices[:, 0]) == planar_height/2.0

    # Min y is -0.5 height
    assert np.min(mesh.vertices[:, 1]) == -planar_width/2.0
    # Max y is 0.5 height
    assert np.max(mesh.vertices[:, 1]) == planar_width/2.0

    # Min z is 0
    assert np.min(mesh.vertices[:, 2]) == 0.0
    # Max z is 0
    assert np.max(mesh.vertices[:, 2]) == 0.0


def test_build_planar_mesh_rotate_Y():
    planar_height = 0.25
    planar_width = 0.4
    num_lateral_divisions = 3
    num_longitudinal_divisions = 4
    rotation_vector_x = 0
    rotation_vector_y = 1
    rotation_vector_z = 0
    rotation_angle = np.pi/2.0
    center_position_x = 0
    center_position_y = 0
    center_position_z = 0

    mesh = build_planar_mesh(planar_height, planar_width, num_lateral_divisions, num_longitudinal_divisions,
                             rotation_vector_x, rotation_vector_y, rotation_vector_z, rotation_angle,
                             center_position_x, center_position_y, center_position_z)

    # Min x is close to 0
    assert np.min(mesh.vertices[:, 0]) >= -0.001
    # Max x is close to 0
    assert np.max(mesh.vertices[:, 0]) <= 0.001

    # Min y is -0.5 width
    assert np.min(mesh.vertices[:, 1]) == -planar_height/2.0
    # Max y is 0.5 width
    assert np.max(mesh.vertices[:, 1]) == planar_height/2.0

    # Min z is -0.5 width
    assert np.min(mesh.vertices[:, 2]) == -planar_width/2.0
    # Max z is 0.5 width
    assert np.max(mesh.vertices[:, 2]) == planar_width/2.0


def test_build_planar_mesh_rotate_X():
    planar_height = 0.25
    planar_width = 0.4
    num_lateral_divisions = 3
    num_longitudinal_divisions = 4
    rotation_vector_x = 1
    rotation_vector_y = 0
    rotation_vector_z = 0
    rotation_angle = np.pi/2.0
    center_position_x = 0
    center_position_y = 0
    center_position_z = 0

    mesh = build_planar_mesh(planar_height, planar_width, num_lateral_divisions, num_longitudinal_divisions,
                             rotation_vector_x, rotation_vector_y, rotation_vector_z, rotation_angle,
                             center_position_x, center_position_y, center_position_z)

    # Min x is -0.5 width
    assert np.min(mesh.vertices[:, 0]) == -planar_width/2.0
    # Max y is 0.5 width
    assert np.max(mesh.vertices[:, 0]) == planar_width/2.0

    # Min y is close to 0
    assert np.min(mesh.vertices[:, 1]) >= -0.001
    # Max y is close to 0
    assert np.max(mesh.vertices[:, 1]) <= 0.001

    # Min z is -0.5 width
    assert np.min(mesh.vertices[:, 2]) == -planar_height/2.0
    # Max z is 0.5 width
    assert np.max(mesh.vertices[:, 2]) == planar_height/2.0


def test_build_planar_mesh_translate():
    planar_height = 0.25
    planar_width = 0.4
    num_lateral_divisions = 3
    num_longitudinal_divisions = 4
    rotation_vector_x = 0
    rotation_vector_y = 0
    rotation_vector_z = 1
    rotation_angle = np.pi/2.0
    center_position_x = 0.1
    center_position_y = 0.2
    center_position_z = 0.3

    mesh = build_planar_mesh(planar_height, planar_width, num_lateral_divisions, num_longitudinal_divisions,
                             rotation_vector_x, rotation_vector_y, rotation_vector_z, rotation_angle,
                             center_position_x, center_position_y, center_position_z)

    # Min x is -0.5 height + x offset
    assert np.min(mesh.vertices[:, 0]) == -planar_height/2.0 + center_position_x
    # Max x is 0.5 height + x offset
    assert np.max(mesh.vertices[:, 0]) == planar_height/2.0 + center_position_x

    # Min y is -0.5 height + y offset
    assert np.min(mesh.vertices[:, 1]) == -planar_width/2.0 + center_position_y
    # Max y is 0.5 height + y offset
    assert np.max(mesh.vertices[:, 1]) == planar_width/2.0 + center_position_y

    # Min z is 0 + z offset
    assert np.min(mesh.vertices[:, 2]) == 0.0 + center_position_z
    # Max z is 0 + z offset
    assert np.max(mesh.vertices[:, 2]) == 0.0 + center_position_z
