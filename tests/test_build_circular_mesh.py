import numpy as np

# Test support
# Code under test
from pyCoilGen.mesh_factory.build_circular_mesh import build_circular_mesh


def test_build_circular_mesh_basic():
    parameters = {
        'radius': 1.0,
        'num_radial_divisions': 10,
        'rotation_vector_x': 0.0,
        'rotation_vector_y': 0.0,
        'rotation_vector_z': 1.0,
        'rotation_angle': 0.0,
        'center_position_x': 0.0,
        'center_position_y': 0.0,
        'center_position_z': 0.0
    }

    mesh = build_circular_mesh(**parameters)

    # Min x is -0.5 radius
    assert np.min(mesh.vertices[:, 0]) == -parameters['radius']
    # Max x is 0.5 radius
    assert np.max(mesh.vertices[:, 0]) == parameters['radius']

    # Min y is -0.5 radius
    assert np.min(mesh.vertices[:, 1]) == -parameters['radius']
    # Max y is 0.5 radius
    assert np.max(mesh.vertices[:, 1]) == parameters['radius']

    # Min z is 0
    assert np.min(mesh.vertices[:, 2]) == parameters['center_position_z']
    # Max z is 0
    assert np.max(mesh.vertices[:, 2]) == parameters['center_position_z']

    # Test shape
    # assert mesh.vertices.shape == ((num_lateral_divisions+1)*(num_longitudinal_divisions+1), 3)


def test_build_planar_mesh_rotate_Z():
    radius = 0.5
    num_radial_divisions = 10
    rotation_vector_x = 0
    rotation_vector_y = 0
    rotation_vector_z = 1
    rotation_angle = np.pi/2.0
    center_position_x = 0
    center_position_y = 0
    center_position_z = 0

    mesh = build_circular_mesh(radius, num_radial_divisions,
                               rotation_vector_x, rotation_vector_y, rotation_vector_z,
                               rotation_angle,
                               center_position_x, center_position_y, center_position_z)

    # Min x is -0.5 height
    assert np.min(mesh.vertices[:, 0]) == -radius
    # Max x is 0.5 height
    assert np.max(mesh.vertices[:, 0]) == radius

    # Min y is -0.5 height
    assert np.min(mesh.vertices[:, 1]) == -radius
    # Max y is 0.5 height
    assert np.max(mesh.vertices[:, 1]) == radius

    # Min z is 0
    assert np.min(mesh.vertices[:, 2]) == 0.0
    # Max z is 0
    assert np.max(mesh.vertices[:, 2]) == 0.0


def test_build_planar_mesh_rotate_Y():
    radius = 0.5
    num_radial_divisions = 10
    rotation_vector_x = 0
    rotation_vector_y = 1
    rotation_vector_z = 0
    rotation_angle = np.pi/2.0
    center_position_x = 0
    center_position_y = 0
    center_position_z = 0

    mesh = build_circular_mesh(radius, num_radial_divisions,
                               rotation_vector_x, rotation_vector_y, rotation_vector_z,
                               rotation_angle,
                               center_position_x, center_position_y, center_position_z)

    # Min x is close to 0
    assert np.min(mesh.vertices[:, 0]) >= -0.001
    # Max x is close to 0
    assert np.max(mesh.vertices[:, 0]) <= 0.001

    # Min y is -0.5 width
    assert np.min(mesh.vertices[:, 1]) == -radius
    # Max y is 0.5 width
    assert np.max(mesh.vertices[:, 1]) == radius

    # Min z is -0.5 width
    assert np.min(mesh.vertices[:, 2]) == -radius
    # Max z is 0.5 width
    assert np.max(mesh.vertices[:, 2]) == radius


def test_build_planar_mesh_rotate_X():
    radius = 0.5
    num_radial_divisions = 10
    rotation_vector_x = 1
    rotation_vector_y = 0
    rotation_vector_z = 0
    rotation_angle = np.pi/2.0
    center_position_x = 0
    center_position_y = 0
    center_position_z = 0

    mesh = build_circular_mesh(radius, num_radial_divisions,
                               rotation_vector_x, rotation_vector_y, rotation_vector_z,
                               rotation_angle,
                               center_position_x, center_position_y, center_position_z)

    # Min x is -0.5 width
    assert np.min(mesh.vertices[:, 0]) == -radius
    # Max y is 0.5 width
    assert np.max(mesh.vertices[:, 0]) == radius

    # Min y is close to 0
    assert np.min(mesh.vertices[:, 1]) >= -0.001
    # Max y is close to 0
    assert np.max(mesh.vertices[:, 1]) <= 0.001

    # Min z is -0.5 width
    assert np.min(mesh.vertices[:, 2]) == -radius
    # Max z is 0.5 width
    assert np.max(mesh.vertices[:, 2]) == radius


def test_build_planar_mesh_translate():
    radius = 0.5
    num_radial_divisions = 10
    rotation_vector_x = 0
    rotation_vector_y = 0
    rotation_vector_z = 1
    rotation_angle = np.pi/2.0
    center_position_x = 0.1
    center_position_y = 0.2
    center_position_z = 0.3

    mesh = build_circular_mesh(radius, num_radial_divisions,
                               rotation_vector_x, rotation_vector_y, rotation_vector_z,
                               rotation_angle,
                               center_position_x, center_position_y, center_position_z)

    # Min x is -0.5 height + x offset
    assert np.min(mesh.vertices[:, 0]) == -radius + center_position_x
    # Max x is 0.5 height + x offset
    assert np.max(mesh.vertices[:, 0]) == radius + center_position_x

    # Min y is -0.5 height + y offset
    assert np.min(mesh.vertices[:, 1]) == -radius + center_position_y
    # Max y is 0.5 height + y offset
    assert np.max(mesh.vertices[:, 1]) == radius + center_position_y

    # Min z is 0 + z offset
    assert np.min(mesh.vertices[:, 2]) == 0.0 + center_position_z
    # Max z is 0 + z offset
    assert np.max(mesh.vertices[:, 2]) == 0.0 + center_position_z
