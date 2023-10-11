import numpy as np

# Test support
from pytest import approx

# Code under test
from pyCoilGen.mesh_factory.build_biplanar_mesh import build_biplanar_mesh


def test_build_biplanar_mesh_basic():
    planar_height = 0.25
    planar_width = 0.4
    num_lateral_divisions = 5
    num_longitudinal_divisions = 4
    target_normal_x = 1.0
    target_normal_y = 0.0
    target_normal_z = 0.0
    center_position_x = 0
    center_position_y = 0
    center_position_z = 0
    plane_distance = 0.2

    mesh = build_biplanar_mesh(planar_height, planar_width,
                               num_lateral_divisions, num_longitudinal_divisions,
                               target_normal_x, target_normal_y, target_normal_z,
                               center_position_x, center_position_y, center_position_z,
                               plane_distance)

    # Min x is -0.5 width
    assert np.min(mesh.vertices[:, 0]) == approx(-plane_distance/2.0)
    # Max x is 0.5 width
    assert np.max(mesh.vertices[:, 0]) == approx(plane_distance/2.0)

    # Min y is -0.5 height
    assert np.min(mesh.vertices[:, 1]) == -planar_height/2.0
    # Max y is 0.5 height
    assert np.max(mesh.vertices[:, 1]) == planar_height/2.0

    # Min z is 0
    assert np.min(mesh.vertices[:, 2]) == -planar_width/2.0
    # Max z is 0
    assert np.max(mesh.vertices[:, 2]) == planar_width/2.0

    # Test shape
    assert mesh.vertices.shape == (2*(num_lateral_divisions+1)*(num_longitudinal_divisions+1), 3)


def test_build_biplanar_mesh_y_axis():
    planar_height = 0.25
    planar_width = 0.4
    num_lateral_divisions = 5
    num_longitudinal_divisions = 4
    target_normal_x = 0.0
    target_normal_y = 1.0
    target_normal_z = 0.0
    center_position_x = 0
    center_position_y = 0
    center_position_z = 0
    plane_distance = 0.2

    mesh = build_biplanar_mesh(planar_height, planar_width,
                               num_lateral_divisions, num_longitudinal_divisions,
                               target_normal_x, target_normal_y, target_normal_z,
                               center_position_x, center_position_y, center_position_z,
                               plane_distance)

    # Min x is -0.5 height
    assert np.min(mesh.vertices[:, 0]) == -planar_width/2.0
    # Max x is 0.5 height
    assert np.max(mesh.vertices[:, 0]) == planar_width/2.0

    # Min y is -0.5 height
    assert np.min(mesh.vertices[:, 1]) == approx(-plane_distance/2.0)
    # Max y is 0.5 height
    assert np.max(mesh.vertices[:, 1]) == approx(plane_distance/2.0)

    # Min z is 0
    assert np.min(mesh.vertices[:, 2]) == -planar_height/2.0
    # Max z is 0
    assert np.max(mesh.vertices[:, 2]) == planar_height/2.0


def test_build_biplanar_mesh_z_axis():
    planar_height = 0.25
    planar_width = 0.4
    num_lateral_divisions = 5
    num_longitudinal_divisions = 4
    target_normal_x = 0.0
    target_normal_y = 0.0
    target_normal_z = 1.0
    center_position_x = 0
    center_position_y = 0
    center_position_z = 0
    plane_distance = 0.2

    mesh = build_biplanar_mesh(planar_height, planar_width,
                               num_lateral_divisions, num_longitudinal_divisions,
                               target_normal_x, target_normal_y, target_normal_z,
                               center_position_x, center_position_y, center_position_z,
                               plane_distance)

    # Min x is -0.5 height
    assert np.min(mesh.vertices[:, 0]) == -planar_width/2.0
    # Max x is 0.5 height
    assert np.max(mesh.vertices[:, 0]) == planar_width/2.0

    # Min y is -0.5 height
    assert np.min(mesh.vertices[:, 1]) == -planar_height/2.0
    # Max y is 0.5 height
    assert np.max(mesh.vertices[:, 1]) == planar_height/2.0

    # Min z is 0
    assert np.min(mesh.vertices[:, 2]) == -plane_distance/2.0
    # Max z is 0
    assert np.max(mesh.vertices[:, 2]) == plane_distance/2.0


def test_build_biplanar_mesh_translate():
    planar_height = 0.25
    planar_width = 0.4
    num_lateral_divisions = 5
    num_longitudinal_divisions = 4
    target_normal_x = 0.0
    target_normal_y = 0.0
    target_normal_z = 0.0
    center_position_x = 1.0
    center_position_y = -0.5
    center_position_z = 0.25
    plane_distance = 0.2

    mesh = build_biplanar_mesh(planar_height, planar_width,
                               num_lateral_divisions, num_longitudinal_divisions,
                               target_normal_x, target_normal_y, target_normal_z,
                               center_position_x, center_position_y, center_position_z,
                               plane_distance)

    # Min x is -0.5 height + x offset
    assert np.min(mesh.vertices[:, 0]) == -planar_width/2.0 + center_position_x
    # Max x is 0.5 height + x offset
    assert np.max(mesh.vertices[:, 0]) == planar_width/2.0 + center_position_x

    # Min y is -0.5 height + y offset
    assert np.min(mesh.vertices[:, 1]) == -planar_height/2.0 + center_position_y
    # Max y is 0.5 height + y offset
    assert np.max(mesh.vertices[:, 1]) == planar_height/2.0 + center_position_y

    # Min z is 0 + z offset
    assert np.min(mesh.vertices[:, 2]) == -plane_distance/2.0 + center_position_z
    # Max z is 0 + z offset
    assert np.max(mesh.vertices[:, 2]) == plane_distance/2.0 + center_position_z
