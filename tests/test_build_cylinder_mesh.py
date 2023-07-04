# Hack code
# Set up paths: Add the project root directory to the Python path
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
# Test support
from pytest import approx
# Code under test
from sub_functions.build_cylinder_mesh import build_cylinder_mesh


def test_build_cylinder_mesh_basic():
    cylinder_height = 0.25
    cylinder_radius = 0.4
    num_circular_divisions = 4
    num_longitudinal_divisions = 3
    rotation_vector_x = 0
    rotation_vector_y = 0
    rotation_vector_z = 1.0
    rotation_angle = 0

    mesh = build_cylinder_mesh(cylinder_height, cylinder_radius,
                               num_circular_divisions, num_longitudinal_divisions,
                               rotation_vector_x, rotation_vector_y, rotation_vector_z, rotation_angle)

    # Min x is -radius
    assert np.min(mesh.vertices[:, 0]) == -cylinder_radius
    # Max x is +radius
    assert np.max(mesh.vertices[:, 0]) == cylinder_radius

    # Min y is -radius
    assert np.min(mesh.vertices[:, 1]) == -cylinder_radius
    # Max y is +radius
    assert np.max(mesh.vertices[:, 1]) == cylinder_radius

    # Min z is -height/2
    assert np.min(mesh.vertices[:, 2]) == -cylinder_height/2.0
    # Max z is +height/2
    assert np.max(mesh.vertices[:, 2]) == cylinder_height/2.0

    # Test shape
    assert mesh.vertices.shape == ((num_circular_divisions)*(num_longitudinal_divisions+1), 3)


def test_build_cylinder_mesh_rotate_Z():
    cylinder_height = 0.25
    cylinder_radius = 0.4
    num_circular_divisions = 4
    num_longitudinal_divisions = 3
    rotation_vector_x = 0
    rotation_vector_y = 0
    rotation_vector_z = 1.0
    rotation_angle = np.pi/2

    mesh = build_cylinder_mesh(cylinder_height, cylinder_radius,
                               num_circular_divisions, num_longitudinal_divisions,
                               rotation_vector_x, rotation_vector_y, rotation_vector_z, rotation_angle)

    # Mesh is unchanged, rotationally invariant for 90 degree rotation about Z
    # Min x is -radius
    assert np.min(mesh.vertices[:, 0]) == -cylinder_radius
    # Max x is +radius
    assert np.max(mesh.vertices[:, 0]) == cylinder_radius

    # Min y is -radius
    assert np.min(mesh.vertices[:, 1]) == -cylinder_radius
    # Max y is +radius
    assert np.max(mesh.vertices[:, 1]) == cylinder_radius

    # Min z is -height/2
    assert np.min(mesh.vertices[:, 2]) == -cylinder_height/2.0
    # Max z is +height/2
    assert np.max(mesh.vertices[:, 2]) == cylinder_height/2.0

    rotation_angle = np.pi/4
    mesh = build_cylinder_mesh(cylinder_height, cylinder_radius,
                               num_circular_divisions, num_longitudinal_divisions,
                               rotation_vector_x, rotation_vector_y, rotation_vector_z, rotation_angle)

    #
    # Min x is -radius * sin(45)
    assert np.min(mesh.vertices[:, 0]) == -cylinder_radius * np.sin(rotation_angle)
    # Max x is +radius
    assert np.max(mesh.vertices[:, 0]) == cylinder_radius * np.cos(rotation_angle)

    # Min y is -radius
    assert np.min(mesh.vertices[:, 1]) == -cylinder_radius * np.cos(rotation_angle)
    # Max y is +radius
    assert np.max(mesh.vertices[:, 1]) == cylinder_radius * np.cos(rotation_angle)

    # Min z is -height/2
    assert np.min(mesh.vertices[:, 2]) == -cylinder_height/2.0
    # Max z is +height/2
    assert np.max(mesh.vertices[:, 2]) == cylinder_height/2.0

    cylinder_height = 0.25
    cylinder_radius = 0.4
    num_circular_divisions = 4
    num_longitudinal_divisions = 3
    rotation_vector_x = 0
    rotation_vector_y = 0
    rotation_vector_z = 1.0
    rotation_angle = np.pi/2

    mesh = build_cylinder_mesh(cylinder_height, cylinder_radius,
                               num_circular_divisions, num_longitudinal_divisions,
                               rotation_vector_x, rotation_vector_y, rotation_vector_z, rotation_angle)

    # Mesh is unchanged, rotationally invariant for 90 degree rotation about Z
    # Min x is -radius
    assert np.min(mesh.vertices[:, 0]) == -cylinder_radius
    # Max x is +radius
    assert np.max(mesh.vertices[:, 0]) == cylinder_radius

    # Min y is -radius
    assert np.min(mesh.vertices[:, 1]) == -cylinder_radius
    # Max y is +radius
    assert np.max(mesh.vertices[:, 1]) == cylinder_radius

    # Min z is -height/2
    assert np.min(mesh.vertices[:, 2]) == -cylinder_height/2.0
    # Max z is +height/2
    assert np.max(mesh.vertices[:, 2]) == cylinder_height/2.0


def test_build_cylinder_mesh_rotate_Y():
    cylinder_height = 0.25
    cylinder_radius = 0.4
    num_circular_divisions = 4
    num_longitudinal_divisions = 3
    rotation_vector_x = 0
    rotation_vector_y = 1.0
    rotation_vector_z = 0
    rotation_angle = np.pi/2

    mesh = build_cylinder_mesh(cylinder_height, cylinder_radius,
                               num_circular_divisions, num_longitudinal_divisions,
                               rotation_vector_x, rotation_vector_y, rotation_vector_z, rotation_angle)

    # Mesh rotated Z onto X
    # Min x is -radius
    assert np.min(mesh.vertices[:, 0]) == approx(-cylinder_height/2.0)
    # Max x is +radius
    assert np.max(mesh.vertices[:, 0]) == approx(+cylinder_height/2.0)

    # Min y is -radius
    assert np.min(mesh.vertices[:, 1]) == -cylinder_radius
    # Max y is +radius
    assert np.max(mesh.vertices[:, 1]) == cylinder_radius

    # Min z is -height/2
    assert np.min(mesh.vertices[:, 2]) == -cylinder_radius
    # Max z is +height/2
    assert np.max(mesh.vertices[:, 2]) == +cylinder_radius


def test_build_cylinder_mesh_rotate_X():
    cylinder_height = 0.25
    cylinder_radius = 0.4
    num_circular_divisions = 4
    num_longitudinal_divisions = 3
    rotation_vector_x = 1.0
    rotation_vector_y = 0
    rotation_vector_z = 0
    rotation_angle = np.pi/2

    mesh = build_cylinder_mesh(cylinder_height, cylinder_radius,
                               num_circular_divisions, num_longitudinal_divisions,
                               rotation_vector_x, rotation_vector_y, rotation_vector_z, rotation_angle)

    # Mesh rotated Z onto Y
    # Min x is -radius
    assert np.min(mesh.vertices[:, 0]) == -cylinder_radius
    # Max x is +radius
    assert np.max(mesh.vertices[:, 0]) == cylinder_radius

    # Min y is -radius
    assert np.min(mesh.vertices[:, 1]) == approx(-cylinder_height/2.0)
    # Max y is +radius
    assert np.max(mesh.vertices[:, 1]) == approx(+cylinder_height/2.0)

    # Min z is -height/2
    assert np.min(mesh.vertices[:, 2]) == -cylinder_radius
    # Max z is +height/2
    assert np.max(mesh.vertices[:, 2]) == +cylinder_radius


def cp():
    log.debug(" rot: %s", rot)
    log.debug(" rot_mat: %s", rot_mat)
    log.debug(" min(rot_mat): %s", np.min(rot_mat))


if __name__ == "__main__":
    import logging
    # Set up logging
    log = logging.getLogger(__name__)
    logging.basicConfig(level=logging.DEBUG)

    test_build_cylinder_mesh_basic()
    test_build_cylinder_mesh_rotate_Z()
    test_build_cylinder_mesh_rotate_Y()
    test_build_cylinder_mesh_rotate_Z()
