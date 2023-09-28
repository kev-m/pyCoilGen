import numpy as np
# Logging
import logging

# Local imports
from .data_structures import DataStructure

log = logging.getLogger(__name__)


def build_planar_mesh(planar_height, planar_width, num_lateral_divisions, num_longitudinal_divisions,
                      rotation_vector_x, rotation_vector_y, rotation_vector_z, rotation_angle,
                      center_position_x, center_position_y, center_position_z):
    """
    Generate a planar mesh with specified dimensions and parameters.

    Args:
        planar_height (float): Height of the planar mesh.
        planar_width (float): Width of the planar mesh.
        num_lateral_divisions (int): Number of divisions in the lateral direction.
        num_longitudinal_divisions (int): Number of divisions in the longitudinal direction.
        rotation_vector_x (float): X component of the rotation vector.
        rotation_vector_y (float): Y component of the rotation vector.
        rotation_vector_z (float): Z component of the rotation vector.
        rotation_angle (float): Rotation angle in radians.
        center_position_x (float): X component of the center position.
        center_position_y (float): Y component of the center position.
        center_position_z (float): Z component of the center position.

    Returns:
        tuple: Tuple containing the vertices and faces of the planar mesh.

    """
    simple_vertices, faces = simple_planar_mesh(
        planar_height, planar_width, num_lateral_divisions, num_longitudinal_divisions)
    vertices, normal_rep = apply_rotation_translation(simple_vertices, rotation_vector_x, rotation_vector_y, rotation_vector_z, rotation_angle,
                                                      center_position_x, center_position_y, center_position_z)
    return DataStructure(vertices=vertices, faces=faces, normal=normal_rep)


def simple_planar_mesh(planar_height, planar_width, num_lateral_divisions, num_longitudinal_divisions):
    """
    Generate a planar mesh with specified dimensions and parameters.

    Args:
        planar_height (float): Height of the planar mesh.
        planar_width (float): Width of the planar mesh.
        num_lateral_divisions (int): Number of divisions in the lateral direction.
        num_longitudinal_divisions (int): Number of divisions in the longitudinal direction.

    Returns:
        tuple: Tuple containing the vertices and faces of the planar mesh.

    """

    # Calculate the step size in the lateral and longitudinal directions
    lateral_step = planar_width / num_lateral_divisions
    longitudinal_step = planar_height / num_longitudinal_divisions

    # Generate the vertices of the planar mesh
    vertices = np.empty(((num_lateral_divisions + 1) *
                        (num_longitudinal_divisions + 1), 3))

    index = 0
    for i in range(num_lateral_divisions + 1):
        for j in range(num_longitudinal_divisions + 1):
            x = i * lateral_step - planar_width / 2
            y = j * longitudinal_step - planar_height / 2
            z = 0.0

            vertices[index] = np.array([x, y, z])
            index += 1

    # Generate the faces of the planar mesh
    faces = np.empty(((num_lateral_divisions) *
                      (num_longitudinal_divisions) * 2, 3), dtype=int)
    index = 0
    for i in range(num_lateral_divisions):
        for j in range(num_longitudinal_divisions):
            # Calculate the indices of the vertices for each face
            v1 = i * (num_longitudinal_divisions + 1) + j
            v2 = (i + 1) * (num_longitudinal_divisions + 1) + j
            v3 = v2 + 1
            v4 = v1 + 1

            # Create the two triangles for each face
            # faces.append([v1, v2, v3])
            faces[index] = np.array([v1, v2, v3])
            index += 1
            # faces.append([v1, v3, v4])
            faces[index] = np.array([v1, v3, v4])
            index += 1

    # return np.array(vertices), np.array(faces)
    return vertices, faces


def apply_rotation_translation(vertices, rotation_vector_x, rotation_vector_y, rotation_vector_z, rotation_angle,
                               center_position_x, center_position_y, center_position_z):
    """
    Apply rotation and translation to a point in 3D space.

    Args:
        vertices (ndarray) : Array of vertics
        rotation_vector_x (float): X component of the rotation vector.
        rotation_vector_y (float): Y component of the rotation vector.
        rotation_vector_z (float): Z component of the rotation vector.
        rotation_angle (float): Rotation angle in radians.
        center_position_x (float): X component of the center position.
        center_position_y (float): Y component of the center position.
        center_position_z (float): Z component of the center position.

    Returns:
        vertices: ndarray containing the transformed input.
        normal_rep: ndarray vector of the new surface normal

    """

    # Apply rotation around the rotation vector
    rotation_matrix = calculate_rotation_matrix(
        rotation_vector_x, rotation_vector_y, rotation_vector_z, rotation_angle)
    rotated_vertices = np.dot(vertices, rotation_matrix)

    # Apply translation
    translated_vertices = rotated_vertices + \
        np.array([center_position_x, center_position_y, center_position_z])

    # Calculate representative normal
    normal = [0.0, 0.0, 1.0]
    normal_rep = np.dot(normal, rotation_matrix)

    return translated_vertices, normal_rep


def calculate_rotation_matrix(rotation_vector_x, rotation_vector_y, rotation_vector_z, rotation_angle):
    """
    Calculate the rotation matrix given the rotation vector and angle.

    Args:
        rotation_vector_x (float): X component of the rotation vector.
        rotation_vector_y (float): Y component of the rotation vector.
        rotation_vector_z (float): Z component of the rotation vector.
        rotation_angle (float): Rotation angle in radians.

    Returns:
        numpy.ndarray: Rotation matrix.

    """

    c = np.cos(rotation_angle)
    s = np.sin(rotation_angle)
    t = 1 - c

    # Calculate the components of the rotation matrix
    xx = rotation_vector_x * rotation_vector_x * t + c
    xy = rotation_vector_x * rotation_vector_y * t - rotation_vector_z * s
    xz = rotation_vector_x * rotation_vector_z * t + rotation_vector_y * s
    yx = rotation_vector_x * rotation_vector_y * t + rotation_vector_z * s
    yy = rotation_vector_y * rotation_vector_y * t + c
    yz = rotation_vector_y * rotation_vector_z * t - rotation_vector_x * s
    zx = rotation_vector_x * rotation_vector_z * t - rotation_vector_y * s
    zy = rotation_vector_y * rotation_vector_z * t + rotation_vector_x * s
    zz = rotation_vector_z * rotation_vector_z * t + c

    # Construct the rotation matrix
    rotation_matrix = np.array([[xx, xy, xz],
                                [yx, yy, yz],
                                [zx, zy, zz]])

    return rotation_matrix
