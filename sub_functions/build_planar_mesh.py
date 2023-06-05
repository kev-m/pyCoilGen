import numpy as np
# Logging
import logging

# Local imports
from data_structures import DataStructure

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
    simple_vertices, faces = simple_planar_mesh(planar_height, planar_width, num_lateral_divisions, num_longitudinal_divisions)
    vertices = apply_rotation_translation(simple_vertices, rotation_vector_x, rotation_vector_y, rotation_vector_z, rotation_angle,
                      center_position_x, center_position_y, center_position_z)
    return DataStructure(vertices=vertices, faces=faces)


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
            #faces.append([v1, v2, v3])
            faces[index] = np.array([v1, v2, v3])
            index += 1
            #faces.append([v1, v3, v4])
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

    """

    # Apply rotation around the rotation vector
    rotation_matrix = calculate_rotation_matrix(
        rotation_vector_x, rotation_vector_y, rotation_vector_z, rotation_angle)
    rotated_vertices = np.dot(vertices, rotation_matrix)

    # Apply translation
    translated_vertices = rotated_vertices + \
        np.array([center_position_x, center_position_y, center_position_z])

    return translated_vertices


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


if __name__ == "__main__":
    planar_height = 2.0
    planar_width = 3.0
    num_lateral_divisions = 4
    num_circular_divisions = 5
    num_longitudinal_divisions = 5
    rotation_vector_x = 1.0
    rotation_vector_y = 0.0
    rotation_vector_z = 0.0
    rotation_angle = 0.0
    center_position_x = 0.0
    center_position_y = 0.0
    center_position_z = 0.0
    mesh = build_planar_mesh(planar_height, planar_width, num_lateral_divisions, num_longitudinal_divisions,
                             rotation_vector_x, rotation_vector_y, rotation_vector_z, rotation_angle,
                             center_position_x, center_position_y, center_position_z)
    print(mesh.vertices)
    print(mesh.faces)

"""
[[-1.5  -1.    0.  ]
 [-1.5  -0.6   0.  ]
 [-1.5  -0.2   0.  ]
 [-1.5   0.2   0.  ]
 [-1.5   0.6   0.  ]
 [-1.5   1.    0.  ]
 [-0.75 -1.    0.  ]
 [-0.75 -0.6   0.  ]
 [-0.75 -0.2   0.  ]
 [-0.75  0.2   0.  ]
 [-0.75  0.6   0.  ]
 [-0.75  1.    0.  ]
 [ 0.   -1.    0.  ]
 [ 0.   -0.6   0.  ]
 [ 0.   -0.2   0.  ]
 [ 0.    0.2   0.  ]
 [ 0.    0.6   0.  ]
 [ 0.    1.    0.  ]
 [ 0.75 -1.    0.  ]
 [ 0.75 -0.6   0.  ]
 [ 0.75 -0.2   0.  ]
 [ 0.75  0.2   0.  ]
 [ 0.75  0.6   0.  ]
 [ 0.75  1.    0.  ]
 [ 1.5  -1.    0.  ]
 [ 1.5  -0.6   0.  ]
 [ 1.5  -0.2   0.  ]
 [ 1.5   0.2   0.  ]
 [ 1.5   0.6   0.  ]
 [ 1.5   1.    0.  ]]
[[ 0  6  7]
 [ 0  7  1]
 [ 1  7  8]
 [ 1  8  2]
 [ 2  8  9]
 [ 2  9  3]
 [ 3  9 10]
 [ 3 10  4]
 [ 4 10 11]
 [ 4 11  5]
 [ 6 12 13]
 [ 6 13  7]
 [ 7 13 14]
 [ 7 14  8]
 [ 8 14 15]
 [ 8 15  9]
 [ 9 15 16]
 [ 9 16 10]
 [10 16 17]
 [10 17 11]
 [12 18 19]
 [12 19 13]
 [13 19 20]
 [13 20 14]
 [14 20 21]
 [14 21 15]
 [15 21 22]
 [15 22 16]
 [16 22 23]
 [16 23 17]
 [18 24 25]
 [18 25 19]
 [19 25 26]
 [19 26 20]
 [20 26 27]
 [20 27 21]
 [21 27 28]
 [21 28 22]
 [22 28 29]
 [22 29 23]]
"""

"""
Convert the Matlab file given by the URL below into Python, with docstring comments:
https://raw.githubusercontent.com/Philipp-MR/CoilGen/main/sub_functions/build_planar_mesh.m
Note that the matlab function signature is build_planar_mesh(planar_height,planar_width,num_lateral_divisions,num_longitudinal_divisions,rotation_vector_x,rotation_vector_y,rotation_vector_z,rotation_angle,center_position_x,center_position_y,center_position_z)
"""
