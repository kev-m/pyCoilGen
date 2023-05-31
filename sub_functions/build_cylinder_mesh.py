import numpy as np
from data_structures import CoilMesh

def build_cylinder_mesh(cylinder_height, cylinder_radius, num_circular_divisions, num_longitudinal_divisions,
                        rotation_vector_x, rotation_vector_y, rotation_vector_z, rotation_angle):
    """
    Builds a cylinder mesh in 3D space.

    Args:
        cylinder_height (float): The height of the cylinder.
        cylinder_radius (float): The radius of the cylinder.
        num_circular_divisions (int): The number of divisions for the circular cross-section.
        num_longitudinal_divisions (int): The number of divisions along the height of the cylinder.
        rotation_vector_x (float): The x-component of the rotation vector.
        rotation_vector_y (float): The y-component of the rotation vector.
        rotation_vector_z (float): The z-component of the rotation vector.
        rotation_angle (float): The rotation angle in radians.

    Returns:
        numpy.ndarray: The coordinates of the vertices and the indices of the faces in the cylinder mesh.
            The vertices are represented as a 2D array of shape ((num_circular_divisions + 1) * (num_longitudinal_divisions + 1), 3),
            and the faces are represented as a 2D array of shape (num_circular_divisions * num_longitudinal_divisions, 3).

    Raises:
        ValueError: If any of the input arguments is not positive.

    Examples:
        >>> cylinder_height = 2.0
        >>> cylinder_radius = 1.0
        >>> num_circular_divisions = 10
        >>> num_longitudinal_divisions = 5
        >>> rotation_vector_x = 1.0
        >>> rotation_vector_y = 0.0
        >>> rotation_vector_z = 0.0
        >>> rotation_angle = np.pi / 4
        >>> vertices, faces = build_cylinder_mesh(cylinder_height, cylinder_radius, num_circular_divisions,
        ...                                      num_longitudinal_divisions, rotation_vector_x, rotation_vector_y,
        ...                                      rotation_vector_z, rotation_angle)
        >>> print(vertices)
        [[ 0.   -1.    0.  ]
         [ 0.   -0.707 0.707]
         [ 0.    0.    1.  ]
         ...
         [ 0.   -0.707 2.707]
         [ 0.   -1.    3.  ]]
        >>> print(faces)
        [[ 0  1 10]
         [ 1  2 10]
         [ 2  3 10]
         ...
         [15  6 16]
         [ 6  7 16]
         [ 7  8 16]]
    """
    if cylinder_height <= 0 or cylinder_radius <= 0 or num_circular_divisions <= 0 or num_longitudinal_divisions <= 0:
        raise ValueError("All input arguments must be positive.")

    # Create the circular cross-section vertices
    angles = np.linspace(0, 2 * np.pi, num_circular_divisions + 1)
    x_circular = cylinder_radius * np.cos(angles)
    y_circular = cylinder_radius * np.sin(angles)
    z_circular = np.zeros(num_circular_divisions + 1)

    # Create the longitudinal division vertices
    z_longitudinal = np.linspace(
        0, cylinder_height, num_longitudinal_divisions + 1)

    # Combine the circular and longitudinal vertices
    vertices = np.empty(((num_circular_divisions + 1) *
                        (num_longitudinal_divisions + 1), 3))
    for i in range(num_longitudinal_divisions + 1):
        vertices[i * (num_circular_divisions + 1): (i + 1) *
                 (num_circular_divisions + 1), 0] = x_circular
        vertices[i * (num_circular_divisions + 1): (i + 1) *
                 (num_circular_divisions + 1), 1] = y_circular
        vertices[i * (num_circular_divisions + 1): (i + 1) *
                 (num_circular_divisions + 1), 2] = z_longitudinal[i]

    # Apply rotation to vertices
    rotation_matrix = get_rotation_matrix(
        rotation_vector_x, rotation_vector_y, rotation_vector_z, rotation_angle)
    vertices = np.dot(vertices, rotation_matrix.T)

    # Create the faces
    tri_1_vert_inds_1 = np.arange(1, (num_circular_divisions)*(num_longitudinal_divisions)+1)
    tri_1_vert_inds_2 = tri_1_vert_inds_1 + 1

    # Take care of index overflow at the end of the rings
    overflow_indices = np.where(np.mod(tri_1_vert_inds_2 - 1, num_circular_divisions) == 0)
    tri_1_vert_inds_2[overflow_indices] -= num_circular_divisions

    tri_1_vert_inds_3 = tri_1_vert_inds_2 + num_circular_divisions

    tri_2_vert_inds_1 = tri_1_vert_inds_1
    tri_2_vert_inds_2 = tri_1_vert_inds_3
    tri_2_vert_inds_3 = np.arange(1, (num_circular_divisions)*(num_longitudinal_divisions)+1) + num_circular_divisions

    faces_1 = np.vstack((tri_1_vert_inds_2, tri_1_vert_inds_1, tri_1_vert_inds_3))
    faces_2 = np.vstack((tri_2_vert_inds_2, tri_2_vert_inds_1, tri_2_vert_inds_3))

    faces = np.hstack((faces_1.T, faces_2.T)).T

    mesh = CoilMesh(vertices, None, faces)
    return mesh


def get_rotation_matrix(rotation_vector_x, rotation_vector_y, rotation_vector_z, rotation_angle):
    """
    Calculates the rotation matrix based on the rotation vector and angle.

    Args:
        rotation_vector_x (float): The x-component of the rotation vector.
        rotation_vector_y (float): The y-component of the rotation vector.
        rotation_vector_z (float): The z-component of the rotation vector.
        rotation_angle (float): The rotation angle in radians.

    Returns:
        numpy.ndarray: The rotation matrix.

    Raises:
        ValueError: If the rotation angle is not finite.

    Examples:
        >>> rotation_vector_x = 1.0
        >>> rotation_vector_y = 0.0
        >>> rotation_vector_z = 0.0
        >>> rotation_angle = np.pi / 4
        >>> rotation_matrix = get_rotation_matrix(rotation_vector_x, rotation_vector_y,
        ...                                       rotation_vector_z, rotation_angle)
        >>> print(rotation_matrix)
        [[ 1.     0.     0.   ]
         [ 0.     0.707 -0.707]
         [ 0.     0.707  0.707]]
    """
    if not np.isfinite(rotation_angle):
        raise ValueError("Rotation angle must be a finite value.")

    cos_theta = np.cos(rotation_angle)
    sin_theta = np.sin(rotation_angle)

    rotation_matrix = np.array([[cos_theta + rotation_vector_x ** 2 * (1 - cos_theta),
                                 rotation_vector_x * rotation_vector_y *
                                 (1 - cos_theta) - rotation_vector_z * sin_theta,
                                 rotation_vector_x * rotation_vector_z * (1 - cos_theta) + rotation_vector_y * sin_theta],
                                [rotation_vector_y * rotation_vector_x * (1 - cos_theta) + rotation_vector_z * sin_theta,
                                 cos_theta + rotation_vector_y ** 2 *
                                 (1 - cos_theta),
                                 rotation_vector_y * rotation_vector_z * (1 - cos_theta) - rotation_vector_x * sin_theta],
                                [rotation_vector_z * rotation_vector_x * (1 - cos_theta) - rotation_vector_y * sin_theta,
                                 rotation_vector_z * rotation_vector_y *
                                 (1 - cos_theta) + rotation_vector_x * sin_theta,
                                 cos_theta + rotation_vector_z ** 2 * (1 - cos_theta)]])

    return rotation_matrix


if __name__ == "__main__":
    cylinder_height = 2.0
    cylinder_radius = 1.0
    num_circular_divisions = 10
    num_longitudinal_divisions = 5
    rotation_vector_x = 1.0
    rotation_vector_y = 0.0
    rotation_vector_z = 0.0
    rotation_angle = np.pi / 4
    vertices, faces = build_cylinder_mesh(cylinder_height, cylinder_radius, num_circular_divisions,
                                          num_longitudinal_divisions, rotation_vector_x, rotation_vector_y,
                                          rotation_vector_z, rotation_angle)
    print(vertices)
    print(faces)

"""
 [[ 1.00000000e+00  0.00000000e+00  0.00000000e+00]
 [ 8.09016994e-01  4.15626938e-01  4.15626938e-01]
 [ 3.09016994e-01  6.72498512e-01  6.72498512e-01]
 [-3.09016994e-01  6.72498512e-01  6.72498512e-01]
 [-8.09016994e-01  4.15626938e-01  4.15626938e-01]
 [-1.00000000e+00  8.65956056e-17  8.65956056e-17]
 [-8.09016994e-01 -4.15626938e-01 -4.15626938e-01]
 [-3.09016994e-01 -6.72498512e-01 -6.72498512e-01]
 [ 3.09016994e-01 -6.72498512e-01 -6.72498512e-01]
 [ 8.09016994e-01 -4.15626938e-01 -4.15626938e-01]
 [ 1.00000000e+00 -1.73191211e-16 -1.73191211e-16]
 [ 1.00000000e+00 -2.82842712e-01  2.82842712e-01]
 [ 8.09016994e-01  1.32784225e-01  6.98469650e-01]
 [ 3.09016994e-01  3.89655799e-01  9.55341224e-01]
 [-3.09016994e-01  3.89655799e-01  9.55341224e-01]
 [-8.09016994e-01  1.32784225e-01  6.98469650e-01]
 [-1.00000000e+00 -2.82842712e-01  2.82842712e-01]
 [-8.09016994e-01 -6.98469650e-01 -1.32784225e-01]
 [-3.09016994e-01 -9.55341224e-01 -3.89655799e-01]
 [ 3.09016994e-01 -9.55341224e-01 -3.89655799e-01]
 [ 8.09016994e-01 -6.98469650e-01 -1.32784225e-01]
 [ 1.00000000e+00 -2.82842712e-01  2.82842712e-01]
 [ 1.00000000e+00 -5.65685425e-01  5.65685425e-01]
 [ 8.09016994e-01 -1.50058487e-01  9.81312363e-01]
 [ 3.09016994e-01  1.06813087e-01  1.23818394e+00]
 [-3.09016994e-01  1.06813087e-01  1.23818394e+00]
 [-8.09016994e-01 -1.50058487e-01  9.81312363e-01]
 [-1.00000000e+00 -5.65685425e-01  5.65685425e-01]
 [-8.09016994e-01 -9.81312363e-01  1.50058487e-01]
 [-3.09016994e-01 -1.23818394e+00 -1.06813087e-01]
 [ 3.09016994e-01 -1.23818394e+00 -1.06813087e-01]
 [ 8.09016994e-01 -9.81312363e-01  1.50058487e-01]
 [ 1.00000000e+00 -5.65685425e-01  5.65685425e-01]
 [ 1.00000000e+00 -8.48528137e-01  8.48528137e-01]
 [ 8.09016994e-01 -4.32901200e-01  1.26415508e+00]
 [ 3.09016994e-01 -1.76029625e-01  1.52102665e+00]
 [-3.09016994e-01 -1.76029625e-01  1.52102665e+00]
 [-8.09016994e-01 -4.32901200e-01  1.26415508e+00]
 [-1.00000000e+00 -8.48528137e-01  8.48528137e-01]
 [-8.09016994e-01 -1.26415508e+00  4.32901200e-01]
 [-3.09016994e-01 -1.52102665e+00  1.76029625e-01]
 [ 3.09016994e-01 -1.52102665e+00  1.76029625e-01]
 [ 8.09016994e-01 -1.26415508e+00  4.32901200e-01]
 [ 1.00000000e+00 -8.48528137e-01  8.48528137e-01]
 [ 1.00000000e+00 -1.13137085e+00  1.13137085e+00]
 [ 8.09016994e-01 -7.15743912e-01  1.54699779e+00]
 [ 3.09016994e-01 -4.58872338e-01  1.80386936e+00]
 [-3.09016994e-01 -4.58872338e-01  1.80386936e+00]
 [-8.09016994e-01 -7.15743912e-01  1.54699779e+00]
 [-1.00000000e+00 -1.13137085e+00  1.13137085e+00]
 [-8.09016994e-01 -1.54699779e+00  7.15743912e-01]
 [-3.09016994e-01 -1.80386936e+00  4.58872338e-01]
 [ 3.09016994e-01 -1.80386936e+00  4.58872338e-01]
 [ 8.09016994e-01 -1.54699779e+00  7.15743912e-01]
 [ 1.00000000e+00 -1.13137085e+00  1.13137085e+00]
 [ 1.00000000e+00 -1.41421356e+00  1.41421356e+00]
 [ 8.09016994e-01 -9.98586625e-01  1.82984050e+00]
 [ 3.09016994e-01 -7.41715050e-01  2.08671207e+00]
 [-3.09016994e-01 -7.41715050e-01  2.08671207e+00]
 [-8.09016994e-01 -9.98586625e-01  1.82984050e+00]
 [-1.00000000e+00 -1.41421356e+00  1.41421356e+00]
 [-8.09016994e-01 -1.82984050e+00  9.98586625e-01]
 [-3.09016994e-01 -2.08671207e+00  7.41715050e-01]
 [ 3.09016994e-01 -2.08671207e+00  7.41715050e-01]
 [ 8.09016994e-01 -1.82984050e+00  9.98586625e-01]
 [ 1.00000000e+00 -1.41421356e+00  1.41421356e+00]]
[[ 2  3  4  5  6  7  8  9 10  1 12 13 14 15 16 17 18 19 20 11 22 23 24 25
  26 27 28 29 30 21 32 33 34 35 36 37 38 39 40 31 42 43 44 45 46 47 48 49
  50 41]
 [ 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24
  25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48
  49 50]
 [12 13 14 15 16 17 18 19 20 11 22 23 24 25 26 27 28 29 30 21 32 33 34 35
  36 37 38 39 40 31 42 43 44 45 46 47 48 49 50 41 52 53 54 55 56 57 58 59
  60 51]
 [12 13 14 15 16 17 18 19 20 11 22 23 24 25 26 27 28 29 30 21 32 33 34 35
  36 37 38 39 40 31 42 43 44 45 46 47 48 49 50 41 52 53 54 55 56 57 58 59
  60 51]
 [ 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24
  25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48
  49 50]
 [11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34
  35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58
  59 60]]
"""

"""
Convert the Matlab file given by the URL below into Python, with docstring comments:
 https://raw.githubusercontent.com/Philipp-MR/CoilGen/main/sub_functions/build_cylinder_mesh.m

Note that the matlab function signature is:
 build_cylinder_mesh(cylinder_height,cylinder_radius,num_circular_divisions,num_longitudinal_divisions,rotation_vector_x,rotation_vector_y,rotation_vector_z,rotation_angle)
"""
