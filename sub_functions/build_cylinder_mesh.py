import numpy as np

# Logging
import logging

# Local imports
from sub_functions.calc_3d_rotation_matrix_by_vector import calc_3d_rotation_matrix_by_vector
from sub_functions.data_structures import DataStructure

log = logging.getLogger(__name__)

def build_cylinder_mesh(
        cylinder_height,
        cylinder_radius,
        num_circular_divisions,
        num_longitudinal_divisions,
        rotation_vector_x,
        rotation_vector_y,
        rotation_vector_z,
        rotation_angle
):
    """
    Create a cylindrical regular mesh in any orientation.

    Args:
        cylinder_height (float): Height of the cylinder.
        cylinder_radius (float): Radius of the cylinder.
        num_circular_divisions (int): Number of circular divisions.
        num_longitudinal_divisions (int): Number of longitudinal divisions.
        rotation_vector_x (float): X-component of the rotation vector.
        rotation_vector_y (float): Y-component of the rotation vector.
        rotation_vector_z (float): Z-component of the rotation vector.
        rotation_angle (float): Rotation angle.

    Returns:
        mesh: DataStructure with 'faces' and 'vertices' arrays of the cylindrical mesh.
    """
    # Calculate positions along the circular divisions
    theta = np.linspace(0, 2*np.pi, num_circular_divisions, endpoint=False)
    x_positions = np.sin(theta) * cylinder_radius
    y_positions = np.cos(theta) * cylinder_radius

    # Calculate positions along the longitudinal divisions
    z_positions = np.linspace(-cylinder_height/2, cylinder_height/2, num_longitudinal_divisions+1)

    # Create vertices
    vertices_temp = []
    for z in z_positions:
        for x, y in zip(x_positions, y_positions):
            vertices_temp.append([x, y, z])
    vertices_temp = np.array(vertices_temp)

    # Create faces
    faces = []
    for i in range(num_longitudinal_divisions):
        for j in range(num_circular_divisions):
            a = i * num_circular_divisions + j
            b = i * num_circular_divisions + (j + 1) % num_circular_divisions
            c = (i + 1) * num_circular_divisions + j
            d = (i + 1) * num_circular_divisions + (j + 1) % num_circular_divisions
            faces.append([a, b, c])
            faces.append([b, d, c])
    faces = np.array(faces)

    # Rotate the cylinder in the desired orientation
    rot_vec = np.array([rotation_vector_x, rotation_vector_y, rotation_vector_z])
    rot_mat = calc_3d_rotation_matrix_by_vector(rot_vec, rotation_angle)

    vertices=np.dot(rot_mat, vertices_temp.T).T
    cylinder_mesh = DataStructure(vertices=vertices, faces=faces)

    return cylinder_mesh



if __name__ == "__main__":
    cylinder_height = 2.0
    cylinder_radius = 1.0
    num_circular_divisions = 10
    num_longitudinal_divisions = 5
    rotation_vector_x = 1.0
    rotation_vector_y = 0.0
    rotation_vector_z = 0.0
    rotation_angle = np.pi / 4
    mesh = build_cylinder_mesh(cylinder_height, cylinder_radius, num_circular_divisions,
                               num_longitudinal_divisions, rotation_vector_x, rotation_vector_y,
                               rotation_vector_z, rotation_angle)
    print("vertices = ", mesh.vertices, mesh.vertices.shape)
    print("faces = ", mesh.faces, np.min(mesh.faces), np.max(mesh.faces))

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
