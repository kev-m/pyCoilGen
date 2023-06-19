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
    # Calculate x, y, z positions of the vertices
    x_positions = np.sin(np.linspace(0, 2 * np.pi, num_circular_divisions+1)) * cylinder_radius
    y_positions = np.cos(np.linspace(0, 2 * np.pi, num_circular_divisions+1)) * cylinder_radius
    x_positions = x_positions[:-1]  # Remove repetition at the end
    y_positions = y_positions[:-1]  # Remove repetition at the end
    z_positions = np.linspace(-cylinder_height / 2, cylinder_height / 2,
                              num_longitudinal_divisions+1)

    # log.debug(" Shape x_positions: %s", np.shape(x_positions))
    # log.debug(" Shape y_positions: %s", np.shape(y_positions))
    # log.debug(" Shape z_positions: %s", np.shape(z_positions))

    # Create the mesh vertices
    vertices_x = np.tile(x_positions, num_longitudinal_divisions+1)
    vertices_y = np.tile(y_positions, num_longitudinal_divisions+1)
    vertices_z = np.repeat(z_positions, len(x_positions))
    # log.debug(" Shape vertices_x: %s", np.shape(vertices_x))
    # log.debug(" Shape vertices_y: %s", np.shape(vertices_y))
    # log.debug(" Shape vertices_z: %s", np.shape(vertices_z))
    vertices = np.vstack((vertices_x, vertices_y, vertices_z))

    # Set the vertices in the center
    # log.debug(" Shape vertices: %s", vertices.shape)
    vertices = vertices - np.mean(vertices, axis=1, keepdims=True)

    # Create the faces for the cylinder mesh
    tri_1_vert_inds_1 = np.arange(num_circular_divisions * num_longitudinal_divisions)
    tri_1_vert_inds_2 = tri_1_vert_inds_1 + 1
    tri_1_vert_inds_2[tri_1_vert_inds_2 % num_circular_divisions == 0] -= num_circular_divisions
    tri_1_vert_inds_3 = tri_1_vert_inds_2 + num_circular_divisions

    tri_2_vert_inds_1 = tri_1_vert_inds_1
    tri_2_vert_inds_2 = tri_1_vert_inds_3
    tri_2_vert_inds_3 = np.arange(num_circular_divisions * num_longitudinal_divisions) + num_circular_divisions

    faces_1 = np.column_stack((tri_1_vert_inds_2, tri_1_vert_inds_1, tri_1_vert_inds_3))
    faces_2 = np.column_stack((tri_2_vert_inds_2, tri_2_vert_inds_1, tri_2_vert_inds_3))
    faces = np.vstack((faces_1, faces_2))  # Subtract 1 due to Matlab index offset starting at 1

    # Rotate the cylinder in the desired orientation
    rot_vec = np.array([rotation_vector_x, rotation_vector_y, rotation_vector_z])
    rot_mat = calc_3d_rotation_matrix_by_vector(rot_vec, rotation_angle)
    vertices = np.dot(vertices.T, rot_mat)

    # Calculate representative normal
    normal = np.array([0.0, 0.0, 1.0])
    normal_rep = np.dot(normal, rot_mat)

    cylinder_mesh = DataStructure(vertices=vertices, faces=faces, normal=normal_rep)

    return cylinder_mesh


if __name__ == "__main__":
    # Set up logging
    log = logging.getLogger(__name__)
    logging.basicConfig(level=logging.DEBUG)

    params = [0.4,  0.1125, 50, 50,  0.,  1.,  0., 0.]
    mesh = build_cylinder_mesh(*params)

    log.debug(" faces: %s, %s, min: %s, max: %s", mesh.faces, mesh.faces.shape, np.min(mesh.faces), np.max(mesh.faces))
    log.debug(" Should be:\n [2500 2499 2549]], (5000, 3)")

    log.debug(" vertices: %s, %s", mesh.vertices, mesh.vertices.shape)
    log.debug(" Should be:\n [-1.40999888e-02  1.11612904e-01  1.96078431e-01]], (2550, 3)")

    from sub_functions.data_structures import Mesh
    tri_mesh = Mesh(vertices=mesh.vertices, faces=mesh.faces)

    t_faces = tri_mesh.get_faces()
    log.debug(" t_faces: %s, %s, min: %s, max: %s", t_faces, t_faces.shape, np.min(t_faces), np.max(t_faces))
    # tri_mesh.display()

"""
DEBUG:__main__: m_faces: [[   1    0   51]
 [   2    1   52]
 [   3    2   53]
 ...
 [2548 2497 2547]
 [2549 2498 2548]
 [2500 2499 2549]], (5000, 3)
DEBUG:__main__: m_vertices: [[ 4.14973067e-20  1.12500000e-01 -1.96078431e-01]
 [ 1.40999888e-02  1.11612904e-01 -1.96078431e-01]
 [ 2.79776123e-02  1.08965606e-01 -1.96078431e-01]
 ...
 [-4.14140122e-02  1.04599855e-01  1.96078431e-01]
 [-2.79776123e-02  1.08965606e-01  1.96078431e-01]
 [-1.40999888e-02  1.11612904e-01  1.96078431e-01]], (2550, 3)
"""

"""    
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
