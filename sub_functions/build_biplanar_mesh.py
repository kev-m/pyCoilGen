import numpy as np

import numpy as np

def build_biplanar_mesh(planar_height, planar_width, num_lateral_divisions, num_longitudinal_divisions, target_normal_x, target_normal_y, target_normal_z, center_position_x, center_position_y, center_position_z, plane_distance):
    """
    Create a biplanar regular mesh in any orientation.
    
    Parameters:
    - planar_height (float): Height of the planar mesh.
    - planar_width (float): Width of the planar mesh.
    - num_lateral_divisions (int): Number of divisions in the lateral direction.
    - num_longitudinal_divisions (int): Number of divisions in the longitudinal direction.
    - target_normal_x (float): X-component of the target normal vector.
    - target_normal_y (float): Y-component of the target normal vector.
    - target_normal_z (float): Z-component of the target normal vector.
    - center_position_x (float): X-coordinate of the center position.
    - center_position_y (float): Y-coordinate of the center position.
    - center_position_z (float): Z-coordinate of the center position.
    - plane_distance (float): Distance between the two planes.
    
    Returns:
    - biplanar_mesh (dict): Dictionary containing the mesh faces and vertices.
    """
    
    # Define the mesh triangles
    tri_1_vert_inds_1 = (np.tile(np.arange(1, num_lateral_divisions + 1), (num_longitudinal_divisions, 1)) + np.tile(np.arange(num_longitudinal_divisions)[:, np.newaxis], (1, num_lateral_divisions)) * (num_lateral_divisions + 1)).T
    tri_1_vert_inds_2 = (np.tile(np.arange(2, num_lateral_divisions + 2), (num_longitudinal_divisions, 1)) + np.tile(np.arange(num_longitudinal_divisions)[:, np.newaxis], (1, num_lateral_divisions)) * (num_lateral_divisions + 1)).T
    tri_1_vert_inds_3 = (np.tile(np.arange(2, num_lateral_divisions + 2), (num_longitudinal_divisions, 1)) + np.tile(np.arange(num_longitudinal_divisions)[:, np.newaxis] - 1, (1, num_lateral_divisions)) * (num_lateral_divisions + 1)).T
    tri_2_vert_inds_1 = tri_1_vert_inds_1
    tri_2_vert_inds_2 = (np.tile(np.arange(1, num_lateral_divisions + 1), (num_longitudinal_divisions, 1)) + np.tile(np.arange(num_longitudinal_divisions)[:, np.newaxis], (1, num_lateral_divisions)) * (num_lateral_divisions + 1)).T
    tri_2_vert_inds_3 = tri_1_vert_inds_2
    faces_1 = np.column_stack((tri_1_vert_inds_1.ravel(), tri_1_vert_inds_2.ravel(), tri_1_vert_inds_3.ravel()))
    faces_2 = np.column_stack((tri_2_vert_inds_1.ravel(), tri_2_vert_inds_2.ravel(), tri_2_vert_inds_3.ravel()))

    # Define the vertex positions
    x_positions = np.linspace(-planar_width/2, planar_width/2, num_lateral_divisions + 1)
    y_positions = np.linspace(-planar_height/2, planar_height/2, num_longitudinal_divisions + 1)
    x, y = np.meshgrid(x_positions, y_positions)
    
    old_normal = np.array([0, 0, 1])
    target_normal = np.array([target_normal_x, target_normal_y, target_normal_z])
    
    if np.linalg.norm(np.cross(old_normal, target_normal)) != 0:
        rot_vec = np.cross(old_normal, target_normal) / np.linalg.norm(np.cross(old_normal, target_normal))
        rot_angle = np.arcsin(np.linalg.norm(np.cross(old_normal, target_normal)) / (np.linalg.norm(old_normal) * np.linalg.norm(target_normal)))
    else:
        rot_vec = np.array([1, 0, 0])
        rot_angle = 0
    
    z1 = np.zeros_like(y) + plane_distance / 2
    z2 = np.zeros_like(y) - plane_distance / 2
    
    rot_mat = calc_3d_rotation_matrix_by_vector(rot_vec, rot_angle)
    
    vertices1 = np.dot(rot_mat, np.vstack((y.ravel(), x.ravel(), z1.ravel()))) + np.array([center_position_x, center_position_y, center_position_z]).reshape(3, 1)
    vertices2 = np.dot(rot_mat, np.vstack((y.ravel(), x.ravel(), z2.ravel()))) + np.array([center_position_x, center_position_y, center_position_z]).reshape(3, 1)
    
    faces_first_plane = np.vstack((faces_1, faces_2))
    faces_second_plane = np.vstack((faces_1, faces_2)) + vertices1.shape[1]
    
    biplanar_mesh = {
        'faces': np.vstack((faces_first_plane, faces_second_plane)),
        'vertices': np.vstack((vertices1.T, vertices2.T))
    }
    
    return biplanar_mesh

def calc_3d_rotation_matrix_by_vector(rot_vec, rot_angle):
    """
    Calculate the 3D rotation matrix around a rotation axis given by a vector and an angle.
    
    Parameters:
    - rot_vec (ndarray): Rotation axis vector.
    - rot_angle (float): Rotation angle.
    
    Returns:
    - rot_mat_out (ndarray): 3D rotation matrix.
    """
    
    rot_vec = rot_vec / np.linalg.norm(rot_vec)  # Normalize rot vector
    u_x = rot_vec[0]
    u_y = rot_vec[1]
    u_z = rot_vec[2]
    tmp1 = np.sin(rot_angle)
    tmp2 = np.cos(rot_angle)
    tmp3 = (1 - np.cos(rot_angle))
    rot_mat_out = np.zeros((3, 3))
    rot_mat_out[0, 0] = tmp2 + u_x * u_x * tmp3
    rot_mat_out[0, 1] = u_x * u_y * tmp3 - u_z * tmp1
    rot_mat_out[0, 2] = u_x * u_z * tmp3 + u_y * tmp1
    rot_mat_out[1, 0] = u_y * u_x * tmp3 + u_z * tmp1
    rot_mat_out[1, 1] = tmp2 + u_y * u_y * tmp3
    rot_mat_out[1, 2] = u_y * u_z * tmp3 - u_x * tmp1
    rot_mat_out[2, 0] = u_z * u_x * tmp3 - u_y * tmp1
    rot_mat_out[2, 1] = u_z * u_y * tmp3 + u_x * tmp1
    rot_mat_out[2, 2] = tmp2 + u_z * u_z * tmp3
    
    return rot_mat_out


if __name__ == "__main__":
    planar_height = 2.0
    planar_width = 3.0
    num_lateral_divisions = 4
    num_longitudinal_divisions = 4
    target_normal_x = 1.0
    target_normal_y = 0.0
    target_normal_z = 0.0
    center_position_x = 0.0
    center_position_y = 0.0
    center_position_z = 0.0
    plane_distance = 0.5
    mesh = build_biplanar_mesh(planar_height, planar_width,
                                          num_lateral_divisions, num_longitudinal_divisions,
                                          target_normal_x, target_normal_y, target_normal_z,
                                          center_position_x, center_position_y, center_position_z,
                                          plane_distance)
    print(mesh['vertices'])
    print(mesh['faces'])

"""
[[ 2.5000000e-01 -1.5000000e+00  1.0000000e+00]
 [ 2.5000000e-01 -7.5000000e-01  1.0000000e+00]
 [ 2.5000000e-01  0.0000000e+00  1.0000000e+00]
 [ 2.5000000e-01  7.5000000e-01  1.0000000e+00]
 [ 2.5000000e-01  1.5000000e+00  1.0000000e+00]
 [ 2.5000000e-01 -1.5000000e+00  5.0000000e-01]
 [ 2.5000000e-01 -7.5000000e-01  5.0000000e-01]
 [ 2.5000000e-01  0.0000000e+00  5.0000000e-01]
 [ 2.5000000e-01  7.5000000e-01  5.0000000e-01]
 [ 2.5000000e-01  1.5000000e+00  5.0000000e-01]
 [ 2.5000000e-01 -1.5000000e+00  1.5308085e-17]
 [ 2.5000000e-01 -7.5000000e-01  1.5308085e-17]
 [ 2.5000000e-01  0.0000000e+00  1.5308085e-17]
 [ 2.5000000e-01  7.5000000e-01  1.5308085e-17]
 [ 2.5000000e-01  1.5000000e+00  1.5308085e-17]
 [ 2.5000000e-01 -1.5000000e+00 -5.0000000e-01]
 [ 2.5000000e-01 -7.5000000e-01 -5.0000000e-01]
 [ 2.5000000e-01  0.0000000e+00 -5.0000000e-01]
 [ 2.5000000e-01  7.5000000e-01 -5.0000000e-01]
 [ 2.5000000e-01  1.5000000e+00 -5.0000000e-01]
 [ 2.5000000e-01 -1.5000000e+00 -1.0000000e+00]
 [ 2.5000000e-01 -7.5000000e-01 -1.0000000e+00]
 [ 2.5000000e-01  0.0000000e+00 -1.0000000e+00]
 [ 2.5000000e-01  7.5000000e-01 -1.0000000e+00]
 [ 2.5000000e-01  1.5000000e+00 -1.0000000e+00]
 [-2.5000000e-01 -1.5000000e+00  1.0000000e+00]
 [-2.5000000e-01 -7.5000000e-01  1.0000000e+00]
 [-2.5000000e-01  0.0000000e+00  1.0000000e+00]
 [-2.5000000e-01  7.5000000e-01  1.0000000e+00]
 [-2.5000000e-01  1.5000000e+00  1.0000000e+00]
 [-2.5000000e-01 -1.5000000e+00  5.0000000e-01]
 [-2.5000000e-01 -7.5000000e-01  5.0000000e-01]
 [-2.5000000e-01  0.0000000e+00  5.0000000e-01]
 [-2.5000000e-01  7.5000000e-01  5.0000000e-01]
 [-2.5000000e-01  1.5000000e+00  5.0000000e-01]
 [-2.5000000e-01 -1.5000000e+00 -1.5308085e-17]
 [-2.5000000e-01 -7.5000000e-01 -1.5308085e-17]
 [-2.5000000e-01  0.0000000e+00 -1.5308085e-17]
 [-2.5000000e-01  7.5000000e-01 -1.5308085e-17]
 [-2.5000000e-01  1.5000000e+00 -1.5308085e-17]
 [-2.5000000e-01 -1.5000000e+00 -5.0000000e-01]
 [-2.5000000e-01 -7.5000000e-01 -5.0000000e-01]
 [-2.5000000e-01  0.0000000e+00 -5.0000000e-01]
 [-2.5000000e-01  7.5000000e-01 -5.0000000e-01]
 [-2.5000000e-01  1.5000000e+00 -5.0000000e-01]
 [-2.5000000e-01 -1.5000000e+00 -1.0000000e+00]
 [-2.5000000e-01 -7.5000000e-01 -1.0000000e+00]
 [-2.5000000e-01  0.0000000e+00 -1.0000000e+00]
 [-2.5000000e-01  7.5000000e-01 -1.0000000e+00]
 [-2.5000000e-01  1.5000000e+00 -1.0000000e+00]]
[[ 1  2 -3]
 [ 6  7  2]
 [11 12  7]
 [16 17 12]
 [ 2  3 -2]
 [ 7  8  3]
 [12 13  8]
 [17 18 13]
 [ 3  4 -1]
 [ 8  9  4]
 [13 14  9]
 [18 19 14]
 [ 4  5  0]
 [ 9 10  5]
 [14 15 10]
 [19 20 15]
 [ 1  1  2]
 [ 6  6  7]
 [11 11 12]
 [16 16 17]
 [ 2  2  3]
 [ 7  7  8]
 [12 12 13]
 [17 17 18]
 [ 3  3  4]
 [ 8  8  9]
 [13 13 14]
 [18 18 19]
 [ 4  4  5]
 [ 9  9 10]
 [14 14 15]
 [19 19 20]
 [26 27 22]
 [31 32 27]
 [36 37 32]
 [41 42 37]
 [27 28 23]
 [32 33 28]
 [37 38 33]
 [42 43 38]
 [28 29 24]
 [33 34 29]
 [38 39 34]
 [43 44 39]
 [29 30 25]
 [34 35 30]
 [39 40 35]
 [44 45 40]
 [26 26 27]
 [31 31 32]
 [36 36 37]
 [41 41 42]
 [27 27 28]
 [32 32 33]
 [37 37 38]
 [42 42 43]
 [28 28 29]
 [33 33 34]
 [38 38 39]
 [43 43 44]
 [29 29 30]
 [34 34 35]
 [39 39 40]
 [44 44 45]]
"""
