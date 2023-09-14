import numpy as np


def calc_3d_rotation_matrix_by_vector(rot_vec, rot_angle):
    """
    Calculate the 3D rotation matrix around a rotation axis given by a vector and an angle.

    Args:
        rot_vec (numpy.ndarray): Rotation axis vector.
        rot_angle (float): Rotation angle in radians.

    Returns:
        numpy.ndarray: 3x3 rotation matrix.
    """
    rot_vec = rot_vec / np.linalg.norm(rot_vec)  # Normalize rotation vector

    u_x = rot_vec[0]
    u_y = rot_vec[1]
    u_z = rot_vec[2]

    tmp1 = np.sin(rot_angle)
    tmp2 = np.cos(rot_angle)
    tmp3 = 1 - np.cos(rot_angle)

    rot_mat_out = np.zeros((3, 3))

    rot_mat_out[0, 0] = tmp2 + u_x * u_x * tmp3
    rot_mat_out[1, 0] = u_x * u_y * tmp3 - u_z * tmp1
    rot_mat_out[2, 0] = u_x * u_z * tmp3 + u_y * tmp1

    rot_mat_out[0, 1] = u_y * u_x * tmp3 + u_z * tmp1
    rot_mat_out[1, 1] = tmp2 + u_y * u_y * tmp3
    rot_mat_out[2, 1] = u_y * u_z * tmp3 - u_x * tmp1

    rot_mat_out[0, 2] = u_z * u_x * tmp3 - u_y * tmp1
    rot_mat_out[1, 2] = u_z * u_y * tmp3 + u_x * tmp1
    rot_mat_out[2, 2] = tmp2 + u_z * u_z * tmp3

    return rot_mat_out
