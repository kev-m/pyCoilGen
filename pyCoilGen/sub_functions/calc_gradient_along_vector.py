import numpy as np


def calc_gradient_along_vector(field, field_coords, target_endcoding_function):
    """
    Calculate the mean gradient in a given direction and angle.

    Args:
        field (array-like): Field values.
        field_coords (array-like): Field coordinates.
        target_endcoding_function (str): Target encoding function for coordinate transformation.

    Returns:
        mean_gradient_strength (float): Mean gradient strength.
        gradient_out (array): Gradient values.
    """
    def my_fun(x, y, z):
        # Define the target encoding function
        return eval(target_endcoding_function)

    norm_dir_x = my_fun(1, 0, 0)
    norm_dir_y = my_fun(0, 1, 0)
    norm_dir_z = my_fun(0, 0, 1)

    target_direction = np.array([0, 0, 1])
    gradient_direction = np.array([norm_dir_x, norm_dir_y, norm_dir_z])
    gradient_direction /= np.linalg.norm(gradient_direction)

    if np.linalg.norm(np.cross(gradient_direction, target_direction)) != 0:
        rot_vector = np.cross(gradient_direction, target_direction) / \
            np.linalg.norm(np.cross(gradient_direction, target_direction))
        rot_angle = np.arcsin(np.linalg.norm(np.cross(gradient_direction, target_direction)) /
                              (np.linalg.norm(target_direction) * np.linalg.norm(gradient_direction)))
    else:
        rot_vector = np.array([1, 0, 0])
        rot_angle = 0

    rot_mat_out = calc_3d_rotation_matrix(rot_vector.reshape(-1, 1), rot_angle)
    rotated_field_coords = np.dot(rot_mat_out, (field_coords - np.mean(field_coords, axis=1).reshape(-1, 1)))
    gradient_out = field[2, :] / rotated_field_coords[2, :]
    gradient_out[np.abs(rotated_field_coords[2, :]) < 1e-6] = np.nan
    mean_gradient_strength = np.nanmean(gradient_out)

    return mean_gradient_strength, gradient_out


def calc_3d_rotation_matrix(rot_vec, rot_angle):
    """
    Calculate the 3D rotation matrix around a rotation axis given by a vector and an angle.

    Args:
        rot_vec (array-like): Rotation vector.
        rot_angle (float): Rotation angle.

    Returns:
        rot_mat_out (array): 3D rotation matrix.

    Raises:
        None
    """
    rot_vec = rot_vec / np.linalg.norm(rot_vec)  # normalize rotation vector
    u_x = rot_vec[0]
    u_y = rot_vec[1]
    u_z = rot_vec[2]
    tmp1 = np.sin(rot_angle)
    tmp2 = np.cos(rot_angle)
    tmp3 = 1 - np.cos(rot_angle)
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
