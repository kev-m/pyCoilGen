import numpy as np


def calc_local_opening_gab(loop, point_1, point_2, opening_gab):
    """
    Calculate the local opening gap based on the given inputs.

    Args:
        loop (object): Loop object.
        point_1 (int): Index of the first point.
        point_2 (int): Index of the second point.
        opening_gab (float): Opening gap value.

    Returns:
        local_opening_gab (float): Local opening gap value.
    """
    if point_2 is not None:  # Two points are specified
        uv_distance = np.linalg.norm(loop.uv[:, point_2] - loop.uv[:, point_1])
        v_distance = np.linalg.norm(loop.v[:, point_2] - loop.v[:, point_1])
        local_opening_gab = opening_gab * uv_distance / v_distance
    else:  # Only one point is specified, find the other to build the direction
        min_ind_2 = np.argmin(np.linalg.norm(loop.uv - point_1, axis=1))
        min_ind_1 = min_ind_2 + 2
        if min_ind_1 < 0:
            min_ind_1 += loop.uv.shape[1]
        if min_ind_1 > loop.uv.shape[1]:
            min_ind_1 -= loop.uv.shape[1]
        uv_distance = np.linalg.norm(loop.uv[:, min_ind_1] + (loop.uv[:, min_ind_2] - loop.uv[:, min_ind_1]) / 1000)
        v_distance = np.linalg.norm(loop.v[:, min_ind_1] + (loop.v[:, min_ind_2] - loop.v[:, min_ind_1]) / 1000)
        local_opening_gab = opening_gab * uv_distance / v_distance

    return local_opening_gab
