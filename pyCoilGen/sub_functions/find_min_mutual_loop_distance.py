import numpy as np

from typing import List

# Local imports
from .data_structures import ContourLine
from pyCoilGen.helpers.common import nearest_approaches


def find_min_mutual_loop_distance(loop_a: ContourLine, loop_b: ContourLine, only_point_flag: bool, only_min_dist=False):
    """
    Calculate the mutual nearest positions and segment indices between two loops.

    Args:
        loop_a (ContourLine): First contour line.
        loop_b (ContourLine): Second contour line.
        only_point_flag (bool): If True, only consider nearest points and neglect positions between points.
        only_min_dist (bool): If True, only calculate min_dist and skip the others.

    Returns:
        tuple: A tuple containing the following elements:
            min_dist (float): The minimum distance between the two loops.
            near_points_a (ContourLine): Nearest points on the first contour line.
            min_ind_a (int): Index of the nearest point on the first contour line.
            near_points_b (ContourLine): Nearest points on the second contour line.
            min_ind_b (int): Index of the nearest point on the second contour line.
    """
    if not only_point_flag:
        loop_a_points = loop_a.v.shape[1]
        loop_b_points = loop_b.v.shape[1]

        near_dists = np.zeros(loop_b_points-1)
        near_t_b = np.zeros(loop_b_points-1)
        near_points_b_v = np.zeros((3, loop_b_points))

        for test_point_ind in range(loop_b_points - 1):
            x1 = np.tile(loop_b.v[:, test_point_ind, np.newaxis], (1, loop_a_points))
            x2 = np.tile(loop_b.v[:, test_point_ind + 1, np.newaxis], (1, loop_a_points))
            t, diff = nearest_approaches(loop_a.v, x1, x2)
            t[t < 0] = 0
            t[t > 1] = 1
            all_near_points_b = x1 + (diff) * np.tile(t, (3, 1))
            all_dists = np.linalg.norm(all_near_points_b - loop_a.v, axis=0)
            min_ind_b = np.argmin(all_dists)
            near_dists[test_point_ind] = all_dists[min_ind_b]
            near_t_b[test_point_ind] = t[min_ind_b]
            near_points_b_v[:, test_point_ind] = all_near_points_b[:, min_ind_b]

        min_dist = np.min(near_dists)
        if only_min_dist:
            return min_dist
        min_ind_b = np.argmin(near_dists)
        near_points_b_v = near_points_b_v[:, min_ind_b]
        near_points_b_uv = loop_b.uv[:, min_ind_b] + \
            (loop_b.uv[:, min_ind_b + 1] - loop_b.uv[:, min_ind_b]) * near_t_b[min_ind_b]
        x1 = loop_a.v[:, :-1]
        x2 = loop_a.v[:, 1:]
        near_points = np.tile(near_points_b_v[:, np.newaxis], (1, loop_a_points-1))
        t, diff = nearest_approaches(near_points, x1, x2)
        t[t < 0] = 0
        t[t > 1] = 1
        all_near_points_a = x1 + (diff) * np.tile(t, (3, 1))
        # all_dists = np.linalg.norm(all_near_points_a - near_points_b_v, axis=0)
        # MATLAB performs (vector - matrix) by decomposition.
        delta_anpa_npbv = np.array(
            [all_near_points_a[:, i] - near_points_b_v for i in range(all_near_points_a.shape[1])]).T
        all_dists = np.linalg.norm(delta_anpa_npbv, axis=0)
        min_ind_a = np.argmin(all_dists)
        near_points_a_v = all_near_points_a[:, min_ind_a]
        near_points_a_uv = loop_a.uv[:, min_ind_a] + \
            (loop_a.uv[:, min_ind_a + 1] - loop_a.uv[:, min_ind_a]) * t[min_ind_a]
        near_points_a = ContourLine(v=near_points_a_v, uv=near_points_a_uv)
        near_points_b = ContourLine(v=near_points_b_v, uv=near_points_b_uv)

    else:
        min_test_ind = np.zeros(loop_b_points)
        min_dist_ind = np.zeros(loop_b_points)

        for test_point_ind in range(loop_b_points):
            min_test_ind[test_point_ind] = np.min(np.linalg.norm(
                loop_a.v - np.tile(loop_b.v[:, test_point_ind], (1, loop_a_points)), axis=0))
            min_dist_ind[test_point_ind] = np.argmin(np.linalg.norm(loop_a.v - loop_b.v[:, test_point_ind], axis=0))

        min_dist = np.min(min_test_ind)
        if only_min_dist:
            return min_dist
        min_ind_b = np.argmin(min_test_ind)
        min_ind_a = min_dist_ind[min_ind_b]

        near_points_a_v = loop_a.v[:, min_ind_a]
        near_points_a_uv = loop_a.uv[:, min_ind_a]

        near_points_b_v = loop_b.v[:, min_ind_b]
        near_points_b_uv = loop_b.uv[:, min_ind_b]

        near_points_a = ContourLine(v=near_points_a_v, uv=near_points_a_uv)
        near_points_b = ContourLine(v=near_points_b_v, uv=near_points_b_uv)

    return min_dist, near_points_a, min_ind_a, near_points_b, min_ind_b
