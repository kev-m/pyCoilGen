import numpy as np

from typing import List

# Local imports
from sub_functions.data_structures import ContourLine


def find_min_mutual_loop_distance(loop_a: ContourLine, loop_b: ContourLine, only_point_flag: bool):
    """
    Calculate the mutual nearest positions and segment indices between two loops.

    Args:
        loop_a (ContourLine): First contour line.
        loop_b (ContourLine): Second contour line.
        only_point_flag (bool): If True, only consider nearest points and neglect positions between points.

    Returns:
        tuple: A tuple containing the following elements:
            min_dist (float): The minimum distance between the two loops.
            near_points_a (ContourLine): Nearest points on the first contour line.
            min_ind_a (int): Index of the nearest point on the first contour line.
            near_points_b (ContourLine): Nearest points on the second contour line.
            min_ind_b (int): Index of the nearest point on the second contour line.
    """
    if not only_point_flag:
        loop_a_points = loop_a.v.shape[1]  # (3,6)
        loop_b_points = loop_b.v.shape[1]  # (3,6)

        near_dists = np.zeros(loop_b_points-1)
        near_t_b = np.zeros(loop_b_points-1)
        near_points_b_v = np.zeros((3, loop_b_points))

        for test_point_ind in range(loop_b_points - 1):
            x1 = np.tile(loop_b.v[:, test_point_ind, np.newaxis], (1, loop_a_points))     # M: 3x8
            x2 = np.tile(loop_b.v[:, test_point_ind + 1, np.newaxis], (1, loop_a_points))  # M: 3x8
            diff = x2 - x1
            p1 = loop_a.v - x1 # (3,8)
            t1dot = [np.dot(p1[:, i], diff[:, 1]) for i in range(loop_a_points)]
            # M: 0.0063    0.0063    0.0063    0.0063    0.0063    0.0063    0.0063    0.0063
            diff_sum = np.sum((diff) ** 2, axis=0)
            t = t1dot/diff_sum
            t[t < 0] = 0
            t[t > 1] = 1
            # M: 1     1     1     1     1     1     1     1

            # M:
            #    -0.0662   -0.0662   -0.0662   -0.0662   -0.0662   -0.0662   -0.0662   -0.0662
            #     0.4913    0.4913    0.4913    0.4913    0.4913    0.4913    0.4913    0.4913
            #    -0.3000   -0.3000   -0.3000   -0.3000   -0.3000   -0.3000   -0.3000   -0.3000
            all_near_points_b = x1 + (diff) * np.tile(t, (3, 1))
            # M: 1.1538    1.1549    1.1547    1.1594    1.1923    1.1761    1.1493    1.1538
            all_dists = np.linalg.norm(all_near_points_b - loop_a.v, axis=0)
            # 7
            min_ind_b = np.argmin(all_dists)
            near_dists[test_point_ind] = all_dists[min_ind_b]
            near_t_b[test_point_ind] = t[min_ind_b]
            near_points_b_v[:, test_point_ind] = all_near_points_b[:, min_ind_b]

        # M:  1.1493    1.1493    1.1546    1.1482    1.1482
        min_dist = np.min(near_dists)
        # M: 4
        min_ind_b = np.argmin(near_dists)
        # M:
        # v	[0.0662;0.4913;-0.3000]	3x1	double
        # uv	[-1.5477;-0.1931]	2x1	double
        near_points_b_v = near_points_b_v[:, min_ind_b]
        near_points_b_uv = loop_b.uv[:, min_ind_b] + \
            (loop_b.uv[:, min_ind_b + 1] - loop_b.uv[:, min_ind_b]) * near_t_b[min_ind_b]

        # M: x1
        #   -0.0090   -0.0000    0.0091    0.0968   -0.0000   -0.0336   -0.0856
        #   -0.4988   -0.5000   -0.4988   -0.4873   -0.5000   -0.4956   -0.4887
        #    0.2896    0.2889    0.2894    0.3000    0.3592    0.3389    0.3000        
        x1 = loop_a.v[:, :-1]

        # M: x2
        #   -0.0000    0.0091    0.0968   -0.0000   -0.0336   -0.0856   -0.0090
        #   -0.5000   -0.4988   -0.4873   -0.5000   -0.4956   -0.4887   -0.4988
        #    0.2889    0.2894    0.3000    0.3592    0.3389    0.3000    0.2896
        x2 = loop_a.v[:, 1:]

        # M:
        #  0.0090    0.0091    0.0877   -0.0968   -0.0336   -0.0520    0.0766
        # -0.0012    0.0012    0.0115   -0.0127    0.0044    0.0068   -0.0101
        # -0.0007    0.0006    0.0106    0.0592   -0.0202   -0.0389   -0.0104
        diff = x2 - x1 # (3,7)

        # M: near_points_b.v (1x3)
        #  0.0662
        #  0.4913
        # -0.3000
        # M: near_points_b.v - x1 (7x3)
        #  0.0752    0.0662    0.0571   -0.0306    0.0662    0.0998    0.1518
        #  0.9901    0.9913    0.9901    0.9785    0.9913    0.9869    0.9800
        # -0.5896   -0.5889   -0.5894   -0.6000   -0.6592   -0.6389   -0.6000
        #p2 = near_points_b_v - x1
        p2 = np.array([near_points_b_v - x1[:,i] for i in range(x1.shape[1])]).T # (3,7)
        t2dot = [np.dot(p2[:, i], diff[:, 1]) for i in range(diff.shape[1])]
        # t = np.dot(p2, diff) / np.sum((diff) ** 2, axis=0)
        t = t2dot/np.sum((diff) ** 2, axis=0)
        t[t < 0] = 0
        t[t > 1] = 1
        all_near_points_a = x1 + (diff) * np.tile(t, (3, 1))

        # Exception: operands could not be broadcast together with shapes (3,7) (3,) 
        all_dists = np.linalg.norm(all_near_points_a - near_points_b_v, axis=0)
        min_ind_a = np.argmin(all_dists)
        near_points_a_v = all_near_points_a[:, min_ind_a]
        near_points_a_uv = loop_a.uv[:, min_ind_a] + \
            (loop_a.uv[:, min_ind_a + 1] - loop_a.uv[:, min_ind_a]) * t[min_ind_a]
        # M: a (transposed)
        # v	[0.0968;-0.4873;0.3000]	3x1	double
        # uv	[0.9287;-0.1977]	2x1	double
        near_points_a = ContourLine(v=near_points_a_v, uv=near_points_a_uv)
        # M: b (transposed)
        # v	[0.0662;0.4913;-0.3000]	3x1	double
        # uv	[-1.5477;-0.1931]	2x1	double
        near_points_b = ContourLine(v=near_points_b_v, uv=near_points_b_uv)

    else:
        min_test_ind = np.zeros(loop_b_points)
        min_dist_ind = np.zeros(loop_b_points)

        for test_point_ind in range(loop_b_points):
            min_test_ind[test_point_ind] = np.min(np.linalg.norm(
                loop_a.v - np.tile(loop_b.v[:, test_point_ind], (1, loop_a_points)), axis=0))
            min_dist_ind[test_point_ind] = np.argmin(np.linalg.norm(loop_a.v - loop_b.v[:, test_point_ind], axis=0))

        min_dist = np.min(min_test_ind)
        min_ind_b = np.argmin(min_test_ind)
        min_ind_a = min_dist_ind[min_ind_b]

        near_points_a_v = loop_a.v[:, min_ind_a]
        near_points_a_uv = loop_a.uv[:, min_ind_a]

        near_points_b_v = loop_b.v[:, min_ind_b]
        near_points_b_uv = loop_b.uv[:, min_ind_b]

        near_points_a = ContourLine(v=near_points_a_v, uv=near_points_a_uv)
        near_points_b = ContourLine(v=near_points_b_v, uv=near_points_b_uv)

    return min_dist, near_points_a, min_ind_a, near_points_b, min_ind_b
