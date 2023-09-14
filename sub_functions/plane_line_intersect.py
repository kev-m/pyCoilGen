import numpy as np

# Logging
import logging

log = logging.getLogger(__name__)


def plane_line_intersect(plane_normal: np.ndarray, plane_pos: np.ndarray, point_0: np.ndarray, point_1: np.ndarray):
    """
    Compute the intersection point between a plane and a line segment defined by two points.

    Args:
        plane_normal (np.ndarray): The normal vector of the plane.
        plane_pos (np.ndarray): A point on the plane.
        point_0 (np.ndarray): The first point of the line segment.
        point_1 (np.ndarray): The second point of the line segment.

    Returns:
        intersec_point (np.ndarray): The intersection point.
        cut_flag (int): A flag indicating the type of intersection.
            - 1: The intersection point is within the line segment.
            - 2: The line segment lies within the plane.
            - 3: The intersection point is outside the line segment.

    """
    intersec_point = np.zeros(3)
    line_vec = point_1 - point_0
    diff_vec = point_0 - plane_pos
    D = np.dot(plane_normal, line_vec)
    N = -np.dot(plane_normal, diff_vec)
    cut_flag = 0

    if abs(D) < 10**-7:  # The segment is parallel to the plane
        if N == 0:  # The segment lies in the plane
            cut_flag = 2
            return intersec_point, cut_flag
        else:
            cut_flag = 0  # no intersection
            return intersec_point, cut_flag

    # Compute the intersection parameter
    sI = N / D
    intersec_point = point_0 + sI * line_vec

    if (sI < -0.0000001 or sI > 1.0000001):
        cut_flag = 3  # The intersection point lies outside the segment, so there is no intersection
    else:
        cut_flag = 1

    return intersec_point, cut_flag


"""
In this Python version, the function plane_line_intersect takes in the required parameters as NumPy arrays and returns
the intersection point and a cut flag indicating the type of intersection. The rest of the code remains similar to the
MATLAB implementation, but with appropriate NumPy syntax for array operations and comparisons.
"""
