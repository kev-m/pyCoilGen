import numpy as np


def calc_plane_line_intersection(n, V0, P0, P1):
    """
    Calculate the intersection point between a plane and a line segment.

    Args:
        n (numpy.ndarray): Plane normal vector.
        V0 (numpy.ndarray): Point on the plane.
        P0 (numpy.ndarray): Start point of the line segment.
        P1 (numpy.ndarray): End point of the line segment.

    Returns:
        numpy.ndarray: Intersection point.
        int: Intersection check code (0 - no intersection, 1 - successful intersection,
             2 - line segment lies in plane, 3 - intersection lies outside the segment).
    """
    I = np.zeros(3)
    u = P1 - P0
    w = P0 - V0
    D = np.dot(n, u)
    N = -np.dot(n, w)
    check = 0

    if np.abs(D) < 1e-7:  # The segment is parallel to the plane
        if N == 0:  # The segment lies in the plane
            check = 2
            return I, check
        else:
            check = 0  # No intersection
            return I, check

    # Compute the intersection parameter
    sI = N / D
    I = P0 + sI * u

    if sI < 0 or sI > 1:
        check = 3  # The intersection point lies outside the segment
    else:
        check = 1

    return I, check
