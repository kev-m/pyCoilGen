from numpy import dot, sum, ndarray


def nearest_approaches(point: ndarray, starts: ndarray, ends: ndarray):
    """
    Calculate the nearest approach of a point to arrays of line segments.

    NOTE: Uses MATLAB shape conventions

    Args:
        point (ndarray): The point of interest (3D coordinates) (1,3).
        starts (ndarray): The line segment starting positions (m,3)
        ends (ndarray): The line segment ending positions (m,3)

    Returns:
       distances, diffs (ndarray, ndarray): The nearest approach distances and the diffs array for re-use.
    """
    diffs = ends - starts
    vec_targets2 = point - starts
    t1 = sum(vec_targets2 * diffs, axis=0) / sum(diffs * diffs, axis=0)
    return t1, diffs
