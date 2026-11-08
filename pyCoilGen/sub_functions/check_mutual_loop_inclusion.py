import numpy as np
from typing import List

# The approximate number of (test vertex, polygon edge) pairs that are evaluated at once
_PAIRS_PER_BLOCK = 100_000


def check_mutual_loop_inclusion(test_poly: np.ndarray, target_poly: np.ndarray) -> bool:
    """
    Check if the test polygon lies fully enclosed within the second polygon.

    This check is done with the winding number algorithm test for each vertex towards the second polygon.

    Args:
        test_poly (np.ndarray): Array representing the test polygon's 2D coordinates (shape: (2, num_vertices)).
        target_poly (np.ndarray): Array representing the target polygon's 2D coordinates (shape: (2, num_vertices)).

    Returns:
        bool: True if the test polygon is fully enclosed within the target polygon, False otherwise.
    """

    if len(test_poly.shape) == 1:
        test_poly = test_poly.reshape((2, 1))
    num_entries = test_poly.shape[1]

    # The edges of the target polygon: from C to B
    edge_start = target_poly[:, :-1]
    edge_end = target_poly[:, 1:]
    num_edges = edge_start.shape[1]

    # Evaluate many test vertices at once, in blocks that keep the temporary arrays small. The winding number of each
    # vertex is independent of the others, so the first vertex that is not enclosed ends the test: the first block is
    # a single vertex, as most loops that are compared are not nested.
    points_per_block = max(1, _PAIRS_PER_BLOCK // max(1, num_edges))
    first = 0
    block_size = 1
    while first < num_entries:
        points = test_poly[:, first:first + block_size]
        vec1_x = edge_start[0][np.newaxis, :] - points[0][:, np.newaxis]  # (points, edges)
        vec1_y = edge_start[1][np.newaxis, :] - points[1][:, np.newaxis]
        vec2_x = edge_end[0][np.newaxis, :] - points[0][:, np.newaxis]
        vec2_y = edge_end[1][np.newaxis, :] - points[1][:, np.newaxis]

        angle = np.arctan2(vec1_x * vec2_y - vec1_y * vec2_x, vec1_x * vec2_x + vec1_y * vec2_y)

        if not np.all(np.round(np.abs(np.sum(angle, axis=1) / (2 * np.pi))) == 1):
            return False
        first += block_size
        block_size = points_per_block

    inside_flag = True

    return inside_flag


"""
In this Python implementation, the function check_mutual_loop_inclusion takes two 2D arrays, test_poly and target_poly,
which represent the coordinates of the test polygon and the target polygon, respectively. The function calculates the
winding numbers for each vertex of the test polygon with respect to the target polygon using the winding number
algorithm. If all winding numbers are equal to 1, it indicates that the test polygon is fully enclosed within the 
target polygon, and the function returns True; otherwise, it returns False.
"""
