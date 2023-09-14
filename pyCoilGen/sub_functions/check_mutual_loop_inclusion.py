import numpy as np
from typing import List


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
    winding_numbers = np.zeros(num_entries)
    for point_ind in range(num_entries):
        A = np.tile(test_poly[:, point_ind, np.newaxis], (1, target_poly.shape[1] - 1))
        B = target_poly[:, 1:]
        C = target_poly[:, :-1]

        vec1 = C - A
        vec2 = B - A

        angle = np.arctan2(vec1[0, :] * vec2[1, :] - vec1[1, :] * vec2[0, :],
                           vec1[0, :] * vec2[0, :] + vec1[1, :] * vec2[1, :])

        winding_numbers[point_ind] = round(abs(np.sum(angle) / (2 * np.pi)))

    inside_flag = np.all(winding_numbers == 1)

    return inside_flag


"""
In this Python implementation, the function check_mutual_loop_inclusion takes two 2D arrays, test_poly and target_poly,
which represent the coordinates of the test polygon and the target polygon, respectively. The function calculates the
winding numbers for each vertex of the test polygon with respect to the target polygon using the winding number
algorithm. If all winding numbers are equal to 1, it indicates that the test polygon is fully enclosed within the 
target polygon, and the function returns True; otherwise, it returns False.
"""
