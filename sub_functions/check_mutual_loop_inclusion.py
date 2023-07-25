import numpy as np
from typing import List

def check_mutual_loop_inclusion(test_poly: np.ndarray, target_poly: np.ndarray) -> bool:
    """
    Check if the test polygon lies fully enclosed within the second polygon.
    This check is done with the winding number algorithm test for each vertex towards the second polygon.

    Args:
        test_poly (np.ndarray): The polygon to test for inclusion.
        target_poly (np.ndarray): The target polygon.

    Returns:
        bool: True if the test polygon is fully enclosed within the target polygon, False otherwise.
    """
    winding_numbers = np.zeros(test_poly.shape[1])

    for point_ind in range(test_poly.shape[1]):
        A = np.tile(test_poly[:, point_ind, np.newaxis], (1, target_poly.shape[1] - 1))
        B = target_poly[:, 1:]
        C = target_poly[:, :-1]

        # ValueError: operands could not be broadcast together with shapes (2,7) (1,14) 
        vec1 = C - A
        vec2 = B - A

        angle = np.arctan2(vec1[0, :] * vec2[1, :] - vec1[1, :] * vec2[0, :], vec1[0, :] * vec2[0, :] + vec1[1, :] * vec2[1, :])

        winding_numbers[point_ind] = round(abs(np.sum(angle) / (2 * np.pi)))

    inside_flag = np.all(winding_numbers == 1)

    return inside_flag
