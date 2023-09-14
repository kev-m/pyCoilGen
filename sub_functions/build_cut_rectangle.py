import numpy as np


def build_cut_rectangle(loop, center_point, segment_ind, cut_width, cut_height_ratio):
    """
    Build a rectangular cut shape.

    Args:
        loop (ndarray): Array of loop coordinates.
        center_point: Center point of the rectangle
        segment_ind: Index of the segment within the loop
        cut_width: Width of the cut rectangle
        cut_height_ratio: Height ratio of the cut rectangle

    Returns:
        cut_rectangle: Array containing the coordinates of the rectangular cut shape
    """

    cut_points_left = loop[:, segment_ind + 1]
    cut_points_right = loop[:, segment_ind]

    longitudinal_vector = cut_points_left - cut_points_right
    longitudinal_vector = longitudinal_vector / np.linalg.norm(longitudinal_vector)
    orthogonal_vector = np.array([longitudinal_vector[1], -longitudinal_vector[0]])
    orthogonal_vector = orthogonal_vector / np.linalg.norm(orthogonal_vector)

    # Scale the vectors to the targeted width and height
    orthogonal_vector = orthogonal_vector * (cut_width * cut_height_ratio)
    longitudinal_vector = longitudinal_vector * cut_width

    # Create the rectangular points
    cut_rectangle = np.array([center_point + longitudinal_vector/2 + orthogonal_vector/2,
                              center_point + longitudinal_vector/2 - orthogonal_vector/2,
                              center_point - longitudinal_vector/2 - orthogonal_vector/2,
                              center_point - longitudinal_vector/2 + orthogonal_vector/2,
                              center_point + longitudinal_vector/2 + orthogonal_vector/2])

    return cut_rectangle
