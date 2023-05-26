import numpy as np

def build_cut_rectangle(loop, center_point, segment_ind, cut_width, cut_height_ratio):
    """
    Build a rectangular cut shape.

    Args:
    - loop (ndarray): Array of loop coordinates.
    - center_point: Center point of the rectangle
    - segment_ind: Index of the segment within the loop
    - cut_width: Width of the cut rectangle
    - cut_height_ratio: Height ratio of the cut rectangle

    Returns:
    - cut_rectangle: Array containing the coordinates of the rectangular cut shape
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




if __name__ == "__main__":
    # Example input values
    loop = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])  # Example loop coordinates
    center_point = np.array([0.5, 0.5])  # Example center point
    segment_ind = 0  # Example segment index
    cut_width = 0.8  # Example cut width
    cut_height_ratio = 0.5  # Example cut height ratio

    # Call the function
    cut_rectangle = build_cut_rectangle(loop, center_point, segment_ind, cut_width, cut_height_ratio)

    # Print the result
    print("Cut Rectangle Points:")
    print(cut_rectangle)

"""
Traceback (most recent call last):
  File "/home/kevin/Dev/CoilGen-Python/sub_functions/build_cut_rectangle.py", line 51, in <module>
    cut_rectangle = build_cut_rectangle(loop, center_point, segment_ind, cut_width, cut_height_ratio)
  File "/home/kevin/Dev/CoilGen-Python/sub_functions/build_cut_rectangle.py", line 31, in build_cut_rectangle
    cut_rectangle = np.array([center_point + longitudinal_vector/2 + orthogonal_vector/2,
ValueError: operands could not be broadcast together with shapes (2,) (3,) 
"""

