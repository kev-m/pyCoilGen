import numpy as np


def build_cut_circle(center_point, cut_width):
    """
    Build a rectangular cut shape in the form of a circular opening.

    Args:
        center_point (ndarray): Array containing the x and y coordinates of the center point.
        cut_width (float): Width of the cut.

    Returns:
        cut_circle (ndarray): Array containing the x and y coordinates of the circular cut shape.
    """

    circular_resolution = 10

    # Build circular cut shapes
    opening_angles = np.linspace(0, 2*np.pi, circular_resolution)
    opening_circle = np.column_stack((np.sin(opening_angles), np.cos(opening_angles)))

    # Create a circular opening cut
    cut_circle = opening_circle * (cut_width / 2) + np.tile(center_point, (circular_resolution, 1))

    return cut_circle
