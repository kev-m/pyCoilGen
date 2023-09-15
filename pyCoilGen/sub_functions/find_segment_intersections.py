import numpy as np
from typing import List

# local imports
from .data_structures import DataStructure


def find_segment_intersections(loop: np.ndarray, test_polygon: np.ndarray):
    """
    Find intersection points between a loop and a polygon (2D).

    Args:
        loop (np.ndarray): Array representing the loop's 2D coordinates (shape: (2, num_vertices)).
        test_polygon (np.ndarray): Array representing the polygon's 2D coordinates (shape: (2, num_vertices)).

    Returns:
        List[dict]: A list of dictionaries, each containing 'segment_inds' and 'uv' keys.
                    'segment_inds' holds indices of the segments where intersections occur.
                    'uv' holds the intersection points as a 2xN array.
                    Values contain np.nan if there is no intersection.
    """

    intersection_points = []
    num_segments = test_polygon.shape[1] - 1

    for seg_ind in range(num_segments):
        x1 = np.tile(test_polygon[0, seg_ind], loop.shape[1] - 1)
        x2 = np.tile(test_polygon[0, seg_ind + 1], loop.shape[1] - 1)
        x3 = loop[0, :-1]
        x4 = loop[0, 1:]

        y1 = np.tile(test_polygon[1, seg_ind], loop.shape[1] - 1)
        y2 = np.tile(test_polygon[1, seg_ind + 1], loop.shape[1] - 1)
        y3 = loop[1, :-1]
        y4 = loop[1, 1:]

        d1x = x2 - x1
        d1y = y2 - y1
        d2x = x4 - x3
        d2y = y4 - y3

        s = (-d1y * (x1 - x3) + d1x * (y1 - y3)) / (-d2x * d1y + d1x * d2y)
        t = (d2x * (y1 - y3) - d2y * (x1 - x3)) / (-d2x * d1y + d1x * d2y)

        intersection_segment_inds = np.where((s >= 0) & (s <= 1) & (t >= 0) & (t <= 1))[0]

        if len(intersection_segment_inds) > 0:
            x_out = x1[intersection_segment_inds] + (t[intersection_segment_inds] * d1x[intersection_segment_inds])
            y_out = y1[intersection_segment_inds] + (t[intersection_segment_inds] * d1y[intersection_segment_inds])

            new_intersection = DataStructure(segment_inds=intersection_segment_inds, uv=np.vstack((x_out, y_out)))
        else:
            new_intersection = DataStructure(segment_inds=np.nan, uv=np.vstack((np.nan, np.nan)))
        intersection_points.append(new_intersection)

    return intersection_points


"""
In this Python implementation, the function find_segment_intersections takes two 2D arrays, loop and test_polygon,
which represent the coordinates of a loop and a test polygon, respectively. The function then finds the intersection
points between each segment of the test polygon and the loop. The result is returned as a list of dictionaries, where
each dictionary contains the indices of the segments with intersections and the corresponding 2D intersection points.

Please note that the implementation might not directly translate to NumPy broadcasting due to the nature of the loop in
the original Matlab code. Instead, the use of NumPy functions like np.tile, np.where, and array slicing helps achieve 
the same functionality.
"""
