import numpy as np

from typing import List

# Logging
import logging

# Local imports
from sub_functions.data_structures import Shape3D

log = logging.getLogger(__name__)

def remove_points_from_loop(loop : Shape3D, points_to_remove: np.ndarray, boundary_threshold: int):
    """
    Remove points with identical uv coordinates from a loop, even with some additional more points around.

    Parameters:
        loop (Shape3D): The loop data containing 'uv' and 'v'.
        points_to_remove (np.ndarray): The points to be removed (shape: (2, num_points)).
        boundary_threshold (int): The number of additional points around each identical point to be removed.

    Returns:
        Shape3D: The updated loop data after removing the specified points.

    """

    rep_u = np.tile(loop.uv[0, :], (points_to_remove.shape[1], 1))
    rep_v = np.tile(loop.uv[1, :], (points_to_remove.shape[1], 1))

    rep_u2 = np.tile(points_to_remove[0, :], (loop.uv.shape[1], 1))
    rep_v2 = np.tile(points_to_remove[1, :], (loop.uv.shape[1], 1))

    identical_point_inds = np.where((rep_u == rep_u2.T) & (rep_v == rep_v2.T))
    identical_point_inds = identical_point_inds[0]

    below_inds = np.arange(min(identical_point_inds) - boundary_threshold, min(identical_point_inds))
    below_inds[below_inds < 0] = below_inds[below_inds < 0] + loop.uv.shape[1]

    abow_inds = np.arange(max(identical_point_inds), max(identical_point_inds) + boundary_threshold + 1)
    abow_inds[abow_inds >= loop.uv.shape[1]] = abow_inds[abow_inds >= loop.uv.shape[1]] - loop.uv.shape[1]

    # Add more points as a "boundary threshold"
    full_point_inds_to_remove = np.concatenate((below_inds, identical_point_inds, abow_inds))

    # Use setdiff to find the indices to keep
    inds_out, _ = np.setdiff1d(np.arange(loop.uv.shape[1]), full_point_inds_to_remove, return_indices=True)

    # Extract the loop with the remaining points
    loop_out_uv = loop.uv[:, inds_out]
    loop_out_v = loop.v[:, inds_out]

    return loop_out_uv, loop_out_v

"""
Note: The code assumes that the data structure Shape3D is already defined elsewhere or imported. Additionally, the
function returns the updated loop data (loop_out_uv and loop_out_v) rather than modifying the original data in-place,
as it is not possible to modify the input loop data due to the immutability of NumPy arrays. The logic and 
functionality of the MATLAB function have been retained in the Python equivalent.
"""