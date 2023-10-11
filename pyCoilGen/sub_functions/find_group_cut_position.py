import numpy as np

from typing import List

# Logging
import logging

# Local imports
from .plane_line_intersect import plane_line_intersect
from .calc_mean_loop_normal import calc_mean_loop_normal
from .data_structures import TopoGroup, Mesh, CutPoint, CutPosition

log = logging.getLogger(__name__)


def find_group_cut_position(loop_group: TopoGroup, group_center: np.ndarray, mesh: Mesh, b_0_direction: np.ndarray, cut_plane_definition) -> List[CutPosition]:
    """
    Define the cut plane orientation for the group.
    Find the opening shapes and cut points, separated into higher and lower cut points.
    Check whether a forced cut selection is given.
    Generate circular cut shape and delete overlapping points.

    Args:
        loop_group (LoopGroup): The loop group containing loops for cut position calculation.
        group_center (np.ndarray): Center point of the loop group.
        mesh (Mesh): The mesh object for mesh-related operations.
        b_0_direction (np.ndarray): Direction of B0 field.
        cut_plane_definition (str): Definition of the cut plane orientation.

    Returns:
        cut_positions(List[CutPosition]): List of cut positions for each loop in the loop group.
    """
    # M: loop_normal	[0.0061;1;-0]	3x1	double [~y]
    loop_normal = calc_mean_loop_normal(loop_group, mesh)

    # Test if loop normal and b_0_direction are not independent enough
    if np.linalg.norm(np.cross(b_0_direction / np.linalg.norm(b_0_direction), loop_normal)) < 0.01:
        if cut_plane_definition == 'nearest':
            min_ind = np.argmin(np.linalg.norm(loop_group.loops[0].v - group_center.reshape(3, 1), axis=0))
            alternative_cut_direction = loop_group.loops[0].v[:, min_ind] - group_center
            alternative_cut_direction = alternative_cut_direction / np.linalg.norm(alternative_cut_direction)
            cut_plane_direction = np.cross(alternative_cut_direction, loop_normal)
            cut_plane_direction = cut_plane_direction / np.linalg.norm(cut_plane_direction)
        else:
            alternative_center_normal = b_0_direction
            coord_vec_min_ind = np.argmin([np.dot(b_0_direction, [1, 0, 0]), np.dot(
                b_0_direction, [0, 1, 0]), np.dot(b_0_direction, [0, 0, 1])])

            if coord_vec_min_ind == 0:
                alternative_center_normal = np.cross(b_0_direction, [1, 0, 0])
            elif coord_vec_min_ind == 1:
                alternative_center_normal = np.cross(b_0_direction, [0, 1, 0])
            elif coord_vec_min_ind == 2:
                alternative_center_normal = np.cross(b_0_direction, [0, 0, 1])

            alternative_center_normal = alternative_center_normal / np.linalg.norm(alternative_center_normal)
            cut_plane_direction = np.cross(b_0_direction, alternative_center_normal)
            cut_plane_direction = cut_plane_direction / np.linalg.norm(cut_plane_direction)
    else:
        cut_plane_direction = np.cross(b_0_direction, loop_normal)
        cut_plane_direction = cut_plane_direction / np.linalg.norm(cut_plane_direction)

    # Calculate the cut shapes
    cut_positions = []
    for loop_ind in range(len(loop_group.loops)):
        cut_position = CutPosition(cut_point=CutPoint(segment_ind=[]), high_cut=CutPoint(), low_cut=CutPoint())
        cut_positions.append(cut_position)

        for point_ind in range(1, loop_group.loops[loop_ind].v.shape[1]):
            point_a = loop_group.loops[loop_ind].v[:, point_ind - 1]
            point_b = loop_group.loops[loop_ind].v[:, point_ind]
            cut_p, cut_flag = plane_line_intersect(cut_plane_direction, group_center, point_a, point_b)

            if cut_flag == 1:  # Test if there is a cut between the line points
                cut_position.cut_point.add_v(cut_p)
                cut_position.cut_point.segment_ind.append(point_ind - 1)
                # Build the corresponding uv point
                cut_point_ratio = np.linalg.norm(point_a - cut_p) / np.linalg.norm(point_a - point_b)
                point_a_uv = loop_group.loops[loop_ind].uv[:, point_ind - 1]
                point_b_uv = loop_group.loops[loop_ind].uv[:, point_ind]
                cut_point_uv = point_a_uv + (point_b_uv - point_a_uv) * cut_point_ratio
                cut_position.cut_point.add_uv(cut_point_uv)

        # NOTE: cut_position.cut_point.v has shape (n,3) i.e. Python not MATLAB convention

        try:
            # Delete repeating degenerate cut points
            arr_abs = np.abs(np.diff(cut_position.cut_point.uv, axis=1))  # Python convention
            repeat_inds = np.where(arr_abs < 1e-10)[0]
            is_repeating_cutpoint = np.any(repeat_inds)
            if is_repeating_cutpoint:
                cut_position.cut_point.v = np.delete(cut_position.cut_point.v, np.where(repeat_inds), axis=1)
                cut_position.cut_point.uv = np.delete(cut_position.cut_point.uv, np.where(repeat_inds), axis=1)
                cut_position.cut_point.segment_ind = np.delete(
                    cut_position.cut_point.segment_ind, np.where(repeat_inds))
        except ValueError as e:
            log.debug("Exception: %s", e)

        # https://github.com/kev-m/pyCoilGen/issues/60
        # Take care of the exception that the cut plane has no intersection with the loop
        if cut_position.cut_point.v is None:
            max_ind = np.argmax(np.dot(loop_group.loops[loop_ind].v.T, cut_plane_direction))
            min_ind = np.argmin(np.dot(loop_group.loops[loop_ind].v.T, cut_plane_direction))

            # To get (n,3)
            cut_position.cut_point.v = np.vstack(
                (loop_group.loops[loop_ind].v[:, min_ind], loop_group.loops[loop_ind].v[:, max_ind]))
            # To get (n,2)
            cut_position.cut_point.uv = np.vstack(
                (loop_group.loops[loop_ind].uv[:, min_ind], loop_group.loops[loop_ind].uv[:, max_ind]))

            if max_ind != loop_group.loops[loop_ind].v.shape[1] - 1:
                cut_position.cut_point.segment_ind = [min_ind, max_ind - 1]
            else:
                cut_position.cut_point.segment_ind = [min_ind, max_ind]

        # Separated into higher and lower cut points:
        # First: use the 2D representation of the loop

        # Since cut points are always alternating between cut points "in" and "out",
        # (seen from the orientation of the cut plane);
        if loop_ind == 0:
            # Select the first pair of the cuts, the cuts with the largest distance to group center
            cut_sort_ind = np.argsort(np.linalg.norm(cut_position.cut_point.v - group_center, axis=1))
            first_pair = [cut_sort_ind[0], cut_sort_ind[1]]

            # Find the direction for which high and low cuts are separated
            cut_direction = cut_position.cut_point.v[first_pair[1], :] - cut_position.cut_point.v[first_pair[0], :]
            cut_direction = cut_direction / np.linalg.norm(cut_direction)
            if np.sum(b_0_direction * cut_direction) < 0:
                cut_direction = cut_direction * (-1)

            # Project the coordinates of the cut pairs
            arr_sum = np.sum(cut_position.cut_point.v[first_pair, :] * cut_direction, axis=1)  # Result (2,)
            min_ind = np.argmin(arr_sum)
            high_ind = first_pair[min_ind]
            low_ind = first_pair[1 - min_ind]

            high_cut_primer = cut_position.cut_point.v[high_ind]
            low_cut_primer = cut_position.cut_point.v[low_ind]

            cut_position.high_cut.segment_ind = cut_position.cut_point.segment_ind[high_ind]
            cut_position.high_cut.v = cut_position.cut_point.v[high_ind]
            cut_position.high_cut.uv = cut_position.cut_point.uv[high_ind]
            cut_position.low_cut.segment_ind = cut_position.cut_point.segment_ind[low_ind]
            cut_position.low_cut.v = cut_position.cut_point.v[low_ind]
            cut_position.low_cut.uv = cut_position.cut_point.uv[low_ind]
            # center_first_cut = (cut_position.high_cut.v + cut_position.low_cut.v) / 2

        else:  # Now for the following inner loops
            # Choose the following cut pairs regarding their distance to the high and low cut of the first loop
            high_dists = np.linalg.norm(cut_position.cut_point.v - high_cut_primer, axis=1)
            low_dists = np.linalg.norm(cut_position.cut_point.v - low_cut_primer, axis=1)

            high_min_ind = np.argmin(high_dists)
            low_min_ind = np.argmin(low_dists)

            cut_position.high_cut.segment_ind = cut_position.cut_point.segment_ind[high_min_ind]
            cut_position.high_cut.v = cut_position.cut_point.v[high_min_ind]
            cut_position.high_cut.uv = cut_position.cut_point.uv[high_min_ind]
            cut_position.low_cut.segment_ind = cut_position.cut_point.segment_ind[low_min_ind]
            cut_position.low_cut.v = cut_position.cut_point.v[low_min_ind]
            cut_position.low_cut.uv = cut_position.cut_point.uv[low_min_ind]

    return cut_positions
