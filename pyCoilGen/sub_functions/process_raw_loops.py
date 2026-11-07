# System imports
import numpy as np
from typing import List

# Logging
import logging

# Local imports
from .constants import *
from .data_structures import CoilPart, WirePart, TargetField
from .smooth_track_by_folding import smooth_track_by_folding
from .wire_kernels import biot_savart_field

log = logging.getLogger(__name__)


def process_raw_loops(coil_parts: List[CoilPart], input_args, target_field: TargetField):
    """
    Process raw loops in the coil mesh.

    Initialises the following properties of a CoilPart:
        - combined_loop_length
        - combined_loop_field
        - loop_significance (double)
        - field_by_loops (3, # , num contours)

    Depends on the following properties of the CoilParts:
        - contour_lines

    Depends on the following input_args:
        - smooth_factor
        - min_loop_significance

    Updates the following properties of a CoilPart:
        - contour_lines

    Args:
        coil_parts (List[CoilPart]): A list of CoilPart structures.
        input: The input parameters.
        target_field: The target field associated with the CoilSolution.

    Returns:
        List[CoilPart]: The updated list of CoilPart structures after processing the raw loops.
    """

    # Smooth the contours if smooth_flag is True
    if input_args.smooth_factor > 1:
        for coil_part in coil_parts:
            for index, contour in enumerate(coil_part.contour_lines):
                smoothed = smooth_track_by_folding(contour.uv, input_args.smooth_factor)
                if get_level() > DEBUG_BASIC:
                    log.debug(" smoothed difference: %f", np.sum(np.abs(contour.uv - smoothed)))
                contour.uv = smoothed

    # Generate the curved coordinates
    for coil_part in coil_parts:
        for p_contour in coil_part.contour_lines:
            p_contour.v, p_contour.uv = coil_part.coil_mesh.uv_to_xyz(p_contour.uv, coil_part.coil_mesh.uv)

    # Evaluate loop significance and remove loops that do not contribute enough to the target field
    coil_parts = evaluate_loop_significance(coil_parts, target_field)
    for coil_part in coil_parts:
        loops_to_delete = coil_part.loop_significance < input_args.min_loop_significance
        if np.any(loops_to_delete):
            coil_part.contour_lines = [loop for i, loop in enumerate(coil_part.contour_lines) if not loops_to_delete[i]]

    # Remove loops that are smaller than the interconnection cut: they cannot be opened by it (the cut sphere would
    # cover the whole loop), and a wire loop of this size cannot be manufactured anyway
    for coil_part in coil_parts:
        tiny = [is_loop_smaller_than(loop.v, input_args.interconnection_cut_width) for loop in coil_part.contour_lines]
        if any(tiny):
            log.warning("Removing %d loop(s) smaller than the interconnection cut width of %g m",
                        sum(tiny), input_args.interconnection_cut_width)
            coil_part.contour_lines = [loop for loop, small in zip(coil_part.contour_lines, tiny) if not small]

    # Close the loops
    for coil_part in coil_parts:
        for p_contour in coil_part.contour_lines:
            if not is_loop_closed(p_contour.uv):
                p_contour.add_uv(p_contour.uv[:, 0][:, None])  # Close the loops
                p_contour.add_v(p_contour.v[:, 0][:, None])  # Close the loops

    # Calculate the combined wire length for the unconnected loops
    coil_parts[-1].combined_loop_length = 0
    for coil_part in coil_parts:
        combined_loop_length = sum(np.sum(np.linalg.norm(
            loop.v[:, 1:] - loop.v[:, :-1], axis=0)) for loop in coil_part.contour_lines)
        coil_part.combined_loop_length = combined_loop_length

    return coil_parts


def is_loop_smaller_than(v: np.ndarray, size: float) -> bool:
    """
    Check whether a loop fits into a box whose diagonal is smaller than a given size.

    Args:
        v (ndarray): The 3D points of the loop, (3, n).
        size (float): The size to compare with, e.g. the interconnection cut width.

    Returns:
        bool: True if the diagonal of the bounding box of the loop is smaller than `size`.
    """
    return bool(np.linalg.norm(np.ptp(v, axis=1)) < size)


def is_loop_closed(uv: np.ndarray, rtol: float = 1e-9) -> bool:
    """
    Check whether a loop ends where it starts, within a tolerance relative to the size of the loop.

    The MATLAB original compares the coordinates of the end and start points for exact (in)equality, and only closes
    a loop if both coordinates differ. A loop whose end point shares one coordinate with its start point (as happens on
    the mesh grid lines) was then left open, depending on the last bit of the contour, i.e. on rounding.

    Args:
        uv (ndarray): The 2D points of the loop, (2, n).
        rtol (float): The tolerance for the distance between the end and the start point, relative to the extent of the loop.

    Returns:
        bool: True if the end point coincides with the start point.
    """
    gap = np.linalg.norm(uv[:, -1] - uv[:, 0])
    extent = np.max(np.ptp(uv, axis=1))
    return bool(gap <= rtol * extent)


def evaluate_loop_significance(coil_parts: List[CoilPart], target_field: TargetField):
    """
    Calculate the relative errors between the different input and output fields and evaluate loop significance.

    Initialises the following properties of a CoilPart:
        - field_by_loops
        - combined_loop_field
        - loop_significance

    Updates the following properties of a CoilPart:
        - None

    Args:
        coil_parts (List[CoilPart]): A list of CoilPart structures.
        target_field (TargetField): The target field associated with the CoilSolution.

    Returns:
        List[CoilPart]: The updated list of CoilPart structures after evaluating loop significance.
    """

    for coil_part in coil_parts:
        num_contours = len(coil_part.contour_lines)
        combined_loop_field = np.zeros((3, target_field.b.shape[1]))
        loop_significance = np.zeros(num_contours)
        # Create field_by_loops
        field_shape = target_field.coords.shape
        coil_part.field_by_loops = np.empty((field_shape[0], field_shape[1], num_contours), dtype=float)

        # Enhancement: This can be parallelised.
        for i, loop in enumerate(coil_part.contour_lines):
            coil_part.field_by_loops[:, :, i] = biot_savart_calc_b(loop.v, target_field) * coil_part.contour_step
            combined_loop_field += coil_part.field_by_loops[:, :, i]

        for i, loop in enumerate(coil_part.contour_lines):
            loop_z_abs = np.abs(coil_part.field_by_loops[2, :, i])
            field_z_abs = np.abs(combined_loop_field[2, :])
            loop_significance[i] = np.max(loop_z_abs) / (np.mean(field_z_abs)) * 100

        coil_part.combined_loop_field = combined_loop_field
        coil_part.loop_significance = loop_significance

    return coil_parts


def biot_savart_calc_b(wire_elements: np.ndarray, target_f: TargetField):
    """
    Calculate the magnetic field using Biot-Savart law for wire elements given as a sequence of coordinate points.

    Args:
        wire_elements (ndarray): A sequence of coordinate points representing wire elements (m,3).
        target_f: The target field.

    Returns:
        ndarray: The magnetic field calculated using Biot-Savart law with shape (3, num_tp).
    """
    # The wire is evaluated in blocks of segments and target points, see `wire_kernels`.
    # NOTE: Using Python (num_vertices, xyz) matrix layout
    seg_coords = ((wire_elements[:, :-1] + wire_elements[:, 1:]) / 2).T
    currents = (wire_elements[:, 1:] - wire_elements[:, :-1]).T
    b_field = biot_savart_field(seg_coords, currents, target_f.coords.T)

    # NOTE: Returning MATLAB 3xM layout
    return b_field.T
