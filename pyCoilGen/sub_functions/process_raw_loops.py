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

    # Close the loops
    for coil_part in coil_parts:
        for p_contour in coil_part.contour_lines:
            # if loop.uv(1, end) ~= loop.uv(1, 1) & loop.uv(2, end) ~= loop.uv(2, 1)
            if p_contour.uv[0, -1] != p_contour.uv[0, 0] and p_contour.uv[1, -1] != p_contour.uv[1, 0]:
                p_contour.add_uv(p_contour.uv[:, 0][:, None])  # Close the loops
                p_contour.add_v(p_contour.v[:, 0][:, None])  # Close the loops

    # Calculate the combined wire length for the unconnected loops
    coil_parts[-1].combined_loop_length = 0
    for coil_part in coil_parts:
        combined_loop_length = sum(np.sum(np.linalg.norm(
            loop.v[:, 1:] - loop.v[:, :-1], axis=0)) for loop in coil_part.contour_lines)
        coil_part.combined_loop_length = combined_loop_length

    return coil_parts


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
