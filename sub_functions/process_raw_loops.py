# System imports
import numpy as np
from typing import List

# Logging
import logging

# Local imports
from sub_functions.constants import *
from sub_functions.data_structures import CoilPart, WirePart, TargetField
from sub_functions.smooth_track_by_folding import smooth_track_by_folding
from sub_functions.uv_to_xyz import uv_to_xyz

log = logging.getLogger(__name__)


def process_raw_loops(coil_parts: List[CoilPart], input_args, target_field: TargetField):
    """
    Process raw loops in the coil mesh.

    Args:
        coil_parts (List[CoilPart]): A list of CoilPart structures.
        input: The input parameters.
        target_field: The target field associated with the CoilSolution.

    Returns:
        List[CoilPart]: The updated list of CoilPart structures after processing the raw loops.
    """

    # Smooth the contours if smooth_flag is True
    if input_args.smooth_flag and input_args.smooth_factor > 1:
        for coil_part in coil_parts:
            for index, contour in enumerate(coil_part.contour_lines):
                smoothed = smooth_track_by_folding(contour.uv, input_args.smooth_factor)
                log.debug(" smoothed difference: %f", np.sum(np.abs(contour.uv - smoothed)))
                contour.uv = smoothed

    # Generate the curved coordinates
    for coil_part in coil_parts:
        curved_mesh = coil_part.coil_mesh.trimesh_obj
        for loop in coil_part.contour_lines:
            # log.debug(" - Loop: %s", loop.potential)
            loop.v, loop.uv = uv_to_xyz(loop.uv, coil_part.coil_mesh.uv, curved_mesh)

    # Evaluate loop significance and remove loops that do not contribute enough to the target field
    coil_parts = evaluate_loop_significance(coil_parts, target_field)
    for coil_part in coil_parts:
        loops_to_delete = coil_part.loop_significance < input_args.min_loop_significance
        if np.any(loops_to_delete):
            coil_part.contour_lines = [loop for i, loop in enumerate(coil_part.contour_lines) if not loops_to_delete[i]]

    # Close the loops
    for coil_part in coil_parts:
        for loop in coil_part.contour_lines:
            # if loop.uv(1, end) ~= loop.uv(1, 1) & loop.uv(2, end) ~= loop.uv(2, 1)
            if loop.uv[0, -1] != loop.uv[0, 0] and loop.uv[1, -1] != loop.uv[1, 0]:
                loop.add_uv(loop.uv[:, 0][:, None])# Close the loops
                loop.add_v(loop.v[:, 0][:, None])  # Close the loops

    # Calculate the combined wire length for the unconnected loops
    coil_parts[-1].combined_loop_length = 0
    for coil_part in coil_parts:
        # coil_part.combined_loop_length = sum(arrayfun(@(x) 
        #   sum(
        #       vecnorm(
        #           coil_part.contour_lines(x).v(:, 2:end) - 
        #           coil_part.contour_lines(x).v(:, 1:end - 1)
        #       )
        #   ), 1:numel(coil_part.contour_lines)));
        combined_loop_length = sum(np.sum(np.linalg.norm(
            loop.v[:, 1:] - loop.v[:, :-1], axis=0)) for loop in coil_part.contour_lines)
        coil_part.combined_loop_length = combined_loop_length

    return coil_parts


def evaluate_loop_significance(coil_parts: List[CoilPart], target_field: TargetField):
    """
    Calculate the relative errors between the different input and output fields and evaluate loop significance.

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
    num_tp = target_f.b.shape[1]  # (3,n)
    track_part_length = 1000

    if wire_elements.shape[1] > track_part_length:  # (3,7)
        track_part_inds = np.arange(0, wire_elements.shape[1], track_part_length)
        track_part_inds = np.append(track_part_inds, wire_elements.shape[1])

        if track_part_inds[-2] == track_part_inds[-1]:
            track_part_inds = track_part_inds[:-1]

        wire_parts = []

        for i in range(len(track_part_inds) - 1):
            part_ind_start = track_part_inds[i]
            part_ind_end = track_part_inds[i + 1]
            coord = wire_elements[:, part_ind_start:part_ind_end]
            seg_coords = (wire_elements[:, part_ind_start:(part_ind_end - 1)] +
                          wire_elements[:, (part_ind_start + 1):part_ind_end]) / 2
            currents = wire_elements[:, (part_ind_start + 1):part_ind_end] - \
                wire_elements[:, part_ind_start:(part_ind_end - 1)]
            wire_part = WirePart(coord=coord, seg_coords=seg_coords, currents=currents)
            wire_parts.append(wire_part)

    else:
        coord = wire_elements
        seg_coords = (wire_elements[:, :-1] + wire_elements[:, 1:]) / 2  # M: 3x6
        currents = wire_elements[:, 1:] - wire_elements[:, :-1]
        wire_part = WirePart(coord=coord, seg_coords=seg_coords, currents=currents)
        wire_parts = [wire_part]

    # Based on code generated by ChatGPT July 20 Version
    mu0 = 4 * np.pi * 1e-7  # Permeability of free space

    # NOTE: Using Python (num_vertices, xyz) matrix layout
    b_field = np.zeros((num_tp, 3))

    # Extract target cooordinates
    target_coords_T = target_f.coords.T

    for wire_part in wire_parts:  # M: target_f.coords => 3x257, wire_part.seg_coords => 3x6
        # Extract the wire segment co-ordinates
        w_seg_coords_T = wire_part.seg_coords.T
        # Extract the wire segment currents
        w_currents_T = wire_part.currents.T  # -> 6,3

        # Calculate the vector from each target point to each conductor fragment
        r = target_coords_T[:, np.newaxis] - w_seg_coords_T

        # Calculate the distance from each target point to each conductor fragment
        r_norm = np.linalg.norm(r, axis=-1)**3

        # Calculate the cross product between r and currents along the last axis
        r_cross_I = np.cross(w_currents_T, r, axis=-1)

        # Calculate the magnetic field for each target point using broadcasting
        dB = r_cross_I / (r_norm)[:, :, np.newaxis]
        b_field_part = mu0 / (4 * np.pi) * np.sum(dB, axis=1)

        #################################
        # MATLAB 3x257
        # -3.2524e-9	-3.1937e-9	-2.9359e-9	-2.6999e-9 ...
        # -1.2585e-8	-1.5238e-8	-1.3577e-8	-1.2096e-8 ...
        # -9.7899e-9	-1.1245e-8	-1.0907e-8	-1.0509e-8 ...
        #################################
        b_field += b_field_part

    # NOTE: Returning MATLAB 3xM layout
    return b_field.T
