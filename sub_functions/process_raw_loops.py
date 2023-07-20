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

def process_raw_loops(coil_parts: List[CoilPart], input, target_field : TargetField):
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
    if input.smooth_flag:
        for coil_part in coil_parts:
            for index, contour in enumerate(coil_part.contour_lines):
                contour.uv = smooth_track_by_folding(contour.uv, input.smooth_factor)

    # Generate the curved coordinates
    for coil_part in coil_parts:
        curved_mesh = coil_part.coil_mesh.trimesh_obj
        for loop in coil_part.contour_lines:
            # log.debug(" - Loop: %s", loop.potential)
            loop.v, loop.uv = uv_to_xyz(loop.uv, coil_part.coil_mesh.uv, curved_mesh)

    # Evaluate loop significance and remove loops that do not contribute enough to the target field
    coil_parts = evaluate_loop_significance(coil_parts, target_field)
    for coil_part in coil_parts:
        loops_to_delete = coil_part.loop_significance < input.min_loop_significance
        coil_part.contour_lines = [loop for i, loop in enumerate(coil_part.contour_lines) if not loops_to_delete[i]]

    # Close the loops
    for coil_part in coil_parts:
        for loop in coil_part.contour_lines:
            if not np.allclose(loop.uv[:, 0], loop.uv[:, -1]) or not np.allclose(loop.uv[:, 1], loop.uv[:, -1]):
                loop.uv = np.hstack((loop.uv, loop.uv[:, 0][:, None]))  # Close the loops
                loop.v = np.hstack((loop.v, loop.v[:, 0][:, None]))  # Close the loops

    # Calculate the combined wire length for the unconnected loops
    coil_parts[-1].combined_loop_length = 0
    for coil_part in coil_parts:
        combined_loop_length = sum(np.sum(np.linalg.norm(loop.v[:, 1:] - loop.v[:, :-1], axis=0)) for loop in coil_part.contour_lines)
        coil_part.combined_loop_length = combined_loop_length

    return coil_parts

def evaluate_loop_significance(coil_parts: List[CoilPart], target_field : TargetField):
    """
    Calculate the relative errors between the different input and output fields and evaluate loop significance.

    Args:
        coil_parts (List[CoilPart]): A list of CoilPart structures.
        target_field (TargetField): The target field associated with the CoilSolution.

    Returns:
        List[CoilPart]: The updated list of CoilPart structures after evaluating loop significance.
    """

    for coil_part in coil_parts:
        combined_loop_field = np.zeros((3, target_field.b.shape[1]))
        loop_significance = np.zeros(len(coil_part.contour_lines))

        for i, loop in enumerate(coil_part.contour_lines):
            coil_part.field_by_loops[:, :, i] = biot_savart_calc_b(loop.v, target_field) * coil_part.contour_step
            combined_loop_field += coil_part.field_by_loops[:, :, i]

        for i, loop in enumerate(coil_part.contour_lines):
            loop_significance[i] = np.max(np.abs(coil_part.field_by_loops[2, :, i])) / (np.mean(np.abs(combined_loop_field[2, :])) / len(coil_part.contour_lines)) * 100

        coil_part.combined_loop_field = combined_loop_field
        coil_part.loop_significance = loop_significance

    return coil_parts

def biot_savart_calc_b(wire_elements : np.ndarray, target_f : TargetField):
    """
    Calculate the magnetic field using Biot-Savart law for wire elements given as a sequence of coordinate points.

    Args:
        wire_elements (ndarray): A sequence of coordinate points representing wire elements (m,3).
        target_f: The target field.

    Returns:
        ndarray: The magnetic field calculated using Biot-Savart law with shape (3, num_tp).
    """
    num_tp = target_f.b.shape[1] # (3,n)
    track_part_length = 1000

    if wire_elements.shape[1] > track_part_length:
        track_part_inds = np.arange(0, wire_elements.shape[1], track_part_length)
        track_part_inds = np.append(track_part_inds, wire_elements.shape[1])

        if track_part_inds[-2] == track_part_inds[-1]:
            track_part_inds = track_part_inds[:-1]

        wire_parts = []

        for i in range(len(track_part_inds) - 1):
            part_ind_start = track_part_inds[i]
            part_ind_end = track_part_inds[i + 1]
            coord=wire_elements[:, part_ind_start:part_ind_end]
            seg_coords = (wire_elements[:, part_ind_start:(part_ind_end - 1)] + wire_elements[:, (part_ind_start + 1):part_ind_end]) / 2
            currents = wire_elements[:, (part_ind_start + 1):part_ind_end] - wire_elements[:, part_ind_start:(part_ind_end - 1)]
            wire_part = WirePart(coord=coord, seg_coords=seg_coords, currents=currents)
            wire_parts.append(wire_part)

    else:
        coord = wire_elements
        seg_coords = (wire_elements[:, :-1] + wire_elements[:, 1:]) / 2
        currents = wire_elements[:, 1:] - wire_elements[:, :-1]        
        wire_part = WirePart(coord=coord, seg_coords=seg_coords, currents=currents)
        wire_parts = [wire_part]

    b_field = np.zeros((3, num_tp))

    for wire_part in wire_parts:
        target_p = np.tile(target_f.coords[:, :, np.newaxis], [1, 1, wire_part.seg_coords.shape[1]])  # copies target point coordinates
        cur_pos = np.tile(wire_part.seg_coords[:, :, np.newaxis], [1, 1, num_tp])  # copies vector of current position
        cur_dir = np.tile(wire_part.currents[:, :, np.newaxis], [1, 1, num_tp])  # copies vector of current direction
        log.debug(" -- here -- ")
        # Exception: operands could not be broadcast together with shapes (3,257,6) (3,6,257)
        len_R = (1 / np.linalg.norm(R, axis=0)) ** 3  # distance factor of Biot-Savart law
        len_R = np.tile(len_R, (3, 1, 1))
        dB = 10 ** (-7) * np.cross(cur_dir, R, axis=0) * len_R
        b_field += np.sum(dB, axis=1)

    return b_field

def biot_savart_calc_b_XXXX(wire_elements: np.ndarray, target_f: TargetField):
    """
    Calculate the magnetic field using Biot-Savart law for wire elements given as a sequence of coordinate points.

    Args:
        wire_elements (ndarray): A sequence of coordinate points representing wire elements (m, 3).
        target_f: The target field.

    Returns:
        ndarray: The magnetic field calculated using Biot-Savart law with shape (num_tp, 3).
    """
    num_tp = target_f.b.shape[1]  # (3, n)
    track_part_length = 1000

    if wire_elements.shape[0] > track_part_length:
        track_part_inds = np.arange(0, wire_elements.shape[0], track_part_length)
        track_part_inds = np.append(track_part_inds, wire_elements.shape[0])

        if track_part_inds[-2] == track_part_inds[-1]:
            track_part_inds = track_part_inds[:-1]

        wire_parts = []

        for i in range(len(track_part_inds) - 1):
            part_ind_start = track_part_inds[i]
            part_ind_end = track_part_inds[i + 1]
            coord = wire_elements[part_ind_start:part_ind_end, :]
            seg_coords = (wire_elements[part_ind_start:(part_ind_end - 1), :] + wire_elements[(part_ind_start + 1):part_ind_end, :]) / 2
            currents = wire_elements[(part_ind_start + 1):part_ind_end, :] - wire_elements[part_ind_start:(part_ind_end - 1), :]
            wire_part = WirePart(coord=coord, seg_coords=seg_coords, currents=currents)
            wire_parts.append(wire_part)

    else:
        coord = wire_elements
        seg_coords = (wire_elements[:-1, :] + wire_elements[1:, :]) / 2
        currents = wire_elements[1:, :] - wire_elements[:-1, :]
        wire_part = WirePart(coord=coord, seg_coords=seg_coords, currents=currents)
        wire_parts = [wire_part]

    b_field = np.zeros((num_tp, 3))

    for wire_part in wire_parts:
        target_p = np.tile(target_f.coords[:, :, np.newaxis], [1, 1, wire_part.seg_coords.shape[0]])  # copies target point coordinates
        cur_pos = np.tile(wire_part.seg_coords[:, :, np.newaxis], [1, 1, num_tp])  # copies vector of current position
        cur_dir = np.tile(wire_part.currents[:, :, np.newaxis], [1, 1, num_tp])  # copies vector of current direction
        #  operands could not be broadcast together with shapes (3,257,6) (6,3,257)
        R = target_p - cur_pos  # distance from current path to each point
        len_R = (1 / np.linalg.norm(R, axis=0)) ** 3  # distance factor of Biot-Savart law
        len_R = np.tile(len_R, (3, 1, 1))
        dB = 10 ** (-7) * np.cross(cur_dir, R, axis=0) * len_R
        b_field += np.sum(dB, axis=1)

    return b_field