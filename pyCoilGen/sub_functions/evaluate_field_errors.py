import numpy as np
from typing import List
import logging

from .data_structures import CoilPart, DataStructure, FieldErrors, SolutionErrors, TargetField
from .process_raw_loops import biot_savart_calc_b

log = logging.getLogger(__name__)


def evaluate_field_errors(coil_parts: List[CoilPart], input_args: DataStructure, target_field: TargetField, sf_b_field: np.ndarray) -> (List[CoilPart], SolutionErrors):
    """
    Calculate relative errors between different input and output fields.

    Initialises the following properties of the CoilParts:
        - field_by_loops
        - field_by_layout

    Depends on the following properties of the CoilParts:
        - wire_path

    Depends on the following input_args:
        - skip_postprocessing

    Updates the following properties of a CoilPart:
        - None

    Args:
        coil_parts (List[CoilPart]): List of CoilPart structures.
        input_args (struct): Input parameters.
        target_field (TargetField): The target magnetic field.
        sf_b_field (np.ndarray): TBD

    Returns:
        solution_errors (SolutionErrors): A structure containing field error values and other information.
    """
    for coil_part in coil_parts:
        # Calculate the combined field of the unconnected contours
        coil_part.field_by_loops2 = np.zeros((3, target_field.b.shape[1]))  # (3,257)
        for loop in coil_part.contour_lines:
            loop_field = biot_savart_calc_b(loop.v, target_field)
            coil_part.field_by_loops2 += loop_field
        coil_part.field_by_loops2 *= coil_part.contour_step

        # Calculate the field of connected, final layouts
        if not input_args.skip_postprocessing:
            coil_part.field_by_layout = biot_savart_calc_b(coil_part.wire_path.v, target_field)
            coil_part.field_by_layout *= coil_part.contour_step  # scaled with the current of the discretization
        else:
            coil_part.field_by_layout = coil_part.field_by_loops2

    """
    The provided code segment initializes the fields in coil_parts, calculates the field by loops and layouts,
    and finds the ideal current strength.
    """
    # End of part 1

    # Part 2: Find the current polarity for the different coil parts regarding the target field
    # Create all possible polarities as binary representations
    possible_polarities = [list(format(x, '0' + str(len(coil_parts)) + 'b')) for x in range(2 ** len(coil_parts))]

    # Convert binary representations to integers and set 0 to -1
    possible_polarities = [[1 if digit == '1' else -1 for digit in polarity] for polarity in possible_polarities]

    # Combine the coil_part fields scaled with all possible polarities and choose the one
    # which is closest to the target field
    combined_field_layout = np.zeros((len(possible_polarities), *coil_part.field_by_layout.shape))
    combined_field_loops = np.zeros((len(possible_polarities), *coil_part.field_by_loops2.shape))
    pol_projections_loops = np.zeros(len(possible_polarities))
    pol_projections_layout = np.zeros(len(possible_polarities))

    for pol_ind in range(len(possible_polarities)):
        for part_ind in range(len(coil_parts)):
            combined_field_layout[pol_ind] += possible_polarities[pol_ind][part_ind] * \
                coil_parts[part_ind].field_by_layout
            combined_field_loops[pol_ind] += possible_polarities[pol_ind][part_ind] * \
                coil_parts[part_ind].field_by_loops2

        # Project the combined field onto the target field
        pol_projections_layout[pol_ind] = np.linalg.norm(combined_field_layout[pol_ind] - target_field.b)  # Fails?!?
        pol_projections_loops[pol_ind] = np.linalg.norm(combined_field_loops[pol_ind] - target_field.b)  # Passes

    """
    This code segment covers the second part of your MATLAB code, including generating possible polarities,
    combining fields with different polarities, and projecting the combined fields onto the target field. 
    """
    # End of part 2
    # Part 3: Choose the best combination and adjust the current direction

    # Find the best direction for layout and loops
    best_dir_layout = np.argmin(pol_projections_layout)
    best_dir_loops = np.argmin(pol_projections_loops)

    # Choose the best combination for layout and loops
    combined_field_layout = combined_field_layout[best_dir_layout]  # Fail
    combined_field_loops = combined_field_loops[best_dir_loops]  # Pass

    # Adjust the current direction for the layout (in case of the wrong direction)
    if not input_args.skip_postprocessing:
        for part_ind in range(len(coil_parts)):
            if possible_polarities[best_dir_layout][part_ind] != 1:
                coil_parts[part_ind].wire_path.v = np.fliplr(coil_parts[part_ind].wire_path.v)
                coil_parts[part_ind].wire_path.uv = np.fliplr(coil_parts[part_ind].wire_path.uv)

    # Adjust the current direction for the loops (in case of the wrong direction)
    for part_ind in range(len(coil_parts)):
        if possible_polarities[best_dir_loops][part_ind] != 1:
            for loop_ind in range(len(coil_parts[part_ind].contour_lines)):
                coil_parts[part_ind].contour_lines[loop_ind].v = np.fliplr(
                    coil_parts[part_ind].contour_lines[loop_ind].v)
                coil_parts[part_ind].contour_lines[loop_ind].uv = np.fliplr(
                    coil_parts[part_ind].contour_lines[loop_ind].uv)

    """
    This code segment covers the third part of your MATLAB code, including choosing the best combination, adjusting
    the current direction for layout and loops if necessary.
    """
    # End of part 3

    # Part 4: Calculate field errors and return results
    # Extract z-components of the fields
    target_z = target_field.b[2, :]  # Tranpose into MATLAB shape (3,n)
    sf_z = sf_b_field[:, 2]  # Field of stream function (Transposed, because it is Python shaped (n,3))
    layout_z = combined_field_layout[2, :]
    loop_z = combined_field_loops[2, :]

    # Calculate relative errors
    field_error_vals = FieldErrors()
    field_error_vals.max_rel_error_layout_vs_target = np.max(
        np.abs((layout_z - target_z) / np.max(np.abs(target_z)))) * 100
    field_error_vals.mean_rel_error_layout_vs_target = np.mean(
        np.abs((layout_z - target_z) / np.max(np.abs(target_z)))) * 100

    field_error_vals.max_rel_error_unconnected_contours_vs_target = np.max(
        np.abs((loop_z - target_z) / np.max(np.abs(target_z)))) * 100
    field_error_vals.mean_rel_error_unconnected_contours_vs_target = np.mean(
        np.abs((loop_z - target_z) / np.max(np.abs(target_z)))) * 100

    field_error_vals.max_rel_error_layout_vs_stream_function_field = np.max(
        np.abs((layout_z - sf_z) / np.max(np.abs(sf_z)))) * 100
    field_error_vals.mean_rel_error_layout_vs_stream_function_field = np.mean(
        np.abs((layout_z - sf_z) / np.max(np.abs(sf_z)))) * 100

    field_error_vals.max_rel_error_unconnected_contours_vs_stream_function_field = np.max(
        np.abs((loop_z - sf_z) / np.max(np.abs(sf_z)))) * 100
    field_error_vals.mean_rel_error_unconnected_contours_vs_stream_function_field = np.mean(
        np.abs((loop_z - sf_z) / np.max(np.abs(sf_z)))) * 100

    # Go back to the fields for 1 Ampere (Unit Current)
    combined_field_layout_per1Amp = np.zeros_like(combined_field_layout)
    combined_field_loops_per1Amp = np.zeros_like(combined_field_layout)

    for part_ind in range(len(coil_parts)):
        combined_field_layout_per1Amp += (coil_parts[part_ind].field_by_layout /
                                          np.max([coil_parts[x].contour_step for x in range(len(coil_parts))])) * possible_polarities[best_dir_layout][part_ind]
        # Passes
        combined_field_loops_per1Amp += (coil_parts[part_ind].field_by_loops2 /
                                         np.max([coil_parts[x].contour_step for x in range(len(coil_parts))])) * possible_polarities[best_dir_loops][part_ind]

    # Calculate the ideal current strength for the connected layout to match the target field
    opt_current_layout = np.abs(np.mean(target_field.b[2, :] / combined_field_layout[2, :]))

    # Return results
    solution_errors = SolutionErrors(field_error_vals=field_error_vals)
    solution_errors.combined_field_layout = combined_field_layout
    solution_errors.combined_field_loops = combined_field_loops
    solution_errors.combined_field_layout_per1Amp = combined_field_layout_per1Amp
    solution_errors.combined_field_loops_per1Amp = combined_field_loops_per1Amp
    solution_errors.opt_current_layout = opt_current_layout

    return coil_parts, solution_errors
