import numpy as np

from typing import List

# Logging
import logging

# Local imports
from .data_structures import CoilPart

log = logging.getLogger(__name__)


def calc_potential_levels(coil_parts: List[CoilPart], combined_mesh, input_args):
    """
    Calculate the potential levels for different coil parts based on the stream function values.

    Initialises the following properties of a CoilPart:
        - contour_step: float
        - potential_level_list: ndarray

    Depends on the following properties of the CoilParts:
        - stream_function

    Depends on the following input_args:
        - levels
        - pot_offset_factor
        - level_set_method

    Updates the following properties of a CoilPart:
        - None

    Args:
        coil_parts (List[CoilPart]): List of coil parts.
        combined_mesh (DataStructure): Combined mesh containing stream function values.
        input (InputParameters): Input parameters.

    Returns:
        coil_parts (List[CoilPart]): Updated list of coil parts
        primary_surface_ind (int):  primary surface index.
    """
    # Extract input parameters
    num_levels = input_args.levels
    level_offset = input_args.pot_offset_factor
    level_set_method = input_args.level_set_method

    primary_surface_ind = 0  # Initialize the index of the primary surface

    for part_ind, coil_part in enumerate(coil_parts):
        if level_set_method == "primary":
            # Calculate the stream function range per part
            sf_range_per_part = [np.max(part.stream_function) - np.min(part.stream_function) for part in coil_parts]

            # Find the primary surface index with the highest range
            primary_surface_ind = np.argmax(sf_range_per_part)

            # Calculate the contour step width based on the primary surface
            contour_step = sf_range_per_part[primary_surface_ind] / (num_levels - 1 + 2 * level_offset)

            # Set the contour step and potential levels for each coil part
            for idx, part in enumerate(coil_parts):
                part.contour_step = contour_step

                if idx == primary_surface_ind:
                    part.potential_level_list = np.arange(
                        num_levels) * contour_step + (np.min(part.stream_function) + level_offset * contour_step)
                else:
                    pot_residual = sf_range_per_part[idx] - 2 * level_offset * contour_step
                    num_pot_steps = int(np.floor(pot_residual / contour_step))
                    part.potential_level_list = np.arange(
                        num_pot_steps) * contour_step + (np.min(part.stream_function) + level_offset * contour_step)
                    dist_to_pot_max = np.max(part.stream_function) - part.potential_level_list[-1]
                    dist_to_pot_min = part.potential_level_list[0] - np.min(part.stream_function)

                    if dist_to_pot_max < dist_to_pot_min:
                        part.potential_level_list -= (dist_to_pot_max - dist_to_pot_min) / 2
                    else:
                        part.potential_level_list += (dist_to_pot_max - dist_to_pot_min) / 2

        elif level_set_method == "combined":
            # Calculate the contour step and potential levels based on the combined mesh
            contour_step = (np.max(combined_mesh.stream_function) -
                            np.min(combined_mesh.stream_function)) / (num_levels - 1 + 2 * level_offset)
            for part in coil_parts:
                part.contour_step = contour_step
                part.potential_level_list = np.arange(num_levels) * contour_step + \
                    (np.min(combined_mesh.stream_function) + level_offset * contour_step)
            primary_surface_ind = 0

        elif level_set_method == "independent":
            # Calculate the contour step and potential levels independently for each part
            for part in coil_parts:
                part.contour_step = (np.max(part.stream_function) - np.min(part.stream_function)) / \
                    (num_levels - 1 + 2 * level_offset)
                part.potential_level_list = np.arange(num_levels) * part.contour_step + \
                    (np.min(part.stream_function) + level_offset * part.contour_step)
            primary_surface_ind = 0

    return coil_parts, primary_surface_ind
