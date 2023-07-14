import numpy as np

from typing import List

# Logging
import logging

# Local imports
from sub_functions.data_structures import CoilPart

log = logging.getLogger(__name__)

def calc_potential_levels(coil_parts: List[CoilPart], combined_mesh, input):
    """
    Calculate the potential levels for different coil parts based on the stream function values.
    
    Args:
        coil_parts (List[CoilPart]): List of coil parts.
        combined_mesh (CombinedMesh): Combined mesh containing stream function values.
        input (InputParameters): Input parameters.
    
    Returns:
        Tuple[List[CoilPart], int]: Updated list of coil parts and primary surface index.
    """
    num_levels = input.levels
    level_offset = input.pot_offset_factor
    level_set_method = input.level_set_method

    if level_set_method == "primary":
        sf_range_per_part = []
        for part in coil_parts:
            sf_range = np.max(part.stream_function) - np.min(part.stream_function)
            sf_range_per_part.append(sf_range)
        primary_surface_ind = np.argmax(sf_range_per_part)
        contour_step = sf_range_per_part[primary_surface_ind] / (num_levels - 1 + 2 * level_offset)
        
        for part_ind, part in enumerate(coil_parts):
            part.contour_step = contour_step
            if part_ind == primary_surface_ind:
                part.potential_level_list = np.arange(num_levels) * contour_step + (np.min(part.stream_function) + level_offset * contour_step)
            else:
                pot_residual = sf_range_per_part[part_ind] - 2 * level_offset * contour_step
                num_pot_steps = int(pot_residual / contour_step)
                part.potential_level_list = np.arange(num_pot_steps) * contour_step + (np.min(part.stream_function) + level_offset * contour_step)
                
                dist_to_pot_max = np.max(part.stream_function) - part.potential_level_list[-1]
                dist_to_pot_min = part.potential_level_list[0] - np.min(part.stream_function)
                if dist_to_pot_max < dist_to_pot_min:
                    part.potential_level_list -= (dist_to_pot_max - dist_to_pot_min) / 2
                else:
                    part.potential_level_list += (dist_to_pot_max - dist_to_pot_min) / 2
        
    elif level_set_method == "combined":
        for part in coil_parts:
            contour_step = (np.max(combined_mesh.stream_function) - np.min(combined_mesh.stream_function)) / (num_levels - 1 + 2 * level_offset)
            part.contour_step = contour_step
            part.potential_level_list = np.arange(num_levels) * contour_step + (np.min(combined_mesh.stream_function) + level_offset * contour_step)
        primary_surface_ind = 0

    elif level_set_method == "independent":
        for part in coil_parts:
            contour_step = (np.max(part.stream_function) - np.min(part.stream_function)) / (num_levels - 1 + 2 * level_offset)
            part.contour_step = contour_step
            part.potential_level_list = np.arange(num_levels) * contour_step + (np.min(part.stream_function) + level_offset * contour_step)
        primary_surface_ind = 0
    
    return coil_parts, primary_surface_ind
