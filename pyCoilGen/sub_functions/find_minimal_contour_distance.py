import numpy as np

from typing import List

# Logging
import logging

# Local imports
from .data_structures import CoilPart
from .find_min_mutual_loop_distance import find_min_mutual_loop_distance

log = logging.getLogger(__name__)


def find_minimal_contour_distance(coil_parts: List[CoilPart], input_args):
    """
    Find the minimal distance in the xyz domain between contours to assign a proper conductor width later.

    Initialises the following properties of a CoilPart:
        - None

    Depends on the following properties of the CoilParts:
        - contour_lines

    Depends on the following input_args:
        - skip_calculation_min_winding_distance

    Updates the following properties of a CoilPart:
        - pcb_track_width

    Args:
        coil_parts (List[CoilPart]): List of CoilPart structures.
        input_args (DataStructure): The application command-line arguments.

    Returns:
        List[CoilPart]: List of CoilPart structures with the 'pcb_track_width' attribute updated.
    """
    for part_ind in range(len(coil_parts)):
        coil_part = coil_parts[part_ind]
        if not input_args.skip_calculation_min_winding_distance:
            min_vals = []
            for ind_1 in range(len(coil_part.contour_lines)):
                for ind_2 in range(ind_1, len(coil_part.contour_lines)):
                    if ind_1 != ind_2:
                        min_dist = find_min_mutual_loop_distance(coil_part.contour_lines[ind_1],
                                                                 coil_part.contour_lines[ind_2],
                                                                 False, only_min_dist=True)
                        min_vals.append(min_dist)
            coil_part.pcb_track_width = min(min_vals)
        else:
            coil_part.pcb_track_width = 0

    return coil_parts
