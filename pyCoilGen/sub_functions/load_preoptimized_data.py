import numpy as np

from typing import List
import logging

from os import path


# Local imports
from .constants import *
from .data_structures import CoilSolution, TargetField, Mesh, DataStructure
from .split_disconnected_mesh import split_disconnected_mesh
from .parameterize_mesh import parameterize_mesh
from .stream_function_optimization import generate_combined_mesh

# For timing
from pyCoilGen.helpers.timing import Timing

# For loading data files
from pyCoilGen.helpers.common import find_file


# Logging
log = logging.getLogger(__name__)


def load_preoptimized_data(input_args, default_dir='Pre_Optimized_Solutions') -> CoilSolution:
    """
    Load pre-calculated data from a previous run.

    Initialises the following properties of the CoilParts:
        - v,n (np.ndarray)  : vertices and vertex normals (m,3), (m,3)
        - f,fn (np.ndarray) : faces and face normals (n,2), (n,3)
        - uv (np.ndarray)   : 2D project of mesh (m,2)
        - boundary (int)    : list of lists boundary vertex indices (n, variable)

    Depends on the following properties of the CoilParts:
        - None

    Depends on the following input_args:
        - sf_source_file

    Updates the following properties of a CoilPart:
        - None

    Args:
        input_args (any): Input arguments for loading pre-optimised data.
        default_dir (str, optional): Default directory to search first. Defaults to 'Pre_Optimized_Solutions'

    Returns:
        coilSolution (CoilSolution): Pre-optimised coil solution containing mesh and stream function information.
    """
    # Load pre-optimised data
    source_file = f'{input_args.sf_source_file}.npy'
    if '/' in source_file or '\\' in source_file:
        filename = source_file
    else:
        filename = find_file(default_dir, source_file)

    # Load data from load_path
    log.info("Loading pre-optimised data from '%s'", filename)
    loaded_data = np.load(filename, allow_pickle=True)[0]

    # Extract loaded data
    coil_mesh = loaded_data.coil_mesh
    # Transpose because data is saved in Python (m,3) format
    target_field = TargetField(b=loaded_data.target_field.b.T, coords=loaded_data.target_field.coords.T)
    stream_function = loaded_data.stream_function

    timer = Timing()

    secondary_target_mesh = None

    # Split the mesh and the stream function into disconnected pieces
    timer.start()
    log.info('Split the mesh and the stream function into disconnected pieces.')
    combined_mesh = Mesh(vertices=coil_mesh.vertices, faces=coil_mesh.faces)
    combined_mesh.normal_rep = [0.0, 0.0, 0.0]  # Invalid value, fix this later if needed
    coil_parts = split_disconnected_mesh(combined_mesh)
    timer.stop()

    # Parameterize the mesh
    timer.start()
    log.info('Parameterise the mesh:')
    coil_parts = parameterize_mesh(coil_parts, input_args)
    timer.stop()

    # Update additional target field properties
    target_field.weights = np.ones(target_field.b.shape[1])
    target_field.target_field_group_inds = np.ones(target_field.b.shape[1])
    is_suppressed_point = np.zeros(target_field.b.shape[1])
    sf_b_field = loaded_data.target_field.b  # MATLAB Shape

    # Generate a combined mesh container
    # TODO: Umm?? Why recreate the combined mesh, when it was created above?
    combined_mesh = generate_combined_mesh(coil_parts)

    # Assign the stream function to the different mesh parts
    for part_ind in range(len(coil_parts)):
        unique_vert_inds = coil_parts[part_ind].coil_mesh.unique_vert_inds
        coil_parts[part_ind].stream_function = stream_function[unique_vert_inds]

    # Return the CoilSolution instance with the pre-optimised data
    return CoilSolution(input_args=input_args, coil_parts=coil_parts, target_field=target_field,
                        is_suppressed_point=is_suppressed_point, combined_mesh=combined_mesh,
                        sf_b_field=sf_b_field)
