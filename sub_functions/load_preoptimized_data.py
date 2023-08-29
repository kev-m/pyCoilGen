import numpy as np

from typing import List
import logging

# Logging
logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)

# Local imports
from sub_functions.constants import *
from sub_functions.data_structures import CoilSolution, TargetField, Mesh, DataStructure
from sub_functions.split_disconnected_mesh import split_disconnected_mesh
from sub_functions.parameterize_mesh import parameterize_mesh
from sub_functions.stream_function_optimization import generate_combined_mesh

# For timing
from helpers.timing import Timing


def load_preoptimized_data(input_args, matlab_data = None) -> CoilSolution:
    """
    Load preoptimized data.

    Args:
        input_args (any): Input arguments for loading preoptimized data.

    Returns:
        CoilSolution: Preoptimized coil solution containing mesh and stream function information.
    """
    # Load preoptimized data
    load_path = 'Pre_Optimized_Solutions/' + input_args.sf_source_file
    
    # Load data from load_path
    loaded_data = np.load(load_path, allow_pickle=True)[0]

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
    combined_mesh.normal_rep = [0.0, 0.0, 0.0] # Invalid value, fix this later if needed
    coil_parts = split_disconnected_mesh(combined_mesh)
    timer.stop()

    # Parameterize the mesh
    timer.start()
    log.info('Parameterize the mesh:')
    coil_parts = parameterize_mesh(coil_parts, input_args, matlab_data)
    timer.stop()

    # Update additional target field properties
    target_field.weights = np.ones(target_field.b.shape[1])
    target_field.target_field_group_inds = np.ones(target_field.b.shape[1])
    is_suppressed_point = np.zeros(target_field.b.shape[1])
    sf_b_field = target_field.b

    # Generate a combined mesh container
    # TODO: Umm?? Why recreate the combined mesh, when it was created above?
    combined_mesh = generate_combined_mesh(coil_parts) 

    # Assign the stream function to the different mesh parts
    for part_ind in range(len(coil_parts)):
        unique_vert_inds = coil_parts[part_ind].coil_mesh.unique_vert_inds
        coil_parts[part_ind].stream_function = stream_function[unique_vert_inds]

    # Return the CoilSolution instance with the preoptimized data
    return CoilSolution(input_args=input_args, coil_parts=coil_parts, target_field=target_field, 
                        is_suppressed_point=is_suppressed_point, combined_mesh=combined_mesh,
                        sf_b_field=sf_b_field)

# Example usage
if __name__ == "__main__":
    # Load preoptimized data using the defined function
    input_args = DataStructure(sf_source_file='source_data_breast_coil.npy')
    preoptimized_solution = load_preoptimized_data(input_args)
