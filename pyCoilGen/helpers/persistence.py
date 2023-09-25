from numpy import save as np_save, load as np_load, asarray, concatenate
from os import path, makedirs

# Logging
import logging

from pyCoilGen.sub_functions.constants import get_level, DEBUG_NONE
from pyCoilGen.sub_functions.data_structures import CoilSolution, DataStructure

log = logging.getLogger(__name__)


def save(output_dir: str, project_name: str, tag: str, solution) -> str:
    """
    Save the solution to the output directory, with a name based on the project_name and tag.

    Creates the output_dir if it does not already exist.

    Args:
        output_dir (str): The default directory to write to.
        project_name (str): The project name.
        tag (str): A tag to distinguish this save from any others.
        solution (CoilSolution): The solution to save.

    Returns:
        filename (str): The actual filename that the solution has been saved to.
    """
    # Create the output_dir if it does not exist
    makedirs(output_dir, exist_ok=True)
    filename = f'{path.join(output_dir,project_name)}_{tag}.npy'
    if get_level() > DEBUG_NONE:
        log.debug("Saving solution to '%s'", filename)
    np_save(filename, asarray([solution], dtype=object))

    return filename


def load(output_dir: str, project_name: str, tag: str) -> CoilSolution:
    """
    Load the solution from the output directory, with a name based on the project_name and tag.

    Args:
        output_dir (str): The default directory to write to.
        project_name (str): The project name.
        tag (str): A tag to distinguish this save from any others.

    Returns:
        solution (CoilSolution): The coil solution.
    """
    # Create the output_dir if it does not exist
    filename = f'{path.join(output_dir,project_name)}_{tag}.npy'
    if get_level() > DEBUG_NONE:
        log.debug("Loading solution from '%s'", filename)
    [solution] = np_load(filename, allow_pickle=True)

    return solution

def save_preoptimised_data(solution: CoilSolution, default_dir = 'Pre_Optimized_Solutions') -> str:
    """
    Writes out the combined coil mesh, stream function and target field data for re-use.

    Load the data with the `--sf_source_file` parameter.

    Depends on the following properties of the CoilSolution:
        - target_field.coords, target_field.b
        - coil_parts[n].stream_function
        - combined_mesh.vertices, combined_mesh.faces
        - input_args.sf_dest_file
    
    Args:
        solution (CoilSolution): The solution data.
        default_dir (str, optional): Default directory to search first. Defaults to 'Pre_Optimized_Solutions'

    Returns:
        filename (str): The filename written to.

    Raises:
        FileNotFoundError: If the file can not be created, e.g. if the directory does not exist.
    """
    if solution.input_args.sf_dest_file != 'none':
        # Extract the TargetField coords and b-field in Python (m,3) format
        target_field = DataStructure(coords=solution.target_field.coords.T, b=solution.target_field.b.T)

        # Create the stream_function from the coil_parts
        stream_function = solution.coil_parts[0].stream_function
        for i in range(1,len(solution.coil_parts)):
            stream_function = concatenate((stream_function, solution.coil_parts[i].stream_function))

        # Extract the vertices and faces of the combined mesh in (m,3) format
        combined_mesh = DataStructure(vertices=solution.combined_mesh.vertices, faces=solution.combined_mesh.faces)

        # Create the carrier data structure
        data = DataStructure(coil_mesh=combined_mesh, target_field=target_field, stream_function=stream_function)

        target_file = solution.input_args.sf_dest_file
        if '/' in target_file or '\\' in target_file:
            filename = f'{target_file}.npy'
        else:
            makedirs(default_dir, exist_ok=True)
            filename = f'{path.join(default_dir, target_file)}.npy'
        log.info("Writing pre-optimised data to '%s'", filename)
        np_save(filename, [data], allow_pickle=True)

        return filename