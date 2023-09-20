from numpy import save as np_save, load as np_load, asarray
from os import path, makedirs

# Logging
import logging

from pyCoilGen.sub_functions.constants import get_level, DEBUG_NONE
from pyCoilGen.sub_functions.data_structures import CoilSolution

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
