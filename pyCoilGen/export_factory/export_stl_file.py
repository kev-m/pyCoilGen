"""Module for exporting meshes to CAD.

This module provides functions for creating STL meshes based on input arguments.
"""

# System
import numpy as np

# Logging
import logging

# Local imports
from pyCoilGen.sub_functions.data_structures import CoilSolution, Mesh

log = logging.getLogger(__name__)

def export_CAD_file(solution: CoilSolution):
    """Export the coil_mesh to a CAD file format.

    This function uses the export functionality of the Mesh. See 

    Args:
        solution (CoilSolution): The current CoilSolution

    Returns:
        None

    Example:
        >>> input_args = argparse.Namespace(
        ...     geometry_source_path='/path/to/geometry/',
        ...     coil_mesh_file='coil_mesh.stl'
        ... )
        >>> coil_mesh = create_stl_mesh(input_args)
    """
    # Support both stl_mesh_filename and coil_mesh_file
    mesh_file = input_args.stl_mesh_filename
    if mesh_file == 'none':
        mesh_file = input_args.coil_mesh_file
        if mesh_file == 'none':
            return None
    log.debug("Loading STL from %s", mesh_file)
    # Load the stl file; read the coil mesh surface
    coil_mesh = Mesh.load_from_file(input_args.geometry_source_path, mesh_file)
    log.info(" Loaded mesh from %s/%s.", input_args.geometry_source_path, mesh_file)
    coil_mesh.normal_rep = np.array([0.0, 0.0, 1.0])
    return coil_mesh

def get_name():
    """
    Template function to retrieve the plugin builder name.
    
    Returns:
        builder_name (str): The builder name, given to 'coil_mesh'.
    """
    return 'export CAD file'

def get_parameters()->list: 
    """
    Template function to retrieve the supported parameters and default values as strings.
    
    Returns:
        list of tuples of parameter name and default value: The additional parameters provided by this builder
    """
    return [('CAD_export_filename', 'none')]

def register_args(parser):
    """Register arguments specific to STL mesh creation.

    This function adds command-line arguments to the provided parser that are
    specific to STL mesh creation.

    Args:
        parser (argparse.ArgumentParser): The parser to which arguments will be added.
    """
    parser.add_argument('--CAD_export_filename', type=str, default='none',
                        help="The filename to export to.")
