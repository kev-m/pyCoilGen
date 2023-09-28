"""Module for creating STL meshes.

This module provides functions for creating STL meshes based on input arguments.
"""

# System
import numpy as np

# Logging
import logging

# Local imports
from pyCoilGen.sub_functions.data_structures import Mesh

log = logging.getLogger(__name__)


def create_stl_mesh(input_args):
    """Create an STL mesh based on input arguments.

    This function loads an STL file specified in the input arguments, reads
    the coil mesh surface, and returns a Mesh object. It assumes a representative
    normal of [0,0,1] if not specified.

    Args:
        input_args (argparse.Namespace): Input arguments containing file paths and settings.

    Returns:
        Mesh: A Mesh object representing the coil.

    Example:
        >>> input_args = argparse.Namespace(
        ...     geometry_source_path='/path/to/geometry/',
        ...     coil_mesh_file='coil_mesh.stl'
        ... )
        >>> coil_mesh = create_stl_mesh(input_args)
    """
    log.debug("Loading STL")
    # Load the stl file; read the coil mesh surface
    coil_mesh = Mesh.load_from_file(input_args.geometry_source_path, input_args.coil_mesh_file)
    log.info(" Loaded mesh from %s/%s.", input_args.geometry_source_path, input_args.coil_mesh_file)
    coil_mesh.normal_rep = np.array([0.0, 0.0, 1.0])
    return coil_mesh


def register_args(parser):
    """Register arguments specific to STL mesh creation.

    This function adds command-line arguments to the provided parser that are
    specific to STL mesh creation.

    Args:
        parser (argparse.ArgumentParser): The parser to which arguments will be added.
    """
    # Add arguments specific to STL mesh creation
    # parser.add_argument('--stl_mesh_filename', type=str, default='none', help="File of the primary coil mesh")
    pass
