# System
import numpy as np

# Logging
import logging

# Local imports
from pyCoilGen.sub_functions.data_structures import Mesh

log = logging.getLogger(__name__)


def create_stl_mesh(input_args):
    log.debug("Loading STL")
    # Load the stl file; read the coil mesh surface
    coil_mesh = Mesh.load_from_file(input_args.geometry_source_path,  input_args.coil_mesh_file)
    # TODO: Need to populate normal_rep with representative normal.
    # HACK: Assume [0,0,1]
    log.warning(" Loaded mesh from STL. Assuming shape representative normal is [0,0,1]!")
    coil_mesh.normal_rep = np.array([0.0, 0.0, 1.0])
    return coil_mesh


def register_args(parser):
    # Add arguments specific to STL mesh creation
    parser.add_argument('--coil_mesh_file', type=str, default='none', help="File of the coil mesh")
