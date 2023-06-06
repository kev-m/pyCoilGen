# System imports
import sys
from pathlib import Path
import numpy as np

# Trimesh
import trimesh

# Logging
import logging

# Local imports
# Add the sub_functions directory to the Python module search path
sub_functions_path = Path(__file__).resolve().parent / '../sub_functions'
sys.path.append(str(sub_functions_path))

from read_mesh import read_mesh, stlread_local, create_unique_noded_mesh
from parse_input import parse_input

# Import the required modules from sub_functions directory

if __name__ == "__main__":
    # Set up logging
    log = logging.getLogger(__name__)
    logging.basicConfig(level=logging.DEBUG)
    # logging.basicConfig(level=logging.INFO)

    # attach to logger so trimesh messages will be printed to console
    trimesh.util.attach_to_log()

    """
    # Use high-level function
    arg_list = ['--coil_mesh_file', 'dental_gradient_ccs_single_low.stl'] # IndexError: index 114 is out of bounds for axis 1 with size 114
    input_parser, input_args = parse_input(arg_list)
    coil_mesh, target_mesh, shielded_mesh = read_mesh(input_args)
    """
    # Use low level function
    coil_mesh = stlread_local(
        'Geometry_Data/dental_gradient_ccs_single_low.stl')
    coil_mesh = create_unique_noded_mesh(coil_mesh)
    # coil_mesh.vertices = coil_mesh.vertices.T
    # coil_mesh.faces = coil_mesh.faces.T

    # log.debug(" coil_mesh: Vertices: %s", coil_mesh.vertices)
    # log.debug(" coil_mesh: Faces: %s", coil_mesh.faces)

    # DEBUG   (checking2.py: 44)  shape vertices: (114, 3)
    log.debug(" shape vertices: %s", coil_mesh.get_vertices().shape)
    # DEBUG:__main__: shape faces: (182, 3)
    log.debug(" shape faces: %s", coil_mesh.get_faces().shape)
    log.debug(" faces min: %d, max: %s", np.min(
        coil_mesh.get_faces()), np.max(coil_mesh.get_faces()))

    # mesh = trimesh.load('Geometry_Data/dental_gradient_ccs_single_low.stl')
    #  <trimesh.Trimesh(vertices.shape=(114, 3), faces.shape=(182, 3))>
    # log.debug(" coil_mesh: %s", mesh)

    # Access implementation
    mesh = coil_mesh.trimesh_obj

    # is the current mesh watertight?
    log.debug("mesh.is_watertight: %s", mesh.is_watertight)

    # what's the euler number for the mesh?
    log.debug("mesh.euler_number: %s", mesh.euler_number)

    mesh.show()
