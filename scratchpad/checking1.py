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
sub_functions_path = Path(__file__).resolve().parent / '..'
sys.path.append(str(sub_functions_path))

# Import the required modules from sub_functions directory
from sub_functions.build_cylinder_mesh import build_cylinder_mesh
from sub_functions.read_mesh import create_unique_noded_mesh

if __name__ == "__main__":
    # Set up logging
    log = logging.getLogger(__name__)
    logging.basicConfig(level=logging.DEBUG)
    # logging.basicConfig(level=logging.INFO)

    # attach to logger so trimesh messages will be printed to console
    # trimesh.util.attach_to_log()

    cylinder_height = 0.8
    cylinder_radius = 0.3
    num_circular_divisions = 4 # 20
    num_longitudinal_divisions = 3 # 20
    rotation_vector_x = 1.0
    rotation_vector_y = 0.0
    rotation_vector_z = 0.0
    rotation_angle = 0.0
    data_mesh = build_cylinder_mesh(cylinder_height, cylinder_radius, num_circular_divisions,
                               num_longitudinal_divisions, rotation_vector_x, rotation_vector_y,
                               rotation_vector_z, rotation_angle)
    coil_mesh = create_unique_noded_mesh(data_mesh)
    
    #log.debug(" coil_mesh: Vertices: %s", coil_mesh.get_vertices())
    #log.debug(" coil_mesh: Faces: %s", coil_mesh.get_faces())
    log.debug(" shape vertices: %s", coil_mesh.get_vertices().shape)
    log.debug(" faces min: %d, max: %s", np.min(coil_mesh.get_faces()), np.max(coil_mesh.get_faces()))

    parts = coil_mesh.separate_into_get_parts()
    log.debug("Parts: %d", len(parts))

    # Access the Trimesh implementation
    mesh = coil_mesh.trimesh_obj

    # is the current mesh watertight?
    log.debug("mesh.is_watertight: %s", mesh.is_watertight)

    # what's the euler number for the mesh?
    log.debug("mesh.euler_number: %s", mesh.euler_number)

    mesh.show()