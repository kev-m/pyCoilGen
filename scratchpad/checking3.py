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

from build_planar_mesh import build_planar_mesh
from build_biplanar_mesh import build_biplanar_mesh

# Import the required modules from sub_functions directory

if __name__ == "__main__":
    # Set up logging
    log = logging.getLogger(__name__)
    logging.basicConfig(level=logging.DEBUG)
    # logging.basicConfig(level=logging.INFO)

    # attach to logger so trimesh messages will be printed to console
    trimesh.util.attach_to_log()


    planar_height = 0.5
    planar_width = 0.75
    num_lateral_divisions = 4
    num_longitudinal_divisions = 4
    rotation_vector_x = 0.0
    rotation_vector_y = 0.0
    rotation_vector_z = 1.0
    rotation_angle = np.pi/8.0
    center_position_x = 0.0
    center_position_y = 0.0
    center_position_z = 0.0
    coil_mesh = build_planar_mesh(planar_height, planar_width, num_lateral_divisions, num_longitudinal_divisions,
                      rotation_vector_x, rotation_vector_y, rotation_vector_z, rotation_angle,
                      center_position_x, center_position_y, center_position_z)


    print(coil_mesh.vertices)
    #log.debug(" vertices: %s", coil_mesh.vertices)
    # DEBUG   (checking2.py: 44)  shape vertices: (114, 3)
    log.debug(" shape vertices: %s", coil_mesh.vertices.shape)
    # DEBUG:__main__: shape faces: (182, 3)
    log.debug(" shape faces: %s", coil_mesh.faces.shape)
    log.debug(" faces min: %d, max: %s", np.min(
        coil_mesh.faces), np.max(coil_mesh.faces))

    # mesh = trimesh.load('Geometry_Data/dental_gradient_ccs_single_low.stl')
    #  <trimesh.Trimesh(vertices.shape=(114, 3), faces.shape=(182, 3))>
    # log.debug(" coil_mesh: %s", mesh)

    mesh = trimesh.Trimesh(vertices=coil_mesh.vertices, faces=coil_mesh.faces)

    # is the current mesh watertight?
    log.debug("mesh.is_watertight: %s", mesh.is_watertight)

    # what's the euler number for the mesh?
    log.debug("mesh.euler_number: %s", mesh.euler_number)

    #mesh.show()

    planar_height = 0.5
    planar_width = 0.75
    num_lateral_divisions = 4
    num_longitudinal_divisions = 4
    target_normal_x = 0.0
    target_normal_y = 0.0
    target_normal_z = 1.0
    center_position_x = 0.0
    center_position_y = 0.0
    center_position_z = 0.0
    plane_distance = 0.25
    coil_mesh2 = build_biplanar_mesh(planar_height, planar_width,
                               num_lateral_divisions, num_longitudinal_divisions,
                               target_normal_x, target_normal_y, target_normal_z,
                               center_position_x, center_position_y, center_position_z,
                               plane_distance)

    print("vertices = ", coil_mesh2.vertices)
    print("faces = ", coil_mesh2.faces, np.min(coil_mesh2.faces), np.max(coil_mesh2.faces))

    mesh2 = trimesh.Trimesh(vertices=coil_mesh2.vertices, faces=coil_mesh2.faces)

    # is the current mesh watertight?
    log.debug("mesh.is_watertight: %s", mesh2.is_watertight)

    # what's the euler number for the mesh?
    log.debug("mesh.euler_number: %s", mesh2.euler_number)

    mesh2.show()

