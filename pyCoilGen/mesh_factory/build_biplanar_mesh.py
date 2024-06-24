# System
import numpy as np

# Logging
import logging

# Local imports
from pyCoilGen.sub_functions.constants import get_level, DEBUG_BASIC
from pyCoilGen.sub_functions.calc_3d_rotation_matrix_by_vector import calc_3d_rotation_matrix_by_vector
from pyCoilGen.sub_functions.data_structures import DataStructure
from .build_planar_mesh import simple_planar_mesh
from pyCoilGen.sub_functions.read_mesh import create_unique_noded_mesh
from pyCoilGen.helpers.common import int_or_float

log = logging.getLogger(__name__)


def build_biplanar_mesh(planar_height, planar_width,
                        num_lateral_divisions, num_longitudinal_divisions,
                        target_normal_x, target_normal_y, target_normal_z,
                        center_position_x, center_position_y, center_position_z,
                        plane_distance):
    """
    Create a biplanar regular mesh in any orientation.

    Args:
        planar_height (float): Height of the planar mesh.
        planar_width (float): Width of the planar mesh.
        num_lateral_divisions (int): Number of divisions in the lateral direction.
        num_longitudinal_divisions (int): Number of divisions in the longitudinal direction.
        target_normal_x (float): X-component of the target normal vector.
        target_normal_y (float): Y-component of the target normal vector.
        target_normal_z (float): Z-component of the target normal vector.
        center_position_x (float): X-coordinate of the center position.
        center_position_y (float): Y-coordinate of the center position.
        center_position_z (float): Z-coordinate of the center position.
        plane_distance (float): Distance between the two planes.

    Returns:
        biplanar_mesh (dict): Dictionary containing the mesh faces and vertices.
    """

    simple_vertices1, faces1 = simple_planar_mesh(planar_height, planar_width, 
                                                  num_lateral_divisions, num_longitudinal_divisions,
                                                  False)
    # Shift the vertices up
    simple_vertices1 -= np.array([0.0, 0.0, plane_distance/2.0])

    if get_level() > DEBUG_BASIC:
        log.debug(" simple_vertices1 shape: %s", simple_vertices1.shape)

    simple_vertices2, faces2 = simple_planar_mesh(planar_height, planar_width, 
                                                  num_lateral_divisions, num_longitudinal_divisions,
                                                  True)
    # Shift the vertices down
    simple_vertices2 += np.array([0.0, 0.0, plane_distance/2.0])

    if get_level() > DEBUG_BASIC:
        log.debug(" simple_vertices2 shape: %s", simple_vertices2.shape)


    # Combine the vertex arrays
    simple_vertices = np.append(simple_vertices1, simple_vertices2, axis=0)
    if get_level() > DEBUG_BASIC:
        log.debug(" simple_vertices shape: %s", simple_vertices.shape)
        log.debug(" faces1 shape: %s", faces1.shape)
    num_faces1 = simple_vertices1.shape[0]
    faces = np.append(faces1, faces2 + num_faces1, axis=0)

    # Translate and shift
    shifted_vertices, normal_rep = translate_and_shift(simple_vertices,
                                                       target_normal_x, target_normal_y, target_normal_z,
                                                       center_position_x, center_position_y, center_position_z)

    return DataStructure(vertices=shifted_vertices, faces=faces, normal=normal_rep)


def translate_and_shift(vertices,
                        target_normal_x, target_normal_y, target_normal_z,
                        center_position_x, center_position_y, center_position_z):
    old_normal = np.array([0, 0, 1])
    target_normal = np.array([target_normal_x, target_normal_y, target_normal_z])

    if np.linalg.norm(np.cross(old_normal, target_normal)) != 0:
        rot_vec = np.cross(old_normal, target_normal) / np.linalg.norm(np.cross(old_normal, target_normal))
        rot_angle = np.arcsin(np.linalg.norm(np.cross(old_normal, target_normal)) /
                              (np.linalg.norm(old_normal) * np.linalg.norm(target_normal)))
    elif np.allclose(target_normal, np.array([0.0, 0.0, -1.0])):
        # Special case: invert the mesh
        rot_vec = np.array([0, 0, 1])
        rot_angle = np.pi
    else:
        rot_vec = np.array([0, 0, 1])
        rot_angle = 0

    # Rotate
    rot_mat = calc_3d_rotation_matrix_by_vector(rot_vec, rot_angle)
    rot_vertices = np.dot(vertices, rot_mat)

    # Calculate representative normal
    # normal = np.array([0.0, 0.0, 1.0])
    normal_rep = target_normal # np.dot(normal, rot_mat)

    # Shift
    shifted_vertices = rot_vertices + np.array([center_position_x, center_position_y, center_position_z])

    return shifted_vertices, normal_rep


def create_bi_planar_mesh(input_args):
    """Template function to create a bi-planar mesh.

    Used when 'input_args.coil_mesh_file' is 'create bi-planar mesh'.
    """
    log.debug("Creating bi-planar mesh with '%s'", input_args.biplanar_mesh_parameter_list)
    mesh_data = build_biplanar_mesh(*input_args.biplanar_mesh_parameter_list)
    coil_mesh = create_unique_noded_mesh(mesh_data)
    return coil_mesh


__default_value__ = [0.25, 0.25, 20, 20, 1, 0, 0, 0, 0, 0, 0.2]


def get_name():
    """
    Template function to retrieve the plugin builder name.

    Returns:
        builder_name (str): The builder name, given to 'coil_mesh'.
    """
    return 'create bi-planar mesh'


def get_parameters() -> list:
    """
    Template function to retrieve the supported parameters and default values as strings.

    Returns:
        list of tuples of parameter name and default value: The additional parameters provided by this builder
    """
    return [('biplanar_mesh_parameter_list', str(__default_value__))]


def register_args(parser):
    """Template function to register arguments specific to bi-planar mesh creation.

    Args:
        parser (argparse.ArgumentParser): The parser to which arguments will be added.
    """
    # Add the parameters for the generation of the (default) biplanar mesh
    parser.add_argument('--biplanar_mesh_parameter_list', nargs='+', type=int_or_float,
                        default=__default_value__,
                        help="Parameters for the generation of the (default) biplanar mesh")
