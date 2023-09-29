import numpy as np
# Logging
import logging

# Local imports
from pyCoilGen.sub_functions.data_structures import DataStructure
from pyCoilGen.sub_functions.read_mesh import create_unique_noded_mesh
from pyCoilGen.helpers.triangulation import Triangulate
from pyCoilGen.sub_functions.calc_3d_rotation_matrix_by_vector import calc_3d_rotation_matrix_by_vector

log = logging.getLogger(__name__)


def build_circular_mesh(radius: float, num_radial_divisions: int,
                        rotation_vector_x: float, rotation_vector_y: float, rotation_vector_z: float,
                        rotation_angle: float,
                        center_position_x: float, center_position_y: float, center_position_z: float) -> DataStructure:
    """
    Create a mono-planar circular mesh in any orientation.

    Args:
        radius (float): Radius of the circular mesh.
        num_radial_divisions (int): Number of radial divisions.
        rotation_vector_x (float): X component of the rotation vector.
        rotation_vector_y (float): Y component of the rotation vector.
        rotation_vector_z (float): Z component of the rotation vector.
        rotation_angle (float): Rotation angle in radians.
        center_position_x (float): X component of the center position.
        center_position_y (float): Y component of the center position.
        center_position_z (float): Z component of the center position.

    Returns:
        DataStructure(vertices, faces): DataStructure containing the vertices and faces of the planar mesh.

    Raises:
        ValueError: If the provided radius is not positive.

    Example:
        >>> parameters = {
        ...     'radius' : 1.0,
        ...     'num_radial_divisions' : 10,
        ...     'rotation_vector_x' : 0.0,
        ...     'rotation_vector_y' : 0.0,
        ...     'rotation_vector_z' : 1.0,
        ...     'rotation_angle' : 0.0,
        ...     'center_position_x' : 0.0,
        ...     'center_position_y' : 0.0,
        ...     'center_position_z' : 0.0
        ... }
        >>> circular_mesh = build_circular_mesh(**parameters)
    """
    if radius <= 0:
        raise ValueError("Radius must be positive.")

    rotation_vector = np.array([rotation_vector_x, rotation_vector_y, rotation_vector_z])
    center_position = np.array([center_position_x, center_position_y, center_position_z])

    # Generate mesh vertices
    x_positions = np.linspace(-radius, radius, num_radial_divisions+1)
    x, y = np.meshgrid(x_positions, x_positions)
    z = np.zeros_like(x)
    vertices = np.vstack((y.flatten(), x.flatten(), z.flatten())).T

    # Generate Delaunay triangles
    tri = Triangulate(vertices[:, :2])

    # Create mesh faces
    faces_1 = np.asarray(tri.get_triangles())
    faces_2 = faces_1[:, [1, 0, 2]]

    # Combine faces
    faces = np.vstack((faces_1, faces_2))

    # Create circular mesh
    circular_mesh = DataStructure(faces=faces,vertices=vertices)

    # Delete vertices outside the circular boundary
    vert_distances = np.linalg.norm(circular_mesh.vertices[:, :2], axis=1)
    vertices_to_delete = np.where(vert_distances > radius * 0.99)[0]
    circular_mesh.vertices = np.delete(circular_mesh.vertices, vertices_to_delete, axis=0)

    # Update faces after deletion
    for delete_ind in sorted(vertices_to_delete, reverse=True):
        circular_mesh.faces[circular_mesh.faces > delete_ind] -= 1
    circular_mesh.faces = circular_mesh.faces.reshape(-1, 3)

    # Morph boundary verts for proper circular shape
    boundary_verts = np.unique(np.concatenate(
        (circular_mesh.faces[:, 0], circular_mesh.faces[:, 1], circular_mesh.faces[:, 2])))
    for vert_ind in boundary_verts:
        vert_radius = np.linalg.norm(circular_mesh.vertices[vert_ind, :2])
        circular_mesh.vertices[vert_ind, :] *= radius / vert_radius

    # Adjust mesh with desired translation and rotation
    rotation_matrix = calc_3d_rotation_matrix_by_vector(rotation_vector, rotation_angle)
    circular_mesh.vertices = np.dot(rotation_matrix, circular_mesh.vertices.T).T + center_position

    return circular_mesh


def create_circular_mesh(input_args):
    """Template function to create a circular mesh.

    Used when 'input_args.coil_mesh_file' is 'create circular mesh'.
    """
    log.debug("Creating cylinder mesh with '%s'", input_args.circular_mesh_parameter_list)
    mesh_data = build_circular_mesh(*input_args.circular_mesh_parameter_list)
    coil_mesh = create_unique_noded_mesh(mesh_data)
    return coil_mesh


def register_args(parser):
    """Template function to register arguments specific to planar mesh creation.

    Args:
        parser (argparse.ArgumentParser): The parser to which arguments will be added.
    """
    # Add the parameters for the generation of the (default) planar mesh
    parser.add_argument('--circular_mesh_parameter_list', nargs='+', type=float,
                        default=[0.25, 20, 1.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0,],
                        help="Parameters for the generation of the (default) planar mesh")
