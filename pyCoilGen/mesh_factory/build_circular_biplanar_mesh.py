"""Mesh builder plugin for a bi-planar mesh of two circular plates.

The plates are meshed on a polar grid (concentric rings of equally spaced vertices), which gives a
regular boundary and evenly sized triangles, compared to the Cartesian grid used by `create circular mesh`.
"""
# System
import numpy as np

# Logging
import logging

# Local imports
from pyCoilGen.sub_functions.data_structures import DataStructure
from pyCoilGen.sub_functions.read_mesh import create_unique_noded_mesh
from pyCoilGen.helpers.common import int_or_float
from .build_biplanar_mesh import translate_and_shift

log = logging.getLogger(__name__)


def simple_circular_mesh(radius: float, num_radial_divisions: int, num_angular_divisions: int,
                         normal_winding_direction: bool = True) -> tuple:
    """
    Generate a flat circular mesh on a polar grid, in the XY plane at z=0.

    The mesh consists of a vertex at the centre, surrounded by `num_radial_divisions` concentric rings, each with
    `num_angular_divisions` equally spaced vertices. The centre vertex is connected to the first ring by a fan of
    triangles; successive rings are connected by pairs of triangles.

    Args:
        radius (float): Radius of the outermost ring.
        num_radial_divisions (int): Number of radial divisions (rings, excluding the centre vertex).
        num_angular_divisions (int): Number of divisions around each ring.
        normal_winding_direction (bool): If True, the triangle normals point along +z (anti-clockwise winding).
            Set to False to reverse the winding, so that the normals point along -z.

    Returns:
        tuple: (vertices, faces), where vertices is an (N, 3) array and faces is an (M, 3) integer array.

    Raises:
        ValueError: If `radius` is not positive, `num_radial_divisions` < 1 or `num_angular_divisions` < 3.
    """
    if radius <= 0:
        raise ValueError("Radius must be positive.")
    if num_radial_divisions < 1:
        raise ValueError("The number of radial divisions must be at least 1.")
    if num_angular_divisions < 3:
        raise ValueError("The number of angular divisions must be at least 3.")

    nr = int(num_radial_divisions)
    nt = int(num_angular_divisions)

    # Vertices: index 0 is the centre; ring i (1..nr) occupies indices 1 + (i-1)*nt ... i*nt
    radii = radius * np.arange(1, nr + 1) / nr
    angles = 2.0 * np.pi * np.arange(nt) / nt
    ring_x = np.outer(radii, np.cos(angles)).ravel()
    ring_y = np.outer(radii, np.sin(angles)).ravel()
    vertices = np.zeros((1 + nr * nt, 3))
    vertices[1:, 0] = ring_x
    vertices[1:, 1] = ring_y

    j = np.arange(nt)
    j_next = (j + 1) % nt

    # Fan of triangles between the centre and the first ring
    fan = np.column_stack((np.zeros(nt, dtype=int), 1 + j, 1 + j_next))

    # Pairs of triangles between ring i and ring i+1
    faces = [fan]
    for i in range(1, nr):
        inner = 1 + (i - 1) * nt
        outer = 1 + i * nt
        v1 = inner + j
        v2 = inner + j_next
        v3 = outer + j_next
        v4 = outer + j
        faces.append(np.column_stack((v1, v3, v2)))
        faces.append(np.column_stack((v1, v4, v3)))
    faces = np.vstack(faces)

    if not normal_winding_direction:
        faces = faces[:, ::-1]

    return vertices, faces


def build_circular_biplanar_mesh(radius: float, num_radial_divisions: int, num_angular_divisions: int,
                                 target_normal_x: float, target_normal_y: float, target_normal_z: float,
                                 center_position_x: float, center_position_y: float, center_position_z: float,
                                 plane_distance: float) -> DataStructure:
    """
    Create a bi-planar mesh of two circular plates, in any orientation.

    The plates are parallel, centred on the origin before any translation, separated by `plane_distance` along the
    target normal. The triangle normals of both plates point away from the other plate.

    Args:
        radius (float): Radius of the circular plates.
        num_radial_divisions (int): Number of radial divisions (rings) of each plate.
        num_angular_divisions (int): Number of angular divisions of each plate.
        target_normal_x (float): X-component of the target normal vector.
        target_normal_y (float): Y-component of the target normal vector.
        target_normal_z (float): Z-component of the target normal vector.
        center_position_x (float): X-coordinate of the center position.
        center_position_y (float): Y-coordinate of the center position.
        center_position_z (float): Z-coordinate of the center position.
        plane_distance (float): Distance between the two planes.

    Returns:
        DataStructure(vertices, faces, normal): The vertices, faces and representative normal of the bi-planar mesh.

    Raises:
        ValueError: If the radius is not positive, or the number of divisions is too small.

    Example:
        >>> mesh = build_circular_biplanar_mesh(0.07, 30, 90, 0, 0, 1, 0, 0, 0, 0.0735)
        >>> mesh.vertices.shape
        (5404, 3)
    """
    vertices1, faces1 = simple_circular_mesh(radius, num_radial_divisions, num_angular_divisions, False)
    vertices1[:, 2] -= plane_distance / 2.0

    vertices2, faces2 = simple_circular_mesh(radius, num_radial_divisions, num_angular_divisions, True)
    vertices2[:, 2] += plane_distance / 2.0

    vertices = np.vstack((vertices1, vertices2))
    faces = np.vstack((faces1, faces2 + vertices1.shape[0]))

    # Rotate to the target normal and translate to the centre position
    shifted_vertices, normal_rep = translate_and_shift(vertices,
                                                       target_normal_x, target_normal_y, target_normal_z,
                                                       center_position_x, center_position_y, center_position_z)

    return DataStructure(vertices=shifted_vertices, faces=faces, normal=normal_rep)


def create_circular_bi_planar_mesh(input_args):
    """Template function to create a circular bi-planar mesh.

    Used when 'input_args.coil_mesh_file' is 'create circular bi-planar mesh'.

    Args:
        input_args (argparse.Namespace): The input arguments, containing `circular_biplanar_mesh_parameter_list`.

    Returns:
        Mesh: The bi-planar mesh of two circular plates.
    """
    log.debug("Creating circular bi-planar mesh with '%s'", input_args.circular_biplanar_mesh_parameter_list)
    mesh_data = build_circular_biplanar_mesh(*input_args.circular_biplanar_mesh_parameter_list)
    coil_mesh = create_unique_noded_mesh(mesh_data)
    return coil_mesh


__default_value__ = [0.07, 30, 90, 0, 0, 1, 0, 0, 0, 0.0735]


def get_name():
    """
    Template function to retrieve the plugin builder name.

    Returns:
        builder_name (str): The builder name, given to 'coil_mesh'.
    """
    return 'create circular bi-planar mesh'


def get_parameters() -> list:
    """
    Template function to retrieve the supported parameters and default values as strings.

    Returns:
        list of tuples of parameter name and default value: The additional parameters provided by this builder
    """
    return [('circular_biplanar_mesh_parameter_list', str(__default_value__))]


def register_args(parser):
    """Template function to register arguments specific to circular bi-planar mesh creation.

    Args:
        parser (argparse.ArgumentParser): The parser to which arguments will be added.
    """
    parser.add_argument('--circular_biplanar_mesh_parameter_list', nargs='+', type=int_or_float,
                        default=__default_value__,
                        help="Parameters for the generation of the circular bi-planar mesh")
