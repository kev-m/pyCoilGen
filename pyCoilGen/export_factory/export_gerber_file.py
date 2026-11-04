"""Module for exporting planar coil parts to Gerber files.

Each planar coil part is exported to a separate set of Gerber (RS-274X) files, suitable for PCB fabrication or
CNC milling workflows:

- `tracks`: The wire path, drawn as a track of width `conductor_cross_section_width`.
- `complement`: The coil surface outline as solid copper, with the wire path cleared. This is what remains when the
  wire path is cut out of a solid plate.
"""

# System
import numpy as np
from os import path, makedirs
from typing import Tuple

# Logging
import logging

# Local imports
from pyCoilGen.sub_functions.data_structures import CoilSolution, CoilPart
from pyCoilGen.helpers.gerber import create_gerber_tracks, create_gerber_complement
from .export_cad_file import gradient_to_filename

log = logging.getLogger(__name__)

_SUPPORTED_LAYERS = ('tracks', 'complement')

# The maximum allowed out-of-plane spread of a coil part, relative to its in-plane extent.
_PLANARITY_TOLERANCE = 0.1


def get_viewing_basis(coil_part: CoilPart) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate the orthonormal basis used to project a planar coil part onto a Gerber layer.

    The coil part is viewed from the side that its coil mesh face normals point towards. For a bi-planar coil, whose
    plates have outward pointing normals, each plate is therefore seen from the outside, as it would be viewed when
    laid face-up on a bench. Looking along the negative normal, `u` points to the right and `w` points up, so the
    projection is never a mirror image of the view.

    The first basis vector, `u`, is the projection of the global x axis onto the plane (or the global y axis, if the
    plane is perpendicular to x).

    Args:
        coil_part (CoilPart): The coil part. Its `coil_mesh` must have consistently oriented faces.

    Returns:
        tuple of np.ndarray: The unit vectors (u, w, n), forming a right-handed system, where n is the viewing normal.

    Raises:
        ValueError: If the coil mesh face normals cancel out, so that no viewing direction can be determined.
    """
    trimesh_obj = coil_part.coil_mesh.trimesh_obj
    normal = np.sum(trimesh_obj.face_normals * trimesh_obj.area_faces[:, np.newaxis], axis=0)
    length = np.linalg.norm(normal)
    if length < 1e-12 * np.sum(trimesh_obj.area_faces):
        raise ValueError("The coil mesh face normals cancel out; unable to determine the viewing direction.")
    normal = normal / length

    reference = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = reference - np.dot(reference, normal) * normal
    u = u / np.linalg.norm(u)
    w = np.cross(normal, u)
    return u, w, normal


def project_to_plane(points: np.ndarray, u: np.ndarray, w: np.ndarray) -> np.ndarray:
    """
    Project 3D points onto a plane, converting from metres to millimetres.

    Args:
        points (np.ndarray): A (3, n) or (n, 3) array of points, in metres.
        u (np.ndarray): The unit vector along the plane x axis.
        w (np.ndarray): The unit vector along the plane y axis.

    Returns:
        np.ndarray: An (n, 2) array of in-plane co-ordinates, in millimetres.
    """
    points = np.asarray(points)
    if points.shape[0] == 3 and points.shape[1] != 3:
        points = points.T
    return 1000.0 * np.column_stack((points @ u, points @ w))


def _polygon_area(points: np.ndarray) -> float:
    """
    Calculate the unsigned area of a polygon, using the shoelace formula.

    Args:
        points (np.ndarray): An (n, 2) array of polygon vertices.

    Returns:
        float: The area of the polygon.
    """
    x, y = points[:, 0], points[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def get_part_outlines(coil_part: CoilPart, u: np.ndarray, w: np.ndarray) -> Tuple[np.ndarray, list]:
    """
    Calculate the outer and inner outlines of a coil part, projected onto the viewing plane.

    Args:
        coil_part (CoilPart): The coil part.
        u (np.ndarray): The unit vector along the plane x axis.
        w (np.ndarray): The unit vector along the plane y axis.

    Returns:
        tuple: The outer outline, an (n, 2) array in millimetres, and a list of inner outlines (holes), each an
            (m, 2) array in millimetres. The outer outline is the boundary that encloses the largest area.
    """
    vertices = coil_part.coil_mesh.get_vertices()
    boundaries = [project_to_plane(vertices[boundary], u, w) for boundary in coil_part.coil_mesh.boundary_indices()]
    areas = [_polygon_area(boundary) for boundary in boundaries]
    outer_index = int(np.argmax(areas))
    holes = [boundary for index, boundary in enumerate(boundaries) if index != outer_index]
    return boundaries[outer_index], holes


def check_planar(coil_part: CoilPart, normal: np.ndarray):
    """
    Check that a coil part is (nearly) planar, as required for a Gerber layer.

    Shifts of the wire path along the surface normal, e.g. from `normal_shift_length`, are permitted: they are small
    compared with the extent of the coil part.

    Args:
        coil_part (CoilPart): The coil part.
        normal (np.ndarray): The unit viewing normal.

    Raises:
        ValueError: If the spread of the coil mesh along the normal is large, compared to its in-plane extent.
    """
    vertices = coil_part.coil_mesh.get_vertices()
    out_of_plane = np.ptp(vertices @ normal)
    in_plane = np.max(np.ptp(vertices, axis=0))
    if out_of_plane > _PLANARITY_TOLERANCE * in_plane:
        raise ValueError("The Gerber exporter only supports planar coil parts. "
                         f"The coil surface deviates {out_of_plane * 1000:.1f} mm from a plane, over an extent of "
                         f"{in_plane * 1000:.1f} mm.")


def _output_filename(export_filename: str, input_args, index: int, layer: str) -> str:
    """
    Expand the filename template and locate it in the output directory.

    Args:
        export_filename (str): The filename template.
        input_args (Namespace): The project input arguments.
        index (int): The zero-based coil part index.
        layer (str): The Gerber layer, 'tracks' or 'complement'.

    Returns:
        str: The final filename. The output directory is only prepended if the template contains no path separator.
    """
    filename = export_filename.format(output_dir=input_args.output_directory, project=input_args.project_name,
                                      field_function=gradient_to_filename(input_args.field_shape_function),
                                      part_index=index, layer=layer)
    if '/' in filename or '\\' in filename:
        return filename
    return path.join(input_args.output_directory, filename)


def export_gerber_file(solution: CoilSolution):
    """Export each planar coil part to Gerber files.

    For each coil part and each of the requested `gerber_layers`, one file is written. The files are written to the
    `output_directory` unless `gerber_filename` contains a slash ('/' or '\\').

    The wire path of each coil part is projected onto the plane of the part, viewed from the side that its coil mesh
    face normals point towards (see `get_viewing_basis`). Any shift of the wire path along the normal is ignored.

    Args:
        solution (CoilSolution): The current CoilSolution.

    Returns:
        list of str: The names of the files that were written.

    Raises:
        ValueError: If a coil part is not planar, has no wire path, if an unsupported layer is requested, or if
            several layers are requested and `gerber_filename` does not contain '{layer}'.
    """
    input_args = solution.input_args
    export_filename = input_args.gerber_filename
    layers = list(input_args.gerber_layers)

    unsupported = [layer for layer in layers if layer not in _SUPPORTED_LAYERS]
    if unsupported:
        raise ValueError(f"Unsupported gerber_layers {unsupported}; supported layers are {list(_SUPPORTED_LAYERS)}.")
    if len(layers) > 1 and '{layer}' not in export_filename:
        raise ValueError("'gerber_filename' must contain '{layer}' when more than one layer is exported.")

    track_width = input_args.conductor_cross_section_width * 1000.0  # mm
    makedirs(input_args.output_directory, exist_ok=True)

    written = []
    for index, coil_part in enumerate(solution.coil_parts):
        if coil_part.wire_path is None:
            raise ValueError(f"Coil part {index} has no wire path; the Gerber export requires post-processing.")

        u, w, normal = get_viewing_basis(coil_part)
        check_planar(coil_part, normal)
        track = project_to_plane(coil_part.wire_path.v, u, w)

        for layer in layers:
            filename = _output_filename(export_filename, input_args, index, layer)
            if layer == 'tracks':
                contents = create_gerber_tracks([track], track_width,
                                                comment=f"{input_args.project_name} part {index} tracks")
            else:
                outline, holes = get_part_outlines(coil_part, u, w)
                contents = create_gerber_complement(outline, [track], track_width, holes,
                                                    comment=f"{input_args.project_name} part {index} complement")
            with open(filename, 'w') as file:
                file.write(contents)
            log.info("Exported Gerber %s layer to %s", layer, filename)
            written.append(filename)

    return written


def get_name():
    """
    Template function to retrieve the plugin exporter name.

    Returns:
        exporter_name (str): The exporter name, given to 'exporter'.
    """
    return 'export gerber file'


def get_parameters() -> list:
    """
    Template function to retrieve the supported parameters and default values as strings.

    Returns:
        list of tuples of parameter name and default value: The additional parameters provided by this exporter.
    """
    return [('gerber_filename', '{project}_{part_index}_{field_function}_{layer}.gbr'),
            ('gerber_layers', "['tracks', 'complement']")]


def register_args(parser):
    """Register arguments specific to Gerber export.

    The following substitutions are available in `gerber_filename` and will be replaced with the corresponding
    content:

    - `{output_dir}` : Replaced with `input_args.output_directory`.
    - `{project}`: Replaced with `input_args.project_name`.
    - `{field_function}`: Replaced with a cleaned up version of `input_args.field_shape_function`.
    - `{part_index}`: Replaced with the current, zero-based, coil index.
    - `{layer}`: Replaced with the layer name, `tracks` or `complement`.

    Args:
        parser (argparse.ArgumentParser): The parser to which arguments will be added.
    """
    parser.add_argument('--gerber_filename', type=str, default='{project}_{part_index}_{field_function}_{layer}.gbr',
                        help="The filename template for the Gerber files.")
    parser.add_argument('--gerber_layers', nargs='+', type=str, default=['tracks', 'complement'],
                        help="The Gerber layers to export: 'tracks' and/or 'complement'.")
