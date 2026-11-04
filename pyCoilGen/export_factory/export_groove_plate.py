"""Module for exporting 3D-printable groove plates of planar coil parts.

For each planar coil part, a solid plate is created behind the coil surface, and the swept conductor is subtracted
from it. The result is a plate with the wire path cut into it as a groove, into which the conductor can be laid.

This exporter requires the optional `manifold3d` package: `pip install pycoilgen[fabrication]`.
"""

# System
import numpy as np
import trimesh
from os import path, makedirs

# Logging
import logging

# Local imports
from pyCoilGen.sub_functions.data_structures import CoilSolution, CoilPart
from pyCoilGen.helpers.planar_layer import get_viewing_basis, project_to_plane, check_planar
from pyCoilGen.helpers.solid_boolean import solid_difference
from .export_cad_file import gradient_to_filename

log = logging.getLogger(__name__)


def create_groove_plate(coil_part: CoilPart, thickness: float) -> trimesh.Trimesh:
    """
    Create the groove plate for a single planar coil part.

    The plate lies behind the coil surface, as seen from the side that the coil mesh face normals point towards: its
    top face is the coil surface and its bottom face is `thickness` below it. The swept conductor, in
    `coil_part.layout_surface_mesh`, is subtracted from the plate to form the groove. The groove is therefore as deep
    as the conductor extends below the coil surface; any part of the wire path that was shifted away from the surface
    (see `normal_shift_length`) is correspondingly shallower, or does not cut into the plate at all.

    The plate is returned in its own frame, laid flat and ready to print: the x and y axes are those of the viewing
    projection (see `get_viewing_basis`), and the z axis points away from the bottom face towards the groove. The
    units are millimetres.

    Args:
        coil_part (CoilPart): The coil part, with a swept conductor in `layout_surface_mesh`.
        thickness (float): The thickness of the plate, in metres.

    Returns:
        trimesh.Trimesh: The plate, in millimetres.

    Raises:
        ValueError: If the coil part is not planar, has no swept conductor, if the conductor does not reach into the
            plate (no groove), or if it cuts all the way through it.
        ImportError: If `manifold3d` is not installed.
    """
    if coil_part.layout_surface_mesh is None:
        raise ValueError("The coil part has no swept conductor mesh; the groove plate requires `skip_sweep` to be "
                         "False.")

    u, w, normal = get_viewing_basis(coil_part)
    check_planar(coil_part, normal)

    # Plate: the coil surface, extruded away from the viewing side
    vertices = coil_part.coil_mesh.get_vertices()
    surface_level = 1000.0 * np.mean(vertices @ normal)
    plate_height = 1000.0 * thickness
    plate = trimesh.creation.extrude_triangulation(project_to_plane(vertices, u, w), coil_part.coil_mesh.get_faces(),
                                                   height=plate_height)

    # Conductor, in the same frame: the z co-ordinate is the height above the bottom of the plate
    conductor_vertices = coil_part.layout_surface_mesh.get_vertices()
    conductor = trimesh.Trimesh(
        vertices=np.column_stack((project_to_plane(conductor_vertices, u, w),
                                  1000.0 * (conductor_vertices @ normal) - (surface_level - plate_height))),
        faces=coil_part.layout_surface_mesh.get_faces())

    groove_depth = plate_height - np.min(conductor.vertices[:, 2])
    if groove_depth <= 0:
        raise ValueError("The conductor does not reach into the plate, so there is no groove. Check that "
                         "`conductor_thickness` or `cross_sectional_points` is larger than `normal_shift_length`.")
    if groove_depth >= plate_height:
        raise ValueError(f"The groove is {groove_depth:.2f} mm deep, which cuts through the {plate_height:.2f} mm "
                         "plate. Increase `groove_plate_thickness` or reduce the conductor size.")
    log.debug("Groove plate: thickness %.2f mm, maximum groove depth %.2f mm", plate_height, groove_depth)

    return solid_difference(plate, conductor)


def export_groove_plate(solution: CoilSolution):
    """Export a 3D-printable groove plate for each planar coil part, as an STL file.

    The files are written to the `output_directory` unless `groove_plate_filename` contains a slash ('/' or '\\').

    Each plate is exported laid flat, with its groove facing up, ready for slicing. The viewing direction and the
    orientation are the same as for the `export gerber file` exporter (see `create_groove_plate`). **NB:** The
    exported STL file is in millimetres, unlike the other exporters, which use metres, because slicers assume
    millimetres.

    Args:
        solution (CoilSolution): The current CoilSolution.

    Returns:
        list of str: The names of the files that were written.

    Raises:
        ValueError: See `create_groove_plate`.
        ImportError: If `manifold3d` is not installed.
    """
    input_args = solution.input_args
    makedirs(input_args.output_directory, exist_ok=True)

    written = []
    for index, coil_part in enumerate(solution.coil_parts):
        plate = create_groove_plate(coil_part, input_args.groove_plate_thickness)

        filename = input_args.groove_plate_filename.format(
            output_dir=input_args.output_directory, project=input_args.project_name,
            field_function=gradient_to_filename(input_args.field_shape_function), part_index=index)
        if '/' not in filename and '\\' not in filename:
            filename = path.join(input_args.output_directory, filename)

        plate.export(filename)
        log.info("Exported groove plate to %s", filename)
        written.append(filename)

    return written


def get_name():
    """
    Template function to retrieve the plugin exporter name.

    Returns:
        exporter_name (str): The exporter name, given to 'exporter'.
    """
    return 'export groove plate'


def get_parameters() -> list:
    """
    Template function to retrieve the supported parameters and default values as strings.

    Returns:
        list of tuples of parameter name and default value: The additional parameters provided by this exporter.
    """
    return [('groove_plate_filename', '{project}_{part_index}_{field_function}_groove_plate.stl'),
            ('groove_plate_thickness', '0.01')]


def register_args(parser):
    """Register arguments specific to groove plate export.

    The following substitutions are available in `groove_plate_filename` and will be replaced with the corresponding
    content:

    - `{output_dir}` : Replaced with `input_args.output_directory`.
    - `{project}`: Replaced with `input_args.project_name`.
    - `{field_function}`: Replaced with a cleaned up version of `input_args.field_shape_function`.
    - `{part_index}`: Replaced with the current, zero-based, coil index.

    Args:
        parser (argparse.ArgumentParser): The parser to which arguments will be added.
    """
    parser.add_argument('--groove_plate_filename', type=str,
                        default='{project}_{part_index}_{field_function}_groove_plate.stl',
                        help="The filename template for the groove plate STL files.")
    parser.add_argument('--groove_plate_thickness', type=float, default=0.01,
                        help="The thickness of the groove plate, in metres.")
