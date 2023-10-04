"""Module for exporting meshes to CAD.

This module provides functions for creating STL meshes based on input arguments.
"""

# System
import numpy as np
from os import path, makedirs

# Logging
import logging

# Local imports
from pyCoilGen.sub_functions.data_structures import CoilSolution, Mesh

log = logging.getLogger(__name__)


def gradient_to_filename(input_str):
    """Convert a stream function expression into a valid filename string."""
    result = input_str
    for char in "\\/ ^*":
        result = result.replace(char, '')
    return result


def export_mesh(index, coil_part, mesh, input_args, export_filename):
    """Export the mesh part to the appropriate directory."""
    output_dir = input_args.output_directory
    project = input_args.project_name
    field_function = gradient_to_filename(input_args.field_shape_function)

    # Replace name elements with the project values.
    filename = export_filename.format(output_dir=output_dir, project=project,
                                      field_function=field_function,
                                      part_index=index, mesh=mesh)

    # Write to the output directory unless the filename already contains a path separator.
    if '/' in filename or '\\' in filename:
        final_filename = filename
    else:
        final_filename = path.join(output_dir, filename)

    # Export the appropriate part
    if mesh == 'surface':
        log.info("Exporting surface mesh to %s", final_filename)
        coil_part.coil_mesh.export(final_filename, file_type=None)
    else:
        if coil_part.layout_surface_mesh is not None:
            log.info("Exporting wire path mesh to %s", final_filename)
            coil_part.layout_surface_mesh.export(final_filename, file_type=None)
        else:
            log.info("The layout_surface_mesh was not computed, skipping export.")


def export_parts(index, coil_part, input_args, export_filename):
    """Iterate over the mesh parts or just the wire path."""
    if '{mesh}' in export_filename:
        # Export both 'surface' and 'wire'
        for mesh in ['surface', 'wire']:
            export_mesh(index, coil_part, mesh, input_args, export_filename)
    else:
        export_mesh(index, coil_part, 'wire', input_args, export_filename)


def export_CAD_file(solution: CoilSolution):
    """Export the coil_mesh to a CAD file format.

    This function uses the export functionality of the Mesh to export each coil_part as a supported file.

    The file will be written to the `output_directory` unless `CAD_export_filename` contains a slash ('/' or '\').

    The file extension determines the file type. Support file types are:

    - STL: [Stereolithography](https://en.wikipedia.org/wiki/STL_(file_format))
    - GLB: [Graphics Library Transmission Format](https://en.wikipedia.org/wiki/GlTF#GLB)
    - PLY: [Polygon](https://en.wikipedia.org/wiki/PLY_(file_format))
    - 3MF: [3D Manufacturing Format](https://en.wikipedia.org/wiki/3D_Manufacturing_Format)
    - OBJ: [Wavefront .obj](https://en.wikipedia.org/wiki/Wavefront_.obj_file)
    - DAE: [COLLADA digital asset exchange](https://en.wikipedia.org/wiki/COLLADA)
    - OFF: [ASCII Object File Format](https://en.wikipedia.org/wiki/OFF_(file_format))

    For a list of supported export formats, see:
    https://github.com/mikedh/trimesh/tree/3.23.5/trimesh/exchange    

    Args:
        solution (CoilSolution): The current CoilSolution

    Returns:
        None
    """

    # Input parameters
    input_args = solution.input_args
    save_mesh = input_args.save_stl_flag

    if save_mesh:
        # Plugin parameters
        export_filename = input_args.CAD_filename

        # Iterate over coil parts and export the mesh/meshes
        for index, coil_part in enumerate(solution.coil_parts):
            export_parts(index, coil_part, input_args, export_filename)


def get_name():
    """
    Template function to retrieve the plugin builder name.

    Returns:
        builder_name (str): The builder name, given to 'coil_mesh'.
    """
    return 'export CAD file'


def get_parameters() -> list:
    """
    Template function to retrieve the supported parameters and default values as strings.

    Returns:
        list of tuples of parameter name and default value: The additional parameters provided by this builder
    """
    return [('CAD_filename', '{project}_{mesh}_{part_index}_{field_function}.stl')]


def register_args(parser):
    """Register arguments specific to STL mesh creation.

    This function adds command-line arguments to the provided parser that are
    specific to STL mesh creation.

    The following substitutions are available and will be replaced with the corresponding content:

    - `{output_dir}` : Replaced with `input_args.output_directory`.
    - `{project}`: Replaced with `input_args.project_name`.
    - `{field_function}`: Replaced with a cleaned up version of `input_args.field_shape_function`.
    - `{part_index}`: Replaced with the current, zero-based, coil index.
    - `{mesh}`: Replaced with `wire` or `surface` for each of the wire path or coil surface.

    If `{mesh}` is not present, only the swept wire path will be exported.   

    Args:
        parser (argparse.ArgumentParser): The parser to which arguments will be added.
    """
    parser.add_argument('--CAD_filename', type=str, default='{project}_{mesh}_{part_index}_{field_function}.stl',
                        help="The filename to export to.")
