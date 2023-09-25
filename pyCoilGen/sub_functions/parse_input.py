import argparse
from pathlib import Path
import platform

# local imports
from .constants import DEBUG_BASIC, DEBUG_VERBOSE


def parse_input(parse_cli=True):
    """
    Parse the input arguments using argparse.

    Args:
        parse_cli (bool): Whether to parse the command-line or not. Default: True
    
    Returns:
        input (argparse.Namespace): Parsed input arguments.
    """

    # Create argument parser
    parser = argparse.ArgumentParser()

    # Add the mesh file that represents the boundary of the target geometry
    parser.add_argument('--temp', type=str, default=None,
                        help="Mesh file representing the boundary of the target geometry")

    # Add the coil mesh file
    parser.add_argument('--coil_mesh_file', type=str,
                        default='none', help="File of the coil mesh")

    # Add the field shape function
    parser.add_argument('--field_shape_function', type=str,
                        default='x', help="Spatial function that defines the field")

    # Add the target gradient strength
    parser.add_argument('--target_gradient_strength', type=float,
                        default=1, help="Target gradient strength in T/m")

    # Add the offset factor for contour levels
    parser.add_argument('--pot_offset_factor', type=float,
                        default=1/2, help="Offset factor for contour levels")

    # Add the target mesh file
    parser.add_argument('--target_mesh_file', type=str,
                        default='none', help="File of the target surface mesh")

    # Add the secondary target mesh file
    parser.add_argument('--secondary_target_mesh_file', type=str,
                        default='none', help="File of the secondary target mesh")

    # Add the secondary target weight
    parser.add_argument('--secondary_target_weight', type=float, default=1, 
                        help="Weight for the secondary target points")

    # Add flag to use only target mesh vertices as target coordinates
    parser.add_argument('--use_only_target_mesh_verts', type=bool, default=False,
                        help="Flag to use only target mesh vertices as target coordinates")

    # Add the stream function source file
    parser.add_argument('--sf_source_file', type=str, default='none',
                        help="File of the already optimized stream function")

    # Add the target field definition file
    parser.add_argument('--target_field_definition_file', type=str,
                        default='none', help="File of the target field definition")

    # Add the target field definition field name
    parser.add_argument('--target_field_definition_field_name', type=str,
                        default='none', help="Field name of the target field definition")

    # Add the stream function optimization method
    parser.add_argument('--sf_opt_method', type=str, default='tikhonov',
                        help="Stream function optimization method")

    # Add the Tikhonov regularization factor
    parser.add_argument('--tikhonov_reg_factor', type=float, default=1,
                        help="Tikhonov regularization factor for the stream function optimization")

    # See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
    # Add the Stream function optimisation method
    parser.add_argument('--minimize_method', type=str, default='SLSQP',
                        help="Stream function minimisation method to use, Default: 'SLSQP'")
    # See: https://docs.scipy.org/doc/scipy/reference/optimize.minimize-slsqp.html
    # Add the Stream function optimisation method parameters
    parser.add_argument('--minimize_method_parameters', type=str, default="{'tol': 1e-6}",
                        help="Parameters for the minimize method parameters. Default: \"{'tol': 1e-6}\"")
    # Add the Stream function optimisation method options.
    parser.add_argument('--minimize_method_options', type=str, default="{'disp': True, 'maxiter' : 100}",
                        help="Parameters for the minimize method options. Default: \"{'disp': True, 'maxiter' : 100}\"")

    # parser.add_argument('--minimize_options', nargs='+', type=float, default=[
    #                    500, 10**10, 1.0e-10, 1.0e-10, 1.0e-10], help="Parameters for the iterative optimization with 'minimize'")

    # Specify whether to save the pre-optimised stream function for later re-use
    parser.add_argument('--sf_dest_file', type=str, default='none',
                        help="File where to save the optimized stream function, for  re-use with 'sf_source_file'.")


    # Add the number of potential levels
    parser.add_argument('--levels', type=int, default=10,
                        help="Number of potential levels")

    # Add the level set method
    parser.add_argument('--level_set_method', type=str, default='primary',
                        help="Method for calculating the level sets")

    """ Unused
    # Add the field type to evaluate
    parser.add_argument('--fieldtype_to_evaluate', type=str, default='field', help="Field type to evaluate")
    """

    # Add flag for cylindrical surface
    parser.add_argument('--surface_is_cylinder_flag', type=bool, default=True,
                        help="Flag for cylindrical surface")

    # Add the circular diameter factor for cylinder parameterization (circular_diameter_factor_cylinder_parameterization)
    parser.add_argument('--circular_diameter_factor', type=float, default=1,
                        help="Circular diameter factor for cylinder parameterization")

    # Add the width in metres of the opening cut for the interconnection of the loops
    parser.add_argument('--interconnection_cut_width', type=float, default=0.01,
                        help="Width in metres of the opening cut for the interconnection of the loops")

    # Add the radius of a spherical target field
    parser.add_argument('--target_region_radius', type=float,
                        default=0.15, help="Radius of a spherical target field")

    # Add the number of target points per dimension within the target region
    parser.add_argument('--target_region_resolution', type=int, default=10,
                        help="Number of target points per dimension within the target region")

    # Add the distance in metres for which crossing lines will be separated along the normal direction of the surface
    parser.add_argument('--normal_shift_length', type=float, default=0.001,
                        help="Distance in metres for which crossing lines will be separated along the normal direction of the surface")

    """ Unused
    # Add the minimal required number of point of a single loop; otherwise loops will be removed
    parser.add_argument('--min_point_loop_number', type=int, default=20,
                        help="Minimal required number of points of a single loop; otherwise loops will be removed")
    """

    # Add the minimal required field contribution (in percent) to the target field; loops that contribute less than that can be deleted
    parser.add_argument('--min_loop_significance', type=int, default=1,
                        help="Minimal required field contribution (in percent) to the target field; loops that contribute less than that can be deleted")

    """ Unused
    # Add additional loop removal criteria which relates to the perimeter to surface ratio of the loop
    parser.add_argument('--area_perimeter_deletion_ratio', type=int, default=5,
                        help="Additional loop removal criteria which relates to the perimeter to surface ratio of the loop")

    # Add the maximum allowed angle of the track of the contours
    parser.add_argument('--max_allowed_angle_within_coil_track', type=int,
                        default=120, help="Maximum allowed angle of the track of the contours")

    # Add the minimum allowed angle of the track of the contours; smaller angles will be converted to straight lines in order to reduce the number of points
    parser.add_argument('--min_allowed_angle_within_coil_track', type=float, default=0.0001,
                        help="Minimum allowed angle of the track of the contours; smaller angles will be converted to straight lines in order to reduce the number of points")

    # Add the minimum relative percentage for which points will be deleted which contribute to segments which are extremely short
    parser.add_argument('--tiny_segment_length_percentage', type=float, default=0,
                        help="Minimum relative percentage for which points will be deleted which contribute to segments which are extremely short")
    """

    # Add the number of refinement iterations of the mesh together with the stream function
    parser.add_argument('--iteration_num_mesh_refinement', type=int, default=0,
                        help="Number of refinement iterations of the mesh together with the stream function")

    # Add the direction (vector) along which the interconnections will be aligned
    parser.add_argument('--b_0_direction', nargs='+', type=float, default=[0, 0, 1], 
                        help="Direction (vector) along which the interconnections will be aligned")

    # Add the directory of the .stl geometry files
    parser.add_argument('--geometry_source_path', type=str, default='Geometry_Data', 
                        help="Directory of the .stl geometry files")

    # Add the output directory
    parser.add_argument('--output_directory', type=str, default=str(Path.cwd()), 
                        help="Output directory")

    # Add the smoothing parameter
    parser.add_argument('--smooth_factor', type=int,
                        default=1, help="Smoothing parameter if tracks should be smoothed (i.e. when > 1).")

    # Add flag to save swept .stl
    parser.add_argument('--save_stl_flag', type=bool, default=True,
                        help="Flag to save swept .stl")

    # Add flag to plot results
    parser.add_argument('--plot_flag', type=bool, default=True,
                        help="Flag to plot results")

    """ Unused
    # Add interconnection_method: Regular or spiral in/out
    parser.add_argument('--interconnection_method', type=str, default='regular',
                        help="Interconnection method: 'regular' or 'spiral' in/out")

    # Add group_interconnection_method: 'straight' or 'crossed'
    parser.add_argument('--group_interconnection_method', type=str, default='crossed',
                        help="Group interconnection method: 'straight' or 'crossed'")
    """

    # Specify or calculate the pcb track width
    parser.add_argument('--skip_calculation_min_winding_distance', type=bool, default=True,
                        help="Flag to skip calculation of minimal winding distance")

    # Add flag to skip post processing
    parser.add_argument('--skip_postprocessing', type=bool, default=False,
                        help="Flag to skip post-processing")

    # Add flag to skip inductance_calculation
    parser.add_argument('--skip_inductance_calculation', type=bool, default=False,
                        help="Flag to skip inductance calculation")

    # Flag to skip the shifting of return paths
    parser.add_argument('--skip_normal_shift', type=bool, default=False,
                        help="Flag to skip the shifting of return paths")

    # Add smoothing parameters regarding the normal shift
    parser.add_argument('--normal_shift_smooth_factors', nargs='+', type=int,
                        default=[2, 3, 2], help="Smoothing parameters regarding the normal shift")

    # Flag to skip the generation of a volumetric (3D) coil body
    parser.add_argument('--skip_sweep', type=bool, default=False,
                        help="Flag to skip the generation of a volumetric (3D) coil body")

    # Flag to generate a rectangular pcb pattern to wrap around a cylinder
    parser.add_argument('--make_cylindrical_pcb', type=bool, default=False,
                        help="Flag to generate a rectangular pcb pattern to wrap around a cylinder")

    # Add the pcb_interconnection_method: 'spiral_in_out' or other options
    parser.add_argument('--pcb_interconnection_method', type=str, default='spiral_in_out',
                        help="Interconnection method for PCB: 'spiral_in_out' or other options")

    # Add the factor of shifting the open ends of the spirals in order to avoid overlaps; in percent
    parser.add_argument('--pcb_spiral_end_shift_factor', type=int, default=10,
                        help="Factor of shifting the open ends of the spirals in order to avoid overlaps; in percent")

    # Add force_cut_selection
    parser.add_argument('--force_cut_selection', nargs='+',
                        default=[], help="Force cut selection")

    # Add the Gauss integration order
    parser.add_argument('--gauss_order', type=int, default=2,
                        help="Gauss integration order")

    # Add flag to set the roi into the geometric center of the mesh
    parser.add_argument('--set_roi_into_mesh_center', type=bool, default=False,
                        help="Flag to set the ROI into the geometric center of the mesh")

    """ Unused
    # In case of pcb layout, specify the track width
    parser.add_argument('--track_width_factor', type=float,
                        default=0.5, help="Track width factor for PCB layout")
    """

    # Add the cross_section_width of the conductor (for the inductance calculation) in metres
    parser.add_argument('--conductor_cross_section_width', type=float, default=0.002,
                        help="Cross-section width of the conductor (for the inductance calculation) in metres")

    # Add the cross_section_height of the conductor (for the inductance calculation) in metres
    parser.add_argument('--conductor_cross_section_height', type=float, default=0.002,
                        help="Cross-section height of the conductor (for the inductance calculation) in metres")

    # Add the conductor conductivity
    parser.add_argument('--specific_conductivity_conductor',
                        type=float, default=0.018e-6, help="Conductor conductivity")

    # Add the thickness of the sheet current density within the stream function representation
    parser.add_argument('--conductor_thickness', type=float, default=0.005,
                        help="Thickness of the sheet current density within the stream function representation")

    # Add the 2D edge points for direct definition of the cross section of the conductor (build circular cut shapes)
    parser.add_argument('--cross_sectional_points', nargs='+', type=float, default=[
                        0, 0], help="2D edge points for direct definition of the cross section of the conductor (build circular cut shapes)")

    # Add the parameters for the generation of the (default) cylindrical mesh
    parser.add_argument('--cylinder_mesh_parameter_list', nargs='+', type=float, default=[
                        0.8, 0.3, 20, 20, 1, 0, 0, 0], help="Parameters for the generation of the (default) cylindrical mesh")

    # Add the parameters for the generation of the (default) planar mesh
    parser.add_argument('--planar_mesh_parameter_list', nargs='+', type=float, default=[
                        0.25, 0.25, 20, 20, 1, 0, 0, 0, 0, 0, 0], help="Parameters for the generation of the (default) planar mesh")

    """ Currently not implemented
    # Add the parameters for the generation of a double cone ("diabolo") shaped mesh
    parser.add_argument('--double_cone_mesh_parameter_list', nargs='+', type=float, default=[
                        0.8, 0.3, 0.3, 0.1, 20, 20, 1, 0, 0, 0], help="Parameters for the generation of a double cone ('diabolo') shaped mesh")

    # Add the parameters for the generation of the (default) circular mesh
    parser.add_argument('--circular_mesh_parameter_list', nargs='+', type=float, default=[
                        0.25, 20, 1, 0, 0, 0, 0, 0, 0], help="Parameters for the generation of the (default) circular mesh")
    """

    # Add the parameters for the generation of the (default) biplanar mesh
    parser.add_argument('--biplanar_mesh_parameter_list', nargs='+', type=float, default=[
                        0.25, 0.25, 20, 20, 1, 0, 0, 0, 0, 0, 0.2], help="Parameters for the generation of the (default) biplanar mesh")

    # Add the parameters for the generation of the (default) biplanar mesh
    parser.add_argument('--debug', type=int, default=0,
                        help=f"Debug verbosity level: 0 = None, {DEBUG_BASIC} = Basic, {DEBUG_VERBOSE} = Verbose")

    # Add the parameters for the generation of the (default) biplanar mesh
    parser.add_argument('--project_name', type=str, default='CoilGen', help=f"Project name for saved data")

    # Add the parameters for the generation of the (default) biplanar mesh
    parser.add_argument('--persistence_dir', type=str, default='debug', help=f"Directory to write persistence data")

    # Add a parameter for the location of the FastHenry2 executable, platform aware
    os_system = platform.system()
    if os_system == "Windows":
        fast_henry_path = 'C:\\Program Files (x86)\\FastFieldSolvers\\FastHenry2\\FastHenry2.exe'
    else: # Assume Linux or equivalent
        fast_henry_path = '/usr/bin/fasthenry'
    parser.add_argument('--fasthenry_bin', type=str, default=fast_henry_path,
                        help=f"Location of the FastHenry2 binary")

    # Parse the input arguments
    if parse_cli == False:
        input = parser.parse_args([])
    else:
        input = parser.parse_args()

    return parser, input


def create_input(dictionary):
    parser, input = parse_input(False)
    for key, value in dictionary.items():
        if hasattr(input, key):
            setattr(input, key, value)
        else:
            raise KeyError(f"Attribute '{key}' is not a valid parameter.")

    return parser, input
