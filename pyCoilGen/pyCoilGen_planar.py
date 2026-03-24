# Logging
import logging

# System imports
import numpy as np
from os import makedirs

# Logging
import logging


# Local imports
from .sub_functions.constants import *
from .sub_functions.data_structures import CoilSolution
import matplotlib.pyplot as plt
import trimesh
import matplotlib.tri as mtri

# For visualisation
from .helpers.visualisation import visualize_vertex_connections, visualize_compare_contours

# For timing
from .helpers.timing import Timing

# For saving Pickle files
from .helpers.persistence import save, save_preoptimised_data

# From original project
from .sub_functions.read_mesh import read_mesh
from .sub_functions.parse_input import parse_input, create_input
from .sub_functions.split_disconnected_mesh import split_disconnected_mesh
from .sub_functions.refine_mesh import refine_mesh_delegated as refine_mesh
# from .sub_functions.refine_mesh import refine_mesh # Broken
from .sub_functions.parameterize_mesh import parameterize_mesh
from .sub_functions.define_target_field import define_target_field
# from .sub_functions.temp_evaluation import temp_evaluation
from .sub_functions.calculate_one_ring_by_mesh import calculate_one_ring_by_mesh
from .sub_functions.calculate_basis_functions import calculate_basis_functions
from .sub_functions.calculate_sensitivity_matrix import calculate_sensitivity_matrix
from .sub_functions.calculate_gradient_sensitivity_matrix import calculate_gradient_sensitivity_matrix
from .sub_functions.calculate_resistance_matrix import calculate_resistance_matrix
from .sub_functions.stream_function_optimization import stream_function_optimization
from .sub_functions.calc_potential_levels import calc_potential_levels
from .sub_functions.calc_contours_by_triangular_potential_cuts import calc_contours_by_triangular_potential_cuts
from .sub_functions.process_raw_loops import process_raw_loops
from .sub_functions.find_minimal_contour_distance import find_minimal_contour_distance
from .sub_functions.topological_loop_grouping import topological_loop_grouping
from .sub_functions.calculate_group_centers import calculate_group_centers
from .sub_functions.interconnect_within_groups import interconnect_within_groups
from .sub_functions.interconnect_among_groups import interconnect_among_groups
from .sub_functions.shift_return_paths import shift_return_paths
from .sub_functions.generate_cylindrical_pcb_print import generate_cylindrical_pcb_print
from .sub_functions.create_sweep_along_surface import create_sweep_along_surface
from .sub_functions.calculate_inductance_by_coil_layout import calculate_inductance_by_coil_layout
from .sub_functions.load_preoptimized_data import load_preoptimized_data
from .sub_functions.evaluate_field_errors import evaluate_field_errors
from .sub_functions.calculate_gradient import calculate_gradient
from .sub_functions.export_data import export_data, check_exporter_help
from .sub_functions.extract_wire_paths import extract_wire_paths


# Set up logging
log = logging.getLogger(__name__)


def pyCoilGen_planar(log, input_args=None):
    """
    Generate optimized planar coil layouts using stream function optimization.
    This module implements the pyCoilGen algorithm for designing planar gradient coils.
    It reads a mesh geometry, optimizes a stream function to match a target magnetic field,
    and extracts wire patterns for coil fabrication.
    The workflow consists of the following main steps:
    1. Read and preprocess the input mesh (refinement, parameterization)
    2. Define target magnetic field specifications
    3. Calculate basis functions and sensitivity matrices
    4. Optimize stream function using constrained optimization
    5. Discretize the stream function into wire contours
    6. Extract and interconnect wire paths
    Args:
        log (logging.Logger): Logger instance for recording execution progress and errors.
        input_args (dict, optional): Dictionary or parsed input arguments containing:
            - project_name (str): Name of the coil project
            - debug (int): Debug verbosity level
            - persistence_dir (str): Directory for saving intermediate results
            - output_directory (str): Directory for saving output files
            - sf_source_file (str): Path to preoptimized stream function file ('none' to skip)
            - sf_dest_file (str): Path to save preoptimized data ('none' to skip)
            - target_field_weighting (bool): Whether to apply radial weighting to target field
            - normal_shift_length (float): Wire width parameter
            - interconnection_cut_width (float): Wire thickness parameter
            Defaults to None, which triggers interactive input parsing.
    Returns:
        list or CoilSolution: List of CoilPart objects containing optimized coil geometry
            and wire patterns. Returns None if mesh loading fails. If exporter help is
            requested, returns empty CoilSolution object.
    Raises:
        Exception: Re-raises any exceptions that occur during optimization with error
            context indicating the last successful runpoint.
    Notes:
        - Adapted from original pyCoilGen with customization for planar gradient coils
        - External dependencies: NumPy, Trimesh, Matplotlib
        - Timing information is logged for performance profiling
        - Intermediate results can be saved/loaded for iterative design
    """

    # Create optimized coil finished coil layout
    # Author: Philipp Amrein, University Freiburg, Medical Center, Radiology, Medical Physics
    # 5.10.2021

    # The following external functions were used in modified form:
    # intreparc@John D'Errico (2010), @matlabcentral/fileexchange
    # The non-cylindrical parameterization is taken from "matlabmesh @ Ryan Schmidt rms@dgp.toronto.edu"
    # based on desbrun et al (2002), "Intrinsic Parameterizations of {Surface} Meshes", NS (2021).
    # Curve intersections (https://www.mathworks.com/matlabcentral/fileexchange/22441-curve-intersections),
    # MATLAB Central File Exchange.



    timer = Timing()
    timer.start()

    # Parse the input variables
    if type(input_args) is dict:
        try:
            if input_args['debug'] >= DEBUG_VERBOSE:
                log.debug(" - converting input dict to input type.")
        except KeyError:
            pass
        input_parser, input_args = create_input(input_args)
    elif input_args is None:
        input_parser, input_args = parse_input(input_args)
    else:
        input_args = input_args

    set_level(input_args.debug)

    project_name = f'{input_args.project_name}'
    persistence_dir = input_args.persistence_dir
    image_dir = input_args.output_directory

    # Create directories if they do not exist
    makedirs(persistence_dir, exist_ok=True)
    makedirs(image_dir, exist_ok=True)

    # Print the input variables
    # DEBUG
    if get_level() >= DEBUG_VERBOSE:
        log.debug('Parse inputs: %s', input_args)


    solution = CoilSolution()
    solution.input_args = input_args

    if check_exporter_help(input_args):
        return solution

    try:
        runpoint_tag = 'test'

        if input_args.sf_source_file == 'none':
            # Read the input mesh
            print('Load geometry:')
            coil_mesh, target_mesh, secondary_target_mesh = read_mesh(input_args)  # 01

            if coil_mesh is None:
                log.info("No coil mesh, exiting.")
                timer.stop()
                return None

            if get_level() >= DEBUG_VERBOSE:
                log.debug(" -- vertices shape: %s", coil_mesh.get_vertices().shape)  # (264,3)
                log.debug(" -- faces shape: %s", coil_mesh.get_faces().shape)  # (480,3)

            if get_level() >= DEBUG_VERBOSE:
                log.debug(" coil_mesh.vertex_faces: %s", coil_mesh.trimesh_obj.vertex_faces[0:10])

            if get_level() > DEBUG_VERBOSE:
                coil_mesh.display()

            # Split the mesh and the stream function into disconnected pieces
            print('Split the mesh and the stream function into disconnected pieces.')
            timer.start()
            coil_parts = split_disconnected_mesh(coil_mesh)  # 00
            timer.stop()
            solution.coil_parts = coil_parts
            runpoint_tag = '00'

            # Upsample the mesh density by subdivision
            print('Upsample the mesh by subdivision:')
            timer.start()
            coil_parts = refine_mesh(coil_parts, input_args)  # 01
            timer.stop()
            runpoint_tag = '01'

            # Parameterize the mesh
            print('Parameterize the mesh:')
            timer.start()
            coil_parts = parameterize_mesh(coil_parts, input_args)  # 02
            timer.stop()
            runpoint_tag = '02'

            if get_level() >= DEBUG_VERBOSE:
                # Export data
                print('Exporting initial data:')
                timer.start()
                export_data(solution)
                timer.stop()

            # Define the target field
            print('Define the target field:')
            timer.start()
            target_field, is_suppressed_point = define_target_field(
                coil_parts, target_mesh, secondary_target_mesh, input_args)
            timer.stop()
            # Can set target field weights here 
            if input_args.target_field_weighting == True:
                coords = target_field.coords
                x = coords[0, :]
                y = coords[1, :]
                z = coords[2, :]
                r = np.sqrt(x**2 + y**2 + z**2)
                R = np.max(r)
                # weights = 1 / (1 + (r/0.01)**4)
                weights = (r / R)**4
                target_field.weights = weights

            solution.target_field = target_field
            solution.is_suppressed_point = is_suppressed_point
            runpoint_tag = '02b'

            if get_level() >= DEBUG_VERBOSE:
                log.debug(" -- target_field.b shape: %s", target_field.b.shape)  # (3, 257)
                log.debug(" -- target_field.coords shape: %s", target_field.coords.shape)  # (3, 257)
                log.debug(" -- target_field.weights shape: %s", target_field.weights.shape)  # (257,)

            # Evaluate the temp data; check whether precalculated values can be used from previous iterations
            # print('Evaluate the temp data:')
            # input_args = temp_evaluation(solution, input_args, target_field)

            # Find indices of mesh nodes for one ring basis functions
            print('Calculate mesh one ring:')
            timer.start()
            coil_parts = calculate_one_ring_by_mesh(coil_parts)  # 03
            timer.stop()
            runpoint_tag = '03'

            # Create the basis function container which represents the current density
            print('Create the basis function container which represents the current density:')
            timer.start()
            coil_parts = calculate_basis_functions(coil_parts)  # 04
            timer.stop()
            runpoint_tag = '04'

            # Calculate the sensitivity matrix Cn
            print('Calculate the sensitivity matrix:')
            timer.start()
            coil_parts = calculate_sensitivity_matrix(coil_parts, target_field, input_args)  # 05
            timer.stop()
            runpoint_tag = '05'
            # Print the condition number of the sensitivity matrix for the first coil part
            if get_level() >= DEBUG_VERBOSE:
                for part_ind in range(len(coil_parts)):
                    coil_part = coil_parts[part_ind]
                    sensitivity_matrix = coil_part.sensitivity_matrix
                    cond_number = np.linalg.cond(sensitivity_matrix.reshape(3, -1))
                    log.info("Condition number of sensitivity matrix for part %d: %e", part_ind, cond_number)




            # Calculate the gradient sensitivity matrix Gn
            print('Calculate the gradient sensitivity matrix:')
            timer.start()
            coil_parts = calculate_gradient_sensitivity_matrix(coil_parts, target_field, input_args)  # 06
            timer.stop()
            runpoint_tag = '06'

            # Calculate the resistance matrix Rmn
            print('Calculate the resistance matrix:')
            timer.start()
            coil_parts = calculate_resistance_matrix(coil_parts, input_args)  # 07
            timer.stop()
            runpoint_tag = '07'

            # Optimize the stream function toward target field and further constraints
            print('Optimize the stream function toward target field and secondary constraints:')
            timer.start()
            coil_parts, combined_mesh, sf_b_field = stream_function_optimization(coil_parts, target_field, input_args)
            timer.stop()
            solution.combined_mesh = combined_mesh
            solution.sf_b_field = sf_b_field
            runpoint_tag = '08'

            if input_args.sf_dest_file != 'none':
                print('Persist pre-optimised data:')
                save_preoptimised_data(solution)

        else:
            # Load the preoptimised data
            print('Load pre-optimised data:')
            timer.start()
            solution = load_preoptimized_data(input_args)
            timer.stop()
            coil_parts = solution.coil_parts
            combined_mesh = solution.combined_mesh
            target_field = solution.target_field


        # Customized workflow for planar gradient design from here on, which can be adapted by the user. The following steps are based on the original workflow of pyCoilGen, but can be modified as needed.
        if get_level() >= DEBUG_VERBOSE:
            psis = coil_parts[0].stream_function
            plt.figure()
            plt.hist(psis, bins=50)
            plt.title("Stream function values distribution")
            plt.xlabel("Stream function value")
            plt.ylabel("Frequency")
            plt.show()



        # Calculate the potential levels for the discretization
        print('Calculate the potential levels for the discretization:')
        timer.start()
        coil_parts, primary_surface_ind = calc_potential_levels(coil_parts, combined_mesh, input_args)  # 09
        timer.stop()
        solution.primary_surface_ind = primary_surface_ind
        runpoint_tag = '09'

        if get_level() >= DEBUG_VERBOSE:
            for part_ind in range(len(coil_parts)):
                coil_part = coil_parts[part_ind]
                log.info("Part %d: potential levels: %s", part_ind, coil_part.potential_level_list)
                mesh = coil_parts[part_ind].coil_mesh  # Assuming you have one coil part
                verts = mesh.v    # Nx3
                faces = mesh.f    # Mx3
                psis = coil_parts[part_ind].stream_function
                # Compute triangle centers
                triang = mtri.Triangulation(verts[:,0], verts[:,1], faces)

                plt.figure(figsize=(6,6))
                plt.tricontourf(triang, psis, levels= coil_parts[part_ind].potential_level_list, cmap='jet')
                plt.colorbar(label="Stream function ψ")
                plt.axis('equal')
                plt.title("Streamfunction on coil surface")
                plt.show()

        # Extract wire patterns

        for part in coil_parts:
            log.info("Extracting wire paths for part with %d vertices and %d faces", part.coil_mesh.get_vertices().shape[0], part.coil_mesh.get_faces().shape[0])
            part.wire_loops = extract_wire_paths(part.coil_mesh.get_vertices(), part.coil_mesh.get_faces(), part.stream_function, 
                                                part.potential_level_list, wire_width = input_args.normal_shift_length, wire_thickness=input_args.interconnection_cut_width, display_debug_plots = True)



    except Exception as e:
        log.error("An error occurred at runpoint %s: %s", runpoint_tag, str(e))
        raise e


    return coil_parts
    





