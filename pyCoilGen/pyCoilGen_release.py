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

# Set up logging
log = logging.getLogger(__name__)


def pyCoilGen(log, input_args=None):
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

            if get_level() > DEBUG_VERBOSE:
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

        # Calculate the potential levels for the discretization
        print('Calculate the potential levels for the discretization:')
        timer.start()
        coil_parts, primary_surface_ind = calc_potential_levels(coil_parts, combined_mesh, input_args)  # 09
        timer.stop()
        solution.primary_surface_ind = primary_surface_ind
        runpoint_tag = '09'

        # Generate the contours
        print('Generate the contours:')
        timer.start()
        coil_parts = calc_contours_by_triangular_potential_cuts(coil_parts)  # 10
        timer.stop()
        runpoint_tag = '10'

        #####################################################
        # Visualisation
        if get_level() > DEBUG_NONE:
            for part_index in range(len(coil_parts)):
                coil_part = coil_parts[part_index]
                coil_mesh = coil_part.coil_mesh

                visualize_compare_contours(coil_mesh.uv, 800, f'{image_dir}/10_{project_name}_contours_{part_index}_p.png',
                                           coil_part.contour_lines)
        #
        #####################################################

        # Process contours
        print('Process contours: Evaluate loop significance')
        timer.start()
        coil_parts = process_raw_loops(coil_parts, input_args, target_field)  # 11
        timer.stop()
        runpoint_tag = '11'

        if not input_args.skip_postprocessing:
            # Find the minimal distance between the contour lines
            print('Find the minimal distance between the contour lines:')
            timer.start()
            coil_parts = find_minimal_contour_distance(coil_parts, input_args)  # 12
            timer.stop()
            runpoint_tag = '12'

            # Group the contour loops in topological order
            print('Group the contour loops in topological order:')
            timer.start()
            coil_parts = topological_loop_grouping(coil_parts)  # 13
            timer.stop()
            runpoint_tag = '13'
            for index, coil_part in enumerate(coil_parts):
                print(f'  -- Part {index} has {len(coil_part.groups)} topological groups')

            # Calculate center locations of groups
            print('Calculate center locations of groups:')
            timer.start()
            coil_parts = calculate_group_centers(coil_parts)  # 14
            timer.stop()
            runpoint_tag = '14'

            #####################################################
            # Visualisation
            if get_level() > DEBUG_NONE:
                for part_index in range(len(coil_parts)):
                    coil_part = coil_parts[part_index]
                    coil_mesh = coil_part.coil_mesh
                    c_group_centers = coil_part.group_centers

                    visualize_compare_contours(coil_mesh.uv, 800, f'{image_dir}/14_{project_name}_contour_centres_{part_index}_p.png',
                                               coil_part.contour_lines, c_group_centers.uv)
            #
            #####################################################

            # Interconnect the single groups
            print('Interconnect the single groups:')
            timer.start()
            coil_parts = interconnect_within_groups(coil_parts, input_args)  # 15
            timer.stop()
            runpoint_tag = '15'

            # Interconnect the groups to a single wire path
            print('Interconnect the groups to a single wire path:')
            timer.start()
            coil_parts = interconnect_among_groups(coil_parts, input_args)  # 16
            timer.stop()
            runpoint_tag = '16'

            #####################################################
            # Visualisation
            if get_level() > DEBUG_NONE:
                for index1 in range(len(coil_parts)):
                    c_part = coil_parts[index1]
                    c_wire_path = c_part.wire_path

                    visualize_vertex_connections(
                        c_wire_path.uv.T, 800, f'{image_dir}/16_{project_name}_wire_path2_uv_{index1}_p.png')
            #
            #####################################################

            # Connect the groups and shift the return paths over the surface
            print('Shift the return paths over the surface:')
            timer.start()
            coil_parts = shift_return_paths(coil_parts, input_args)  # 17
            timer.stop()
            runpoint_tag = '17'

            # Create Cylindrical PCB Print
            print('Create PCB Print:')
            timer.start()
            coil_parts = generate_cylindrical_pcb_print(coil_parts, input_args)  # 18
            timer.stop()
            runpoint_tag = '18'

            # Create Sweep Along Surface
            print('Create sweep along surface:')
            timer.start()
            coil_parts = create_sweep_along_surface(coil_parts, input_args)
            timer.stop()
            runpoint_tag = '19'

        # Calculate the inductance by coil layout
        print('Calculate the inductance by coil layout:')
        # coil_inductance, radial_lumped_inductance, axial_lumped_inductance, radial_sc_inductance, axial_sc_inductance
        timer.start()
        solution = calculate_inductance_by_coil_layout(solution, input_args)
        timer.stop()
        runpoint_tag = '20'

        # Evaluate the field errors
        print('Evaluate the field errors:')
        timer.start()
        coil_parts, solution_errors = evaluate_field_errors(
            coil_parts, input_args, solution.target_field, solution.sf_b_field)
        timer.stop()
        solution.solution_errors = solution_errors
        log.info("Layout error: Mean: %f, Max: %f",
                 solution_errors.field_error_vals.mean_rel_error_layout_vs_target,
                 solution_errors.field_error_vals.max_rel_error_layout_vs_target
                 )
        runpoint_tag = '21'

        # Calculate the gradient
        print('Calculate the gradient:')
        timer.start()
        coil_gradient = calculate_gradient(coil_parts, input_args, target_field)
        timer.stop()
        solution.coil_gradient = coil_gradient
        runpoint_tag = '22'

        # Export data
        print('Exporting data:')
        timer.start()
        export_data(solution)
        timer.stop()
        runpoint_tag = '23'

        # Finally, save the completed solution.
        runpoint_tag = 'final'
        print(f'Solution saved to "{save(persistence_dir, project_name, runpoint_tag, solution)}"')

        timer.stop()
    except Exception as e:
        log.error("Caught exception: %s", e)
        save(persistence_dir, project_name, f'{runpoint_tag}_exception', solution)
        raise e
    return solution
