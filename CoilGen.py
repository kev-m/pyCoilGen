# System imports
import sys
from pathlib import Path

# Logging
import logging

# Local imports
# Import the required modules from sub_functions directory
from sub_functions.constants import *
from sub_functions.data_structures import DataStructure, CoilSolution, OptimisationParameters

# From original project
from sub_functions.read_mesh import read_mesh
from sub_functions.parse_input import parse_input, create_input
from sub_functions.split_disconnected_mesh import split_disconnected_mesh
from sub_functions.refine_mesh import refine_mesh_delegated as refine_mesh
from sub_functions.parameterize_mesh import parameterize_mesh
from sub_functions.define_target_field import define_target_field
# from sub_functions.temp_evaluation import temp_evaluation
from sub_functions.calculate_one_ring_by_mesh import calculate_one_ring_by_mesh
from sub_functions.calculate_basis_functions import calculate_basis_functions
from sub_functions.calculate_sensitivity_matrix import calculate_sensitivity_matrix
"""
from calculate_gradient_sensitivity_matrix import calculate_gradient_sensitivity_matrix
from calculate_resistance_matrix import calculate_resistance_matrix
from stream_function_optimization import stream_function_optimization
from calc_potential_levels import calc_potential_levels
from calc_contours_by_triangular_potential_cuts import calc_contours_by_triangular_potential_cuts
from process_raw_loops import process_raw_loops
from find_minimal_contour_distance import find_minimal_contour_distance
from topological_loop_grouping import topological_loop_grouping
from calculate_group_centers import calculate_group_centers
from interconnect_within_groups import interconnect_within_groups
from interconnect_among_groups import interconnect_among_groups
from shift_return_paths import shift_return_paths
from generate_cylindrical_pcb_print import generate_cylindrical_pcb_print
from create_sweep_along_surface import create_sweep_along_surface
from calculate_inductance_by_coil_layout import calculate_inductance_by_coil_layout
from evaluate_field_errors import evaluate_field_errors
from calculate_gradient import calculate_gradient
from load_preoptimized_data import load_preoptimized_data
"""


def CoilGen(log, input=None):
    # Create optimized coil finished coil layout
    # Autor: Philipp Amrein, University Freiburg, Medical Center, Radiology, Medical Physics
    # 5.10.2021

    # the following external functions were used in modified form:
    # intreparc@John D'Errico (2010), @matlabcentral/fileexchange
    # The non-cylindrical parameterization is taken from "matlabmesh @ Ryan
    # Schmidt rms@dgp.toronto.edu" based on desbrun et al (2002), "Intrinsic Parameterizations of {Surface} Meshes",
    # NS (2021). Curve intersections (https://www.mathworks.com/matlabcentral/fileexchange/22441-curve-intersections), MATLAB Central File Exchange.

    # Parse the input variables
    if type(input) is dict:
        # DEBUG
        if input['debug'] >= DEBUG_VERBOSE:
            log.debug(" - converting input dict to input type.")
        input_parser, input_args = create_input(input)
    elif input is None:
        input_parser, input_args = parse_input(input)
    else:
        input_args = input

    # Print the input variables
    # DEBUG
    if input['debug'] >= DEBUG_VERBOSE:
        log.debug('Parse inputs: %s', input_args)

    solution = CoilSolution()

    if input_args.sf_source_file == 'none':
        # Read the input mesh
        print('Load geometry:')
        coil_mesh, target_mesh, secondary_target_mesh = read_mesh(input_args)
        # log.debug(" coil_mesh.faces: %s", coil_mesh.faces)

        if input['debug'] > DEBUG_VERBOSE:
            coil_mesh.display()


        # Split the mesh and the stream function into disconnected pieces
        print('Split the mesh and the stream function into disconnected pieces.')
        coil_parts = split_disconnected_mesh(coil_mesh)

        # Upsample the mesh density by subdivision
        print('Upsample the mesh by subdivision:')
        coil_parts = refine_mesh(coil_parts, input_args)
        # log.debug("coil_parts: %s", coil_parts)

        # Parameterize the mesh
        print('Parameterize the mesh:')
        coil_parts = parameterize_mesh(coil_parts, input_args)
        solution.coil_parts = coil_parts

        # Define the target field
        print('Define the target field:')
        target_field, is_suppressed_point = define_target_field(
            coil_parts, target_mesh, secondary_target_mesh, input_args)
        solution.target_field = target_field
        solution.is_suppressed_point = is_suppressed_point

        # Evaluate the temp data; check whether precalculated values can be used from previous iterations
        # print('Evaluate the temp data:')
        # input_args = temp_evaluation(solution, input_args, target_field)

        # Find indices of mesh nodes for one ring basis functions
        print('Calculate mesh one ring:')
        coil_parts = calculate_one_ring_by_mesh(solution, coil_parts, input_args)

        # Create the basis function container which represents the current density
        print('Create the basis function container which represents the current density:')
        coil_parts = calculate_basis_functions(solution, coil_parts, input_args)

        # Calculate the sensitivity matrix Cn
        print('Calculate the sensitivity matrix:')
        coil_parts = calculate_sensitivity_matrix(solution, coil_parts, target_field, input_args)

        # WIP
        solution.coil_parts = coil_parts
        return solution

        # Calculate the gradient sensitivity matrix Gn
        print('Calculate the gradient sensitivity matrix:')
        coil_parts = calculate_gradient_sensitivity_matrix(coil_parts, target_field, input_args)

        # Calculate the resistance matrix Rmn
        print('Calculate the resistance matrix:')
        coil_parts = calculate_resistance_matrix(coil_parts, input_args)

        # Optimize the stream function toward target field and further constraints
        print('Optimize the stream function toward target field and secondary constraints:')
        coil_parts, combined_mesh, sf_b_field = stream_function_optimization(coil_parts, target_field, input_args)

    else:
        # Load the preoptimized data
        print('Load preoptimized data:')
        coil_parts, _, _, combined_mesh, sf_b_field, target_field, is_suppressed_point = load_preoptimized_data(
            input_args)

    # Calculate the potential levels for the discretization
    print('Calculate the potential levels for the discretization:')
    coil_parts, primary_surface_ind = calc_potential_levels(coil_parts, combined_mesh, input_args)

    # Generate the contours
    print('Generate the contours:')
    coil_parts = calc_contours_by_triangular_potential_cuts(coil_parts)

    # Process contours
    print('Process contours: Evaluate loop significance')
    coil_parts = process_raw_loops(coil_parts, input_args, target_field)

    if not input_args['skip_postprocessing']:
        # Find the minimal distance between the contour lines
        print('Find the minimal distance between the contour lines:')
        coil_parts = find_minimal_contour_distance(coil_parts, input_args)

        # Group the contour loops in topological order
        print('Group the contour loops in topological order:')
        coil_parts = topological_loop_grouping(coil_parts, input_args)

        # Calculate center locations of groups
        print('Calculate center locations of groups:')
        coil_parts = calculate_group_centers(coil_parts)

        # Interconnect the single groups
        print('Interconnect the single groups:')
        coil_parts = interconnect_within_groups(coil_parts, input_args)

        # Interconnect the groups to a single wire path
        print('Interconnect the groups to a single wire path:')
        coil_parts = interconnect_among_groups(coil_parts, input_args)

        # Connect the groups and shift the return paths over the surface
        print('Shift the return paths over the surface:')
        coil_parts = shift_return_paths(coil_parts, input_args)

        # Create Cylindrical PCB Print
        print('Create PCB Print:')
        coil_parts = generate_cylindrical_pcb_print(coil_parts, input_args)

        # Create Sweep Along Surface
        print('Create sweep along surface:')
        coil_parts = create_sweep_along_surface(coil_parts, input_args)

    # Calculate the inductance by coil layout
    print('Calculate the inductance by coil layout:')
    coil_inductance, radial_lumped_inductance, axial_lumped_inductance, radial_sc_inductance, axial_sc_inductance = calculate_inductance_by_coil_layout(
        coil_parts, input_args)

    # Evaluate the field errors
    print('Evaluate the field errors:')
    field_errors, _, _ = evaluate_field_errors(coil_parts, target_field, input_args)

    # Calculate the gradient
    print('Calculate the gradient:')
    coil_gradient = calculate_gradient(coil_parts, target_field, input_args)

    return coil_parts, combined_mesh, sf_b_field, target_field, coil_inductance, radial_lumped_inductance, axial_lumped_inductance, radial_sc_inductance, axial_sc_inductance, field_errors, coil_gradient, is_suppressed_point


if __name__ == "__main__":
    # Set up logging
    log = logging.getLogger(__name__)
    logging.basicConfig(level=logging.DEBUG)
    # logging.basicConfig(level=logging.INFO)

    # DEBUG:split_disconnected_mesh:Shape: (400, 6), (3, 441)
    input = {'debug': DEBUG_VERBOSE, 'coil_mesh_file': 'create cylinder mesh'}  # Runs OK

    # split_disconnected_mesh.py", line 60, in split_disconnected_mesh
    # DEBUG:split_disconnected_mesh:Shape: (800, 3), (3, 441)
    # arg_list = ['--coil_mesh_file', 'create planary mesh'] # IndexError: index 441 is out of bounds for axis 1 with size 441
    # arg_list = ['--coil_mesh_file', 'create bi-planary mesh'] # IndexError: index 882 is out of bounds for axis 1 with size 882
    # DEBUG:split_disconnected_mesh:Shape: (124, 3), (3, 64)
    # arg_list = ['--coil_mesh_file', 'closed_cylinder_length_300mm_radius_150mm.stl'] # IndexError: index 64 is out of bounds for axis 1 with size 64
    # arg_list = ['--coil_mesh_file', 'dental_gradient_ccs_single_low.stl'] # IndexError: index 114 is out of bounds for axis 1 with size 114
    solution = CoilGen(log, input=input)
