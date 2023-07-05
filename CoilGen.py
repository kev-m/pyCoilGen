# System imports
import sys
from pathlib import Path
import numpy as np

# Logging
import logging

# Debug and development checking imports
from helpers.extraction import get_element_by_name, load_matlab, get_and_show_debug
from helpers.visualisation import compare, compare_contains, visualize_vertex_connections, visualize_multi_connections, visualize_compare_vertices
from sub_functions.data_structures import Mesh

# Local imports
from sub_functions.constants import *
from sub_functions.data_structures import DataStructure, CoilSolution, OptimisationParameters

# From original project
from sub_functions.read_mesh import read_mesh
from sub_functions.parse_input import parse_input, create_input
from sub_functions.split_disconnected_mesh import split_disconnected_mesh
from sub_functions.refine_mesh import refine_mesh_delegated as refine_mesh
# from sub_functions.refine_mesh import refine_mesh # Broken
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

    set_level(input_args.debug)

    ######################################################################################
    # DEVELOPMENT: Remove this
    if input_args.coil_mesh_file == 'create cylinder mesh':
        log.debug(" Loading comparison data from generate_halbch_gradient_system")
        mat_contents = load_matlab('debug/generate_halbch_gradient_system')
        matlab_data = mat_contents['x_channel']
    else:
        log.debug(" Loading comparison data from result_y_gradient")
        mat_contents = load_matlab('debug/result_y_gradient')
        matlab_data = mat_contents['coil_layouts']
    m_faces = get_element_by_name(matlab_data, 'out.coil_parts[0].coil_mesh.faces')-1
    m_vertices = get_element_by_name(matlab_data, 'out.coil_parts[0].coil_mesh.vertices')
    log.debug(" m_faces shape: %s", m_faces.shape)
    log.debug(" m_vertices shape: %s", m_vertices.shape)

    # Mesh parameterisation
    m_v = get_and_show_debug(matlab_data, 'out.coil_parts[0].coil_mesh.v', False)
    m_fn = get_and_show_debug(matlab_data, 'out.coil_parts[0].coil_mesh.fn', False)
    m_n = get_and_show_debug(matlab_data, 'out.coil_parts[0].coil_mesh.n')

    # Sanity check
    assert (compare(m_vertices, m_v))

    m_uv = get_and_show_debug(matlab_data, 'out.coil_parts[0].coil_mesh.uv')
    m_boundary_x = get_and_show_debug(matlab_data, 'out.coil_parts[0].coil_mesh.boundary')-1
    log.debug(" m_boundary_x: %s", m_boundary_x.shape)
    m_boundary_points = m_boundary_x[0].shape[0]
    m_boundary = np.ndarray((2, m_boundary_points), dtype=int)
    m_boundary[0] = m_boundary_x[0].reshape((m_boundary_points))
    m_boundary[1] = m_boundary_x[1].reshape((m_boundary_points))

    # Target field
    # b, coords, weights, target_field_group_inds, target_gradient_dbdxyz
    m_target_field = get_and_show_debug(matlab_data, 'out.target_field')
    log.debug("m_target_field :%s", m_target_field._fieldnames)
    m_tf_b = m_target_field.b
    m_tf_coords = m_target_field.coords
    m_tf_weights = m_target_field.weights
    m_tf_target_field_group_inds = m_target_field.target_field_group_inds
    m_tf_target_gradient_dbdxyz = m_target_field.target_gradient_dbdxyz

    # One Ring List
    m_c_part = get_and_show_debug(matlab_data, 'out.coil_parts[0]')
    m_or_one_ring_list = m_c_part.one_ring_list - 1
    # Transpose the entries
    for index in range(len(m_or_one_ring_list)):
        m_or_one_ring_list[index] = m_or_one_ring_list[index].T
    m_or_node_triangles = m_c_part.node_triangles - 1
    m_or_node_triangle_mat = m_c_part.node_triangle_mat

    # END of Remove this
    ######################################################################################

    # Print the input variables
    # DEBUG
    if get_level() >= DEBUG_VERBOSE:
        log.debug('Parse inputs: %s', input_args)

    solution = CoilSolution()

    if input_args.sf_source_file == 'none':
        # Read the input mesh
        print('Load geometry:')
        coil_mesh, target_mesh, secondary_target_mesh = read_mesh(input_args)
        if get_level() > DEBUG_VERBOSE:
            log.debug(" coil_mesh.vertex_faces: %s", coil_mesh.trimesh_obj.vertex_faces[0:10])

        assert (compare(coil_mesh.get_faces(), m_faces))
        assert (compare(coil_mesh.get_vertices(), m_vertices))

        if get_level() > DEBUG_VERBOSE:
            coil_mesh.display()

        # Split the mesh and the stream function into disconnected pieces
        print('Split the mesh and the stream function into disconnected pieces.')
        coil_parts = split_disconnected_mesh(coil_mesh)

        # Upsample the mesh density by subdivision
        print('Upsample the mesh by subdivision:')
        coil_parts = refine_mesh(coil_parts, input_args)
        # log.debug("coil_parts: %s", coil_parts)

        assert (compare(coil_parts[0].coil_mesh.get_faces(), m_faces))
        # coil_parts[0].coil_mesh.display()

        # Parameterize the mesh
        print('Parameterize the mesh:')
        coil_parts = parameterize_mesh(coil_parts, input_args)
        solution.coil_parts = coil_parts

        ######################################################
        # Verify: v, fn, n, boundary, uv
        coil_mesh = coil_parts[0].coil_mesh
        assert (compare(coil_mesh.v, m_v))      # Pass
        assert (compare(coil_mesh.fn, m_fn))    # Pass
        assert (compare(coil_mesh.n, m_n, double_tolerance=0.1))      # Pass only at 0.1

        # Plot the two boundaries and see the difference
        if get_level() >= DEBUG_VERBOSE:
            visualize_vertex_connections(coil_mesh.v, 800, 'images/uv1_coil_mesh_boundary.png', coil_mesh.boundary)
            visualize_vertex_connections(coil_mesh.v, 800, 'images/uv1_m_boundary.png', m_boundary)

        if get_level() > DEBUG_VERBOSE:
            log.debug(" coil_mesh.boundary: %s", coil_mesh.boundary)
            log.debug(" m_boundary: %s", m_boundary)
        # Question: Does order matter?
        assert (compare_contains(coil_mesh.boundary, m_boundary))  # Pass

        # Plot the two UV and see the difference
        if get_level() >= DEBUG_VERBOSE:
            visualize_compare_vertices(m_uv, coil_mesh.uv, 800, 'images/uvdiff_m_uv.png')
        assert (compare(coil_mesh.uv, m_uv, 0.0001))    # Pass

        # Define the target field
        print('Define the target field:')
        target_field, is_suppressed_point = define_target_field(
            coil_parts, target_mesh, secondary_target_mesh, input_args)
        solution.target_field = target_field
        solution.is_suppressed_point = is_suppressed_point

        #####################################################
        # Verify:  b, coords, weights, target_field_group_inds, target_gradient_dbdxyz
        assert (compare(target_field.b, m_tf_b))               # Pass
        assert (compare(target_field.weights, m_tf_weights))   # Pass
        assert (compare(target_field.coords, m_tf_coords))     # Pass
        assert (compare(target_field.target_field_group_inds, m_tf_target_field_group_inds))  # Pass
        assert (compare(target_field.target_gradient_dbdxyz, m_tf_target_gradient_dbdxyz))  # Pass
        #####################################################

        # Evaluate the temp data; check whether precalculated values can be used from previous iterations
        # print('Evaluate the temp data:')
        # input_args = temp_evaluation(solution, input_args, target_field)


        ###########################################################
        # DEBUG
        if get_level() >= DEBUG_VERBOSE:
            print("m_or_one_ring_list[0:3]:")
            print(m_or_one_ring_list[0])
            print(m_or_one_ring_list[1])
            print(m_or_one_ring_list[2])
        #
        ###########################################################

        # Find indices of mesh nodes for one ring basis functions
        print('Calculate mesh one ring:')
        coil_parts = calculate_one_ring_by_mesh(coil_parts)

        #####################################################
        # DEBUG
        # Verify:  one_ring_list, vertex_triangles, node_triangle_mat
        c_part = coil_parts[0]
        one_ring_list = c_part.one_ring_list
        node_triangles = c_part.node_triangles
        node_triangle_mat = c_part.node_triangle_mat

        # DEBUG:__main__: -- m_or_one_ring_list shape: (264,)
        log.debug(" -- m_or_one_ring_list len: %s", m_or_one_ring_list.shape)
        log.debug(" -- one_ring_list len: %s", one_ring_list.shape)  # DEBUG:__main__: -- one_ring_list shape: (264,)

        log.debug(" -- m_or_node_triangles len: %s", m_or_node_triangles.shape)  # 264,
        log.debug(" -- vertex_triangles len: %s", node_triangles.shape)  # 264,

        log.debug(" -- m_or_node_triangle_mat len: %s", m_or_node_triangle_mat.shape)  # 264,480
        log.debug(" -- node_triangle_mat shape: %s", node_triangle_mat.shape)  # 264,480

        visualize_multi_connections(coil_mesh.uv, 800, 'images/one-1-ring_list.png', one_ring_list[0:25])
        visualize_multi_connections(coil_mesh.uv, 800, 'images/one-1-ring_list_m.png', m_or_one_ring_list[0:25])

        assert (compare_contains(one_ring_list, m_or_one_ring_list))   # PASS - different order!
        assert (compare_contains(node_triangles, m_or_node_triangles)) # PASS - different order!
        assert (compare(node_triangle_mat, m_or_node_triangle_mat)) # PASS
        #
        #####################################################

        # Create the basis function container which represents the current density
        print('Create the basis function container which represents the current density:')
        coil_parts = calculate_basis_functions(coil_parts)

        #####################################################
        # DEVELOPMENT: Remove this
        # DEBUG
        # Verify: is_real_triangle_mat, triangle_corner_coord_mat, current_mat, area_mat, face_normal_mat, current_density_mat
        # Basis functions
        m_is_real_triangle_mat = m_c_part.is_real_triangle_mat
        m_triangle_corner_coord_mat = m_c_part.triangle_corner_coord_mat
        m_current_mat = m_c_part.current_mat
        m_area_mat = m_c_part.area_mat
        m_face_normal_mat = m_c_part.face_normal_mat
        m_current_density_mat = m_c_part.current_density_mat

        log.debug(" -- m_is_real_triangle_mat shape: %s", m_is_real_triangle_mat.shape)  # 264,
        log.debug(" -- c_part.is_real_triangle_mat shape: %s", c_part.is_real_triangle_mat.shape)  # 
        
        log.debug(" -- m_triangle_corner_coord_mat shape: %s", m_triangle_corner_coord_mat.shape)  # 264,
        log.debug(" -- c_part.triangle_corner_coord_mat shape: %s", c_part.triangle_corner_coord_mat.shape)  # 
        
        assert (compare(c_part.is_real_triangle_mat, m_is_real_triangle_mat)) # PASS
        assert (compare_contains(c_part.triangle_corner_coord_mat, m_triangle_corner_coord_mat)) # Pass
        assert (compare_contains(c_part.current_mat, m_current_mat)) # Pass
        assert (compare(c_part.area_mat, m_area_mat)) # Pass
        assert (compare_contains(c_part.face_normal_mat, m_face_normal_mat)) # Pass
        assert (compare(c_part.current_density_mat, m_current_density_mat)) # Pass

        #
        #####################################################


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

    # create cylinder mesh: 0.4,  0.1125, 50, 50. Copy from example
    arg_dict0 = {
        'field_shape_function': 'y',  # % definition of the target field
        # 'coil_mesh_file': 'cylinder_radius500mm_length1500mm.stl',
        'coil_mesh_file': 'create cylinder mesh',
        'cylinder_mesh_parameter_list': [0.4,  0.1125, 50, 50,  0.,  1.,  0., 0.],
        'target_mesh_file': 'none',
        'secondary_target_mesh_file': 'none',
        'secondary_target_weight': 0.5,
        'target_region_radius': 0.15,  # ...  % in meter
        'use_only_target_mesh_verts': False,
        'sf_source_file': 'none',
        # % the number of potential steps that determines the later number of windings (Stream function discretization)
        'levels': 20,
        'pot_offset_factor': 0.25,  # % a potential offset value for the minimal and maximal contour potential ; must be between 0 and 1
        'surface_is_cylinder_flag': True,
        'interconnection_cut_width': 0.1,  # % the width for the interconnections are interconnected; in meter
        'normal_shift_length': 0.025,  # % the length for which overlapping return paths will be shifted along the surface normals; in meter
        'iteration_num_mesh_refinement': 0,  # % the number of refinements for the mesh;
        'set_roi_into_mesh_center': True,
        'force_cut_selection': {'high'},  # ...
        'level_set_method': 'primary',  # ... %Specify one of the three ways the level sets are calculated: "primary","combined", or "independent"
        'interconnection_method': 'regular',
        'skip_postprocessing': False,
        'skip_inductance_calculation': False,
        'make_cylndrical_pcb': True,
        'conductor_cross_section_width': 0.015,
        'cross_sectional_points': np.array([np.sin(np.linspace(0, 2 * np.pi, 10)),
                                            np.cos(np.linspace(0, 2 * np.pi, 10))]) * 0.01,
        'sf_opt_method': 'tikkonov',  # ...
        'fmincon_parameter': [1000.0, 10 ^ 10, 1.000000e-10, 1.000000e-10, 1.000000e-10],
        'tikonov_reg_factor': 100,  # %Tikonov regularization factor for the SF optimization
        'debug': DEBUG_VERBOSE,
        # 'debug': DEBUG_BASIC,
    }

    # create cylinder mesh: 0.4, 0.1125, 50, 50, copy from Matlab
    arg_dict1 = {
        "area_perimeter_deletion_ratio": 5,
        "b_0_direction": [0, 0, 1],
        "biplanar_mesh_parameter_list": [0.25, 0.25, 20, 20, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2],
        # "circular_diameter_factor_cylinder_parameterization": 1,
        "circular_diameter_factor": 1.0,
        "circular_mesh_parameter_list": [0.25, 20.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "coil_mesh_file": "create cylinder mesh",
        "conductor_cross_section_height": 0.002,
        "conductor_cross_section_width": 0.003,
        "conductor_thickness": 0.005,
        "cross_sectional_points": [[-0.005, 0.0, 0.0004578154048673232, 0.000878541328365346, 0.0012280930583256696, 0.0014781519924510923, 0.0016084598430565155, 0.0016084598430565157, 0.0014781519924510925, 0.0012280930583256698, 0.0008785413283653464, 0.0004578154048673232, 1.990051048614449e-19, -0.005, -0.005], [0.001625, 0.001625, 0.0015591760821235582, 0.0013670369908506694, 0.0010641486926610882, 0.0006750493961280654, 0.0002312616121940883, -0.00023126161219408811, -0.0006750493961280653, -0.001064148692661088, -0.0013670369908506692, -0.0015591760821235582, -0.001625, -0.001625, 0.001625]],
        "cylinder_mesh_parameter_list": [0.4, 0.1125, 50, 50, 0.0, 1.0, 0.0, 0.0],
        "double_cone_mesh_parameter_list": [0.8, 0.3, 0.3, 0.1, 20.0, 20.0, 1.0, 0.0, 0.0, 0.0],
        "field_shape_function": "x",
        "fieldtype_to_evaluate": ['', 'MCOS', 'string', [3707764736,          2,          1,          1,          2,                2]],
        "fmincon_parameter": [500.0, 10000000000.0, 1e-10, 1e-10, 1e-10],
        "force_cut_selection": ['high', 'high', 'low', 'high'],
        "gauss_order": 2,
        # "geometry_source_path": "C:\\Users\\amrein\\Documents\\PhD_Work\\CoilGen\\Geometry_Data",
        "group_interconnection_method": "crossed",
        "interconnection_cut_width": 0.02,
        "interconnection_method": "regular",
        "iteration_num_mesh_refinement": 0,
        "level_set_method": "primary",
        "levels": 14,
        "make_cylndrical_pcb": 0,
        "max_allowed_angle_within_coil_track": 120,
        "min_allowed_angle_within_coil_track": 0.0001,
        "min_loop_signifcance": 3,
        "min_point_loop_number": 20,
        "normal_shift_length": 0.004,
        "normal_shift_smooth_factors": [2, 3, 2],
        # "output_directory": "C:\\Users\\amrein\\Documents\\PhD_Work\\CoilGen",
        "pcb_interconnection_method": "regular",
        "pcb_spiral_end_shift_factor": 10,
        "planar_mesh_parameter_list": [0.25, 0.25, 20.0, 20.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "plot_flag": 1,
        "pot_offset_factor": 0.5,
        "save_stl_flag": 1,
        "secondary_target_mesh_file": "none",
        "secondary_target_weight": 1,
        "set_roi_into_mesh_center": 0,
        "sf_opt_method": "tikkonov",
        "sf_source_file": "none",
        "skip_calculation_min_winding_distance": 1,
        "skip_inductance_calculation": 0,
        "skip_normal_shift": 0,
        "skip_postprocessing": 0,
        "skip_sweep": 0,
        "smooth_factor": 0,
        "smooth_flag": 1,
        "specific_conductivity_conductor": 1.8e-8,
        "surface_is_cylinder_flag": 1,
        "target_field_definition_field_name": "none",
        "target_field_definition_file": "none",
        "target_gradient_strength": 1,
        "target_mesh_file": "none",
        "target_region_radius": 0.05,
        "target_region_resolution": 5,
        "temp": [],
        "tikonov_reg_factor": 30000,
        "tiny_segment_length_percentage": 0,
        "track_width_factor": 0.5,
        "use_only_target_mesh_verts": 0,
        "debug": DEBUG_VERBOSE,
    }

    # cylinder_radius500mm_length1500mm
    arg_dict2 = {
        "area_perimeter_deletion_ratio": 5,
        "b_0_direction": 0,
        "biplanar_mesh_parameter_list": [
            0.25,
            0.25,
            20.0,
            20.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.2
        ],
        "circular_diameter_factor_cylinder_parameterization": 1,
        "circular_mesh_parameter_list": [
            0.25,
            20.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0
        ],
        "coil_mesh_file": "cylinder_radius500mm_length1500mm.stl",
        "conductor_cross_section_height": 0.002,
        "conductor_cross_section_width": 0.015,
        "conductor_thickness": 0.005,
        "cross_sectional_points": [
            0.0,
            0.006427876096865392,
            0.00984807753012208,
            0.008660254037844387,
            0.0034202014332566888,
            -0.0034202014332566867,
            -0.008660254037844389,
            -0.009848077530122082,
            -0.006427876096865396,
            -2.4492935982947065e-18
        ],
        "cylinder_mesh_parameter_list": [
            0.8,
            0.3,
            20.0,
            20.0,
            1.0,
            0.0,
            0.0,
            0.0
        ],
        "double_cone_mesh_parameter_list": [
            0.8,
            0.3,
            0.3,
            0.1,
            20.0,
            20.0,
            1.0,
            0.0,
            0.0,
            0.0
        ],
        "field_shape_function": "y",
        # "fieldtype_to_evaluate": [
        #     "",
        #     "MCOS",
        #     "string",
        #     "[[3707764736], [2], [1], [1], [10], [3]]"
        # ],
        "fmincon_parameter": [
            1000.0,
            10000000000.0,
            1e-10,
            1e-10,
            1e-10
        ],
        "force_cut_selection": ['high'],
        "gauss_order": 2,
        # "geometry_source_path": "C:\\Users\\amrein\\Documents\\PhD_Work\\CoilGen\\Geometry_Data",
        "group_interconnection_method": "crossed",
        "interconnection_cut_width": 0.1,
        "interconnection_method": "regular",
        "iteration_num_mesh_refinement": 0,
        "level_set_method": "primary",
        "levels": 20,
        "make_cylndrical_pcb": 1,
        "max_allowed_angle_within_coil_track": 120,
        "min_allowed_angle_within_coil_track": 0.0001,
        "min_loop_signifcance": 3,
        "min_point_loop_number": 20,
        "normal_shift_length": 0.025,
        "normal_shift_smooth_factors": [
            2,
            3,
            2
        ],
        # "output_directory": "C:\\Users\\amrein\\Documents\\PhD_Work\\CoilGen",
        "pcb_interconnection_method": "spiral_in_out",
        "pcb_spiral_end_shift_factor": 10,
        "planar_mesh_parameter_list": [
            0.25,
            0.25,
            20.0,
            20.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0
        ],
        "plot_flag": 1,
        "pot_offset_factor": 0.25,
        "save_stl_flag": 1,
        "secondary_target_mesh_file": "none",
        "secondary_target_weight": 0.5,
        "set_roi_into_mesh_center": 1,
        "sf_opt_method": "tikkonov",
        "sf_source_file": "none",
        "skip_calculation_min_winding_distance": 1,
        "skip_inductance_calculation": 0,
        "skip_normal_shift": 0,
        "skip_postprocessing": 0,
        "skip_sweep": 0,
        "smooth_factor": 1,
        "smooth_flag": 1,
        "specific_conductivity_conductor": 1.8e-8,
        "surface_is_cylinder_flag": 1,
        "target_field_definition_field_name": "none",
        "target_field_definition_file": "none",
        "target_gradient_strength": 1,
        "target_mesh_file": "none",
        "target_region_radius": 0.15,
        "target_region_resolution": 5,
        "tikonov_reg_factor": 100,
        "tiny_segment_length_percentage": 0,
        "track_width_factor": 0.5,
        "use_only_target_mesh_verts": 0,
        "debug": DEBUG_VERBOSE,
    }

    solution = CoilGen(log, arg_dict2)
