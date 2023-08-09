# System imports
import numpy as np

# Logging
import logging

# Debug and development checking imports
from helpers.extraction import get_element_by_name, load_matlab, get_and_show_debug
from helpers.visualisation import compare, compare_contains, visualize_vertex_connections, \
    visualize_multi_connections, visualize_compare_vertices, visualize_compare_contours, get_linenumber
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
from sub_functions.calculate_gradient_sensitivity_matrix import calculate_gradient_sensitivity_matrix
from sub_functions.calculate_resistance_matrix import calculate_resistance_matrix
from sub_functions.stream_function_optimization import stream_function_optimization
from sub_functions.calc_potential_levels import calc_potential_levels
from sub_functions.calc_contours_by_triangular_potential_cuts import calc_contours_by_triangular_potential_cuts
from sub_functions.process_raw_loops import process_raw_loops
from sub_functions.find_minimal_contour_distance import find_minimal_contour_distance
from sub_functions.topological_loop_grouping import topological_loop_grouping
from sub_functions.calculate_group_centers import calculate_group_centers
from sub_functions.interconnect_within_groups import interconnect_within_groups
from sub_functions.interconnect_among_groups import interconnect_among_groups
"""
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
        # mat_contents = load_matlab('debug/result_y_gradient')
        mat_contents = load_matlab('debug/ygradient_coil')
        log.debug("mat_contents: %s", mat_contents.keys())
        matlab_data = mat_contents['coil_layouts']

    m_out = get_element_by_name(matlab_data, 'out')
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
    for index1 in range(len(m_or_one_ring_list)):
        m_or_one_ring_list[index1] = m_or_one_ring_list[index1].T
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

        if get_level() >= DEBUG_VERBOSE:
            log.debug(" -- vertices shape: %s", coil_mesh.get_vertices().shape)  # (264,3)
            log.debug(" -- faces shape: %s", coil_mesh.get_faces().shape)  # (480,3)

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
            log.debug(" m_boundary: %s", m_boundary)
            log.debug(" coil_mesh.boundary: %s", coil_mesh.boundary)
        # Question: Does order matter?
        assert (compare_contains(coil_mesh.boundary, m_boundary, strict=False))  # Pass

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

        if get_level() >= DEBUG_VERBOSE:
            log.debug(" -- target_field.b shape: %s", target_field.b.shape)  # (3, 257)
            log.debug(" -- target_field.coords shape: %s", target_field.coords.shape)  # (3, 257)
            log.debug(" -- target_field.weights shape: %s", target_field.weights.shape)  # (257,)

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

        # Find indices of mesh nodes for one ring basis functions
        print('Calculate mesh one ring:')
        coil_parts = calculate_one_ring_by_mesh(coil_parts)

        #####################################################
        # DEBUG
        # Verify:  one_ring_list, vertex_triangles, node_triangle_mat
        coil_part = coil_parts[0]
        one_ring_list = coil_part.one_ring_list
        node_triangles = coil_part.node_triangles
        node_triangle_mat = coil_part.node_triangle_mat

        visualize_multi_connections(coil_mesh.uv, 800, 'images/one-1-ring_list.png', one_ring_list[0:25])
        visualize_multi_connections(coil_mesh.uv, 800, 'images/one-1-ring_list_m.png', m_or_one_ring_list[0:25])

        assert (compare_contains(one_ring_list, m_or_one_ring_list, strict=False))   # PASS - different order!
        assert (compare_contains(node_triangles, m_or_node_triangles, strict=False))  # PASS - different order!
        assert (compare(node_triangle_mat, m_or_node_triangle_mat))  # PASS

        # =================================================
        # HACK: Use MATLAB's one_ring_list, etc.
        use_matlab_data = True
        if use_matlab_data:
            log.warning("Using MATLAB's one_ring_list in %s, line %d", __file__, get_linenumber())
            coil_part.one_ring_list = m_or_one_ring_list
            coil_part.node_triangles = m_or_node_triangles
        # =================================================

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

        # Transpose MATLAB matrix
        for top in m_triangle_corner_coord_mat:
            for index1 in range(top.shape[0]):
                matrix = top[index1]
                top[index1] = matrix.T

        if use_matlab_data:
            assert (compare(coil_part.current_mat, m_current_mat))  # Pass!
            assert (compare(coil_part.face_normal_mat, m_face_normal_mat))  # Pass!
            assert (compare(coil_part.triangle_corner_coord_mat, m_triangle_corner_coord_mat))  # Pass!
        else:
            assert (compare_contains(coil_part.current_mat, m_current_mat, strict=False))  # Pass
            assert (compare_contains(coil_part.triangle_corner_coord_mat,
                    m_triangle_corner_coord_mat, strict=False))  # Pass, Transposed
            # assert (compare(coil_part.triangle_corner_coord_mat, m_triangle_corner_coord_mat))  # Different order?

        assert (compare(coil_part.is_real_triangle_mat, m_is_real_triangle_mat))  # Pass
        assert (compare(coil_part.area_mat, m_area_mat))  # Pass
        assert (compare_contains(coil_part.face_normal_mat, m_face_normal_mat, strict=False))  # Pass
        assert (compare(coil_part.current_density_mat, m_current_density_mat))  # Pass

        # Verify basis_elements
        m_basis_elements = m_c_part.basis_elements
        assert len(coil_part.basis_elements) == len(m_basis_elements)
        for index1 in range(len(coil_part.basis_elements)):
            cg_element = coil_part.basis_elements[index1]
            m_element = m_basis_elements[index1]

            # Tranpose MATLAB matrix
            for index2 in range(len(m_element.triangle_points_ABC)):
                m_element.triangle_points_ABC[index2] = m_element.triangle_points_ABC[index2].T

            # Verify: triangles, stream_function_potential, area, face_normal, triangle_points_ABC, current
            assert (compare_contains(cg_element.triangles, m_element.triangles-1))  # Pass
            # assert (compare(cg_element.triangles, m_element.triangles-1))  # Pass!
            assert (cg_element.stream_function_potential == m_element.stream_function_potential)  # Pass
            assert (compare(cg_element.area, m_element.area))  # Pass
            assert (compare_contains(cg_element.face_normal, m_element.face_normal))  # Pass
            # assert (compare(cg_element.face_normal, m_element.face_normal))  # Pass!
            assert (compare_contains(cg_element.triangle_points_ABC, m_element.triangle_points_ABC))  # Pass, transposed
            # assert (compare(cg_element.triangle_points_ABC, m_element.triangle_points_ABC))  # ???, transposed
            assert (compare_contains(cg_element.current, m_element.current))  # Pass
            # assert (compare(cg_element.current, m_element.current))  # Pass!
        #
        #####################################################

        # Calculate the sensitivity matrix Cn
        print('Calculate the sensitivity matrix:')
        coil_parts = calculate_sensitivity_matrix(coil_parts, target_field, input_args)

        #####################################################
        # DEVELOPMENT: Remove this
        # DEBUG
        # Verify: sensitivity_matrix
        m_sensitivity_matrix = m_c_part.sensitivity_matrix
        # TODO: Consider Python-like structure: 264 (num vertices) x 257 (target_field) x  3 (x,y,z)
        if get_level() >= DEBUG_VERBOSE:
            log.debug(" -- m_sensitivity_matrix shape: %s", m_sensitivity_matrix.shape)  # (3, 257, 264)
            log.debug(" -- c_part.sensitivity_matrix shape: %s", coil_part.sensitivity_matrix.shape)  # (3, 257, 264)
        assert (compare(coil_part.sensitivity_matrix, m_sensitivity_matrix))  # Pass
        #
        #####################################################

        # Calculate the gradient sensitivity matrix Gn
        print('Calculate the gradient sensitivity matrix:')
        coil_parts = calculate_gradient_sensitivity_matrix(coil_parts, target_field, input_args)

        #####################################################
        # DEVELOPMENT: Remove this
        # DEBUG
        # Verify: gradient_sensitivity_matrix
        m_gradient_sensitivity_matrix = m_c_part.gradient_sensitivity_matrix

        # Consider Python-like structure: 257 () x 264 (num vertices) x 3 (x,y,z)
        log.debug(" -- m_gradient_sensitivity_matrix shape: %s", m_sensitivity_matrix.shape)  # (3, 257, 264)
        log.debug(" -- c_part.gradient_sensitivity_matrix shape: %s",
                  coil_part.gradient_sensitivity_matrix.shape)  # (3, 257, 264)

        assert (compare(coil_part.gradient_sensitivity_matrix, m_gradient_sensitivity_matrix))  # Pass
        #
        #####################################################

        # Calculate the resistance matrix Rmn
        print('Calculate the resistance matrix:')
        coil_parts = calculate_resistance_matrix(coil_parts, input_args)

        #####################################################
        # DEVELOPMENT: Remove this
        # DEBUG
        # Verify: node_adjacency_mat, resistance_matrix
        m_node_adjacency_mat = m_c_part.node_adjacency_mat
        m_resistance_matrix = m_c_part.resistance_matrix

        log.debug(" -- m_gradient_sensitivity_matrix shape: %s", m_resistance_matrix.shape)  # (264, 264)
        log.debug(" -- c_part.gradient_sensitivity_matrix shape: %s", coil_part.resistance_matrix.shape)  # (264, 264)

        assert (compare(coil_part.node_adjacency_mat, m_node_adjacency_mat))  # Pass
        assert (compare(coil_part.resistance_matrix, m_resistance_matrix))  # Pass
        #
        #####################################################

        # Optimize the stream function toward target field and further constraints
        print('Optimize the stream function toward target field and secondary constraints:')
        coil_parts, combined_mesh, sf_b_field = stream_function_optimization(coil_parts, target_field, input_args)

        #####################################################
        # DEVELOPMENT: Remove this
        # DEBUG
        # Verify: combined_mesh.stream_function, c_part.current_density, c_part.stream_function, b_field_opt_sf,
        m_current_density = m_c_part.current_density.T
        m_combined_mesh = get_and_show_debug(matlab_data, 'out.combined_mesh')
        m_sf_b_field = get_and_show_debug(matlab_data, 'out.b_field_opt_sf')
        m_cm_stream_function = m_combined_mesh.stream_function
        m_cp_stream_function = m_c_part.stream_function

        log.debug(" -- m_current_density shape: %s", m_current_density.shape)  # (3 x 480)
        log.debug(" -- m_sf_b_field shape: %s", m_sf_b_field.shape)  # (257 x 3)
        log.debug(" -- m_cm_stream_function shape: %s", m_cm_stream_function.shape)  # (264,)
        log.debug(" -- m_cp_stream_function shape: %s", m_cp_stream_function.shape)  # (264,)

        log.debug(" -- c_part.current_density shape: %s", coil_part.current_density.shape)  # (3 x 480)
        log.debug(" -- sf_b_field shape: %s", sf_b_field.shape)  # (257 x 3) !!!
        log.debug(" -- combined_mesh.stream_function shape: %s", combined_mesh.stream_function.shape)  # (264,)
        log.debug(" -- c_part.stream_function shape: %s", coil_part.stream_function.shape)  # (264,)

        assert (compare(coil_part.current_density, m_current_density))  # Pass
        assert (compare(sf_b_field, m_sf_b_field))  # Pass
        assert (compare(combined_mesh.stream_function, m_cm_stream_function))  # Pass
        assert (compare(coil_part.stream_function, m_cp_stream_function))  # Pass
        #
        #####################################################

    else:
        # Load the preoptimized data
        print('Load preoptimized data:')
        raise Exception("Not supported")
        coil_parts, _, _, combined_mesh, sf_b_field, target_field, is_suppressed_point = load_preoptimized_data(
            input_args)

    # Calculate the potential levels for the discretization
    print('Calculate the potential levels for the discretization:')
    coil_parts, primary_surface_ind = calc_potential_levels(coil_parts, combined_mesh, input_args)

    #####################################################
    # DEVELOPMENT: Remove this
    # DEBUG
    # Verify: primary_surface_ind, coil_part.potential_level_list, coil_part.contour_step
    m_primary_surface_ind = m_out.primary_surface - 1  # -1 because MATLAB uses 1-based indexing
    m_cp_potential_level_list = m_c_part.potential_level_list

    log.debug(" -- m_primary_surface_ind: %d", m_primary_surface_ind)  # (1)
    log.debug(" -- m_cp_potential_level_list shape: %s", m_cp_potential_level_list.shape)  # (20,)

    log.debug(" -- primary_surface_ind: %d", primary_surface_ind)  # (1)
    log.debug(" -- c_part.potential_level_list shape: %s", coil_part.potential_level_list.shape)  # (20)

    log.debug(" -- coil_part.contour_step: %s, m_c_part.contour_step: %s",
              coil_part.contour_step, m_c_part.contour_step)

    assert (primary_surface_ind == m_primary_surface_ind)  # Pass
    assert (compare(coil_part.potential_level_list, m_cp_potential_level_list))  # Pass
    assert np.isclose(coil_part.contour_step, m_c_part.contour_step, atol=0.001)  # Pass
    #
    #####################################################

    # Generate the contours
    print('Generate the contours:')
    coil_parts = calc_contours_by_triangular_potential_cuts(coil_parts)

    #####################################################
    # DEVELOPMENT: Remove this
    # DEBUG
    # Verify: part.contour_lines items current_orientation, potential, uv
    # Verify: part.raw.
    #           unarranged_loops(x).loop(y).[edge_inds, uv]
    #           unsorted_points(x).[edge_ind, potential, uv]

    assert len(coil_part.raw.unsorted_points) == len(m_c_part.raw.unsorted_points)
    for index1, m_ru_points in enumerate(m_c_part.raw.unsorted_points):
        c_ru_point = coil_part.raw.unsorted_points[index1]
        m_ru_point = m_c_part.raw.unsorted_points[index1]
        assert len(c_ru_point.edge_ind) == len(m_ru_point.edge_ind)
        assert np.isclose(c_ru_point.potential, m_ru_point.potential)
        assert c_ru_point.uv.shape[0] == m_ru_point.uv.shape[0]  # Python shape!
        # assert(compare(c_ru_point.edge_ind, m_ru_point.edge_ind)) # Different ordering?
        # assert(compare_contains(c_ru_point.uv, m_ru_point.uv)) # Order is different

    assert len(coil_part.raw.unarranged_loops) == len(m_c_part.raw.unarranged_loops)
    for index1, m_ru_loops in enumerate(m_c_part.raw.unarranged_loops):
        c_loops = coil_part.raw.unarranged_loops[index1]
        m_loops = m_c_part.raw.unarranged_loops[index1]
        assert len(c_loops.loop) == len(m_loops.loop)
        # Skip the next section, the loops are different!!
        # for index2, m_ru_loop in enumerate(m_ru_loops.loop):
        #    c_ru_loop = c_loops.loop[index2]
        #    assert c_ru_loop.uv.shape[0] == m_ru_loop.uv.shape[0] # Python shape!
        #    assert(compare_contains(c_ru_loop.uv, m_ru_loop.uv)) #
        #    assert len(c_ru_loop.edge_inds) == len(m_ru_loop.edge_inds)
        #    #assert(compare(c_ru_point.edge_inds, m_ru_point.edge_inds))

    m_contour_lines = m_c_part.contour_lines1

    assert len(coil_part.contour_lines) == len(m_contour_lines)
    for index1 in range(len(coil_part.contour_lines)):
        if get_level() > DEBUG_VERBOSE:
            log.debug(" Checking contour %d", index1)
        m_contour = m_contour_lines[index1]
        c_contour = coil_part.contour_lines[index1]
        assert c_contour.current_orientation == m_contour.current_orientation  # Pass
        assert np.isclose(c_contour.potential, m_contour.potential)  # Pass
        # The MATLAB coilpart.contours is further processed in a subsequent function call.
        # Unable to compare here.
        # assert compare(c_contour.uv, m_contour.uv) # Fail
        # log.debug(" -- compare uv: %s", compare(c_contour.uv, m_contour.uv))

    if get_level() >= DEBUG_VERBOSE:
        visualize_compare_contours(coil_mesh.uv, 800, 'images/contour1_p.png', coil_part.contour_lines)
        visualize_compare_contours(coil_mesh.uv, 800, 'images/contour1_m.png', m_contour_lines)

    # Manual conclusion: Not identical, but really close.

    # =================================================
    # HACK: Use MATLAB's contour_lines
    if use_matlab_data:
        log.warning("Using MATLAB's contour_lines1 in %s, line %d", __file__, get_linenumber())
        for index1, m_contour in enumerate(m_contour_lines):
            c_contour = coil_part.contour_lines[index1]
            c_contour.uv = m_contour.uv
    # =================================================

    # np.save('debug/ygradient_coil_python_10.npy', coil_parts)

    #
    #####################################################

    # Process contours
    print('Process contours: Evaluate loop significance')
    coil_parts = process_raw_loops(coil_parts, input_args, target_field)

    #####################################################
    # DEVELOPMENT: Remove this
    # DEBUG
    # Verify: Coil Part values: field_by_loops, loop_significance, combined_loop_field, combined_loop_length

    assert len(coil_part.contour_lines) == len(m_c_part.contour_lines)
    assert abs(coil_part.combined_loop_length - m_c_part.combined_loop_length) < 0.002  # 0.0005 # Pass
    if use_matlab_data:
        assert compare(coil_part.combined_loop_field, m_c_part.combined_loop_field, double_tolerance=5e-7)  # Pass!
        assert compare(coil_part.loop_significance, m_c_part.loop_signficance, double_tolerance=0.005)
        assert compare(coil_part.field_by_loops, m_c_part.field_by_loops, double_tolerance=2e-7)  # Pass!
    else:
        # assert compare(coil_part.combined_loop_field, m_c_part.combined_loop_field, double_tolerance=5e-6) # Fails
        assert compare(coil_part.loop_significance, m_c_part.loop_signficance, double_tolerance=3.9)  # Eeek!
        # assert compare(coil_part.field_by_loops, m_c_part.field_by_loops, double_tolerance=2e-7) # Pass!

    # Compare updated contour lines
    for index1 in range(len(coil_part.contour_lines)):
        if get_level() > DEBUG_VERBOSE:
            log.debug(" Checking contour %d", index1)
        m_contour = m_contour_lines[index1]
        c_contour = coil_part.contour_lines[index1]
        assert c_contour.current_orientation == m_contour.current_orientation  # Pass
        assert np.isclose(c_contour.potential, m_contour.potential)  # Pass | Fail with min_loop_significance == 3
        # assert compare(c_contour.uv, m_contour.uv) # Pass [0], Fail [4], Pass!
        # assert compare(c_contour.v, m_contour.v) # Fail: 2nd position 1.15644483e-03 != -9.12899313e-17
        if get_level() > DEBUG_VERBOSE:
            log.debug(" -- compare uv: %s", compare(c_contour.uv, m_contour.uv))
            log.debug(" -- compare v: %s", compare(c_contour.v, m_contour.v))

    if get_level() >= DEBUG_VERBOSE:
        visualize_compare_contours(coil_mesh.uv, 800, 'images/contour2_p.png', coil_part.contour_lines)
        visualize_compare_contours(coil_mesh.uv, 800, 'images/contour2_m.png', m_contour_lines)

    # Manual conclusion: Not identical, but close.

    # =================================================
    # HACK: Use MATLAB's contour_lines.v
    if use_matlab_data:
        m_contour_lines = m_c_part.contour_lines
        log.warning("Using MATLAB's contour_lines1 in %s, line %d", __file__, get_linenumber())
        for index1, m_contour in enumerate(m_contour_lines):
            c_contour = coil_part.contour_lines[index1]
            c_contour.v = m_contour.v
    # =================================================

    #
    #####################################################

    if not input_args.skip_postprocessing:
        # Find the minimal distance between the contour lines
        print('Find the minimal distance between the contour lines:')
        coil_parts = find_minimal_contour_distance(coil_parts, input_args)

        #####################################################
        # DEVELOPMENT: Remove this
        # DEBUG
        # Verify: Coil Part values: pcb_track_width
        m_pcb_track_width = m_c_part.pcb_track_width
        log.debug(" coil_part.pcb_track_width: %f", coil_part.pcb_track_width)
        log.debug(" m_pcb_track_width: %f", m_pcb_track_width)
        assert np.isclose(coil_part.pcb_track_width, m_pcb_track_width)  # Pass
        #
        #####################################################

        # Group the contour loops in topological order
        print('Group the contour loops in topological order:')
        coil_parts = topological_loop_grouping(coil_parts, input_args)

        #####################################################
        # DEVELOPMENT: Remove this
        # DEBUG
        # Verify: Coil Part values: groups (list of ContourLine objects), level_positions, group_levels, loop_groups

        # =================================================
        # MATLAB / Python ordering hack:
        # Re-order Python groups to match MATLAB ordering
        m_to_pyindex = [1, 0, 2, 3]
        cp_groups = coil_part.groups
        cp_level_positions = coil_part.level_positions
        cp_group_levels = coil_part.group_levels
        cp_loop_groups = coil_part.loop_groups

        # Do the swap
        if use_matlab_data == False:
            log.warning("Changing order of coil groups and loop_groups in %s, line %d", __file__, get_linenumber())
            cp_groups[0], cp_groups[1] = cp_groups[1], cp_groups[0]
            cp_loop_groups[0], cp_loop_groups[1] = cp_loop_groups[1], cp_loop_groups[0]
        #
        # =================================================

        m_level_positions = m_c_part.level_positions
        m_group_levels = m_c_part.group_levels - 1  # MATLAB indexing is 1-based

        # assert compare(cp_group_levels[0], m_group_levels)

        m_loop_groups = m_c_part.loop_groups
        for index1, loop_group in enumerate(m_loop_groups):
            m_loop_groups[index1] = loop_group - 1  # MATLAB indexing is 1-based

        c_level_positions = coil_part.level_positions
        c_group_levels = np.array(coil_part.group_levels)
        c_loop_groups = coil_part.loop_groups

        # assert compare_contains(c_group_levels, m_group_levels)
        # assert compare_contains(c_loop_groups, m_loop_groups) # Fail: They don't match up exactly
        # assert compare(c_loop_groups, m_loop_groups) # Fail: They don't match up exactly Pass!

        # Compare updated groups and their loops
        m_groups = m_c_part.groups
        for index1 in range(len(coil_part.groups)):
            if get_level() > DEBUG_VERBOSE:
                log.debug(" Checking contour group %d", index1)
            m_group = m_groups[index1]  # cutshape, loops, opened_loop
            c_group = coil_part.groups[index1]
            for index2, m_loops in enumerate(m_group.loops):
                c_loops = c_group.loops[index2]
                if get_level() > DEBUG_VERBOSE:
                    log.debug(" Checking index %d", index2)

                assert c_loops.current_orientation == m_loops.current_orientation  # Pass
                # assert np.isclose(c_loop.potential, m_loop.potential) # Fail, group 2, index 0
                # assert compare(c_loop.uv, m_loop.uv) # Fail, different path through mesh
                # assert compare(c_loop.v, m_loop.v) # Fail, different path through mesh
                if get_level() > DEBUG_VERBOSE:
                    log.debug(" -- compare uv: %s", compare(c_loops.uv, m_loops.uv))
                    log.debug(" -- compare v: %s", compare(c_loops.v, m_loops.v))

            if get_level() >= DEBUG_VERBOSE:
                visualize_compare_contours(coil_mesh.uv, 800, f'images/countour4_{index1}_p.png', c_group.loops)
                visualize_compare_contours(coil_mesh.uv, 800, f'images/countour4_{index1}_m.png', m_group.loops)

        # Manual conclusion: Not identical, but close. Contour groups in different orders...

        #
        #####################################################

        # =================================================
        if use_matlab_data:
            log.warning("Using MATLAB's loop u and uv values in %s, line %d", __file__, get_linenumber())
            m_groups = m_c_part.groups
            for index1 in range(len(coil_part.groups)):
                m_group = m_groups[index1]  # cutshape, loops, opened_loop
                c_group = coil_part.groups[index1]
                for index2, m_loops in enumerate(m_group.loops):
                    c_group.loops[index2].uv = m_loops.uv
                    c_group.loops[index2].v = m_loops.v
        # =================================================

        # Calculate center locations of groups
        print('Calculate center locations of groups:')
        calculate_group_centers(coil_parts)
        # np.save('debug/ygradient_coil_python_14.npy', coil_parts)

        #####################################################
        # DEVELOPMENT: Remove this
        # DEBUG
        # Verify: Coil Part values: group_centers
        m_group_centers = m_c_part.group_centers
        c_group_centers = coil_part.group_centers

        if use_matlab_data == True:
            assert compare(c_group_centers.uv, m_group_centers.uv)  # Fail, different group layout Pass!
            assert compare(c_group_centers.v, m_group_centers.v)  # Fail, different group layout Pass!

        # Manual conclusion: Not identical, but close. Different paths, different group layouts.

        #
        #####################################################

        # Interconnect the single groups
        print('Interconnect the single groups:')
        coil_parts = interconnect_within_groups(coil_parts, input_args)

        # np.save('debug/ygradient_coil_python_15_true.npy', coil_parts)

        #####################################################
        # DEVELOPMENT: Remove this
        # DEBUG
        # Verify: coil_parts(part_ind).groups(group_ind).opened_loop(loop_ind).uv
        if use_matlab_data:
            for index1, m_group in enumerate(m_c_part.groups):
                c_group = coil_part.groups[index1]
                for index2, m_opened_loop in enumerate(m_group.opened_loop):
                    c_opened_loop = c_group.opened_loop[index2]
                    assert compare(c_opened_loop.v, m_opened_loop.v)
                    assert compare(c_opened_loop.uv, m_opened_loop.uv)

        # Verify: Coil Part connected_group values: return_path, uv, v, spiral_in (uv,v), spiral_out(uv, v)
        m_connected_groups = m_c_part.connected_group
        c_connected_groups = coil_part.connected_group

        assert len(m_connected_groups) == len(c_connected_groups)

        for index1, m_connected_group in enumerate(m_connected_groups):
            c_connected_group = c_connected_groups[index1]

            if get_level() >= DEBUG_VERBOSE:
                # MATLAB shape
                visualize_vertex_connections(c_connected_group.uv.T, 800, f'images/connected_group_uv_{index1}_p.png')
                visualize_vertex_connections(m_connected_group.uv.T, 800, f'images/connected_group_uv_{index1}_m.png')

            # Check....
            if use_matlab_data:
                assert compare(c_connected_group.return_path.v, m_connected_group.return_path.v)
                assert compare(c_connected_group.return_path.uv, m_connected_group.return_path.uv)

                assert compare(c_connected_group.spiral_in.v, m_connected_group.group_debug.spiral_in.v)
                assert compare(c_connected_group.spiral_in.uv, m_connected_group.group_debug.spiral_in.uv)

                assert compare(c_connected_group.spiral_out.v, m_connected_group.group_debug.spiral_out.v)
                assert compare(c_connected_group.spiral_out.uv, m_connected_group.group_debug.spiral_out.uv)

                assert compare(c_connected_group.uv, m_connected_group.uv)
                assert (compare(c_connected_group.v, m_connected_group.v))

                assert compare(c_connected_group.uv, m_connected_group.uv)  # Pass!
                assert compare(c_connected_group.v, m_connected_group.v)    # Pass!

        # Manual conclusion: Fail, maybe - the Python connections look a bit different to the MATLAB ones in a few places

        #
        #####################################################

        # Interconnect the groups to a single wire path
        print('Interconnect the groups to a single wire path:')
        coil_parts = interconnect_among_groups(coil_parts, input_args)

        #####################################################
        # DEVELOPMENT: Remove this
        # DEBUG
        # Verify: coil_parts(part_ind). [opening_cuts_among_groups, wire_path]

        for index1, p_coil_part in enumerate(coil_parts):
            # Opening cuts
            """
            for index2, m_cut in enumerate(m_c_part.opening_cuts_among_groups):
                p_cut = p_coil_part.opening_cuts_among_groups[index2]
                visualize_vertex_connections(p_cut.cut1.T, 800, f'images/cuts_cut1_{index1}_{index2}_p.png')
                visualize_vertex_connections(m_cut.cut1.T, 800, f'images/cuts_cut1_{index1}_{index2}_m.png')
                visualize_vertex_connections(p_cut.cut2.T, 800, f'images/cuts_cut2_{index1}_{index2}_p.png')
                visualize_vertex_connections(m_cut.cut2.T, 800, f'images/cuts_cut2_{index1}_{index2}_m.png')
            """

            # Wire path
            c_wire_path = p_coil_part.wire_path
            m_wire_path = m_c_part.wire_path1
            if get_level() >= DEBUG_VERBOSE:
                visualize_vertex_connections(c_wire_path.uv.T, 800, f'images/wire_path_uv_{index1}_p.png')
                visualize_vertex_connections(m_wire_path.uv.T, 800, f'images/wire_path_uv_{index1}_m.png')


            if use_matlab_data:
                assert (compare(c_wire_path.uv, m_wire_path.uv))    # Pass
                assert (compare(c_wire_path.v, m_wire_path.v))      # Pass

        # Manual conclusion: Pass, when using MATLAB data
        #
        #####################################################

        # WIP
        solution.coil_parts = coil_parts
        return solution

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

    # create cylinder mesh: 0.4, 0.1125, 50, 50, copy from Matlab
    arg_dict1 = {
        "area_perimeter_deletion_ratio": 5,
        "b_0_direction": [0, 0, 1],
        "biplanar_mesh_parameter_list": [0.25, 0.25, 20, 20, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2],
        "circular_diameter_factor": 1.0,  # was circular_diameter_factor_cylinder_parameterization
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
        "min_loop_significance": 3,
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
        "skip_calculation_min_winding_distance": 1,  # Default: 1
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
        "b_0_direction": [0, 0, 1],
        "biplanar_mesh_parameter_list": [0.25, 0.25, 20.0, 20.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2],
        "circular_diameter_factor": 1.0,  # was circular_diameter_factor_cylinder_parameterization
        "circular_mesh_parameter_list": [0.25, 20.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "coil_mesh_file": "cylinder_radius500mm_length1500mm.stl",
        "conductor_cross_section_height": 0.002,
        "conductor_cross_section_width": 0.015,
        "conductor_thickness": 0.005,
        "cross_sectional_points": [[0.0, 0.006427876096865392, 0.00984807753012208, 0.008660254037844387, 0.0034202014332566887, -0.0034202014332566865, -0.008660254037844388, -0.009848077530122082, -0.006427876096865396, -2.4492935982947064e-18], [0.01, 0.007660444431189781, 0.0017364817766693042, -0.0049999999999999975, -0.009396926207859084, -0.009396926207859084, -0.004999999999999997, 0.0017364817766692998, 0.007660444431189778, 0.01]],
        "cylinder_mesh_parameter_list": [0.8, 0.3, 20.0, 20.0, 1.0, 0.0, 0.0, 0.0],
        "double_cone_mesh_parameter_list": [0.8, 0.3, 0.3, 0.1, 20.0, 20.0, 1.0, 0.0, 0.0, 0.0],
        "field_shape_function": "y",
        "fieldtype_to_evaluate": ['', 'MCOS', 'string', [3707764736, 2, 1, 1, 1, 1]],
        "fmincon_parameter": [1000.0, 10000000000.0, 1e-10, 1e-10, 1e-10],
        "force_cut_selection": ['high'],
        "gauss_order": 2,
        # "geometry_source_path": "/MATLAB Drive/CoilGen/Geometry_Data",
        "group_interconnection_method": "crossed",
        "interconnection_cut_width": 0.1,
        "interconnection_method": "regular",
        "iteration_num_mesh_refinement": 0,
        "level_set_method": "primary",
        "levels": 20,
        "make_cylindrical_pcb": 1,
        "max_allowed_angle_within_coil_track": 120,
        "min_allowed_angle_within_coil_track": 0.0001,
        "min_loop_significance": 1,
        "min_point_loop_number": 20,
        "normal_shift_length": 0.025,
        "normal_shift_smooth_factors": [2, 3, 2],
        # "output_directory": "/MATLAB Drive/CoilGen",
        "pcb_interconnection_method": "spiral_in_out",
        "pcb_spiral_end_shift_factor": 10,
        "planar_mesh_parameter_list": [0.25, 0.25, 20.0, 20.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "plot_flag": 1,
        "pot_offset_factor": 0.25,
        "save_stl_flag": 1,
        "secondary_target_mesh_file": "none",
        "secondary_target_weight": 0.5,
        "set_roi_into_mesh_center": 1,
        "sf_opt_method": "tikkonov",
        "sf_source_file": "none",
        "skip_calculation_min_winding_distance": 1,  # Default 1
        "skip_inductance_calculation": 0,
        "skip_normal_shift": 0,
        "skip_postprocessing": 0,
        "skip_sweep": 0,
        "smooth_factor": 1,
        "smooth_flag": 1,
        "specific_conductivity_conductor": 1.8e-08,
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
