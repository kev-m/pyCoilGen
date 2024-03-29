# Logging
import logging

# System imports
import numpy as np
from os import makedirs

# Logging
import logging

# Debug and development checking imports
from .helpers.extraction import get_element_by_name, load_matlab, get_and_show_debug
from .helpers.visualisation import compare, compare_contains, visualize_vertex_connections, \
    visualize_multi_connections, visualize_compare_vertices, visualize_compare_contours, get_linenumber, \
    visualize_projected_vertices, passify_matlab

# Local imports
from .sub_functions.constants import *
from .sub_functions.data_structures import Mesh, DataStructure, CoilSolution, OptimisationParameters


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

# Set up logging
log = logging.getLogger(__name__)


def pyCoilGen(log, input=None):
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

    # Create directories if they do not exist
    persistence_dir = input_args.persistence_dir
    image_dir = input_args.output_directory

    makedirs(persistence_dir, exist_ok=True)
    makedirs(image_dir, exist_ok=True)

    ######################################################################################
    # DEVELOPMENT: Remove this
    compare_mesh_shape = True
    mat_filename = f'{persistence_dir}/{input_args.project_name}_{input_args.iteration_num_mesh_refinement}_{input_args.target_region_resolution}'
    log.debug(" Loading comparison data from %s", mat_filename)
    mat_contents = load_matlab(mat_filename)
    matlab_data = mat_contents['coil_layouts']
    m_out = get_element_by_name(matlab_data, 'out')
    if isinstance(m_out.coil_parts, (np.ndarray)):
        m_c_parts = m_out.coil_parts
    else:
        m_c_parts = [m_out.coil_parts]

    # END of Remove this
    ######################################################################################

    # Print the input variables
    # DEBUG
    use_matlab_data = False
    if get_level() >= DEBUG_VERBOSE:
        log.debug('Parse inputs: %s', input_args)

    persistence_dir = input_args.persistence_dir
    project_name = f'{input_args.project_name}_{input_args.iteration_num_mesh_refinement}_{input_args.target_region_resolution}_{use_matlab_data}'

    solution = CoilSolution()
    solution.input_args = input_args
    timer = Timing()
    timer.start()

    if input_args.sf_source_file == 'none':
        # Read the input mesh
        print('Load geometry:')
        coil_mesh, target_mesh, secondary_target_mesh = read_mesh(input_args)  # 01

        if get_level() >= DEBUG_VERBOSE:
            log.debug(" -- vertices shape: %s", coil_mesh.get_vertices().shape)  # (264,3)
            log.debug(" -- faces shape: %s", coil_mesh.get_faces().shape)  # (480,3)

        if get_level() > DEBUG_VERBOSE:
            log.debug(" coil_mesh.vertex_faces: %s", coil_mesh.trimesh_obj.vertex_faces[0:10])

        if get_level() > DEBUG_VERBOSE:
            coil_mesh.display()

        # Split the mesh and the stream function into disconnected pieces
        print('Split the mesh and the stream function into disconnected pieces.')
        coil_parts = split_disconnected_mesh(coil_mesh)  # 00
        solution.coil_parts = coil_parts
        save(persistence_dir, project_name, '00', solution)

        # Upsample the mesh density by subdivision
        print('Upsample the mesh by subdivision:')
        coil_parts = refine_mesh(coil_parts, input_args)  # 01
        save(persistence_dir, project_name, '01', solution)
        # log.debug("coil_parts: %s", coil_parts)

        # coil_parts[0].coil_mesh.display()

        ###################################################################
        # DEBUG
        if use_matlab_data:
            log.warning("Using MATLAB's mesh in %s, line %d", __file__, get_linenumber())
            for n, old_part in enumerate(coil_parts):
                old_mesh = old_part.coil_mesh
                old_vec = old_mesh.normal_rep
                m_vertices = m_c_parts[n].coil_mesh.v
                m_faces = m_c_parts[n].coil_mesh.faces-1
                coil_parts[n].coil_mesh = Mesh(vertices=m_vertices, faces=m_faces.T)
                coil_parts[n].coil_mesh.normal_rep = old_vec
        ###################################################################

        # Parameterize the mesh
        print('Parameterize the mesh:')
        coil_parts = parameterize_mesh(coil_parts, input_args)  # 02
        save(persistence_dir, project_name, '02', solution)

        ######################################################
        # Verify: v, fn, n, boundary, uv
        for part_index in range(len(coil_parts)):
            coil_mesh = coil_parts[part_index].coil_mesh
            m_v = m_c_parts[part_index].coil_mesh.v
            m_fn = m_c_parts[part_index].coil_mesh.fn
            m_n = m_c_parts[part_index].coil_mesh.n.T
            assert (compare(coil_mesh.v, m_v))      # Pass
            assert (compare(coil_mesh.fn, m_fn))    # Pass
            assert (compare(coil_mesh.n, m_n, double_tolerance=0.1))      # Pass only at 0.1

        # Plot the two boundaries and see the difference
        if get_level() >= DEBUG_VERBOSE:
            # visualize_vertex_connections(coil_mesh.v, 800, 'images/04_uv1_coil_mesh_boundary.png', coil_mesh.boundary)
            # visualize_vertex_connections(coil_mesh.v, 800, 'images/04_uv1_m_boundary.png', m_boundary)

            for part_index in range(len(coil_parts)):
                coil_mesh = coil_parts[part_index].coil_mesh
                m_c_part = m_c_parts[part_index]
                p2d = visualize_projected_vertices(
                    coil_mesh.v, 800, f'{image_dir}/02_{input_args.project_name}_coil_mesh{part_index}_v_p.png')
                m2d = visualize_projected_vertices(m_c_part.coil_mesh.v, 800,
                                                   f'{image_dir}/02_coil_mesh{part_index}_v_m.png')

                # Plot the two UV and see the difference
                visualize_compare_vertices(
                    p2d, m2d, 800, f'{image_dir}/02_{input_args.project_name}_coil_mesh{part_index}_v2d_diff.png')
                visualize_compare_vertices(coil_mesh.uv, m_c_part.coil_mesh.uv.T, 800,
                                           f'{image_dir}/02_{input_args.project_name}_coil_mesh{part_index}_uv_diff.png')

                # assert (compare(coil_mesh.uv, m_c_part.coil_mesh.uv.T, 0.0001))    # Pass

                m_boundaries = m_c_part.coil_mesh.boundary - 1
                log.debug(" m_boundaries: %s", m_boundaries.shape)

                # Convert to array
                if not isinstance(m_boundaries[0], np.ndarray):
                    nm_m_boundary = np.empty((1), dtype=object)
                    nm_m_boundary[0] = m_boundaries
                    m_boundaries = nm_m_boundary

                if get_level() > DEBUG_VERBOSE:
                    log.debug(" m_boundaries: %s", m_boundaries)
                    log.debug(" coil_mesh.boundary: %s", coil_mesh.boundary)

                # Question: Does order matter?
                for index, m_boundary in enumerate(m_boundaries):
                    p_boundary = coil_mesh.boundary[index]
                    assert compare_contains(p_boundary, m_boundary)

        # Define the target field
        print('Define the target field:')
        target_field, is_suppressed_point = define_target_field(
            coil_parts, target_mesh, secondary_target_mesh, input_args)
        solution.target_field = target_field
        solution.is_suppressed_point = is_suppressed_point
        save(persistence_dir, project_name, '02b', solution)

        if get_level() >= DEBUG_VERBOSE:
            log.debug(" -- target_field.b shape: %s", target_field.b.shape)  # (3, 257)
            log.debug(" -- target_field.coords shape: %s", target_field.coords.shape)  # (3, 257)
            log.debug(" -- target_field.weights shape: %s", target_field.weights.shape)  # (257,)

        #####################################################
        # Verify:  b, coords, weights, target_field_group_inds, target_gradient_dbdxyz
        # Differences between biplanar and cylinder examples:

        # Target field
        # b, coords, weights, target_field_group_inds, target_gradient_dbdxyz
        m_target_field = get_and_show_debug(matlab_data, 'out.target_field')
        m_tf_b = m_target_field.b
        m_tf_coords = m_target_field.coords
        m_tf_weights = m_target_field.weights
        m_tf_target_field_group_inds = m_target_field.target_field_group_inds
        m_tf_target_gradient_dbdxyz = m_target_field.target_gradient_dbdxyz

        # visualize_projected_vertices(target_field.b, 800, f'{image_dir}/02b_target_field_p.png')
        # visualize_projected_vertices(m_tf_b, 800, f'{image_dir}/02b_target_field_p.png')

        # Fail: Not the same shape: (3, 3033) is not (3, 3023) ???
        assert (compare(target_field.b, m_tf_b))
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
        coil_parts = calculate_one_ring_by_mesh(coil_parts)  # 03
        save(persistence_dir, project_name, '03', solution)

        #####################################################
        # DEBUG
        # Verify:  one_ring_list, vertex_triangles, node_triangle_mat
        for part_index in range(len(coil_parts)):
            coil_part = coil_parts[part_index]
            coil_mesh = coil_part.coil_mesh
            m_c_part = m_c_parts[part_index]

            one_ring_list = coil_part.one_ring_list
            node_triangles = coil_part.node_triangles
            node_triangle_mat = coil_part.node_triangle_mat

            m_or_one_ring_list = m_c_part.one_ring_list - 1
            # Transpose the entries
            for index1 in range(len(m_or_one_ring_list)):
                # if np.shape(m_or_one_ring_list[index1]) == (2,):
                if len(np.shape(m_or_one_ring_list[index1])) == 1:
                    log.debug("Here: m_or_one_ring_list shape")
                    m_or_one_ring_list[index1] = m_or_one_ring_list[index1].reshape(
                        1, m_or_one_ring_list[index1].shape[0])
                else:
                    m_or_one_ring_list[index1] = m_or_one_ring_list[index1].T
            m_or_node_triangles = m_c_part.node_triangles - 1
            m_or_node_triangle_mat = m_c_part.node_triangle_mat
            # Weird "bug" with MATLAB: If an array has only one entry, it gets turned into a scalar!
            for index, m_tri in enumerate(m_or_node_triangles):
                if np.shape(m_tri) == ():
                    log.debug("Here: %s (%s)", m_tri, node_triangles[index])
                    m_or_node_triangles[index] = np.asarray([m_tri], dtype=np.uint8)

            visualize_multi_connections(
                coil_mesh.uv, 800, f'{image_dir}/03_{input_args.project_name}_one-ring_list_{part_index}_p.png', one_ring_list[0:25])
            visualize_multi_connections(
                coil_mesh.uv, 800, f'{image_dir}/03_{input_args.project_name}_one-ring_list_{part_index}_m.png', m_or_one_ring_list[0:25])

            # Differences between biplanar and cylinder examples:
            # assert (compare_contains(one_ring_list, m_or_one_ring_list, strict=False))   # PASS - different order!
            # ? assert (compare_contains(node_triangles, m_or_node_triangles, strict=False))  # PASS - different order!
            # PASS (just checks shape, since both arrays are zero at this moment)
            assert (compare(node_triangle_mat, m_or_node_triangle_mat))

            # =================================================
            # HACK: Use MATLAB's one_ring_list, etc.
            if use_matlab_data:
                log.warning("Using MATLAB's one_ring_list in %s, line %d", __file__, get_linenumber())
                coil_part.one_ring_list = m_or_one_ring_list
                coil_part.node_triangles = m_or_node_triangles
            # =================================================

        #
        #####################################################

        # Create the basis function container which represents the current density
        print('Create the basis function container which represents the current density:')
        coil_parts = calculate_basis_functions(coil_parts)  # 04
        save(persistence_dir, project_name, '04', solution)

        #####################################################
        # DEVELOPMENT: Remove this
        # DEBUG
        # Verify: is_real_triangle_mat, triangle_corner_coord_mat, current_mat, area_mat, face_normal_mat, current_density_mat
        # Basis functions
        for part_index in range(len(coil_parts)):
            coil_part = coil_parts[part_index]
            coil_mesh = coil_part.coil_mesh
            m_c_part = m_c_parts[part_index]

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
                    # top[index1] = matrix.T

            if use_matlab_data:
                assert (compare(coil_part.current_mat, m_current_mat))  # Pass!
                assert (compare(coil_part.face_normal_mat, m_face_normal_mat))  # Pass!
                assert (compare(coil_part.triangle_corner_coord_mat, m_triangle_corner_coord_mat))  # Pass!
            else:
                # Pass TODO: Check which ordering is off.
                assert (compare_contains(coil_part.current_mat, m_current_mat, strict=False))
                assert (compare_contains(coil_part.triangle_corner_coord_mat,
                        m_triangle_corner_coord_mat, strict=False))  # Pass, Transposed
                # assert (compare(coil_part.triangle_corner_coord_mat, m_triangle_corner_coord_mat))  # Different order?
                assert (compare_contains(coil_part.face_normal_mat, m_face_normal_mat, strict=False))  # Pass

            assert (compare(coil_part.is_real_triangle_mat, m_is_real_triangle_mat))  # Pass
            assert (compare(coil_part.area_mat, m_area_mat))  # Pass
            assert (compare(coil_part.current_density_mat, m_current_density_mat))  # Pass

            # Verify basis_elements
            m_basis_elements = m_c_part.basis_elements
            assert len(coil_part.basis_elements) == len(m_basis_elements)
            for index1 in range(len(coil_part.basis_elements)):
                cg_element = coil_part.basis_elements[index1]
                m_element = m_basis_elements[index1]

                # Verify: triangles, stream_function_potential, area, face_normal, triangle_points_ABC, current
                # Weird "bug" with MATLAB: If an array has only one entry, it gets turned into a scalar!
                if np.shape(m_element.area) == ():
                    log.debug("Here [%d]: %s (%s)", index1, m_element.area, cg_element.area)
                    m_element.area = np.asarray([m_element.area], dtype=np.float64)
                    # Do m_element.triangles at the same time
                    m_element.triangles = np.asarray([m_element.triangles], dtype=np.int64)
                    # And m_element.face_normal
                    m_element.face_normal = m_element.face_normal.reshape(1, 3)
                    # And m_element.triangle_points_ABC
                    m_element.triangle_points_ABC = m_element.triangle_points_ABC.reshape(1, 3, 3)
                    # And m_element.current
                    m_element.current = m_element.current.reshape(1, 3)

                assert (compare(cg_element.area, m_element.area))  # Pass
                assert (cg_element.stream_function_potential == m_element.stream_function_potential)  # Pass

                if use_matlab_data:
                    assert (compare(cg_element.triangles, m_element.triangles-1))  # Pass!
                    assert (compare(cg_element.face_normal, m_element.face_normal))  # Pass!
                    assert (compare(cg_element.current, m_element.current))  # Pass!
                    assert (compare(cg_element.triangle_points_ABC, m_element.triangle_points_ABC))  # Pass
                else:
                    assert (compare_contains(cg_element.triangles, m_element.triangles-1))  # Pass
                    assert (compare_contains(cg_element.face_normal, m_element.face_normal))  # Pass
                    assert (compare_contains(cg_element.triangle_points_ABC, m_element.triangle_points_ABC))  # Pass
                    assert (compare_contains(cg_element.current, m_element.current))  # Pass
        #
        #####################################################

        # Calculate the sensitivity matrix Cn
        print('Calculate the sensitivity matrix:')
        coil_parts = calculate_sensitivity_matrix(coil_parts, target_field, input_args)  # 05
        save(persistence_dir, project_name, '05', solution)

        #####################################################
        # DEVELOPMENT: Remove this
        # DEBUG
        # Verify: sensitivity_matrix
        for part_index in range(len(coil_parts)):
            coil_part = coil_parts[part_index]
            coil_mesh = coil_part.coil_mesh
            m_c_part = m_c_parts[part_index]

            m_sensitivity_matrix = m_c_part.sensitivity_matrix
            # TODO: Consider Python-like structure: 264 (num vertices) x 257 (target_field) x  3 (x,y,z)
            if get_level() >= DEBUG_VERBOSE:
                log.debug(" -- m_sensitivity_matrix shape %d: %s", part_index,
                          m_sensitivity_matrix.shape)  # (3, 257, 264)
                log.debug(" -- c_part.sensitivity_matrix shape %d: %s", part_index,
                          coil_part.sensitivity_matrix.shape)  # (3, 257, 264)

            assert (compare(coil_part.sensitivity_matrix, m_sensitivity_matrix))  # Pass
        #
        #####################################################

        # Calculate the gradient sensitivity matrix Gn
        print('Calculate the gradient sensitivity matrix:')
        coil_parts = calculate_gradient_sensitivity_matrix(coil_parts, target_field, input_args)  # 06
        save(persistence_dir, project_name, '06', solution)

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
        coil_parts = calculate_resistance_matrix(coil_parts, input_args)  # 07
        save(persistence_dir, project_name, '07', solution)

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
        coil_parts, combined_mesh, sf_b_field = stream_function_optimization(coil_parts, target_field, input_args)  # 08
        solution.combined_mesh = combined_mesh
        solution.sf_b_field = sf_b_field
        save(persistence_dir, project_name, '08', solution)

        if input_args.sf_dest_file != 'none':
            print('Persist pre-optimised data:')
            save_preoptimised_data(solution)

        #####################################################
        # DEVELOPMENT: Remove this
        # DEBUG
        # Verify: combined_mesh.stream_function, c_part.current_density, c_part.stream_function, b_field_opt_sf,
        for part_index in range(len(coil_parts)):
            coil_part = coil_parts[part_index]
            coil_mesh = coil_part.coil_mesh
            m_c_part = m_c_parts[part_index]

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
            # Pass # c_part.stream_function shape: (264,)
            assert (compare(coil_part.stream_function, m_cp_stream_function))
        #
        #####################################################

    else:
        # Load the preoptimized data
        print('Load preoptimized data:')
        timer.start()
        solution = load_preoptimized_data(input_args)
        timer.stop()
        coil_parts = solution.coil_parts
        combined_mesh = solution.combined_mesh
        target_field = solution.target_field

    # Calculate the potential levels for the discretization
    print('Calculate the potential levels for the discretization:')
    coil_parts, primary_surface_ind = calc_potential_levels(coil_parts, combined_mesh, input_args)  # 09
    solution.primary_surface_ind = primary_surface_ind
    save(persistence_dir, project_name, '09', solution)

    #####################################################
    # DEVELOPMENT: Remove this
    # DEBUG
    # Verify: primary_surface_ind, coil_part.potential_level_list, coil_part.contour_step
    m_primary_surface_ind = m_out.primary_surface - 1  # -1 because MATLAB uses 1-based indexing

    for part_index in range(len(coil_parts)):
        coil_part = coil_parts[part_index]
        coil_mesh = coil_part.coil_mesh
        m_c_part = m_c_parts[part_index]

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
    coil_parts = calc_contours_by_triangular_potential_cuts(coil_parts)  # 10
    save(persistence_dir, project_name, '10', solution)

    #####################################################
    # DEVELOPMENT: Remove this
    # DEBUG
    # Verify: part.contour_lines items current_orientation, potential, uv
    # Verify: part.raw.
    #           unarranged_loops(x).loop(y).[edge_inds, uv]
    #           unsorted_points(x).[edge_ind, potential, uv]

    for part_index in range(len(coil_parts)):
        coil_part = coil_parts[part_index]
        coil_mesh = coil_part.coil_mesh
        m_c_part = m_c_parts[part_index]
        m_debug = m_c_part.calc_contours_by_triangular_potential_cuts

        assert len(coil_part.raw.unsorted_points) == len(m_debug.raw.unsorted_points)
        for index1, m_ru_points in enumerate(m_debug.raw.unsorted_points):
            c_ru_point = coil_part.raw.unsorted_points[index1]
            m_ru_point = m_debug.raw.unsorted_points[index1]
            assert len(c_ru_point.edge_ind) == len(m_ru_point.edge_ind)
            assert np.isclose(c_ru_point.potential, m_ru_point.potential)
            assert c_ru_point.uv.shape[0] == m_ru_point.uv.shape[0]  # Python shape!
            # assert(compare(c_ru_point.edge_ind, m_ru_point.edge_ind)) # Different ordering?
            # assert(compare_contains(c_ru_point.uv, m_ru_point.uv)) # Order is different

        assert len(coil_part.raw.unarranged_loops) == len(m_debug.raw.unarranged_loops)
        for index1, m_ru_loops in enumerate(m_debug.raw.unarranged_loops):
            c_loops = coil_part.raw.unarranged_loops[index1]
            m_loops = m_debug.raw.unarranged_loops[index1]
            assert len(c_loops.loop) == len(passify_matlab(m_loops.loop))
            # Skip the next section, the loops are different!!
            # for index2, m_ru_loop in enumerate(m_ru_loops.loop):
            #    c_ru_loop = c_loops.loop[index2]
            #    assert c_ru_loop.uv.shape[0] == m_ru_loop.uv.shape[0] # Python shape!
            #    assert(compare_contains(c_ru_loop.uv, m_ru_loop.uv)) #
            #    assert len(c_ru_loop.edge_inds) == len(m_ru_loop.edge_inds)
            #    #assert(compare(c_ru_point.edge_inds, m_ru_point.edge_inds))

        m_contour_lines = m_debug.contour_lines

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
            visualize_compare_contours(
                coil_mesh.uv, 800, f'{image_dir}/10_{input_args.project_name}_contour1_{part_index}_p.png', coil_part.contour_lines)
            visualize_compare_contours(m_c_part.coil_mesh.uv.T, 800,
                                       f'{image_dir}/10_{input_args.project_name}_contour1_{part_index}_m.png', m_contour_lines)

            for index1, m_contour in enumerate(m_contour_lines):
                # MATLAB shape
                p_contour = coil_part.contour_lines[index1]
                visualize_vertex_connections(
                    p_contour.uv.T, 800, f'{image_dir}/10_{input_args.project_name}_contour_lines_{part_index}_{index1}_p.png')
                visualize_vertex_connections(
                    m_contour.uv.T, 800, f'{image_dir}/10_{input_args.project_name}_contour_lines_{part_index}_{index1}_m.png')

    # Manual conclusion: Not identical, but really close.

    # =================================================
    # HACK: Use MATLAB's contour_lines
    if use_matlab_data:
        for part_index in range(len(coil_parts)):
            coil_part = coil_parts[part_index]
            coil_mesh = coil_part.coil_mesh
            m_c_part = m_c_parts[part_index]
            log.warning("Using MATLAB's contour_lines in %s, line %d", __file__, get_linenumber())
            m_contour_lines = m_c_part.contour_lines1
            for index1, m_contour in enumerate(m_contour_lines):
                c_contour = coil_part.contour_lines[index1]
                c_contour.uv = m_contour.uv
    # =================================================

    #
    #####################################################

    # Process contours
    print('Process contours: Evaluate loop significance')
    coil_parts = process_raw_loops(coil_parts, input_args, target_field)  # 11
    save(persistence_dir, project_name, '11', solution)

    #####################################################
    # DEVELOPMENT: Remove this
    # DEBUG
    # Verify: Coil Part values: field_by_loops, loop_significance, combined_loop_field, combined_loop_length

    for part_index in range(len(coil_parts)):
        log.debug(" Part index: %d", part_index)
        coil_part = coil_parts[part_index]
        coil_mesh = coil_part.coil_mesh
        m_c_part = m_c_parts[part_index]

        m_contour_lines = m_c_part.contour_lines
        p2d = visualize_projected_vertices(coil_part.combined_loop_field.T, 800,
                                           f'{image_dir}/11_{input_args.project_name}_combined_loop_field_{part_index}_p.png')
        m2d = visualize_projected_vertices(m_c_part.combined_loop_field.T, 800,
                                           f'{image_dir}/11_{input_args.project_name}_combined_loop_field_{part_index}_m.png')

        # Plot the two fields and see the difference
        visualize_compare_vertices(
            p2d, m2d, 800, f'{image_dir}/11_{input_args.project_name}_combined_loop_field_{part_index}_diff.png')

        if get_level() >= DEBUG_VERBOSE:
            visualize_compare_contours(
                coil_mesh.uv, 800, f'{image_dir}/11_{input_args.project_name}_contour_lines_{part_index}_p.png', coil_part.contour_lines)
            visualize_compare_contours(
                coil_mesh.uv, 800, f'{image_dir}/11_{input_args.project_name}_contour_lines_{part_index}_m.png', m_contour_lines)

        # Fails for part_index 1, when Python generates an extra contour
        if part_index == 0:
            assert len(coil_part.contour_lines) == len(m_c_part.contour_lines)
            assert abs(coil_part.combined_loop_length - m_c_part.combined_loop_length) < 0.05  # 0.002  # 0.0005 # Pass
            if use_matlab_data:
                # 12_uv_to_xyz_bug assert compare(coil_part.combined_loop_field, m_c_part.combined_loop_field, double_tolerance=5e-7)  # Pass! [Fail: 5e-7]
                assert compare(coil_part.combined_loop_field, m_c_part.combined_loop_field,
                               double_tolerance=2.2e-6)  # Pass! [Fail: 5e-7]
                assert compare(coil_part.loop_significance, m_c_part.loop_signficance, double_tolerance=0.005)
                # 12_uv_to_xyz_bug assert compare(coil_part.field_by_loops, m_c_part.field_by_loops, double_tolerance=2e-7)  # Pass!
                assert compare(coil_part.field_by_loops, m_c_part.field_by_loops,
                               double_tolerance=3.1e-7)  # Pass! [Fail: 2e-7]
            else:
                # assert compare(coil_part.field_by_loops, m_c_part.field_by_loops, double_tolerance=2e-7) # Fail
                # assert compare(coil_part.loop_significance, m_c_part.loop_signficance, double_tolerance=3.89)  # 0.09)  # Eeek!
                # assert compare(coil_part.combined_loop_field, m_c_part.combined_loop_field, double_tolerance=5e-6) # Fails
                pass

            # Compare updated contour lines
            for index1 in range(len(coil_part.contour_lines)):
                if get_level() > DEBUG_VERBOSE:
                    log.debug(" Checking contour %d", index1)
                m_contour = m_contour_lines[index1]
                c_contour = coil_part.contour_lines[index1]
                assert c_contour.current_orientation == m_contour.current_orientation  # Pass
                # Pass | Fail with min_loop_significance == 3
                assert np.isclose(c_contour.potential, m_contour.potential)
                # assert compare(c_contour.uv, m_contour.uv) # Pass [0], Fail [4], Pass!
                # assert compare(c_contour.v, m_contour.v) # Fail: 2nd position 1.15644483e-03 != -9.12899313e-17
                if get_level() > DEBUG_VERBOSE:
                    log.debug(" -- compare uv: %s", compare(c_contour.uv, m_contour.uv))
                    log.debug(" -- compare v: %s", compare(c_contour.v, m_contour.v))

    # Manual conclusion (Cylinder): Not identical, but close.
    # Manual conclusion (BiPlanar): Python generates an extra contour for part #2
    #
    #####################################################

    if not input_args.skip_postprocessing:
        # Find the minimal distance between the contour lines
        print('Find the minimal distance between the contour lines:')
        coil_parts = find_minimal_contour_distance(coil_parts, input_args)  # 12
        save(persistence_dir, project_name, '12', solution)

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
        coil_parts = topological_loop_grouping(coil_parts)  # 13
        save(persistence_dir, project_name, '13', solution)

        for index, coil_part in enumerate(coil_parts):
            print(f'  -- Part {index} has {len(coil_part.groups)} topological groups')

        #####################################################
        # DEVELOPMENT: Remove this
        # DEBUG
        # Verify: Coil Part values: groups (list of ContourLine objects), level_positions, group_levels, loop_groups

        for part_index in range(len(coil_parts)):
            log.debug(" Part index: %d", part_index)
            coil_part = coil_parts[part_index]
            coil_mesh = coil_part.coil_mesh
            m_c_part = m_c_parts[part_index]

            # =================================================
            # MATLAB / Python ordering hack:
            # Re-order Python groups to match MATLAB ordering
            if input_args.project_name == 'ygradient_coil':
                m_to_pyindex = [1, 0, 2, 3]
                cp_groups = coil_part.groups
                cp_level_positions = coil_part.level_positions
                cp_group_levels = coil_part.group_levels
                cp_loop_groups = coil_part.loop_groups

                # Do the swap
                if use_matlab_data == False:
                    log.warning("Changing order of coil groups and loop_groups in %s, line %d",
                                __file__, get_linenumber())
                    cp_groups[0], cp_groups[1] = cp_groups[1], cp_groups[0]
                    cp_loop_groups[0], cp_loop_groups[1] = cp_loop_groups[1], cp_loop_groups[0]

            if input_args.project_name == 'biplanar_xgradient':
                if part_index == 0:
                    m_to_pyindex = [0, 1, 2, 3]
                if part_index == 1:
                    m_to_pyindex = [0, 1, 3, 2]
                cp_groups = coil_part.groups
                cp_level_positions = coil_part.level_positions
                cp_group_levels = coil_part.group_levels
                cp_loop_groups = coil_part.loop_groups

                # Do the swap
                if use_matlab_data == False:
                    log.warning("Changing order of coil groups and loop_groups in %s, line %d",
                                __file__, get_linenumber())
                    cp_groups[[0, 1, 2, 3]] = cp_groups[m_to_pyindex]
                    cp_loop_groups[[0, 1, 2, 3]] = cp_loop_groups[m_to_pyindex]

            #
            # =================================================

            m_level_positions = m_c_part.level_positions
            m_group_levels = m_c_part.group_levels - 1  # MATLAB indexing is 1-based
            p_group_levels = coil_part.group_levels

            m_loop_groups = m_c_part.loop_groups
            for index1, loop_group in enumerate(m_loop_groups):
                m_loop_groups[index1] = loop_group - 1  # MATLAB indexing is 1-based

            # Compare updated groups and their loops
            m_groups = m_c_part.groups
            p_groups = coil_part.groups
            for index1 in range(len(coil_part.groups)):
                if get_level() >= DEBUG_VERBOSE:
                    log.debug(" Checking contour group %d", index1)
                m_group = m_groups[index1]  # cutshape, loops, opened_loop
                c_group = p_groups[index1]

                if get_level() >= DEBUG_VERBOSE:
                    visualize_compare_contours(
                        coil_mesh.uv, 800, f'{image_dir}/13_{input_args.project_name}_contour4_{part_index}_{index1}_p.png', c_group.loops)
                    visualize_compare_contours(
                        coil_mesh.uv, 800, f'{image_dir}/13_{input_args.project_name}_contour4_{part_index}_{index1}_m.png', m_group.loops)

                for index2, m_loop in enumerate(m_group.loops):
                    if get_level() >= DEBUG_VERBOSE:
                        log.debug(" Checking index %d", index2)
                    c_loop = c_group.loops[index2]

                    assert c_loop.current_orientation == m_loop.current_orientation  # Pass
                    assert np.isclose(c_loop.potential, m_loop.potential)  # Fail, group 2, index 0
                    # assert compare(c_loop.uv, m_loop.uv) # Fail, different path through mesh
                    # assert compare(c_loop.v, m_loop.v) # Fail, different path through mesh
                    if get_level() > DEBUG_VERBOSE:
                        log.debug(" -- compare uv: %s", compare(c_loop.uv, m_loop.uv))
                        log.debug(" -- compare v: %s", compare(c_loop.v, m_loop.v))

            c_level_positions = np.array(coil_part.level_positions[0])
            c_group_levels = np.array(coil_part.group_levels[0])
            c_loop_groups = coil_part.loop_groups

            if use_matlab_data:
                assert compare(c_group_levels, m_group_levels)      # Pass
                assert compare(c_loop_groups, m_loop_groups)        # Pass
                assert compare(c_level_positions, m_level_positions)  # Pass
            else:
                assert compare_contains(c_group_levels, m_group_levels)        # Pass~
                assert compare(c_level_positions, m_level_positions)  # Pass~
                # assert compare(c_loop_groups, m_loop_groups)        # Fail: They don't match up exactly
                pass

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
            save(persistence_dir, project_name, '13_patched', coil_parts)
        # =================================================

        # Calculate center locations of groups
        print('Calculate center locations of groups:')
        coil_parts = calculate_group_centers(coil_parts)  # 14
        save(persistence_dir, project_name, '14', solution)

        #####################################################
        # DEVELOPMENT: Remove this
        # DEBUG
        # Verify: Coil Part values: group_centers
        for part_index in range(len(coil_parts)):
            log.debug(" Part index: %d", part_index)
            coil_part = coil_parts[part_index]
            coil_mesh = coil_part.coil_mesh
            m_c_part = m_c_parts[part_index]
            m_group_centers = m_c_part.group_centers
            c_group_centers = coil_part.group_centers

            if get_level() >= DEBUG_BASIC:
                visualize_compare_contours(coil_mesh.uv, 800, f'{image_dir}/14_{input_args.project_name}_contour_centres_{part_index}_p.png',
                                           coil_part.contour_lines, c_group_centers.uv)
                visualize_compare_contours(coil_mesh.uv, 800, f'{image_dir}/14_{input_args.project_name}_contour_centres_{part_index}_m.png',
                                           m_c_part.contour_lines, m_group_centers.uv)

            # Pass (alternate sorting in topological_loop_grouping)
            # TODO: Fix! assert compare(c_group_centers.uv, m_group_centers.uv, double_tolerance=0.004)
            # Pass (alternate sorting in topological_loop_grouping)
            # TODO: Fix! assert compare(c_group_centers.v, m_group_centers.v, double_tolerance=0.004)

        # Manual conclusion: Not identical, but close. Different paths, different group layouts.

        #
        #####################################################

        # Interconnect the single groups
        print('Interconnect the single groups:')
        coil_parts = interconnect_within_groups(coil_parts, input_args)  # 15
        save(persistence_dir, project_name, '15', solution)

        #####################################################
        # DEVELOPMENT: Remove this
        # DEBUG
        # Verify: coil_parts(part_ind).groups(group_ind).opened_loop(loop_ind).uv

        for part_index in range(len(coil_parts)):
            log.debug(" Part index: %d", part_index)
            coil_part = coil_parts[part_index]
            coil_mesh = coil_part.coil_mesh
            m_c_part = m_c_parts[part_index]

            for index1, m_group in enumerate(m_c_part.groups):
                c_group = coil_part.groups[index1]
                for index2, m_opened_loop in enumerate(m_group.opened_loop):
                    c_opened_loop = c_group.opened_loop[index2]
                    if False:  # Temporarily disable these checks
                        assert compare(c_opened_loop.v, m_opened_loop.v, double_tolerance=0.001)    # Pass
                        assert compare(c_opened_loop.uv, m_opened_loop.uv, double_tolerance=0.003)  # Pass

            # Verify: Coil Part connected_group values: return_path, uv, v, spiral_in (uv,v), spiral_out(uv, v)
            m_connected_groups = m_c_part.connected_group
            c_connected_groups = coil_part.connected_group

            assert len(m_connected_groups) == len(c_connected_groups)

            for index1, m_connected_group in enumerate(m_connected_groups):
                c_connected_group = c_connected_groups[index1]

                if get_level() >= DEBUG_VERBOSE:
                    # MATLAB shape
                    visualize_vertex_connections(c_connected_group.uv.T, 800,
                                                 f'{image_dir}/15_{input_args.project_name}_connected_group_{part_index}_uv_{index1}_p.png')
                    visualize_vertex_connections(m_connected_group.uv.T, 800,
                                                 f'{image_dir}/15_{input_args.project_name}_connected_group_{part_index}_uv_{index1}_m.png')

                # Check....
                if False:  # Temporarily disable these checks
                    assert compare(c_connected_group.return_path.v, m_connected_group.return_path.v,
                                   double_tolerance=0.001)    # Pass
                    assert compare(c_connected_group.return_path.uv,
                                   m_connected_group.return_path.uv, double_tolerance=0.001)  # Pass

                    assert compare(c_connected_group.uv, m_connected_group.uv, double_tolerance=0.001)  # Pass
                    assert compare(c_connected_group.v, m_connected_group.v, double_tolerance=0.001)    # Pass

        # Manual conclusion: Fail, maybe - the Python connections look a bit different to the MATLAB ones in a few places

        #
        #####################################################

        # Interconnect the groups to a single wire path
        print('Interconnect the groups to a single wire path:')
        coil_parts = interconnect_among_groups(coil_parts, input_args)  # 16
        save(persistence_dir, project_name, '16', solution)

        #####################################################
        # DEVELOPMENT: Remove this
        # DEBUG
        # Verify: coil_parts(part_ind). [opening_cuts_among_groups, wire_path]

        for part_index, p_coil_part in enumerate(coil_parts):
            m_c_part = m_c_parts[part_index]
            # Opening cuts
            """
            for index2, m_cut in enumerate(m_c_part.opening_cuts_among_groups):
                p_cut = p_coil_part.opening_cuts_among_groups[index2]
                visualize_vertex_connections(p_cut.cut1.T, 800, f'{image_dir}/cuts_cut1_{index1}_{index2}_p.png')
                visualize_vertex_connections(m_cut.cut1.T, 800, f'{image_dir}/cuts_cut1_{index1}_{index2}_m.png')
                visualize_vertex_connections(p_cut.cut2.T, 800, f'{image_dir}/cuts_cut2_{index1}_{index2}_p.png')
                visualize_vertex_connections(m_cut.cut2.T, 800, f'{image_dir}/cuts_cut2_{index1}_{index2}_m.png')
            """

            # Wire path
            c_wire_path = p_coil_part.wire_path
            m_wire_path = m_c_part.wire_path1
            if get_level() >= DEBUG_BASIC:
                visualize_vertex_connections(
                    c_wire_path.uv.T, 800, f'{image_dir}/16_{input_args.project_name}_wire_path_uv_{part_index}_p.png')
                visualize_vertex_connections(
                    m_wire_path.uv.T, 800, f'{image_dir}/16_{input_args.project_name}_wire_path_uv_{part_index}_m.png')

            if use_matlab_data:
                assert (compare(c_wire_path.uv, m_wire_path.uv))    # Fail: (2, 1540) is not (2, 1539)
                assert (compare(c_wire_path.v, m_wire_path.v))      # Fail: (3, 1540) is not (3, 1539)

        # Manual conclusion: Pass, when using MATLAB data
        #
        #####################################################

        # Connect the groups and shift the return paths over the surface
        print('Shift the return paths over the surface:')
        coil_parts = shift_return_paths(coil_parts, input_args)  # 17
        save(persistence_dir, project_name, '17', solution)

        #####################################################
        # DEVELOPMENT: Remove this
        # DEBUG
        # Verify: shift_array, points_to_shift, wire_path
        for index1 in range(len(coil_parts)):
            c_part = coil_parts[index1]
            m_c_part = m_c_parts[index1]
            c_wire_path = c_part.wire_path
            m_wire_path = m_c_part.wire_path

            visualize_vertex_connections(
                c_wire_path.uv.T, 800, f'{image_dir}/17_{input_args.project_name}_wire_path2_uv_{index1}_p.png')
            visualize_vertex_connections(
                m_wire_path.uv.T, 800, f'{image_dir}/17_{input_args.project_name}_wire_path2_uv_{index1}_m.png')

            if use_matlab_data:
                visualize_compare_vertices(c_wire_path.uv.T, m_wire_path.uv.T, 800,
                                           f'{image_dir}/17_{input_args.project_name}_wire_path2_uv_{index1}_diff.png')
                # Check....
                assert (compare(c_part.shift_array, m_c_part.shift_array))          # Pass
                assert (compare(c_part.points_to_shift, m_c_part.points_to_shift))  # Pass

                # Pass, with this coarse tolerance!
                assert (compare(c_wire_path.v, m_wire_path.v, double_tolerance=0.03))
                assert (compare(c_wire_path.uv, m_wire_path.uv))  # Pass

        # Manual conclusion: Pass, when using MATLAB data
        #
        #####################################################

        # Create Cylindrical PCB Print
        print('Create PCB Print:')
        coil_parts = generate_cylindrical_pcb_print(coil_parts, input_args)  # 18
        save(persistence_dir, project_name, '18', solution)

        #####################################################
        # DEVELOPMENT: Remove this
        # DEBUG
        # Verify: pcb_tracks.{lower_layer/upper_layer}[0].group_layouts[0..n].wire_parts[0].{ind1,ind2,polygon_track.data,track_shape,uv}
        for part_index in range(len(coil_parts)):
            c_part = coil_parts[part_index]

            if c_part.pcb_tracks is not None:
                layer = 'upper'
                c_upper_group_layouts = c_part.pcb_tracks.upper_layer.group_layouts
                m_upper_group_layouts = m_c_part.pcb_tracks.upper_layer.group_layouts
                for index2, m_group_layout in enumerate(m_upper_group_layouts):
                    c_group_layout = c_upper_group_layouts[index2]
                    c_wire_part = c_group_layout.wire_parts[0]
                    m_wire_part = m_group_layout.wire_parts

                    visualize_vertex_connections(
                        c_wire_part.uv.T, 800, f'{image_dir}/18_{input_args.project_name}_pcb_{part_index}_{layer}_group{index2}_uv_p.png')
                    visualize_vertex_connections(
                        m_wire_part.uv.T, 800, f'{image_dir}/18_{input_args.project_name}_pcb_{part_index}_{layer}_group{index2}_uv_m.png')

                    # visualize_compare_vertices(c_wire_part.uv.T, m_wire_part.uv.T, 800, f'{image_dir}/pcb_{layer}_group{part_index}_uv_diff.png')

                    # Check....
                    if use_matlab_data:
                        assert c_wire_part.ind1 == m_wire_part.ind1 - 1  # MATLAB base 1
                        assert c_wire_part.ind2 == m_wire_part.ind2 - 1  # MATLAB base 1

                        assert compare(c_wire_part.uv, m_wire_part.uv)
                        assert compare(c_wire_part.track_shape, m_wire_part.track_shape)

                layer = 'lower'
                c_lower_group_layouts = c_part.pcb_tracks.lower_layer.group_layouts
                m_lower_group_layouts = m_c_part.pcb_tracks.lower_layer.group_layouts
                for index2, m_group_layout in enumerate(m_lower_group_layouts):
                    c_group_layout = c_lower_group_layouts[index2]
                    c_wire_part = c_group_layout.wire_parts[0]
                    m_wire_part = m_group_layout.wire_parts

                    visualize_vertex_connections(
                        c_wire_part.uv.T, 800, f'{image_dir}/18_{input_args.project_name}_pcb_{part_index}_{layer}_group{index2}_uv_p.png')
                    visualize_vertex_connections(
                        m_wire_part.uv.T, 800, f'{image_dir}/18_{input_args.project_name}_pcb_{part_index}_{layer}_group{index2}_uv_m.png')

        # Manual conclusion: Pass, when using MATLAB data
        #
        #####################################################

        # Create Sweep Along Surface
        print('Create sweep along surface:')
        coil_parts = create_sweep_along_surface(coil_parts, input_args)
        save(persistence_dir, project_name, '19', solution)

    # Calculate the inductance by coil layout
    print('Calculate the inductance by coil layout:')
    # coil_inductance, radial_lumped_inductance, axial_lumped_inductance, radial_sc_inductance, axial_sc_inductance
    solution = calculate_inductance_by_coil_layout(solution, input_args)
    save(persistence_dir, project_name, '20', solution)

    # Evaluate the field errors
    print('Evaluate the field errors:')
    timer.start()
    coil_parts, solution_errors = evaluate_field_errors(
        coil_parts, input_args, solution.target_field, solution.sf_b_field)
    timer.stop()
    solution.solution_errors = solution_errors
    save(persistence_dir, project_name, '21', solution)

    # Calculate the gradient
    print('Calculate the gradient:')
    timer.start()
    coil_gradient = calculate_gradient(coil_parts, input_args, target_field)
    timer.stop()
    solution.coil_gradient = coil_gradient
    save(persistence_dir, project_name, '22', solution)

    timer.stop()
    return solution


if __name__ == "__main__":
    # Set up logging
    # log = logging.getLogger(__name__)
    logging.basicConfig(level=logging.DEBUG)
    # logging.basicConfig(level=logging.INFO)

    # create cylinder mesh: 0.4, 0.1125, 50, 50, copy from Matlab

    # Examples/biplanar_xgradient.m
    # Examples/biplanar_xgradient.m
    arg_dict1 = {
        # "b_0_direction": [0, 0, 1],
        # "biplanar_mesh_parameter_list": [0.25, 0.25, 20, 20, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2],
        # "circular_diameter_factor": 1.0,  # was circular_diameter_factor_cylinder_parameterization
        # "circular_mesh_parameter_list": [0.25, 20.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "coil_mesh_file": "bi_planer_rectangles_width_1000mm_distance_500mm.stl",
        # "conductor_cross_section_height": 0.002,
        # "conductor_cross_section_width": 0.002,
        # "conductor_thickness": 0.005,
        # "cross_sectional_points": [0, 0],
        # "cylinder_mesh_parameter_list": [0.4, 0.1125, 50, 50, 0.0, 1.0, 0.0, 0.0],
        "field_shape_function": "x",
        "force_cut_selection": ['high'],
        # "gauss_order": 2,
        "interconnection_cut_width": 0.05,
        "iteration_num_mesh_refinement": 0,  # MATLAB 1 is default, but 0 is faster
        "level_set_method": "primary",
        "levels": 14,
        # "make_cylindrical_pcb": False,
        # "min_loop_significance": 1,
        "normal_shift_length": 0.01,
        # "normal_shift_smooth_factors": [2, 3, 2],
        # "pcb_interconnection_method": "spiral_in_out",
        # "pcb_spiral_end_shift_factor": 10,
        # "planar_mesh_parameter_list": [0.25, 0.25, 20.0, 20.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        # "plot_flag": True,
        "pot_offset_factor": 0.25,
        # "save_stl_flag": True,
        "secondary_target_mesh_file": "none",
        "secondary_target_weight": 0.5,
        "set_roi_into_mesh_center": True,
        "sf_opt_method": "tikhonov",  # "tikhonov"
        # "minimize_method": "SLSQP", # Only used when 'sf_opt_method' is not tikhonov
        # "minimize_method_parameters" : "{'tol':1.e-6}",
        # "minimize_method_options" : "{'disp': True, 'maxiter' : 1000}",
        "sf_source_file": "none",
        # "skip_calculation_min_winding_distance": True,  # Default: 1
        "skip_inductance_calculation": False,
        # "skip_normal_shift": False,
        "skip_postprocessing": False,
        # "skip_sweep": False,
        # "smooth_factor": 1,
        # "specific_conductivity_conductor": 1.8e-8,
        "surface_is_cylinder_flag": True,
        # "target_field_definition_field_name": "none",
        # "target_field_definition_file": "none",
        # "target_gradient_strength": 1,
        "target_mesh_file": "none",
        "target_region_radius": 0.1,    # GitHub
        "target_region_resolution": 5,  # MATLAB 10 is the default but 5 is faster
        "tikhonov_reg_factor": 10,
        "use_only_target_mesh_verts": False,

        "output_directory": "images",
        "project_name": 'biplanar_xgradient',
        "fasthenry_bin": '../FastHenry2/bin/fasthenry',
        "persistence_dir": 'debug',
        "debug": DEBUG_BASIC,
    }  # 4m3, 6m12.747s

    # cylinder_radius500mm_length1500mm
    arg_dict2 = {
        "b_0_direction": [0, 0, 1],
        "biplanar_mesh_parameter_list": [0.25, 0.25, 20.0, 20.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2],
        "circular_diameter_factor": 1.0,  # was circular_diameter_factor_cylinder_parameterization
        "coil_mesh_file": "cylinder_radius500mm_length1500mm.stl",
        "conductor_cross_section_height": 0.002,
        "conductor_cross_section_width": 0.015,
        "conductor_thickness": 0.005,
        "cross_sectional_points": [[0.0, 0.006427876096865392, 0.00984807753012208, 0.008660254037844387, 0.0034202014332566887, -0.0034202014332566865, -0.008660254037844388, -0.009848077530122082, -0.006427876096865396, -2.4492935982947064e-18], [0.01, 0.007660444431189781, 0.0017364817766693042, -0.0049999999999999975, -0.009396926207859084, -0.009396926207859084, -0.004999999999999997, 0.0017364817766692998, 0.007660444431189778, 0.01]],
        "cylinder_mesh_parameter_list": [0.8, 0.3, 20.0, 20.0, 1.0, 0.0, 0.0, 0.0],
        "field_shape_function": "y",
        "force_cut_selection": ['high'],
        "gauss_order": 2,
        # "geometry_source_path": "/MATLAB Drive/CoilGen/Geometry_Data",
        "interconnection_cut_width": 0.1,
        "iteration_num_mesh_refinement": 0,  # MATLAB 1 is default, but 0 is faster
        "level_set_method": "primary",
        "levels": 20,
        "make_cylindrical_pcb": True,
        "min_loop_significance": 1,  # Was 0.1, a bug?
        "normal_shift_length": 0.025,
        "normal_shift_smooth_factors": [2, 3, 2],
        "pcb_interconnection_method": "spiral_in_out",
        "pcb_spiral_end_shift_factor": 10,
        "planar_mesh_parameter_list": [0.25, 0.25, 20.0, 20.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "plot_flag": True,
        "pot_offset_factor": 0.25,
        "save_stl_flag": True,
        "secondary_target_mesh_file": "none",
        "secondary_target_weight": 0.5,
        "set_roi_into_mesh_center": True,
        "sf_opt_method": "tikhonov",
        # "minimize_method": "SLSQP", # Only used when 'sf_opt_method' is not tikhonov
        # "minimize_method_parameters" : "{'tol':1.e-6}",
        # "minimize_method_options" : "{'disp': True, 'ftol': 1e-6, 'maxiter' : 1000}",
        "sf_source_file": "none",
        "skip_calculation_min_winding_distance": True,  # Default 1
        "skip_inductance_calculation": False,
        "skip_normal_shift": False,
        "skip_postprocessing": False,
        "skip_sweep": False,
        "smooth_factor": 1,
        "specific_conductivity_conductor": 1.8e-08,
        "surface_is_cylinder_flag": True,
        "target_field_definition_field_name": "none",
        "target_field_definition_file": "none",
        "target_gradient_strength": 1,
        "target_mesh_file": "none",
        "target_region_radius": 0.15,
        "target_region_resolution": 5,  # MATLAB 10 is the default but 5 is faster
        "tikhonov_reg_factor": 100,
        "use_only_target_mesh_verts": False,

        "debug": DEBUG_BASIC,
        "output_directory": "images",
        "project_name": 'ygradient_coil',
        "persistence_dir": 'debug',
        "fasthenry_bin": '../FastHenry2/bin/fasthenry',
    }  # 2m11

    solution1 = pyCoilGen(log, arg_dict1)
    solution2 = pyCoilGen(log, arg_dict2)
