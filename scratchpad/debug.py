# System imports
import sys
from pathlib import Path
import numpy as np
import json

# Logging
import logging

# Local imports
# Add the sub_functions directory to the Python module search path
sub_functions_path = Path(__file__).resolve().parent / '..'
print(sub_functions_path)
sys.path.append(str(sub_functions_path))

# Do not move import from here!
from helpers.visualisation import visualize_vertex_connections, visualize_3D_boundary, compare, get_linenumber, \
    visualize_compare_vertices, visualize_projected_vertices
from helpers.extraction import load_matlab
from sub_functions.data_structures import DataStructure, Mesh, CoilPart
from sub_functions.read_mesh import create_unique_noded_mesh
from sub_functions.parameterize_mesh import parameterize_mesh
from sub_functions.refine_mesh import refine_mesh_delegated as refine_mesh
from CoilGen import CoilGen


def debug1():
    print("Planar mesh")
    from sub_functions.build_planar_mesh import build_planar_mesh

    planar_height = 2.0
    planar_width = 3.0
    num_lateral_divisions = 4
    num_longitudinal_divisions = 5
    rotation_vector_x = 1.0
    rotation_vector_y = 0.0
    rotation_vector_z = 0.0
    rotation_angle = 0.0
    center_position_x = 0.0
    center_position_y = 0.0
    center_position_z = 0.0
    mesh = build_planar_mesh(planar_height, planar_width, num_lateral_divisions, num_longitudinal_divisions,
                             rotation_vector_x, rotation_vector_y, rotation_vector_z, rotation_angle,
                             center_position_x, center_position_y, center_position_z)

    # Create debug data
    planar_mesh = {'vertices': mesh.vertices.tolist(), 'faces': mesh.faces.tolist(), 'normal': mesh.normal.tolist()}
    with open('tests/test_data/planar_mesh.json', 'w') as file:
        json.dump(planar_mesh, file)

    mesh = Mesh(vertices=mesh.vertices, faces=mesh.faces)
    print(mesh.trimesh_obj.is_watertight)

    vertices = mesh.get_vertices()
    faces = mesh.get_faces()
    log.debug(" Vertices shape: %s", vertices.shape)
    vertex_counts = np.bincount(faces.flatten())
    print("vertex_counts: ", vertex_counts)

    visualize_vertex_connections(vertices, 800, 'images/debug1_planar_0.png')
    # mesh.display()

    coil_parts = [CoilPart(coil_mesh=mesh)]
    input_args = DataStructure(sf_source_file='none', iteration_num_mesh_refinement=1)
    coil_parts = refine_mesh(coil_parts, input_args)

    vertices2 = mesh.get_vertices()
    visualize_vertex_connections(vertices2, 800, 'images/debug1_planar_1.png')

    # input_params = DataStructure(surface_is_cylinder_flag=False, circular_diameter_factor=0.0)
    # result = parameterize_mesh(parts, input_params)


# Save a small bi-planar mesh to file
def debug1b():
    print("Bi-planar mesh")
    from sub_functions.build_biplanar_mesh import build_biplanar_mesh

    planar_height = 2.0
    planar_width = 3.0
    num_lateral_divisions = 4
    num_longitudinal_divisions = 5
    target_normal_x = 0.0
    target_normal_y = 0.0
    target_normal_z = 1.0
    center_position_x = 0.0
    center_position_y = 0.0
    center_position_z = 0.0
    plane_distance = 0.5

    mesh = build_biplanar_mesh(planar_height, planar_width, num_lateral_divisions, num_longitudinal_divisions,
                               target_normal_x, target_normal_y, target_normal_z,
                               center_position_x, center_position_y, center_position_z, plane_distance)

    # Create debug data
    planar_mesh = {'vertices': mesh.vertices.tolist(), 'faces': mesh.faces.tolist(), 'normal': mesh.normal.tolist()}
    with open('tests/test_data/biplanar_mesh.json', 'w') as file:
        json.dump(planar_mesh, file)


# A Planar mesh with a hole in the middle
def debug2():
    print("Planar mesh with hole")

    # Create small planar mesh with a hole
    # Define the mesh parameters
    x_min, x_max = -1.0, 1.0
    y_min, y_max = -1.0, 1.0
    hole_min, hole_max = -0.25, 0.25
    num_rows, num_cols = 8, 8

    # Calculate step sizes
    x_step = (x_max - x_min) / num_rows
    y_step = (y_max - y_min) / num_cols

    # Generate vertices
    vertices = []
    for i in range(num_rows + 1):
        for j in range(num_cols + 1):
            x = x_min + i * x_step
            y = y_min + j * y_step

            # Check if the vertex is inside the hole region
            # if hole_min <= x <= hole_max and hole_min <= y <= hole_max:
            #    continue  # Skip this vertex

            z = 0.0  # Z-coordinate is 0 for a planar mesh
            vertices.append([x, y, z])

    vertices = np.array(vertices)

    # Generate faces
    faces = []
    for i in range(num_rows):
        for j in range(num_cols):
            v1 = i * (num_cols + 1) + j
            v2 = v1 + 1
            v3 = v1 + num_cols + 2
            v4 = v1 + num_cols + 1

            # Check if any vertex is inside the hole region
            if (hole_min <= vertices[v1, 0] <= hole_max and hole_min <= vertices[v1, 1] <= hole_max) or \
                    (hole_min <= vertices[v2, 0] <= hole_max and hole_min <= vertices[v2, 1] <= hole_max) or \
                    (hole_min <= vertices[v3, 0] <= hole_max and hole_min <= vertices[v3, 1] <= hole_max) or \
                    (hole_min <= vertices[v4, 0] <= hole_max and hole_min <= vertices[v4, 1] <= hole_max):
                continue  # Skip this face

            faces.append([v3, v2, v1])
            faces.append([v4, v3, v1])

    faces = np.array(faces)
    # Print the vertices and faces arrays
    # print("Vertices:\n", vertices)
    # print("Faces:\n", faces)

    log.debug(" Original vertices shape: %s", vertices.shape)
    log.debug(" Original faces shape: %s", faces.shape)

    planar_mesh = DataStructure(vertices=vertices, faces=faces, normal=np.array([0, 0, 1]))

    mesh = create_unique_noded_mesh(planar_mesh)
    # mesh.display()
    # print (mesh.trimesh_obj.is_watertight)
    vertices = mesh.get_vertices()
    faces = mesh.get_faces()

    vertex_counts = np.bincount(faces.flatten())
    print("vertex_counts: ", vertex_counts)

    log.debug(" Vertices shape: %s", vertices.shape)
    log.debug(" Faces shape: %s", faces.shape)

    parts = [DataStructure(coil_mesh=mesh)]

    input_params = DataStructure(surface_is_cylinder_flag=True, circular_diameter_factor=1.0)
    coil_parts = parameterize_mesh(parts, input_params)
    mesh_part = coil_parts[0].coil_mesh
    visualize_vertex_connections(mesh_part.uv, 800, 'images/planar_hole_projected2.png')


# Planar mesh from a file
def debug3():
    arg_dict = {
        'coil_mesh_file': 'dental_gradient_ccs_single_low.stl',
        'iteration_num_mesh_refinement': 0,  # the number of refinements for the mesh;
        'field_shape_function': 'x',  # definition of the target field
        'debug': 0
    }
    x = CoilGen(log, arg_dict)

    mesh_part = x.coil_parts[0].coil_mesh
    # visualize_vertex_connections(mesh_part.uv, 800, 'images/dental_gradient_projected2.png')
    # mesh_part.display()
    # log.debug(" Target field: %s", x.target_field)
    # log.debug(" coil_parts[0].one_ring_list: %s", x.coil_parts[0].one_ring_list)


# Plain cylindrical mesh
def debug4():
    print("Cylindrical mesh")
    from sub_functions.build_cylinder_mesh import build_cylinder_mesh

    # planar_mesh_parameter_list
    cylinder_height = 0.5
    cylinder_radius = 0.25
    num_circular_divisions = 8
    num_longitudinal_divisions = 6
    rotation_vector_x = 0.0
    rotation_vector_y = 1.0
    rotation_vector_z = 0.0
    rotation_angle = np.pi/4.0  # 0.0

    # cylinder_mesh_parameter_list

    cylinder_mesh = build_cylinder_mesh(cylinder_height, cylinder_radius, num_circular_divisions,
                                        num_longitudinal_divisions, rotation_vector_x, rotation_vector_y,
                                        rotation_vector_z, rotation_angle)

    mesh = create_unique_noded_mesh(cylinder_mesh)
    log.debug(" Normal: %s", mesh.normal_rep)
    print(mesh.trimesh_obj.is_watertight)
    vertices = mesh.get_vertices()
    faces = mesh.get_faces()
    log.debug(" Vertices shape: %s", vertices.shape)

    # DEBUG
    mesh.display()

    from sub_functions.data_structures import DataStructure
    parts = [DataStructure(coil_mesh=mesh)]

    input_params = DataStructure(surface_is_cylinder_flag=True, circular_diameter_factor=1.0, debug=1)
    coil_parts = parameterize_mesh(parts, input_params)
    mesh_part = coil_parts[0].coil_mesh
    visualize_vertex_connections(mesh_part.uv, 800, 'images/cylinder_projected2.png')

# Test Mesh refinement


def debug5():
    # Test 1: Trivial case: A single face
    vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]])
    faces = np.array([[0, 1, 2]])
    mesh = Mesh(vertices=vertices, faces=faces)

    mesh.refine(inplace=True)

    new_vertices = mesh.get_vertices()
    new_faces = mesh.get_faces()

    log.debug(" New vertices: %s -> %s", new_vertices.shape, new_vertices)
    log.debug(" New faces: %s -> %s", new_faces.shape, new_faces)

    # mesh.display()
    coil_parts = [CoilPart(coil_mesh=mesh)]
    input_args = DataStructure(sf_source_file='none', iteration_num_mesh_refinement=1)
    coil_parts = refine_mesh(coil_parts, input_args)

    mesh = coil_parts[0].coil_mesh
    mesh.display()

    new_vertices = mesh.get_vertices()
    new_faces = mesh.get_faces()

    log.debug(" New vertices: %s -> %s", new_vertices.shape, new_vertices)
    log.debug(" New faces: %s -> %s", new_faces.shape, new_faces)


def debug6():
    # Load mesh
    from sub_functions.read_mesh import stlread_local
    output = stlread_local('Geometry_Data/cylinder_radius500mm_length1500mm.stl')
    log.debug(" cylinder_radius500mm_length1500mm: vertices: %s, faces: %s, normals: %s",
              np.shape(output.vertices), np.shape(output.faces), np.shape(output.normals))


def get_connected_vertices(vertex_index, face_indices):
    connected_vertices = set()

    for face in face_indices:
        if vertex_index in face:
            connected_vertices.update(face)

    connected_vertices.remove(vertex_index)  # Remove the input vertex index itself
    return list(connected_vertices)


def develop_calculate_one_ring_by_mesh():  # PAUSED
    from sub_functions.calculate_one_ring_by_mesh import calculate_one_ring_by_mesh

    class MockMesh():
        def __init__(self, m_coil_part) -> None:
            self.m_coil_part = m_coil_part
            self._vertices = self.m_coil_part.coil_mesh.v.copy()  # (264,3)
            m_faces = self.m_coil_part.coil_mesh.faces.copy() - 1
            self._faces = m_faces.T  # (480,3)
            self._vertex_faces = None
            self.n = m_coil_part.coil_mesh.n.T.copy()

        def get_vertices(self):
            return self._vertices

        def get_faces(self):
            return self._faces

        def vertex_faces(self):
            """
            Get all the vertex face connections.

            For each vertex, fetch the indices of the faces that it is connected to.

            Returns:
                ndarray: An array of arrays of vertex indices.

            Debug:
                index[1] ==> [0, 5, 4, 15, 6, 7, 1]
            """
            if self._vertex_faces is None:
                num_vertices = self.m_coil_part.coil_mesh.v.shape[0]
                _vertex_faces = np.empty((num_vertices), dtype=object)
                for i in range(num_vertices):
                    _vertex_faces[i] = []

                for face_index, face in enumerate(self._faces):
                    for vertex_index in face:
                        if vertex_index == 1:
                            log.debug(" Adding %d to vertex 1", face_index)
                        _vertex_faces[vertex_index].append(face_index)

                self._vertex_faces = _vertex_faces

            return self._vertex_faces

    # MATLAB saved data
    mat_data = load_matlab('debug/cylinder_coil')
    mat_data_out = mat_data['coil_layouts'].out
    m_coil_parts = mat_data_out.coil_parts
    m_coil_part = m_coil_parts

    debug_data = m_coil_part
    mock_mesh = MockMesh(m_coil_part)
    p_coil_parts = [DataStructure(coil_mesh=mock_mesh)]
    ###################################################################################
    # Function under test
    coil_parts2 = calculate_one_ring_by_mesh(p_coil_parts)
    ###################################################################################

    m_or_one_ring_list = m_coil_part.one_ring_list - 1
    # Transpose the entries
    for index1 in range(len(m_or_one_ring_list)):
        m_or_one_ring_list[index1] = m_or_one_ring_list[index1].T
    m_or_node_triangles = m_coil_part.node_triangles - 1
    m_or_node_triangle_mat = m_coil_part.node_triangle_mat

    p_coil_part = coil_parts2[0]
    assert (compare(p_coil_part.node_triangle_mat, m_or_node_triangle_mat))  # Pass
    assert (compare(p_coil_part.node_triangles, m_or_node_triangles))       # Fail: Order is different
    assert (compare(p_coil_part.one_ring_list, m_or_one_ring_list))         # Fail: Order is different


def develop_calculate_basis_functions():
    from sub_functions.calculate_basis_functions import calculate_basis_functions
    # Given the MATLAB inputs, below:
    # - node_triangles
    # - one_ring_list
    # does it produce exactly the MATLAB outputs?

    # MATLAB saved data
    mat_data = load_matlab('debug/cylinder_coil')
    mat_data_out = mat_data['coil_layouts'].out
    m_coil_parts = mat_data_out.coil_parts
    m_c_part = m_coil_parts

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Transform MATLAB shapes, indices, etc, to Python...
    m_or_one_ring_list = m_c_part.one_ring_list - 1
    # Transpose the entries
    for index1 in range(len(m_or_one_ring_list)):
        m_or_one_ring_list[index1] = m_or_one_ring_list[index1].T
    m_node_triangles = m_c_part.node_triangles - 1
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # Load Python data
    p_coil_parts = np.load(f'debug/cylinder_coil_python_03_False.npy', allow_pickle=True)

    # Use certain MATLAB inputs:
    p_coil_parts[0].node_triangles = m_node_triangles.copy()
    p_coil_parts[0].one_ring_list = m_or_one_ring_list.copy()

    ###################################################################################
    # Function under test
    coil_parts2 = calculate_basis_functions(p_coil_parts, m_c_part)
    ###################################################################################
    coil_part = coil_parts2[0]

    # Verify:
    #  coil_part.basis_elements:
    #   stream_function_potential, triangles, one_ring, area, face_normal, triangle_points_ABC, current
    for index, m_basis_element in enumerate(m_coil_parts.basis_elements):
        p_basis_element = coil_part.basis_elements[index]
        assert p_basis_element.stream_function_potential == p_basis_element.stream_function_potential
        assert (compare(p_basis_element.triangles, m_basis_element.triangles-1))  # Pass!
        assert (compare(p_basis_element.face_normal, m_basis_element.face_normal))  # Pass!
        assert (compare(p_basis_element.current, m_basis_element.current))  # Pass!
        assert (compare(p_basis_element.triangle_points_ABC, m_basis_element.triangle_points_ABC))  # Pass, transposed

    #  - is_real_triangle_mat, triangle_corner_coord_mat, current_mat, area_mat, face_normal_mat, current_density_mat
    assert (compare(coil_part.is_real_triangle_mat, m_c_part.is_real_triangle_mat))  # Pass
    assert (compare(coil_part.triangle_corner_coord_mat, m_c_part.triangle_corner_coord_mat))  # Pass
    assert (compare(coil_part.current_mat, m_c_part.current_mat))  # Pass
    assert (compare(coil_part.area_mat, m_c_part.area_mat))  # Pass
    assert (compare(coil_part.face_normal_mat, m_c_part.face_normal_mat))  # Pass
    assert (compare(coil_part.current_density_mat, m_c_part.current_density_mat))  # Pass


def develop_calculate_sensitivity_matrix():
    from sub_functions.calculate_sensitivity_matrix import calculate_sensitivity_matrix
    # Given the MATLAB inputs, below:
    # - node_triangles
    # - one_ring_list
    # does it produce exactly the MATLAB outputs?

    # MATLAB saved data
    mat_data = load_matlab('debug/cylinder_coil')
    mat_data_out = mat_data['coil_layouts'].out
    m_coil_parts = mat_data_out.coil_parts
    m_c_part = m_coil_parts

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Transform MATLAB shapes, indices, etc, to Python...

    #
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # Load Python data: calculate_basis_functions
    p_coil_parts = np.load(f'debug/cylinder_coil_python_04_False.npy', allow_pickle=True)
    # Use certain MATLAB inputs:
    target_field = mat_data_out.target_field
    #
    ###################################################################################
    # Function under test
    # Uses: basis_elements
    input_args = DataStructure(gauss_order=2)
    coil_parts2 = calculate_sensitivity_matrix(p_coil_parts, target_field, input_args)
    ###################################################################################
    coil_part = coil_parts2[0]

    # Verify:
    #  coil_part.sensitivity_matrix
    assert (compare(coil_part.sensitivity_matrix, m_c_part.sensitivity_matrix))  #


def develop_stream_function_optimization():
    from sub_functions.stream_function_optimization import stream_function_optimization

    which = 'biplanar'
    # MATLAB saved data

    if which == 'biplanar':
        mat_data = load_matlab('debug/biplanar_xgradient')
        # Python saved data 07 : After calculate_resistance_matrix
        [target_field, is_suppressed_point, p_coil_parts] = np.load('debug/biplanar_coil_python_07_False.npy', allow_pickle=True)
    else:
        mat_data = load_matlab('debug/ygradient_coil')
        # Python saved data 07 : After calculate_resistance_matrix
        #[target_field, is_suppressed_point, p_coil_parts] = np.load('debug/cylinder_coil_python_07_True.npy', allow_pickle=True)
        [target_field, is_suppressed_point, p_coil_parts] = np.load('debug/cylinder_coil_python_07_False.npy', allow_pickle=True)

    mat_data_out = mat_data['coil_layouts'].out
    m_coil_parts = mat_data_out.coil_parts
    m_coil_part = m_coil_parts

    input_args = DataStructure(tikonov_reg_factor=10, sf_opt_method='tikkonov', fmincon_parameter=[500.0, 10000000000.0, 1e-10, 1e-10, 1e-10])
    #target_field = mat_data_out.target_field

    debug_data = mat_data_out    
    ###################################################################################
    # Function under test
    coil_parts2 = stream_function_optimization(p_coil_parts, target_field, input_args, debug_data)
    ###################################################################################


def develop_calc_contours_by_triangular_potential_cuts():
    from sub_functions.calc_contours_by_triangular_potential_cuts import calc_contours_by_triangular_potential_cuts

    # MATLAB saved data
    mat_data = load_matlab('debug/cylinder_coil')
    mat_data_out = mat_data['coil_layouts'].out
    m_coil_parts = mat_data_out.coil_parts
    m_c_part = m_coil_parts

    # Python saved data 09: calc_potential_levels
    # [target_field, is_suppressed_point, p_coil_parts] = np.load('debug/cylinder_coil_python_09_False.npy', allow_pickle=True)
    [target_field, is_suppressed_point, p_coil_parts] = np.load('debug/cylinder_coil_python_09_True.npy', allow_pickle=True)

    ###################################################################################
    # Function under test
    coil_parts2 = calc_contours_by_triangular_potential_cuts(p_coil_parts)
    ###################################################################################

    coil_part = coil_parts2[0]

    # TODO: Check the top-level ordering (20 elements). Can the Python output match the MATLAB output
    #       if I re-order there?

    for index1, m_ru_point in enumerate(m_c_part.raw.unsorted_points):
        c_ru_point = coil_part.raw.unsorted_points[index1]
        visualize_vertex_connections(c_ru_point.uv, 800, f'images/10_ru_point_uv_{index1}_p.png')
        visualize_vertex_connections(m_ru_point.uv, 800, f'images/10_ru_point_uv_{index1}_m.png')

    assert len(coil_part.raw.unsorted_points) == len(m_c_part.raw.unsorted_points)
    for index1, m_ru_point in enumerate(m_c_part.raw.unsorted_points):
        c_ru_point = coil_part.raw.unsorted_points[index1]
        assert len(c_ru_point.edge_ind) == len(m_ru_point.edge_ind)
        assert np.isclose(c_ru_point.potential, m_ru_point.potential)
        assert c_ru_point.uv.shape[0] == m_ru_point.uv.shape[0]  # Python shape!
        assert (compare(c_ru_point.uv, m_ru_point.uv))  # Order is different
        assert (compare(c_ru_point.edge_ind, m_ru_point.edge_ind))  # Completely different!!

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


def develop_process_raw_loops():
    from sub_functions.process_raw_loops import process_raw_loops

    # MATLAB saved data
    mat_data = load_matlab('debug/cylinder_coil')
    mat_data_out = mat_data['coil_layouts'].out
    m_coil_parts = mat_data_out.coil_parts
    m_coil_part = m_coil_parts

    # Python saved data 10 : Just between calc_contours_by_triangular_potential_cuts and process_raw_loops
    # [target_field, is_suppressed_point, p_coil_parts] = np.load('debug/cylinder_coil_python_10_False.npy', allow_pickle=True)
    [target_field, is_suppressed_point, p_coil_parts] = np.load('debug/cylinder_coil_python_10_True.npy', allow_pickle=True)

    input_args = DataStructure(smooth_flag=1, smooth_factor=1, min_loop_significance=1)
    target_field = mat_data_out.target_field

    debug_data = m_coil_part
    ###################################################################################
    # Function under test
    coil_parts2 = process_raw_loops(p_coil_parts, input_args, target_field)
    ###################################################################################

    # Verification
    coil_part = coil_parts2[0]
    assert len(coil_part.contour_lines) == len(m_coil_part.contour_lines)
    assert compare(coil_part.combined_loop_field, m_coil_part.combined_loop_field, double_tolerance=5e-7)  # Pass!
    assert compare(coil_part.loop_significance, m_coil_part.loop_signficance, double_tolerance=0.005)
    assert compare(coil_part.field_by_loops, m_coil_part.field_by_loops, double_tolerance=2e-7)  # Pass!

    # Checks:
    coil_part = coil_parts2[0]
    assert abs(coil_part.combined_loop_length - m_coil_part.combined_loop_length) < 0.0005  # Pass
    assert compare(coil_part.combined_loop_field, m_coil_part.combined_loop_field, double_tolerance=5e-7)  # Pass
    assert compare(coil_part.loop_significance, m_coil_part.loop_signficance, double_tolerance=0.005)
    assert compare(coil_part.field_by_loops, m_coil_part.field_by_loops, double_tolerance=2e-7)  # Pass!


def develop_calculate_group_centers():
    from sub_functions.calculate_group_centers import calculate_group_centers
    mat_data = load_matlab('debug/cylinder_coil')
    m_coil_parts = mat_data['coil_layouts'].out.coil_parts
    m_c_part = m_coil_parts
    [target_field, is_suppressed_point, p_coil_parts] = np.load('debug/cylinder_coil_python_13_False_patched.npy', allow_pickle=True)
    # [target_field, is_suppressed_point, p_coil_parts] = np.load('debug/cylinder_coil_python_13_True_patched.npy', allow_pickle=True)

    ###################################################################################
    # Function under test
    coil_parts = calculate_group_centers(p_coil_parts)
    ###################################################################################

    # And now!!
    coil_part = coil_parts[0]

    m_group_centers = m_c_part.group_centers
    c_group_centers = coil_part.group_centers

    assert compare(c_group_centers.uv, m_group_centers.uv)  #
    assert compare(c_group_centers.v, m_group_centers.v)    #


def develop_interconnect_within_groups():
    from sub_functions.interconnect_within_groups import interconnect_within_groups

    which = 'biplanar'
    # MATLAB saved data

    # Python saved data 14 : After calculate_group_centers
    if which == 'biplanar':
        mat_data = load_matlab('debug/biplanar_xgradient')
        [target_field, is_suppressed_point, p_coil_parts] = np.load('debug/biplanar_coil_python_14_False.npy', allow_pickle=True)
    else:
        mat_data = load_matlab('debug/ygradient_coil')
        #[target_field, is_suppressed_point, p_coil_parts] = np.load('debug/cylinder_coil_python_07_True.npy', allow_pickle=True)
        [target_field, is_suppressed_point, p_coil_parts] = np.load('debug/cylinder_coil_python_14_False.npy', allow_pickle=True)

    m_coil_parts = mat_data['coil_layouts'].out.coil_parts
    m_c_part = m_coil_parts

    input_args = DataStructure(force_cut_selection=['high'], b_0_direction=[0, 0, 1], interconnection_cut_width=0.1)

    ###################################################################################
    # Function under test
    coil_parts = interconnect_within_groups(p_coil_parts, input_args)#, m_c_part)
    ###################################################################################

    # And now!!
    coil_part = coil_parts[0]

    # Verify: connected_group, groups.opened_loop

    # Part groups
    m_groups = m_c_part.groups
    c_groups = coil_part.groups
    assert len(m_groups) == len(c_groups)
    for index1, m_group in enumerate(m_groups):
        c_group = c_groups[index1]
        for index2, m_loop in enumerate(m_group.opened_loop):
            c_loop = c_group.opened_loop[index2]
            assert compare(c_loop.uv, m_loop.uv)
            assert compare(c_loop.v, m_loop.v)

    # Connected Groups
    m_connected_groups = m_c_part.connected_group
    c_connected_groups = coil_part.connected_group
    assert len(m_connected_groups) == len(c_connected_groups)
    for index1, m_connected_group in enumerate(m_connected_groups):
        c_connected_group = c_connected_groups[index1]

        # MATLAB shape
        visualize_vertex_connections(c_connected_group.uv.T, 800, f'images/connected_group_uv1_{index1}_p.png')
        visualize_vertex_connections(m_connected_group.group_debug.uv.T, 800,
                                     f'images/connected_group_uv1_{index1}_m.png')

        log.debug(" Here: uv values in %s, line %d", __file__, get_linenumber())

        # Check....
        log.debug(" return_path.v shape: %s", c_connected_group.return_path.v.shape)
        log.debug(" c_connected_group.return_path.v: %s", compare(
            c_connected_group.return_path.v, m_connected_group.return_path.v))  # True
        log.debug(" c_connected_group.return_path.v: %s", compare(
            c_connected_group.return_path.uv, m_connected_group.return_path.uv))

        # Not the same shape: (3, 373) is not (3, 379)
        log.debug(" spiral_in.v shape: %s", c_connected_group.spiral_in.v.shape)
        log.debug(" spiral_in.v: %s", compare(c_connected_group.spiral_in.v, m_connected_group.group_debug.spiral_in.v))
        log.debug(" spiral_in.uv: %s", compare(c_connected_group.spiral_in.uv, m_connected_group.group_debug.spiral_in.uv))

        # Not the same shape: (3, 321) is not (3, 379)
        log.debug(" spiral_out.v: %s", compare(c_connected_group.spiral_out.v, m_connected_group.group_debug.spiral_out.v))
        log.debug(" spiral_out.uv: %s", compare(c_connected_group.spiral_out.uv,
                  m_connected_group.group_debug.spiral_out.uv))

        # Not the same shape: (3, 384) is not (3, 390)
        log.debug(" compare uv: %s", compare(c_connected_group.uv, m_connected_group.uv))
        log.debug(" compare v: %s", compare(c_connected_group.v, m_connected_group.v))


def develop_interconnect_among_groups():
    from sub_functions.interconnect_among_groups import interconnect_among_groups
    mat_data = load_matlab('debug/cylinder_coil')
    m_coil_parts = mat_data['coil_layouts'].out.coil_parts
    m_c_part = m_coil_parts
    # The Python paths and the MATLAB paths are close but slightly different. This prevents detailed debugging.
    # Actually, there is no bug!
    [target_field, is_suppressed_point, p_coil_parts] = np.load('debug/cylinder_coil_python_15_True.npy', allow_pickle=True)

    input_args = DataStructure(interconnection_cut_width=0.1)
    coil_parts = interconnect_among_groups(p_coil_parts, input_args, m_c_part)

    # Wire path
    for index1 in range(len(coil_parts)):
        c_wire_path = coil_parts[index1].wire_path
        m_wire_path = m_c_part.wire_path1

        visualize_vertex_connections(c_wire_path.uv.T, 800, f'images/wire_path_uv_{index1}_p.png')
        visualize_vertex_connections(m_wire_path.uv.T, 800, f'images/wire_path_uv_{index1}_m.png')

        # Check....
        assert (compare(c_wire_path.v, m_wire_path.v))  # Pass!
        assert (compare(c_wire_path.uv, m_wire_path.uv))  # Pass!

        visualize_compare_vertices(c_wire_path.uv.T, m_wire_path.uv.T, 800, f'images/wire_path_uv_{index1}_diff.png')


def test_smooth_track_by_folding():
    from sub_functions.smooth_track_by_folding import smooth_track_by_folding  # 1
    mat_data = load_matlab('debug/cylinder_coil')
    m_coil_parts = mat_data['coil_layouts'].out.coil_parts
    m_c_part = m_coil_parts

    for index1, m_group_layout in enumerate(m_c_part.pcb_tracks.upper_layer.group_layouts):
        m_wire_part = m_group_layout.wire_parts
        input = m_wire_part.point_debug.uv2[:, 1:-1]
        m_debug = m_wire_part.wire_debug
        arr2 = smooth_track_by_folding(input, 3, m_debug)
        assert (compare(arr2, m_debug.arr2))  # Pass


def develop_shift_return_paths():
    from sub_functions.shift_return_paths import shift_return_paths
    mat_data = load_matlab('debug/cylinder_coil')
    m_coil_parts = mat_data['coil_layouts'].out.coil_parts
    m_c_part = m_coil_parts
    [target_field, is_suppressed_point, p_coil_parts] = np.load('debug/cylinder_coil_python_16_True.npy', allow_pickle=True)

    input_args = DataStructure(interconnection_cut_width=0.1,
                               skip_normal_shift=0,
                               smooth_flag=1,
                               smooth_factor=1,
                               normal_shift_smooth_factors=[2, 3, 2],
                               normal_shift_length=0.025)
    coil_parts = shift_return_paths(p_coil_parts, input_args)  # , m_c_part)

    # Verify: shift_array, points_to_shift, wire_path
    for index1 in range(len(coil_parts)):
        c_part = coil_parts[index1]
        c_wire_path = c_part.wire_path
        m_wire_path = m_c_part.wire_path

        visualize_vertex_connections(c_wire_path.uv.T, 800, f'images/17_wire_path2_uv_{index1}_p.png')
        visualize_vertex_connections(m_wire_path.uv.T, 800, f'images/17_wire_path2_uv_{index1}_m.png')

        visualize_compare_vertices(c_wire_path.uv.T, m_wire_path.uv.T, 800,
                                   f'images/17_wire_path2_uv_{index1}_diff.png')

        # Check....
        assert (compare(c_part.shift_array, m_c_part.shift_array))          # Pass
        assert (compare(c_part.points_to_shift, m_c_part.points_to_shift))  # Pass

        assert (compare(c_wire_path.v, m_wire_path.v, double_tolerance=0.03))  # Pass, with this coarse tolerance!
        assert (compare(c_wire_path.uv, m_wire_path.uv))  # Pass


def develop_generate_cylindrical_pcb_print():
    from sub_functions.generate_cylindrical_pcb_print import generate_cylindrical_pcb_print
    mat_data = load_matlab('debug/cylinder_coil')
    m_coil_parts = mat_data['coil_layouts'].out.coil_parts
    m_c_part = m_coil_parts
    [target_field, is_suppressed_point, p_coil_parts] = np.load('debug/cylinder_coil_python_17_True.npy', allow_pickle=True)

    input_args = DataStructure(conductor_cross_section_width=0.015,
                               cylinder_mesh_parameter_list=[0.8, 0.3, 20.0, 20.0, 1.0, 0.0, 0.0, 0.0],
                               surface_is_cylinder_flag=1,
                               make_cylindrical_pcb=1,
                               pcb_interconnection_method='spiral_in_out',
                               pcb_spiral_end_shift_factor=10)
    coil_parts = generate_cylindrical_pcb_print(p_coil_parts, input_args, m_c_part)

    # Verify: pcb_tracks.{lower_layer/upper_layer}[0].group_layouts[0..n].wire_parts[0].{ind1,ind2,polygon_track.data,track_shape,uv}
    for index1 in range(len(coil_parts)):
        c_part = coil_parts[index1]
        c_upper_group_layouts = c_part.pcb_tracks.upper_layer.group_layouts
        m_upper_group_layouts = m_c_part.pcb_tracks.upper_layer.group_layouts

        layer = 'upper'
        for index2, m_group_layout in enumerate(m_upper_group_layouts):
            log.debug(" Checking upper %d", index2)
            c_group_layout = c_upper_group_layouts[index2]
            c_wire_part = c_group_layout.wire_parts[0]
            m_wire_part = m_group_layout.wire_parts

            visualize_vertex_connections(c_wire_part.uv.T, 800, f'images/pcb_{layer}_group{index2}_uv_p.png')
            visualize_vertex_connections(m_wire_part.uv.T, 800, f'images/pcb_{layer}_group{index2}_uv_m.png')

            visualize_compare_vertices(c_wire_part.uv.T, m_wire_part.uv.T, 800,
                                       f'images/pcb_{layer}_group{index2}_uv__diff.png')

            # Check....
            assert c_wire_part.ind1 == m_wire_part.ind1 - 1  # MATLAB base 1
            assert c_wire_part.ind2 == m_wire_part.ind2 - 1  # MATLAB base 1

            assert compare(c_wire_part.uv, m_wire_part.uv)
            assert compare(c_wire_part.track_shape, m_wire_part.track_shape)


def develop_create_sweep_along_surface():
    from sub_functions.create_sweep_along_surface import create_sweep_along_surface
    mat_data = load_matlab('debug/cylinder_coil')
    m_coil_parts = mat_data['coil_layouts'].out.coil_parts
    m_c_part = m_coil_parts
    #[target_field, is_suppressed_point, p_coil_parts] = np.load('debug/cylinder_coil_python_18_True.npy', allow_pickle=True)
    [target_field, is_suppressed_point, p_coil_parts] = np.load('debug/cylinder_coil_python_18_False.npy', allow_pickle=True)

    points = [[0.0, 0.006427876096865392, 0.00984807753012208, 0.008660254037844387, 0.0034202014332566887, -0.0034202014332566865, -0.008660254037844388, -0.009848077530122082, -0.006427876096865396, -2.4492935982947064e-18],
              [0.01, 0.007660444431189781, 0.0017364817766693042, -0.0049999999999999975, -0.009396926207859084, -0.009396926207859084, -0.004999999999999997, 0.0017364817766692998, 0.007660444431189778, 0.01]]
    input_args = DataStructure(skip_sweep=0, cross_sectional_points=points, save_stl_flag=1,
                               specific_conductivity_conductor=1.8e-08, output_directory='images', field_shape_function='y')
    ###################################################################################
    # Function under development
    log.warning(" Using MATLAB wirepath")
    p_coil_parts[0].wire_path.v = m_c_part.wire_path.v
    p_coil_parts[0].wire_path.uv = m_c_part.wire_path.uv

    coil_parts = create_sweep_along_surface(p_coil_parts, input_args)  # , m_c_part)
    ###################################################################################

    # Verify: layout_surface_mesh, ohmian_resistance
    for index1 in range(len(coil_parts)):
        c_part = coil_parts[index1]
        m_ohmian_resistance = m_c_part.ohmian_resistance

        c_surface = c_part.layout_surface_mesh
        m_surface = m_c_part.layout_surface_mesh

        visualize_projected_vertices(c_surface.get_vertices(), 2048, f'images/19_layout_surface_{index1}_uv_p.png')
        visualize_projected_vertices(m_c_part.create_sweep_along_surface.swept_vertices,
                                     2048, f'images/19_layout_surface_{index1}_uv_m.png')

        # visualize_compare_vertices(c_surface.uv.T, m_surface.uv.T, 800,
        #                            f'19_layout_surface_{index1}_uv_diff.png')

        # Check....
        assert c_part.ohmian_resistance == m_ohmian_resistance
        assert compare(c_surface.get_vertices(),
                       m_c_part.create_sweep_along_surface.swept_vertices, double_tolerance=0.003)
        # assert compare(c_wire_part.track_shape, m_wire_part.track_shape)


if __name__ == "__main__":
    # Set up logging
    log = logging.getLogger(__name__)
    logging.basicConfig(level=logging.DEBUG)

    # debug1() # Planar mesh
    # debug1b() # Planar mesh
    # debug2() # Planar mesh with a hole
    # debug3() # Planar mesh from file
    # debug4() # Cylindrical mesh
    # debug5() # Refine a simple 1 face mesh into four.
    # debug6() # Examine cylinder_radius500mm_length1500mm.stl
    # test_add_nearest_ref_point_to_curve()
    # develop_calculate_one_ring_by_mesh()
    # develop_calculate_basis_functions()
    # develop_calculate_sensitivity_matrix()
    # calculate_gradient_sensitivity_matrix
    # calculate_resistance_matrix
    # develop_stream_function_optimization()
    # calc_potential_levels
    # develop_calc_contours_by_triangular_potential_cuts()
    # develop_process_raw_loops()
    # develop_calculate_group_centers()
    develop_interconnect_within_groups()
    # develop_interconnect_among_groups()
    # develop_shift_return_paths()
    # develop_generate_cylindrical_pcb_print()
    # develop_create_sweep_along_surface()
    # test_smooth_track_by_folding()
    #from tests.test_split_disconnected_mesh import test_split_disconnected_mesh_stl_file1, \
    #        test_split_disconnected_mesh_stl_file2, test_split_disconnected_mesh_simple_planar_mesh, \
    #        test_split_disconnected_mesh_biplanar_mesh
    # test_split_disconnected_mesh_simple_planar_mesh()
    # test_split_disconnected_mesh_biplanar_mesh()
    # test_split_disconnected_mesh_stl_file1()
    #test_split_disconnected_mesh_stl_file2()
    # from tests.test_mesh import test_get_face_index2
    # test_get_face_index2()