# System imports
import numpy as np
import json

# Logging
import logging

# Local imports
# Add the sub_functions directory to the Python module search path
import sys
from pathlib import Path
sub_functions_path = Path(__file__).resolve().parent / '..'
print(sub_functions_path)
sys.path.append(str(sub_functions_path))

# Do not move import from here!
from CoilGen import CoilGen
from sub_functions.refine_mesh import refine_mesh_delegated as refine_mesh
from sub_functions.parameterize_mesh import parameterize_mesh
from sub_functions.read_mesh import create_unique_noded_mesh
from sub_functions.data_structures import DataStructure, Mesh, CoilPart, CoilSolution, ConnectedGroup, Shape3D, \
    TopoGroup, ContourLine
from helpers.timing import Timing
from helpers.extraction import load_matlab
from helpers.visualisation import visualize_vertex_connections, visualize_3D_boundary, compare, compare_contains, \
    get_linenumber, visualize_compare_vertices, visualize_projected_vertices, visualize_compare_contours, \
    passify_matlab


def load_numpy(filename) -> CoilSolution:
    [solution] = np.load(filename, allow_pickle=True)
    return solution


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


def develop_split_disconnected_mesh():
    from sub_functions.split_disconnected_mesh import split_disconnected_mesh

    # which = 'shielded_ygradient_coil'
    project_name = 'Preoptimzed_SVD_Coil'

    # Python saved data 10 : Just between calc_contours_by_triangular_potential_cuts and process_raw_loops
    if project_name == 'biplanar':
        matlab_data = load_matlab('debug/biplanar_xgradient')
        mat_data_out = matlab_data['coil_layouts'].out

        solution = load_numpy('debug/coilgen_biplanar_False_10.npy')
    elif project_name == 'ygradient_coil':
        matlab_data = load_matlab('debug/ygradient_coil')
        mat_data_out = matlab_data['coil_layouts'].out

        # solution = load_numpy('debug/coilgen_cylinder_False_10.npy')
        solution = load_numpy('debug/coilgen_cylinder_True_10.npy')
        p_coil_parts = solution.coil_parts
    elif project_name == 'Preoptimzed_SVD_Coil':
        matlab_data = load_matlab(f'debug/{project_name}')
        mat_data_out = matlab_data['coil_layouts'].out

        # Load preoptimized data
        load_path = 'Pre_Optimized_Solutions/source_data_SVD_coil.npy'
        # Load data from load_path
        loaded_data = np.load(load_path, allow_pickle=True)[0]
        # Extract loaded data
        coil_mesh = loaded_data.coil_mesh
        coil_mesh_in = Mesh(vertices=coil_mesh.vertices, faces=coil_mesh.faces)
    else:
        matlab_data = load_matlab(f'debug/{project_name}')
        mat_data_out = matlab_data['coil_layouts'].out

    if not isinstance(mat_data_out.coil_parts, np.ndarray):
        m_c_parts = [mat_data_out.coil_parts]
    else:
        m_c_parts = mat_data_out.coil_parts

    ###################################################################################
    # Function under test
    coil_parts = split_disconnected_mesh(coil_mesh_in)
    ###################################################################################

    # DEBUG
    assert len(coil_parts) == len(m_c_parts)
    for part_ind, m_c_part in enumerate(m_c_parts):
        coil_part = coil_parts[part_ind]
        coil_mesh = coil_part.coil_mesh
        m_coil_mesh = m_c_part.coil_mesh
        assert compare(coil_mesh.get_vertices(), m_coil_mesh.vertices.T)
        assert compare(coil_mesh.get_faces(), m_coil_mesh.faces.T-1)
        assert compare(coil_mesh.unique_vert_inds, m_coil_mesh.unique_vert_inds-1)


def develop_parameterize_mesh():
    from sub_functions.parameterize_mesh import parameterize_mesh
    project_name = 'Preoptimzed_SVD_Coil'
    # project_name = 'ygradient_coil'
    # project_name = 'biplanar'

    # Python saved data 01 : Just after parameterize_mesh
    if project_name == 'biplanar':
        matlab_data = load_matlab('debug/biplanar_xgradient')
        mat_data_out = matlab_data['coil_layouts'].out
        solution = load_numpy('debug/coilgen_biplanar_False_01.npy')

    elif project_name == 'ygradient_coil':
        matlab_data = load_matlab('debug/ygradient_coil')
        mat_data_out = matlab_data['coil_layouts'].out
        # solution = load_numpy('debug/coilgen_cylinder_False_01.npy')
        solution = load_numpy('debug/coilgen_cylinder_True_01.npy')

    elif project_name.startswith('Preoptimzed'):
        # Load preoptimized data
        load_path = 'Pre_Optimized_Solutions/source_data_SVD_coil.npy'
        # Load data from load_path
        loaded_data = np.load(load_path, allow_pickle=True)[0]
        # Extract loaded data
        coil_mesh = loaded_data.coil_mesh
        coil_mesh_in = Mesh(vertices=coil_mesh.vertices, faces=coil_mesh.faces)

        matlab_data = load_matlab(f'debug/{project_name}')
        mat_data_out = matlab_data['coil_layouts'].out

        solution = load_numpy(f'debug/{project_name}_09.npy')
    else:
        matlab_data = load_matlab(f'debug/{project_name}')
        mat_data_out = matlab_data['coil_layouts'].out

        solution = load_numpy(f'debug/{project_name}_01.npy')

    if not isinstance(mat_data_out.coil_parts, np.ndarray):
        m_c_parts = [mat_data_out.coil_parts]
    else:
        m_c_parts = mat_data_out.coil_parts

    input_args = DataStructure(surface_is_cylinder_flag=solution.input_args.surface_is_cylinder_flag,
                               circular_diameter_factor=solution.input_args.circular_diameter_factor,
                               debug=1)

    p_coil_parts = solution.coil_parts

    ###################################################################################
    # Function under test
    p_coil_parts = parameterize_mesh(p_coil_parts, input_args)
    ###################################################################################

    for part_ind, m_c_part in enumerate(m_c_parts):
        coil_mesh = p_coil_parts[part_ind].coil_mesh
        m_c_mesh = m_c_part.coil_mesh

        # Check:
        # - v,n (np.ndarray)  : vertices and vertex normals (m,3), (m,3)
        # - f,fn (np.ndarray) : faces and face normals (n,2), (n,3)
        # - uv (np.ndarray)   : 2D project of mesh (m,2)
        # - boundary (int)    : list of lists boundary vertex indices (n, variable)

        assert compare(coil_mesh.v, m_c_mesh.v)         # M: nv,3
        # assert compare(coil_mesh.n, m_c_mesh.n.T)     # M: 3,nv (vertexNormal(triangulation,.... calculates differently)
        assert compare(coil_mesh.f, m_c_mesh.faces.T-1)  # M: 2,nf
        assert compare(coil_mesh.fn, m_c_mesh.fn)       # M: nf,3
        assert compare(coil_mesh.boundary, m_c_mesh.boundary-1)
        assert compare(coil_mesh.uv, m_c_mesh.uv.T)

        visualize_vertex_connections(coil_mesh.uv, 800, f'images/01_mesh_uv_d_{part_ind}_p.png')
        visualize_vertex_connections(m_c_mesh.uv.T, 800, f'images/01_mesh_uv_d_{part_ind}_m.png')
        visualize_compare_vertices(coil_mesh.uv, m_c_mesh.uv.T, 800, f'images/01_mesh_uv_d_{part_ind}_diff.png')


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
    solution = load_numpy('debug/coilgen_cylinder_False_03.npy')
    p_coil_parts = solution.coil_parts

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
    solution = load_numpy('debug/coilgen_cylinder_False_04.npy')
    p_coil_parts = solution.coil_parts
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
        solution = load_numpy('debug/coilgen_biplanar_False_07.npy')
    else:
        mat_data = load_matlab('debug/ygradient_coil')
        # Python saved data 07 : After calculate_resistance_matrix
        # solution = load_numpy('debug/coilgen_cylinder_True_07.npy')
        solution = load_numpy('debug/coilgen_cylinder_False_07.npy')
    p_coil_parts = solution.coil_parts
    target_field = solution.target_field

    mat_data_out = mat_data['coil_layouts'].out
    m_coil_parts = mat_data_out.coil_parts
    m_coil_part = m_coil_parts

    input_args = DataStructure(tikhonov_reg_factor=10, sf_opt_method='tikhonov',
                               fmincon_parameter=[500.0, 10000000000.0, 1e-10, 1e-10, 1e-10])
    # target_field = mat_data_out.target_field

    debug_data = mat_data_out
    ###################################################################################
    # Function under test
    coil_parts2 = stream_function_optimization(p_coil_parts, target_field, input_args, debug_data)
    ###################################################################################


def develop_calc_potential_levels():
    from sub_functions.calc_potential_levels import calc_potential_levels

    which = 'shielded_gradient_coil'
    # MATLAB saved data

    # Python saved data 14 : After calculate_group_centers
    if which == 'biplanar':
        mat_data = load_matlab('debug/biplanar_xgradient')
        solution = load_numpy('debug/coilgen_biplanar_False_14.npy')
    elif which == 'cylinder':
        mat_data = load_matlab('debug/ygradient_coil')
        # solution = load_numpy('debug/coilgen_cylinder_True_14.npy')
        solution = load_numpy('debug/coilgen_cylinder_False_14.npy')
    else:
        mat_data = load_matlab(f'debug/{which}')
        solution = load_numpy(f'debug/{which}_08.npy')
        input_args = DataStructure(levels=26, pot_offset_factor=0.25, level_set_method='primary')

    p_coil_parts = solution.coil_parts
    m_out = mat_data['coil_layouts'].out

    ###################################################################################
    # Function under test
    coil_parts = calc_potential_levels(p_coil_parts, solution.combined_mesh, input_args)  # , m_out)
    ###################################################################################

    # And now!!
    coil_part = coil_parts[0]


def develop_calc_contours_by_triangular_potential_cuts():
    from sub_functions.calc_contours_by_triangular_potential_cuts import calc_contours_by_triangular_potential_cuts

    # which = 'shielded_ygradient_coil'
    which = 'Preoptimzed_Breast_Coil_0_10'
    # which = 'Preoptimzed_SVD_Coil_0_10'

    # Python saved data 10 : Just between calc_contours_by_triangular_potential_cuts and process_raw_loops
    if which == 'biplanar':
        matlab_data = load_matlab('debug/biplanar_xgradient')
        m_out = matlab_data['coil_layouts'].out

        solution = load_numpy('debug/coilgen_biplanar_False_10.npy')
    elif which == 'ygradient_coil':

        mat_data = load_matlab('debug/cylinder_coil')
        m_out = matlab_data['coil_layouts'].out

        # Python saved data 09: calc_potential_levels
        # solution = load_numpy('debug/coilgen_cylinder_False_09.npy')
        p_coil_parts = solution.coil_parts
        solution = load_numpy('debug/coilgen_cylinder_False_09_True.npy')
        p_coil_parts = solution.coil_parts

    else:
        matlab_data = load_matlab(f'debug/{which}')
        m_out = matlab_data['coil_layouts'].out
        # Put some logic to turn m_c_parts into a list if m_out.coil_parts is not an array.
        solution = load_numpy(f'debug/{which}_09.npy')

    m_c_parts = m_out.coil_parts
    if not isinstance(m_c_parts, np.ndarray):
        m_c_parts = np.asarray([m_c_parts])

    c_coil_parts = solution.coil_parts

    # Use MATLAB input data
    #######################################################
    # Testing the algorithm, so use MATLAB data as input:
    # coil_part.
    #   mesh
    #   stream_function
    p_coil_parts = []
    for index1, m_c_part in enumerate(m_c_parts):
        c_coil_part = c_coil_parts[index1]
        p_coil_part = CoilPart()
        # mesh
        p_coil_part.coil_mesh = Mesh(vertices=m_c_part.coil_mesh.vertices.T, faces=m_c_part.coil_mesh.faces.T-1)
        p_coil_part.coil_mesh.uv = m_c_part.coil_mesh.uv.T
        # stream_function
        p_coil_part.stream_function = m_c_part.stream_function
        p_coil_part.potential_level_list = m_c_part.potential_level_list
        p_coil_parts.append(p_coil_part)

    ###################################################################################
    # Function under test
    coil_parts2 = calc_contours_by_triangular_potential_cuts(p_coil_parts)
    ###################################################################################

    for part_ind, m_c_part in enumerate(m_c_parts):
        coil_part = coil_parts2[part_ind]
        m_debug = m_c_part.calc_contours_by_triangular_potential_cuts

        """
        assert coil_part.num_edges == m_debug.edge_attached_triangles_inds.shape[0]
        # assert edge_nodes.shape[0] == m_debug.edge_nodes2.shape[0]
        # assert compare_contains(edge_nodes,  m_debug.edge_nodes2-1) # Pass, but very slow

        assert len(coil_part.edge_attached_triangles) == len(m_debug.edge_attached_triangles)
        assert coil_part.edge_opposed_nodes.shape[0] == m_debug.edge_opposed_nodes.shape[0]
        # Fail - different route through the mesh!
        # assert compare_contains(edge_opposed_nodes, m_debug.edge_opposed_nodes-1, strict=False)
        assert compare(coil_part.potential_level_list, m_c_part.potential_level_list)
        """

        assert len(coil_part.contour_lines) == len(m_debug.contour_lines)  # Sanity check
        # A bit of visualisation for references
        coil_mesh = coil_part.coil_mesh
        visualize_compare_contours(coil_mesh.uv, 800,
                                   f'images/10_contour1_{part_ind}_d_p.png', coil_part.contour_lines)
        visualize_compare_contours(m_c_part.coil_mesh.uv.T, 800,
                                   f'images/10_contour1_{part_ind}_d_m.png', m_debug.contour_lines)
        """
        for index1, m_contour in enumerate(m_debug.contour_lines):
            p_contour = coil_part.contour_lines[index1]
            visualize_vertex_connections(p_contour.uv.T, 800, 
                                         f'images/10_contour_lines_{part_ind}_{index1}_p.png')
            visualize_vertex_connections(m_contour.uv.T, 800, 
                                         f'images/10_contour_lines_{part_ind}_{index1}_m.png')
        """
        # Now do the actual checking of values
        for index, m_contour in enumerate(m_debug.contour_lines):
            # The ordering is different, so try and find the Python equivalent by looking for
            # the contour line with the same shape and potential as the current MATLAB one....
            found = 0
            found_index = -1
            pcons = []
            for i, pcon in enumerate(coil_part.contour_lines):
                if pcon.uv.shape[0] == m_contour.uv.shape[0] and np.isclose(pcon.uv.min(), m_contour.uv.min()):
                    found += 1
                    found_index = i
                    pcons.append(pcon)
                    if found > 1:
                        log.debug("Trouble!!")
            assert found == 1
            p_contour = coil_part.contour_lines[found_index]
            # Check: potential, current_orientation, uv, v
            assert p_contour.potential == m_contour.potential
            assert p_contour.current_orientation == m_contour.current_orientation
            # assert compare(p_contour.uv, m_contour.uv) # Sometimes contours are rotated around vertices
            assert p_contour.uv.shape == m_contour.uv.shape

        """ Raw parts are the temporary workings for the contours. Only the contours matter!
        assert len(coil_part.raw.unsorted_points) == len(m_debug.raw.unsorted_points)
        for index1, m_ru_point in enumerate(m_debug.raw.unsorted_points):
            c_ru_point = coil_part.raw.unsorted_points[index1]
            assert len(c_ru_point.edge_ind) == len(m_ru_point.edge_ind)
            assert np.isclose(c_ru_point.potential, m_ru_point.potential)
            assert c_ru_point.uv.shape[0] == m_ru_point.uv.shape[0]  # Python shape!
            ## assert (compare(c_ru_point.uv, m_ru_point.uv))  # Order is different
            ## assert (compare(c_ru_point.edge_ind, m_ru_point.edge_ind-1))  # Completely different!!


        assert len(coil_part.raw.unarranged_loops) == len(m_debug.raw.unarranged_loops)
        for index1, m_potential_group in enumerate(m_debug.raw.unsorted_points):
            p_potential_group = coil_part.raw.unarranged_loops[index1]
            for index2, m_unarranged_loops in enumerate(m_debug.raw.unarranged_loops):
                m_loop_container = m_unarranged_loops.loop
                if not isinstance(m_loop_container, np.ndarray): # MATLAB has annoying habit of making single element arrays into items.
                    m_loop_container = [m_loop_container]

                p_loop_container = coil_part.raw.unarranged_loops[index2]
                assert len(p_loop_container.loop) == len(m_loop_container)
                for index3, m_loop in enumerate(m_loop_container):

                    # The ordering is different, so try and find the Python equivalent by looking for
                    # the loop with the same shape and potential as the current MATLAB one....
                    found = 0
                    found_index = -1
                    plops = []
                    for i, plop in enumerate(p_loop_container.loop):
                        if plop.uv.shape[0] == m_loop.uv.shape[0] and plop.uv.min() == m_loop.uv.min():
                            found += 1
                            found_index = i
                            plops.append(plop)
                            if found > 1:
                                log.debug('Trouble!!')
                    assert found == 1

                    p_loop = p_loop_container.loop[found_index]
                    assert len(p_loop.edge_inds) == len(m_loop.edge_inds)                        
                    assert compare_contains(p_loop.edge_inds, m_loop.edge_inds-1)
                    assert p_loop.uv.shape == m_loop.uv.shape
                    # Never used. Python orientation is sometimes opposite to MATLAB
                    # assert p_loop.current_orientation == m_loop.current_orientation
            """


def develop_process_raw_loops():
    from sub_functions.process_raw_loops import process_raw_loops

    # which = 'ygradient_coil_0_5'
    # which = 'biplanar_xgradient_0_5'
    # which = 'shielded_ygradient_coil'
    # which = 'Preoptimzed_Breast_Coil_0_10'
    which = 'Preoptimzed_SVD_Coil_0_10'

    # Python saved data 10 : Just between calc_contours_by_triangular_potential_cuts and process_raw_loops
    if which == 'biplanar':
        matlab_data = load_matlab('debug/biplanar_xgradient')
        m_out = matlab_data['coil_layouts'].out
        m_c_parts = m_out.coil_parts

        solution = load_numpy('debug/coilgen_biplanar_False_10.npy')
    elif which == 'ygradient_coil':
        matlab_data = load_matlab('debug/ygradient_coil')
        m_out = matlab_data['coil_layouts'].out
        m_c_parts = m_out.coil_parts

        # solution = load_numpy('debug/coilgen_cylinder_False_10.npy')
        solution = load_numpy('debug/coilgen_cylinder_True_10.npy')
    else:
        matlab_data = load_matlab(f'debug/{which}')
        m_out = matlab_data['coil_layouts'].out

        solution = load_numpy(f'debug/{which}_10.npy')

    input_args = DataStructure(smooth_flag=solution.input_args.smooth_flag, smooth_factor=solution.input_args.smooth_factor,
                               min_loop_significance=solution.input_args.min_loop_significance)

    m_c_parts = m_out.coil_parts
    if not isinstance(m_c_parts, np.ndarray):
        m_c_parts = np.asarray([m_c_parts])

    target_field = m_out.target_field
    # target_field = solution.target_field

    assert solution.input_args.min_loop_significance == m_out.input_data.min_loop_signifcance
    assert solution.input_args.smooth_flag == m_out.input_data.smooth_flag
    assert solution.input_args.smooth_factor == m_out.input_data.smooth_factor

    c_coil_parts = solution.coil_parts

    # Use MATLAB contour lines
    #######################################################
    # Testing the algorithm, so use MATLAB data as input:
    # coil_part.
    #   contour_lines
    #   contour_step
    #   coil_mesh
    p_coil_parts = []
    for index1, m_c_part in enumerate(m_c_parts):
        p_coil_part = CoilPart()
        # contour_lines
        p_coil_part.contour_lines = []
        m_debug = m_c_part.calc_contours_by_triangular_potential_cuts
        for index2, m_contour in enumerate(passify_matlab(m_debug.contour_lines)):
            p_line = ContourLine(uv=m_contour.uv)
            p_coil_part.contour_lines.append(p_line)
        # contour_step
        p_coil_part.contour_step = m_c_part.contour_step
        # coil_mesh (Just use Python's)
        p_coil_part.coil_mesh = c_coil_parts[index1].coil_mesh
    p_coil_parts.append(p_coil_part)

    timer = Timing()
    ###################################################################################
    # Function under test
    timer.start()
    coil_parts2 = process_raw_loops(c_coil_parts, input_args, target_field)
    timer.stop()
    # Preoptimzed_Breast_Coil_0_10
    #   Using sub_functions.uv_to_xyz: 7.306278 seconds
    #   Using Mesh.uv_to_xyz: 13.067177 seconds, but 100% idenitical to MATLAB
    ###################################################################################

    # Tiny logic to tweak double_tolerances when using c_coil_parts
    using_matlab = False
    if using_matlab == False:
        double_tolerance_uv = 0.03
        double_tolerance_v = 0.015

    # Verification
    for part_ind, m_coil_part in enumerate(m_c_parts):
        coil_part = coil_parts2[part_ind]

        coil_mesh = coil_part.coil_mesh
        visualize_compare_contours(coil_mesh.uv, 800,
                                   f'images/11_contour1_{part_ind}_d_p.png', coil_part.contour_lines)
        visualize_compare_contours(m_c_part.coil_mesh.uv.T, 800,
                                   f'images/11_contour1_{part_ind}_d_m.png', m_coil_part.contour_lines)

        # Now check the actual contour values (uv updated, v initialised)
        mapping = np.arange(len(m_coil_part.contour_lines))
        p_to_m = mapping.copy()
        for index, m_contour in enumerate(m_coil_part.contour_lines):
            # p_contour = coil_part.contour_lines[index]
            # Check: uv, v - same input, output should be identical

            # Order is different, so search for matching Python contour index
            found = 0
            found_index = -1
            pcons = []
            for i, pcon in enumerate(coil_part.contour_lines):
                if pcon.uv.shape[0] == m_contour.uv.shape[0] and np.isclose(pcon.uv.min(), m_contour.uv.min()):
                    found += 1
                    found_index = i
                    pcons.append(pcon)
                    if found > 1:
                        log.debug("Trouble!!")
            assert found == 1
            p_contour = coil_part.contour_lines[found_index]
            p_to_m[index] = found_index

            assert p_contour.uv.shape == m_contour.uv.shape
            assert compare(p_contour.uv, m_contour.uv, double_tolerance=double_tolerance_uv)
            assert compare(p_contour.v, m_contour.v, double_tolerance=double_tolerance_v)

        if using_matlab == False:
            coil_part.loop_significance[mapping] = coil_part.loop_significance[p_to_m]
            coil_part.field_by_loops[:, :, mapping] = coil_part.field_by_loops[:, :, p_to_m]  # Can this be done?

        assert len(coil_part.contour_lines) == len(m_coil_part.contour_lines)
        assert abs(coil_part.combined_loop_length - m_coil_part.combined_loop_length) < 1e-10  # Pass
        assert compare(coil_part.combined_loop_field, m_coil_part.combined_loop_field)  # Pass
        assert compare(coil_part.loop_significance, m_coil_part.loop_signficance, double_tolerance=1e-9)
        assert compare(coil_part.field_by_loops, m_coil_part.field_by_loops)  # Pass!


def develop_topological_loop_grouping():
    from sub_functions.topological_loop_grouping import topological_loop_grouping

    # which = 'shielded_ygradient_coil'
    # which = 'Preoptimzed_Breast_Coil_0_10'
    which = 'Preoptimzed_SVD_Coil_0_10'
    # which = 'biplanar_xgradient_0_5'

    # Python saved data 12 : After find_minimal_contour_distance
    if which == 'biplanar_xgradient':
        matlab_data = load_matlab('debug/biplanar_xgradient')
        m_out = matlab_data['coil_layouts'].out
        solution = load_numpy('debug/coilgen_biplanar_False_16.npy')
    elif which == 'ygradient_coil':
        matlab_data = load_matlab('debug/ygradient_coil')
        m_out = matlab_data['coil_layouts'].out
        solution = load_numpy('debug/coilgen_cylinder_True_16.npy')
        # solution = load_numpy('debug/coilgen_cylinder_False_16.npy')
    else:
        matlab_data = load_matlab(f'debug/{which}')
        m_out = matlab_data['coil_layouts'].out
        solution = load_numpy(f'debug/{which}_12.npy')

    m_c_parts = m_out.coil_parts
    if not isinstance(m_c_parts, np.ndarray):
        m_c_parts = np.asarray([m_c_parts])

    input_args = DataStructure(project_name=which)

    c_coil_parts = solution.coil_parts

    # Use MATLAB contour lines
    #######################################################
    # Testing the algorithm, so use MATLAB data as input:
    # coil_part.
    #   contour_lines
    p_coil_parts = []
    for index1, m_c_part in enumerate(m_c_parts):
        p_coil_part = CoilPart()
        # contour_lines
        p_coil_part.contour_lines = []
        for index2, m_contour in enumerate(passify_matlab(m_c_part.contour_lines)):
            p_line = ContourLine(uv=m_contour.uv, v=m_contour.v)
            p_coil_part.contour_lines.append(p_line)
    p_coil_parts.append(p_coil_part)

    ######################################################################################################
    # Function under test
    coil_parts = topological_loop_grouping(p_coil_parts, input_args, m_c_parts)
    ######################################################################################################

    # And now, check the following:
    # loop_groups, group_levels, level_positions, groups
    assert len(coil_parts) == len(m_c_parts)
    for part_ind, m_c_part in enumerate(m_c_parts):
        m_debug = m_c_part.topological_loop_grouping
        coil_part = coil_parts[part_ind]

        # Check that the groups and loops are the same
        # 1. groups
        assert len(coil_part.groups) == len(m_c_part.groups)
        for index1, m_group in enumerate(m_c_part.groups):
            p_group = coil_part.groups[index1]

            m_group_loops = passify_matlab(m_group.loops)

            # 2. loops
            assert len(p_group.loops) == len(m_group_loops)
            for index2, m_loop in enumerate(m_group_loops):
                p_loop = p_group.loops[index2]
                assert compare(p_loop.uv, m_loop.uv)
                assert compare(p_loop.v, m_loop.v, double_tolerance=1e-4)

        # 3. loop_groups
        m_passified = np.empty((len(m_c_part.loop_groups)), dtype=object)
        for i in range(len(m_debug.loop_groups2)):
            m_passified[i] = passify_matlab(m_c_part.loop_groups[i]-1)

        assert len(coil_part.loop_groups) == len(m_passified)
        assert compare(np.array(coil_part.loop_groups, dtype=object), m_passified)

        # 4. level_positions
        m_passified = []
        for i in range(len(m_c_part.level_positions)):
            m_passified.append(passify_matlab(m_c_part.level_positions[i]-1).tolist())

        assert len(coil_part.level_positions) == len(m_passified)
        assert compare(coil_part.level_positions, m_passified)


def develop_calculate_group_centers():
    from sub_functions.calculate_group_centers import calculate_group_centers

    # which = 'shielded_ygradient_coil_0_9'
    # which = 'Preoptimzed_Breast_Coil_0_10'
    which = 'Preoptimzed_SVD_Coil_0_10'
    # which = 'ygradient_coil'
    # which = 'biplanar_xgradient_0_5'

    # Python saved data 13 : After topological_loop_grouping
    if which == 'biplanar':
        matlab_data = load_matlab('debug/biplanar_xgradient')
        m_out = matlab_data['coil_layouts'].out
        solution = load_numpy('debug/coilgen_biplanar_False_13.npy')
    elif which == 'cylinder':
        matlab_data = load_matlab('debug/ygradient_coil')
        m_out = matlab_data['coil_layouts'].out
        solution = load_numpy('debug/coilgen_cylinder_False_13_patched.npy')
    else:
        matlab_data = load_matlab(f'debug/{which}')
        m_out = matlab_data['coil_layouts'].out
        solution = load_numpy(f'debug/{which}_13.npy')

    m_c_parts = m_out.coil_parts
    if not isinstance(m_c_parts, np.ndarray):
        m_c_parts = np.asarray([m_c_parts])
    c_coil_parts = solution.coil_parts

    #######################################################
    # Testing the algorithm, so use MATLAB data as input:
    # coil_part.
    #   mesh
    #   groups
    #       loops.uv
    p_coil_parts = []
    for index1, m_c_part in enumerate(m_c_parts):
        p_coil_part = CoilPart()
        # mesh
        p_coil_part.coil_mesh = Mesh(vertices=m_c_part.coil_mesh.vertices.T, faces=m_c_part.coil_mesh.faces.T-1)
        p_coil_part.coil_mesh.uv = m_c_part.coil_mesh.uv.T
        # groups
        p_coil_part.groups = []
        for index2, m_group in enumerate(m_c_part.groups):
            p_group = TopoGroup()
            # loops
            p_group.loops = []
            for index2, m_loop in enumerate(passify_matlab(m_group.loops)):
                p_line = ContourLine(uv=m_loop.uv, v=m_loop.v)
                p_line.current_orientation = m_loop.current_orientation
                p_line.potential = m_loop.potential
                p_group.loops.append(p_line)
            p_coil_part.groups.append(p_group)
        p_coil_parts.append(p_coil_part)

    ###################################################################################
    # Function under test
    coil_parts = calculate_group_centers(p_coil_parts, m_c_parts)
    ###################################################################################
    # And now!!
    assert len(coil_parts) == len(m_c_parts)
    for part_index, m_c_part in enumerate(m_c_parts):
        coil_part = coil_parts[part_index]

        coil_mesh = coil_part.coil_mesh
        c_group_centers = coil_part.group_centers
        visualize_compare_contours(coil_mesh.uv, 800, f'images/14_{which}_contour_centres_d_{part_index}_p.png',
                                   m_c_part.contour_lines, c_group_centers.uv)
        m_coil_mesh = m_c_part.coil_mesh
        m_group_centers = m_c_part.group_centers
        visualize_compare_contours(m_coil_mesh.uv.T, 800, f'images/14_{which}_contour_centres_d_{part_index}_m.png',
                                   m_c_part.contour_lines, m_group_centers.uv)

        assert len(coil_part.groups) == len(m_c_part.groups)

        m_group_centers = m_c_part.group_centers
        c_group_centers = coil_part.group_centers

        assert c_group_centers.uv.shape == m_group_centers.uv.shape

        assert compare(c_group_centers.uv, m_group_centers.uv)  #
        assert compare(c_group_centers.v, m_group_centers.v)    #


def develop_open_loop_with_3d_sphere():
    from sub_functions.open_loop_with_3d_sphere import open_loop_with_3d_sphere

    # which = 'shielded_ygradient_coil'
    # which = 'Preoptimzed_Breast_Coil'
    which = 'Preoptimzed_SVD_Coil_0_10'
    # which = 'ygradient_coil'
    # which = 'biplanar_xgradient_0_5'
    project_name = 'Preoptimzed_SVD_Coil'

    # Python saved data 13 : After topological_loop_grouping
    if which == 'biplanar':
        matlab_data = load_matlab('debug/biplanar_xgradient')
        m_out = matlab_data['coil_layouts'].out
        solution = load_numpy('debug/coilgen_biplanar_False_15.npy')
    elif which == 'cylinder':
        matlab_data = load_matlab('debug/ygradient_coil')
        m_out = matlab_data['coil_layouts'].out
        solution = load_numpy('debug/coilgen_cylinder_False_15_patched.npy')
        # The Python paths and the MATLAB paths are close but slightly different.
        # This prevents detailed debugging.
        # solution = load_numpy('debug/coilgen_cylinder_True_15.npy')
    else:
        matlab_data = load_matlab(f'debug/{which}')
        m_out = matlab_data['coil_layouts'].out
        solution = load_numpy(f'debug/{which}_15.npy')

    m_c_parts = m_out.coil_parts
    if not isinstance(m_c_parts, np.ndarray):
        m_c_parts = np.asarray([m_c_parts])

    c_coil_parts = solution.coil_parts

    ###################################################################################
    # Function under test : open_loop_with_3d_sphere
    #
    counter = 0
    for part_ind, m_c_part in enumerate(m_c_parts):
        m_top_debug = m_c_part.interconnect_among_groups
        # debug_single_level_ind_debugPreoptimzed_SVD_Coil_2
        mat_data = load_matlab(f'debug/debug_single_level_ind_debug{project_name}_{counter+1}')
        m_level_debug = passify_matlab(mat_data['single_level_ind_debug'])

        m_levels_to_process = m_top_debug.level_ind_debug[counter].levels_to_process
        levels_to_process = passify_matlab(m_levels_to_process-1)

        for single_level_ind in range(len(levels_to_process)):  # 1 level
            mat_data = load_matlab(
                f'debug/debug_interconnect_among_groups_{project_name}_{counter+1}_{single_level_ind+1}')
            m_single_level_debug = mat_data['connection_ind_debug']

            num_connections_to_do = m_level_debug[single_level_ind].num_connections_to_do
            for connect_ind in range(num_connections_to_do):  # 3
                m_connection_ind_debug = m_single_level_debug[connect_ind]
                m_sphere_debug_in = m_connection_ind_debug.open_loop_with_3d_sphere
                m_sphere_debug_out = m_connection_ind_debug.sphere_debug_out1

                curve_points_in = Shape3D(v=m_sphere_debug_in.debug_open1.input.curve_in.v,
                                          uv=m_sphere_debug_in.debug_open1.input.curve_in.uv)
                sphere_center_p = m_sphere_debug_in.debug_open1.input.sphere_centre
                sphere_center = [[sphere_center_p[0]], [sphere_center_p[1]], [sphere_center_p[2]]]  # MATLAB shape
                sphere_diameter = m_sphere_debug_in.debug_open1.input.sphere_diameter

                ###################################################################################
                opened_loop, uv_cut, cut_points = open_loop_with_3d_sphere(
                    curve_points_in, sphere_center, sphere_diameter)
                ###################################################################################
                m_cut_shape_1 = m_c_part.opening_cuts_among_groups[connect_ind].cut1
                m_opened_group_1 = m_connection_ind_debug.opened_group_1
                m_cut_points = m_sphere_debug_out.cut_points

                assert compare(opened_loop.uv, m_opened_group_1.uv)
                assert compare(opened_loop.v, m_opened_group_1.v)
                assert compare(uv_cut, m_cut_shape_1)

        counter += 1

    ###################################################################################


def develop_interconnect_within_groups():
    from sub_functions.interconnect_within_groups import interconnect_within_groups

    # which = 'shielded_ygradient_coil'
    # which = 'Preoptimzed_Breast_Coil'
    which = 'Preoptimzed_SVD_Coil_0_10'
    # which = 'ygradient_coil_0_5'
    # which = 'biplanar_xgradient_0_5'

    # Python saved data 13 : After topological_loop_grouping
    if which == 'biplanar':
        matlab_data = load_matlab('debug/biplanar_xgradient')
        m_out = matlab_data['coil_layouts'].out
        solution = load_numpy('debug/coilgen_biplanar_False_14.npy')
    elif which == 'cylinder':
        matlab_data = load_matlab('debug/ygradient_coil')
        m_out = matlab_data['coil_layouts'].out
        # solution = load_numpy('debug/coilgen_cylinder_True_14.npy')
        solution = load_numpy('debug/coilgen_cylinder_False_14.npy')
    else:
        matlab_data = load_matlab(f'debug/{which}')
        m_out = matlab_data['coil_layouts'].out
        solution = load_numpy(f'debug/{which}_14.npy')

    m_c_parts = m_out.coil_parts
    if not isinstance(m_c_parts, np.ndarray):
        m_c_parts = np.asarray([m_c_parts])

    p_coil_parts = solution.coil_parts

    # input_args = DataStructure(force_cut_selection=['high'], b_0_direction=[0, 0, 1], interconnection_cut_width=0.1)
    input_args = DataStructure(force_cut_selection=solution.input_args.force_cut_selection,
                               b_0_direction=solution.input_args.b_0_direction,
                               interconnection_cut_width=solution.input_args.interconnection_cut_width)

    ###################################################################################
    # Function under test
    coil_parts = interconnect_within_groups(p_coil_parts, input_args)  # , m_c_part)
    ###################################################################################

    # And now!!
    for part_ind, m_c_part in enumerate(m_c_parts):
        coil_part = coil_parts[part_ind]
        # Verify: connected_group, groups.opened_loop
        # Part groups
        m_groups = m_c_part.groups
        # MATLAB to Python order mapping for Preoptimzed_SVD_Coil
        mapping = np.arange(len(m_groups))
        if which == 'Preoptimzed_SVD_Coil_0_10':
            m_to_m = [12, 13, 15, 16, 17, 18, 19, 20]
            m_to_py = [13, 12, 16, 17, 16, 17, 18, 19]
            mapping[m_to_m] = mapping[m_to_py]
        c_groups = coil_part.groups
        assert len(m_groups) == len(c_groups)
        for index1, m_group in enumerate(m_groups):
            c_group = c_groups[mapping[index1]]
            # Just check the cutshape size, as a representative proxy
            if len(c_group.loops) > 1:
                for index2, m_cutshape in enumerate(m_group.cutshape):
                    c_cutshape = c_group.cutshape[index2]
                    assert c_cutshape.uv.shape == m_cutshape.uv.shape
                    assert compare(c_cutshape.uv, m_cutshape.uv, double_tolerance=0.005)

            else:
                assert c_group.cutshape[0].uv.shape == m_group.cutshape.uv.shape

        # Connected Groups
        m_connected_groups = m_c_part.connected_group
        c_connected_groups = coil_part.connected_group
        assert len(m_connected_groups) == len(c_connected_groups)
        for index1, m_connected_group in enumerate(m_connected_groups):
            c_connected_group = c_connected_groups[mapping[index1]]

            # MATLAB shape
            if False:
                visualize_vertex_connections(c_connected_group.uv.T, 800,
                                             f'images/15_connected_group_{part_ind}_uv_d_{index1}_p.png')
                visualize_vertex_connections(m_connected_group.group_debug.uv.T, 800,
                                             f'images/15_connected_group_{part_ind}_uv_d_{index1}_m.png')

            log.debug("Here in %s, line %d - Part: %d, Index: %d", __file__, get_linenumber(), part_ind, index1)

            assert c_connected_group.uv.shape == m_connected_group.uv.shape
            assert c_connected_group.v.shape == m_connected_group.v.shape
            assert c_connected_group.return_path.v.shape == m_connected_group.return_path.v.shape
            assert c_connected_group.spiral_in.v.shape == m_connected_group.spiral_in.v.shape
            assert c_connected_group.spiral_out.v.shape == m_connected_group.spiral_out.v.shape

            # Check....
            """
            log.debug(" return_path.v shape: %s", c_connected_group.return_path.v.shape)
            ### log.debug(" c_connected_group.return_path.v: %s", compare(
            ###    c_connected_group.return_path.v, m_connected_group.return_path.v))  # True
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
            """


def develop_interconnect_among_groups():
    from sub_functions.interconnect_among_groups import interconnect_among_groups

    # which = 'shielded_ygradient_coil'
    # which = 'Preoptimzed_Breast_Coil_0_10'
    which = 'Preoptimzed_SVD_Coil_0_10'
    # which = 'ygradient_coil'
    # which = 'biplanar_xgradient_0_5'

    project_name = 'Preoptimzed_SVD_Coil'
    # project_name = 'Preoptimzed_Breast_Coil'

    # Python saved data 13 : After topological_loop_grouping
    if which == 'biplanar':
        matlab_data = load_matlab('debug/biplanar_xgradient')
        m_out = matlab_data['coil_layouts'].out
        solution = load_numpy('debug/coilgen_biplanar_False_15.npy')
    elif which == 'cylinder':
        matlab_data = load_matlab('debug/ygradient_coil')
        m_out = matlab_data['coil_layouts'].out
        solution = load_numpy('debug/coilgen_cylinder_False_15_patched.npy')
        # The Python paths and the MATLAB paths are close but slightly different.
        # This prevents detailed debugging.
        # solution = load_numpy('debug/coilgen_cylinder_True_15.npy')
    else:
        matlab_data = load_matlab(f'debug/{which}')
        m_out = matlab_data['coil_layouts'].out
        solution = load_numpy(f'debug/{which}_15.npy')

    m_c_parts = m_out.coil_parts
    if not isinstance(m_c_parts, np.ndarray):
        m_c_parts = np.asarray([m_c_parts])

    c_coil_parts = solution.coil_parts

    # Wire path
    for part_ind, m_c_part in enumerate(m_c_parts):
        m_wire_path = m_c_part.wire_path1
        visualize_vertex_connections(m_wire_path.uv.T, 800, f'images/16_{which}_wire_path2_uv_d_{part_ind}_m.png')

    #######################################################
    # Testing the algorithm, so use MATLAB data as input:
    # coil_part.
    #   connected_group
    #       return_path.uv
    #   level_positions
    #   group_levels
    p_coil_parts = []
    for index1, m_c_part in enumerate(m_c_parts):
        p_coil_part = CoilPart()
        # connected_group
        p_coil_part.connected_group = []
        for index2, m_connected_group in enumerate(m_c_part.connected_group):
            p_connected_group = ConnectedGroup()
            p_connected_group.uv = m_connected_group.uv
            p_connected_group.v = m_connected_group.v
            p_connected_group.return_path = Shape3D(uv=passify_matlab(m_connected_group.return_path.uv, magic=1))
            p_coil_part.connected_group.append(p_connected_group)
        p_coil_parts.append(p_coil_part)

        # level_positions
        if project_name == 'Preoptimzed_Breast_Coil':
            p_coil_part.level_positions = [[]]  # [[]]
            m_c_part.level_positions = p_coil_part.level_positions
        else:
            p_coil_part.level_positions = []
        for index2, m_level_position in enumerate(passify_matlab(m_c_part.level_positions)):
            p_coil_part.level_positions.append(passify_matlab(m_level_position).tolist())

        # group_levels
        p_coil_part.group_levels = np.empty((len(m_c_part.group_levels)), dtype=object)
        if project_name == 'Preoptimzed_Breast_Coil':
            p_coil_part.group_levels[0] = m_c_part.group_levels - 1
        else:
            for index2, m_group_level in enumerate(m_c_part.group_levels):
                p_coil_part.group_levels[index2] = m_group_level-1

    #
    #######################################################
    assert solution.input_args.interconnection_cut_width == m_out.input_data.interconnection_cut_width

    input_args = DataStructure(interconnection_cut_width=solution.input_args.interconnection_cut_width,
                               project_name=project_name)
    ###################################################################################
    # Function under test
    coil_parts = interconnect_among_groups(p_coil_parts, input_args, m_c_parts)
    ###################################################################################

    # Wire path
    for part_ind, m_c_part in enumerate(m_c_parts):
        c_wire_path = coil_parts[part_ind].wire_path
        m_wire_path = m_c_part.wire_path1
        visualize_vertex_connections(c_wire_path.uv.T, 800, f'images/16_{which}_wire_path2_uv_d_{part_ind}_p.png')

        # Check....
        assert (compare(c_wire_path.v, m_wire_path.v))  # Pass!
        assert (compare(c_wire_path.uv, m_wire_path.uv))  # Pass!

        visualize_compare_vertices(c_wire_path.uv.T, m_wire_path.uv.T, 800,
                                   f'images/16_{which}_wire_path2_uv_d_{part_ind}_diff.png')


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

    # which = 'shielded_ygradient_coil'
    # which = 'Preoptimzed_Breast_Coil_0_10'
    which = 'Preoptimzed_SVD_Coil_0_10'
    # which = 'ygradient_coil'
    # which = 'biplanar_xgradient_0_5'

    project_name = 'Preoptimzed_SVD_Coil'
    # project_name = 'Preoptimzed_Breast_Coil'

    # Python saved data 16 : After interconnect_among_groups
    if which == 'biplanar':
        matlab_data = load_matlab('debug/biplanar_xgradient')
        m_out = matlab_data['coil_layouts'].out
        solution = load_numpy('debug/coilgen_biplanar_False_16.npy')
    elif which == 'cylinder':
        matlab_data = load_matlab('debug/ygradient_coil')
        m_out = matlab_data['coil_layouts'].out
        solution = load_numpy('debug/coilgen_cylinder_True_16.npy')
        # The Python paths and the MATLAB paths are close but slightly different.
        # This prevents detailed debugging.
        # solution = load_numpy('debug/coilgen_cylinder_True_15.npy')
    else:
        matlab_data = load_matlab(f'debug/{which}')
        m_out = matlab_data['coil_layouts'].out
        solution = load_numpy(f'debug/{which}_16.npy')

    m_c_parts = m_out.coil_parts
    if not isinstance(m_c_parts, np.ndarray):
        m_c_parts = np.asarray([m_c_parts])

    c_coil_parts = solution.coil_parts

    # Wire path
    for part_ind, m_c_part in enumerate(m_c_parts):
        m_wire_path = m_c_part.wire_path1
        visualize_vertex_connections(m_wire_path.uv.T, 800, f'images/17_{which}_wire_path2_uv_d_{part_ind}_m.png')

    input_args = DataStructure(interconnection_cut_width=solution.input_args.interconnection_cut_width,
                               skip_normal_shift=solution.input_args.skip_normal_shift,
                               smooth_flag=solution.input_args.smooth_flag,
                               smooth_factor=solution.input_args.smooth_factor,
                               normal_shift_smooth_factors=solution.input_args.normal_shift_smooth_factors,
                               normal_shift_length=solution.input_args.normal_shift_length)

    c_coil_parts = solution.coil_parts

    # Make doubly sure that the c_coil_parts model ran with the same input parameters as MATLAB
    assert solution.input_args.interconnection_cut_width == m_out.input_data.interconnection_cut_width
    assert solution.input_args.skip_normal_shift == m_out.input_data.skip_normal_shift
    assert solution.input_args.smooth_factor == m_out.input_data.smooth_factor
    assert solution.input_args.normal_shift_smooth_factors == m_out.input_data.normal_shift_smooth_factors.tolist()
    assert solution.input_args.normal_shift_length == m_out.input_data.normal_shift_length

    #######################################################
    # Testing the algorithm, so use MATLAB data as input:
    # coil_part.
    #   wire_path
    #   coil_mesh.uv
    p_coil_parts = []
    for index1, m_c_part in enumerate(m_c_parts):
        p_coil_part = CoilPart()
        # coil_mesh
        p_coil_part.coil_mesh = Mesh(vertices=m_c_part.coil_mesh.vertices.T, faces=m_c_part.coil_mesh.faces.T-1)
        p_coil_part.coil_mesh.uv = m_c_part.coil_mesh.uv.T
        # wire_path
        p_coil_part.wire_path = Shape3D(uv=m_c_part.wire_path1.uv, v=m_c_part.wire_path1.v)
        # Add part to collective
        p_coil_parts.append(p_coil_part)

    ######################################################################################################
    # Function under test
    coil_parts = shift_return_paths(p_coil_parts, input_args)  # , m_c_part)
    ######################################################################################################

    # Verify: shift_array, points_to_shift, wire_path
    for part_ind, m_c_part in enumerate(m_c_parts):
        coil_part = coil_parts[part_ind]

        c_wire_path = coil_parts[part_ind].wire_path
        m_wire_path = m_c_part.wire_path

        visualize_vertex_connections(c_wire_path.uv.T, 800,
                                     f'images/17_{which}_wire_path2_uv_d_{part_ind}_p.png')

        visualize_compare_vertices(c_wire_path.uv.T, m_wire_path.uv.T, 800,
                                   f'images/17_{which}_wire_path2_uv_d_{part_ind}_diff.png')

        # Check....
        assert (compare(coil_part.shift_array, m_c_part.shift_array))          # Pass
        assert (compare(coil_part.points_to_shift, m_c_part.points_to_shift))  # Pass

        assert (compare(c_wire_path.v, m_wire_path.v, double_tolerance=0.03))  # Pass, with this coarse tolerance!
        assert (compare(c_wire_path.uv, m_wire_path.uv))  # Pass


def develop_generate_cylindrical_pcb_print():
    from sub_functions.generate_cylindrical_pcb_print import generate_cylindrical_pcb_print
    mat_data = load_matlab('debug/cylinder_coil')
    m_coil_parts = mat_data['coil_layouts'].out.coil_parts
    m_c_part = m_coil_parts
    solution = load_numpy('debug/coilgen_cylinder_False_17_True.npy')
    p_coil_parts = solution.coil_parts

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

    which = 'shielded_ygradient_coil'
    # MATLAB saved data

    # Python saved data 14 : After calculate_group_centers
    if which == 'biplanar':
        mat_data = load_matlab('debug/biplanar_xgradient')
        solution = load_numpy('debug/coilgen_biplanar_False_14.npy')
    elif which == 'cylinder':
        mat_data = load_matlab('debug/cylinder_coil')
        m_coil_parts = mat_data['coil_layouts'].out.coil_parts
        solution = load_numpy('debug/coilgen_cylinder_False_18.npy')
        points = [[0.0, 0.006427876096865392, 0.00984807753012208, 0.008660254037844387, 0.0034202014332566887, -0.0034202014332566865, -0.008660254037844388, -0.009848077530122082, -0.006427876096865396, -2.4492935982947064e-18],
                  [0.01, 0.007660444431189781, 0.0017364817766693042, -0.0049999999999999975, -0.009396926207859084, -0.009396926207859084, -0.004999999999999997, 0.0017364817766692998, 0.007660444431189778, 0.01]]
        input_args = DataStructure(skip_sweep=0, cross_sectional_points=points, save_stl_flag=True,
                                   specific_conductivity_conductor=1.8e-08, output_directory='images', field_shape_function='y')
    else:
        mat_data = load_matlab(f'debug/{which}_0_9')
        m_out = mat_data['coil_layouts'].out
        solution = load_numpy(f'debug/{which}_18.npy')
        points = [0.0]
        input_args = DataStructure(skip_sweep=0, cross_sectional_points=points, save_stl_flag=True, conductor_thickness=0.005,
                                   specific_conductivity_conductor=1.8e-08, output_directory='images', field_shape_function='y',
                                   project_name=which)

    m_c_parts = m_out.coil_parts
    p_coil_parts = solution.coil_parts

    ###################################################################################
    # Function under development
    # log.warning(" Using MATLAB wirepath")
    # p_coil_parts[0].wire_path.v = m_c_part.wire_path.v
    # p_coil_parts[0].wire_path.uv = m_c_part.wire_path.uv

    coil_parts = create_sweep_along_surface(p_coil_parts, input_args)  # , m_c_part)
    ###################################################################################

    # Verify: layout_surface_mesh, ohmian_resistance
    for index1 in range(len(coil_parts)):
        c_part = coil_parts[index1]
        m_c_part = m_c_parts[index1]
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


def develop_calculate_inductance_by_coil_layout():
    from sub_functions.calculate_inductance_by_coil_layout import calculate_inductance_by_coil_layout

    which = 'biplanarX'
    # MATLAB saved data

    # Python saved data 16 : After interconnect_among_groups (which calculates wire_path)
    if which == 'biplanar':
        mat_data = load_matlab('debug/biplanar_xgradient')
        # solution = load_numpy('debug/coilgen_biplanar_False_16.npy')
        # solution = load_numpy('debug/biplanar_16.npy')
        solution = load_numpy('debug/biplanar_xgradient_16.npy')
        width = 0.002
    else:
        mat_data = load_matlab('debug/ygradient_coil')
        # solution = load_numpy('debug/coilgen_cylinder_True_16.npy')
        # solution = load_numpy('debug/coilgen_cylinder_False_16.npy')
        # solution = load_numpy('debug/cylinder_16.npy')
        solution = load_numpy('debug/ygradient_coil_16.npy')
        width = 0.015

    m_coil_parts = mat_data['coil_layouts'].out.coil_parts
    m_c_part = m_coil_parts

    input_args = DataStructure(conductor_cross_section_width=width, conductor_cross_section_height=0.002,
                               skip_inductance_calculation=False, fasthenry_bin='../FastHenry2/bin/fasthenry')

    ###################################################################################
    # Function under test
    solution = calculate_inductance_by_coil_layout(solution, input_args)  # , m_c_part)
    ###################################################################################

    # And now!!
    for coil_part in solution.coil_parts:
        log.debug("Coil part:")
        log.debug(" coil_part.coil_resistance    = %f", coil_part.coil_resistance)
        log.debug(" coil_part.coil_inductance    = %f", coil_part.coil_inductance)
        log.debug(" coil_part.coil_length        = %f", coil_part.coil_length)
        log.debug(" coil_part.coil_cross_section = %f", coil_part.coil_cross_section)


def develop_load_preoptimized_data():
    from sub_functions.load_preoptimized_data import load_preoptimized_data

    project_name = 'Preoptimzed_Breast_Coil'

    mat_data = load_matlab(f'debug/{project_name}')
    mat_data_out = mat_data['coil_layouts'].out

    input_args = DataStructure(sf_source_file='source_data_breast_coil.npy', debug=1,
                               surface_is_cylinder_flag=False, circular_diameter_factor=1,
                               project_name=project_name)
    ###################################################################################
    # Function under test
    solution = load_preoptimized_data(input_args, matlab_data=mat_data_out)
    ###################################################################################

    log.debug(" Here!")


def develop_evaluate_field_errors():
    from sub_functions.evaluate_field_errors import evaluate_field_errors

    # which = 'ygradient_coil_0_5'              # All Pass!
    # which = 'biplanar_xgradient_0_5'          # All Pass!
    # which = 'biplanar_xgradient_1_10'         # All Pass!
    # which = 'shielded_ygradient_coil_0_9'     # All Pass!
    # which = 'Preoptimzed_Breast_Coil_0_10'    # All Pass!
    which = 'Preoptimzed_SVD_Coil_0_10'       # All Pass!

    # Python saved data 13 : After topological_loop_grouping
    if which == 'biplanar':
        matlab_data = load_matlab('debug/biplanar_xgradient')
        m_out = matlab_data['coil_layouts'].out
        solution = load_numpy('debug/coilgen_biplanar_False_19.npy')
    elif which == 'cylinder':
        matlab_data = load_matlab('debug/ygradient_coil')
        m_out = matlab_data['coil_layouts'].out
        solution = load_numpy('debug/coilgen_cylinder_False_19_patched.npy')
    else:
        matlab_data = load_matlab(f'debug/{which}')
        m_out = matlab_data['coil_layouts'].out
        solution = load_numpy(f'debug/{which}_19.npy')

    m_c_parts = m_out.coil_parts
    if not isinstance(m_c_parts, np.ndarray):
        m_c_parts = np.asarray([m_c_parts])

    c_coil_parts = solution.coil_parts

    #######################################################
    # Testing the algorithm, so use MATLAB data as input:
    # coil_part.
    #   contour_step
    #   contour_lines
    #   wire_path
    #   field_by_loops, if skip_postprocessing = True
    p_coil_parts = []
    m_debug1 = m_c_parts[0].evaluate_field_errors
    for index1, m_c_part in enumerate(m_c_parts):
        m_debug = m_c_part.evaluate_field_errors

        p_coil_part = CoilPart()

        # contour_step
        p_coil_part.contour_step = m_c_part.contour_step

        # contour_lines
        p_coil_part.contour_lines = []
        for index2, m_contour in enumerate(passify_matlab(m_debug.debug1.input.contour_lines)):
            p_line = ContourLine(v=m_contour.v, uv=m_contour.uv.astype(np.float64))
            p_coil_part.contour_lines.append(p_line)

        # wire_path
        p_coil_part.wire_path = m_debug.debug1.input.wire_path

        # Add this coil_part to the list
        p_coil_parts.append(p_coil_part)

    target_field = m_out.target_field  # MATLAB shape (3,n)
    sf_b_field = m_out.b_field_opt_sf.T  # Python shape (n,3) Transpose because Python shape is normally used
    #######################################################
    assert solution.input_args.skip_postprocessing == m_out.input_data.skip_postprocessing

    assert compare(target_field.b, m_debug1.debug2.target_field.b)
    assert compare(target_field.coords, m_debug1.debug2.target_field.coords)

    input_args = DataStructure(skip_postprocessing=solution.input_args.skip_postprocessing)
    ###################################################################################
    # Function under test
    timer = Timing()
    timer.start()
    t_coil_parts, solution_errors = evaluate_field_errors(p_coil_parts, input_args, target_field, sf_b_field)
    # Preoptimzed_SVD_Coil_0_10:Total elapsed time: 120.130148 seconds
    timer.stop()
    ###################################################################################

    # Now, check the computed values:
    # coil_part
    #   field_by_loops
    #   field_by_layout
    for index, m_c_part in enumerate(m_c_parts):
        t_coil_part = t_coil_parts[index]
        assert compare(t_coil_part.field_by_loops2, m_c_part.field_by_loops)  # Pass
        assert compare(t_coil_part.field_by_layout, m_c_part.field_by_layout) # Pass

    assert compare(float(solution_errors.opt_current_layout),
                   m_out.needed_current_layout) # Pass

    assert compare(solution_errors.combined_field_layout, m_out.field_by_layout) # Pass
    assert compare(solution_errors.combined_field_layout_per1Amp, m_out.field_layout_per1Amp) # Pass

    assert compare(solution_errors.combined_field_loops, m_out.field_by_unconnected_loops)  # Pass
    assert compare(solution_errors.combined_field_loops_per1Amp, m_out.field_loops_per1Amp)  # Pass

    fe = solution_errors.field_error_vals
    # Convert all from np.float to float
    for attribute, value in fe.__dict__.items():
        fe.__dict__[attribute] = float(value)
    m_fe = m_out.error_vals
    assert compare(fe.max_rel_error_layout_vs_target, m_fe.max_rel_error_layout_vs_target)  # Pass
    assert compare(fe.mean_rel_error_layout_vs_target, m_fe.mean_rel_error_layout_vs_target)  # Pass

    assert compare(fe.max_rel_error_unconnected_contours_vs_target,
                   m_fe.max_rel_error_unconnected_contours_vs_target)  # Pass
    assert compare(fe.mean_rel_error_unconnected_contours_vs_target,
                   m_fe.mean_rel_error_unconnected_contours_vs_target)  # Pass

    assert compare(fe.max_rel_error_layout_vs_stream_function_field,
                   m_fe.max_rel_error_layout_vs_stream_function_field) # Pass
    assert compare(fe.mean_rel_error_layout_vs_stream_function_field,
                   m_fe.mean_rel_error_layout_vs_stream_function_field) # Pass

    assert compare(fe.max_rel_error_unconnected_contours_vs_stream_function_field,
                   m_fe.max_rel_error_unconnected_contours_vs_stream_function_field)  # Pass
    assert compare(fe.mean_rel_error_unconnected_contours_vs_stream_function_field,
                   m_fe.mean_rel_error_unconnected_contours_vs_stream_function_field)  # Pass


def develop_calculate_gradient():
    from sub_functions.calculate_gradient import calculate_gradient

    # which = 'ygradient_coil_0_5'              # All Pass!
    # which = 'biplanar_xgradient_0_5'          # All Pass!
    # which = 'biplanar_xgradient_1_10'         # All Pass!
    # which = 'shielded_ygradient_coil_0_9'     # All Pass!
    # which = 'Preoptimzed_Breast_Coil_0_10'    # All Pass!
    which = 'Preoptimzed_SVD_Coil_0_10'       # All Pass!

    # Python saved data 13 : After topological_loop_grouping
    matlab_data = load_matlab(f'debug/{which}')
    m_out = matlab_data['coil_layouts'].out
    solution = load_numpy(f'debug/{which}_20.npy')

    m_c_parts = m_out.coil_parts
    if not isinstance(m_c_parts, np.ndarray):
        m_c_parts = np.asarray([m_c_parts])

    c_coil_parts = solution.coil_parts

    #######################################################
    # Testing the algorithm, so use MATLAB data as input:
    # coil_part.
    #   contour_step
    #   contour_lines
    #   wire_path
    #   field_by_loops, if skip_postprocessing = True
    p_coil_parts = []
    for index1, m_c_part in enumerate(m_c_parts):

        p_coil_part = CoilPart()

        # contour_step
        p_coil_part.contour_step = m_c_part.contour_step

        # contour_lines
        p_coil_part.contour_lines = []
        for index2, m_contour in enumerate(passify_matlab(m_c_part.contour_lines)):
            p_line = ContourLine(v=m_contour.v, uv=m_contour.uv.astype(np.float64))
            p_coil_part.contour_lines.append(p_line)

        # wire_path
        p_coil_part.wire_path = m_c_part.wire_path

        # Add this coil_part to the list
        p_coil_parts.append(p_coil_part)

    target_field = m_out.target_field  # MATLAB shape (3,n)
    #######################################################
    assert solution.input_args.field_shape_function == m_out.input_data.field_shape_function

    input_args = DataStructure(field_shape_function=solution.input_args.field_shape_function)
    ###################################################################################
    # Function under test
    timer = Timing()
    timer.start()
    layout_gradient = calculate_gradient(p_coil_parts, input_args, target_field)  # , m_c_parts)
    timer.stop()
    ###################################################################################
    # DEBUG: Preoptimzed_SVD_Coil_0_10
    # direct_biot_savart_gradient_calc_2: helpers.timing:Total elapsed time: 121.136640 seconds
    # direct_biot_savart_gradient_calc_3: helpers.timing:Total elapsed time: 92.586744 seconds
    #   89.131132 seconds
    # Now, check the computed values:

    m_layout_gradient = m_out.layout_gradient  # MATLAB shape (3,n)
    assert compare(layout_gradient.dBxdxyz, m_layout_gradient.dBxdxyz.T) # Pass
    assert compare(layout_gradient.dBydxyz, m_layout_gradient.dBydxyz.T) # Pass
    assert compare(layout_gradient.dBzdxyz, m_layout_gradient.dBzdxyz.T) # Pass

    assert compare(layout_gradient.gradient_in_target_direction,
                   m_layout_gradient.gradient_in_target_direction) # Pass
    assert compare(float(layout_gradient.mean_gradient_in_target_direction),
                   m_layout_gradient.mean_gradient_in_target_direction)  # Pass
    assert compare(float(layout_gradient.std_gradient_in_target_direction),
                   m_layout_gradient.std_gradient_in_target_direction, double_tolerance=1e-4) # Pass


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
    # develop_split_disconnected_mesh()
    # develop_refine_mesh
    # develop_parameterize_mesh()
    # develop_define_target_field
    # develop_calculate_one_ring_by_mesh()
    # develop_calculate_basis_functions()
    # develop_calculate_sensitivity_matrix()
    # calculate_gradient_sensitivity_matrix
    # calculate_resistance_matrix
    # develop_stream_function_optimization()
    # develop_calc_potential_levels()
    # develop_calc_contours_by_triangular_potential_cuts()
    # develop_process_raw_loops()
    # develop_topological_loop_grouping()
    # develop_calculate_group_centers()
    # develop_open_loop_with_3d_sphere()
    # develop_interconnect_within_groups()
    # develop_interconnect_among_groups()
    # develop_shift_return_paths()
    # develop_generate_cylindrical_pcb_print()
    # develop_create_sweep_along_surface()
    # develop_calculate_inductance_by_coil_layout()
    # develop_load_preoptimized_data()
    develop_evaluate_field_errors()
    # develop_calculate_gradient()
    #
    # test_smooth_track_by_folding()
    # from tests.test_split_disconnected_mesh import test_split_disconnected_mesh_stl_file1, \
    #        test_split_disconnected_mesh_stl_file2, test_split_disconnected_mesh_simple_planar_mesh, \
    #        test_split_disconnected_mesh_biplanar_mesh
    # test_split_disconnected_mesh_simple_planar_mesh()
    # test_split_disconnected_mesh_biplanar_mesh()
    # test_split_disconnected_mesh_stl_file1()
    # test_split_disconnected_mesh_stl_file2()
    #
    # from tests.test_mesh import test_get_face_index2
    # test_get_face_index2()
    #
    # from tests.test_mesh import test_uv_to_xyz_planar
    # test_uv_to_xyz_planar()
    #
    # from tests.test_biot_savart_calc_b import test_biot_savart_calc_b_arrays2
    # test_biot_savart_calc_b_arrays2()

