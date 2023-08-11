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
from helpers.visualisation import visualize_vertex_connections, visualize_3D_boundary, compare, get_linenumber, visualize_compare_vertices
from helpers.extraction import load_matlab
from sub_functions.data_structures import DataStructure, Mesh, CoilPart
from sub_functions.read_mesh import create_unique_noded_mesh
from sub_functions.parameterize_mesh import parameterize_mesh, get_boundary_loop_nodes
from sub_functions.refine_mesh import  refine_mesh_delegated as refine_mesh
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


def test_interconnect_within_groups():
    from sub_functions.interconnect_within_groups import interconnect_within_groups
    mat_data = load_matlab('debug/ygradient_coil')
    m_coil_parts = mat_data['coil_layouts'].out.coil_parts
    m_c_part = m_coil_parts
    p_coil_parts = np.load('debug/ygradient_coil_python.npy', allow_pickle=True)

    input_args = DataStructure(force_cut_selection=['high'], b_0_direction=[0, 0, 1], interconnection_cut_width=0.1)
    coil_parts = interconnect_within_groups(p_coil_parts, input_args, m_c_part)

    # And now!!
    coil_part = coil_parts[0]

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

        """
        shape_name = 'cutshape'
        for index2, m_shape in enumerate(m_group.cutshape):
            p_shape = c_group.cutshape[index2]
            visualize_vertex_connections(p_shape.uv.T, 800, f'images/connected_group_{shape_name}_uv_{index1}_{index2}_p.png')
            visualize_vertex_connections(m_shape.uv.T, 800, f'images/connected_group_{shape_name}_uv_{index1}_{index2}_m.png')

        shape_name = 'opened_loop'
        for index2, m_shape in enumerate(m_group.opened_loop):
            p_shape = c_group.opened_loop[index2]
            visualize_vertex_connections(p_shape.uv.T, 800, f'images/connected_group_{shape_name}_uv_{index1}_{index2}_p.png')
            visualize_vertex_connections(m_shape.uv.T, 800, f'images/connected_group_{shape_name}_uv_{index1}_{index2}_m.png')
        """

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


def brute_test_process_raw_loops_brute():
    from sub_functions.process_raw_loops import process_raw_loops

    # MATLAB saved data
    mat_data = load_matlab('debug/ygradient_coil')
    mat_data_out = mat_data['coil_layouts'].out
    m_coil_parts = mat_data_out.coil_parts
    m_coil_part = m_coil_parts

    # Python saved data 10 : Just between calc_contours_by_triangular_potential_cuts and process_raw_loops
    p_coil_parts = np.load('debug/ygradient_coil_python_10.npy', allow_pickle=True)

    input_args = DataStructure(smooth_flag=1, smooth_factor=1, min_loop_significance=1)
    target_field = mat_data_out.target_field

    debug_data = m_coil_part
    ###################################################################################
    # Function under test
    coil_parts2 = process_raw_loops(p_coil_parts, input_args, target_field)
    ###################################################################################

    # Checks:
    coil_part = coil_parts2[0]
    assert len(coil_part.contour_lines) == len(m_coil_part.contour_lines)
    assert abs(coil_part.combined_loop_length - m_coil_part.combined_loop_length) < 0.0005  # Pass
    assert compare(coil_part.combined_loop_field, m_coil_part.combined_loop_field, double_tolerance=5e-7)  # Pass
    assert compare(coil_part.loop_significance, m_coil_part.loop_signficance, double_tolerance=0.005)
    assert compare(coil_part.field_by_loops, m_coil_part.field_by_loops, double_tolerance=2e-7)  # Pass!


def test_interconnect_among_groups():
    from sub_functions.interconnect_among_groups import interconnect_among_groups
    mat_data = load_matlab('debug/ygradient_coil')
    m_coil_parts = mat_data['coil_layouts'].out.coil_parts
    m_c_part = m_coil_parts
    p_coil_parts = np.load('debug/ygradient_coil_python_15_true.npy', allow_pickle=True)

    input_args = DataStructure(interconnection_cut_width=0.1)
    coil_parts = interconnect_among_groups(p_coil_parts, input_args, m_c_part)

    # Wire path
    for index1 in range(len(coil_parts)):
        c_wire_path = coil_parts[index1].wire_path
        m_wire_path = m_c_part.wire_path1

        visualize_vertex_connections(c_wire_path.uv.T, 800, f'images/wire_path_uv_{index1}_p.png')
        visualize_vertex_connections(m_wire_path.uv.T, 800, f'images/wire_path_uv_{index1}_m.png')

        visualize_compare_vertices(c_wire_path.uv.T, m_wire_path.uv.T, 800, f'images/wire_path_uv_{index1}_diff.png')

        # Check....
        assert (compare(c_wire_path.v, m_wire_path.v))  # Pass!
        assert (compare(c_wire_path.uv, m_wire_path.uv))  # Pass!


def develop_shift_return_paths():
    from sub_functions.shift_return_paths import shift_return_paths
    mat_data = load_matlab('debug/ygradient_coil')
    m_coil_parts = mat_data['coil_layouts'].out.coil_parts
    m_c_part = m_coil_parts
    p_coil_parts = np.load('debug/ygradient_coil_python_16_true.npy', allow_pickle=True)

    input_args = DataStructure(interconnection_cut_width=0.1,
                               skip_normal_shift=0,
                               smooth_flag=1,
                               smooth_factor=1,
                               normal_shift_smooth_factors=[2, 3, 2],
                               normal_shift_length=0.025)
    coil_parts = shift_return_paths(p_coil_parts, input_args)#, m_c_part)


    # Verify: shift_array, points_to_shift, wire_path
    for index1 in range(len(coil_parts)):
        c_part = coil_parts[index1]
        c_wire_path = c_part.wire_path
        m_wire_path = m_c_part.wire_path

        visualize_vertex_connections(c_wire_path.uv.T, 800, f'images/wire_path2_uv_{index1}_p.png')
        visualize_vertex_connections(m_wire_path.uv.T, 800, f'images/wire_path2_uv_{index1}_m.png')

        visualize_compare_vertices(c_wire_path.uv.T, m_wire_path.uv.T, 800, f'images/wire_path2_uv_{index1}_diff.png')

        # Check....
        assert (compare(c_part.shift_array, m_c_part.shift_array))          # Pass
        assert (compare(c_part.points_to_shift, m_c_part.points_to_shift))  # Pass

        assert (compare(c_wire_path.v, m_wire_path.v, double_tolerance=0.03))  # Pass, with this coarse tolerance!
        assert (compare(c_wire_path.uv, m_wire_path.uv))  # Pass

def develop_generate_cylindrical_pcb_print():
    from sub_functions.generate_cylindrical_pcb_print import generate_cylindrical_pcb_print
    mat_data = load_matlab('debug/ygradient_coil')
    m_coil_parts = mat_data['coil_layouts'].out.coil_parts
    m_c_part = m_coil_parts
    p_coil_parts = np.load('debug/ygradient_coil_python_17_true.npy', allow_pickle=True)

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
        c_upper_group_layouts = c_part.pcb_tracks.upper_layer[0].group_layouts
        m_upper_group_layouts = m_c_part.pcb_tracks.upper_layer.group_layouts

        layer = 'upper'
        for index1, m_group_layout in enumerate(m_upper_group_layouts):
            c_group_layout = c_upper_group_layouts[index1]
            c_wire_part = c_group_layout.wire_parts[0]
            m_wire_part = m_group_layout.wire_parts

            visualize_vertex_connections(c_wire_part.uv.T, 800, f'images/pcb_{layer}_group{index1}_uv_p.png')
            visualize_vertex_connections(m_wire_part.uv.T, 800, f'images/pcb_{layer}_group{index1}_uv_m.png')

            visualize_compare_vertices(c_wire_part.uv.T, m_wire_part.uv.T, 800, f'images/pcb_{layer}_group{index1}_uv__diff.png')

            # Check....
            assert c_wire_part.ind1 == m_wire_part.ind1 - 1 # MATLAB base 1
            assert c_wire_part.ind2 == m_wire_part.ind2 - 1 # MATLAB base 1

            assert compare(c_wire_part.uv, m_wire_part.uv)
            assert compare(c_wire_part.track_shape, m_wire_part.track_shape)



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
    # brute_test_process_raw_loops_brute()
    # test_interconnect_within_groups()
    # test_interconnect_among_groups()
    # develop_shift_return_paths()
    develop_generate_cylindrical_pcb_print()
