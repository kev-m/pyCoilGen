import numpy as np
import json

# Test support
from pyCoilGen.helpers.visualisation import compare
from pyCoilGen.sub_functions.data_structures import Mesh
# Code under test
from pyCoilGen.sub_functions.split_disconnected_mesh import split_disconnected_mesh


def test_split_disconnected_mesh_simple_planar_mesh():
    with open('tests/test_data/planar_mesh.json', 'r') as file:
        mesh_data = json.load(file)

    mesh = Mesh(vertices=mesh_data['vertices'], faces=mesh_data['faces'])
    test_mesh = Mesh(vertices=mesh_data['vertices'], faces=mesh_data['faces'])
    parts = split_disconnected_mesh(mesh)

    assert len(parts) == 1
    split_mesh = parts[0].coil_mesh

    assert compare(test_mesh.get_vertices(), split_mesh.get_vertices())
    assert compare(test_mesh.get_faces(), split_mesh.get_faces())
    assert compare(mesh.trimesh_obj.vertex_faces, split_mesh.trimesh_obj.vertex_faces)


def test_split_disconnected_mesh_biplanar_mesh():
    with open('tests/test_data/biplanar_mesh.json', 'r') as file:
        mesh_data = json.load(file)


    # Create faces ndarray from list
    num_faces = len(mesh_data['faces'])
    nd_faces = np.empty((num_faces, 3), dtype=int)
    index = 0
    for vertex in mesh_data['faces']:
        nd_faces[index] = vertex
        index += 1

    # Create vertices ndarray from list
    num_vertices = len(mesh_data['vertices'])
    nd_vertices = np.empty((num_vertices, 3), dtype=float)
    index = 0
    for vertex in mesh_data['vertices']:
        nd_vertices[index] = vertex
        index += 1

    mesh = Mesh(vertices=nd_vertices, faces=nd_faces)

    assert compare(mesh.get_faces(), nd_faces)
    assert compare(mesh.get_vertices(), nd_vertices)

    parts = split_disconnected_mesh(mesh)

    assert len(parts) == 2

    split_mesh0 = parts[0].coil_mesh
    part0_vertices = nd_vertices[:int(num_vertices/2),:]
    part0_faces = nd_faces[:int(num_faces/2),:]
    assert compare(split_mesh0.get_vertices(), part0_vertices)
    assert compare(split_mesh0.get_faces(), part0_faces)

    split_mesh1 = parts[1].coil_mesh
    part1_vertices = nd_vertices[int(num_vertices/2):,:] # 2nd half
    part1_faces = nd_faces[int(num_faces/2):,:] # 2nd half
    part1_faces = part1_faces - np.min(part1_faces) # Re-zero the vertex indices
    assert compare(split_mesh1.get_vertices(), part1_vertices)
    assert compare(split_mesh1.get_faces(), part1_faces)

def test_split_disconnected_mesh_stl_file1():
    mesh = Mesh.load_from_file('Geometry_Data', 'cylinder_radius500mm_length1500mm.stl')
    test_vertices = mesh.get_vertices()
    test_faces = mesh.get_faces()

    ##################################################
    # Function under test
    parts = split_disconnected_mesh(mesh)
    ##################################################

    assert len(parts) == 1
    split_mesh = parts[0].coil_mesh
    assert compare(split_mesh.get_faces(), test_faces)
    assert compare(split_mesh.get_vertices(), test_vertices)

def test_split_disconnected_mesh_stl_file2():
    mesh = Mesh.load_from_file('Geometry_Data', 'bi_planer_rectangles_width_1000mm_distance_500mm.stl')
    test_vertices = mesh.get_vertices()
    test_faces = mesh.get_faces()

    ##################################################
    # Function under test
    parts = split_disconnected_mesh(mesh)
    ##################################################

    assert len(parts) == 2
    split_mesh = parts[0].coil_mesh

    assert len(split_mesh.get_faces()) == len(test_faces)/2
    assert len(split_mesh.get_vertices()) == len(test_vertices)/2

def test_load_from_file():
    ##################################################
    # Function under test
    mesh = Mesh.load_from_file('Geometry_Data', 'cylinder_radius500mm_length1500mm.stl')
    ##################################################


if __name__ == "__main__":
    import logging
    # Set up logging
    log = logging.getLogger(__name__)
    logging.basicConfig(level=logging.DEBUG)

    """
    # Run test
    with open('tests/test_data/biplanar_mesh.json', 'r') as file:
        mesh_data = json.load(file)

    num_faces = len(mesh_data['faces'])
    nd_faces = np.empty((num_faces, 3), dtype=int)
    index = 0
    for face in mesh_data['faces']:
        nd_faces[index] = face
        index += 1

    lists_of_connected_faces, face_set_vertices = compute_connected_face_indices(
        vertices=mesh_data['vertices'], faces=nd_faces)

    print(len(lists_of_connected_faces) == 2, len(lists_of_connected_faces))
    print(len(lists_of_connected_faces[0]) == 40, len(lists_of_connected_faces[0]))
    print(len(lists_of_connected_faces[1]) == 40, len(lists_of_connected_faces[1]))
    """    

    # test_split_disconnected_mesh_simple_planar_mesh()
    # test_split_disconnected_mesh_biplanar_mesh()
    test_split_disconnected_mesh_stl_file2()