# Hack code
# Set up paths
import sys
import os    
# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Test code
import json
from sub_functions.data_structures import Mesh
from sub_functions.split_disconnected_mesh import split_disconnected_mesh
from helpers.visualisation import compare

import numpy as np
def compute_connected_face_indices(vertices, faces):
    num_vertices = len(vertices)
    num_faces = len(faces)
    connected_indices = []

    # Initialise variables
    face_set_indices = np.full((num_vertices), -1, dtype=int) # Unused
    face_sets = np.ndarray((num_vertices), dtype=object)
    face_sets[0] = faces[0] # Add the indices of the first face 
    face_set_index = 0

    # Check intersection of faces with face_sets
    connected_faces = []
    list_of_connections = [connected_faces]
    for i in range(0, num_faces-1):
        intersection = np.intersect1d(face_sets[face_set_index], faces[i])
        if intersection.size > 0:
            # Merge the sets if faces have a common vertex
            face_sets[face_set_index] = np.union1d(face_sets[face_set_index], faces[i])
            face_set_indices[faces[i]] = face_set_index
            connected_faces.append(faces[1])
        else:
            print(faces[i], "not in", face_sets)
            face_set_index += 1
            #face_sets[face_set_index] = faces[i]
            face_sets = np.append(face_sets, faces[i])
            face_set_indices[faces[i]] = face_set_index
            list_of_connections.append(connected_faces)
            connected_faces = []

    return list_of_connections


def test_split_disconnected_mesh():
    with open('tests/test_data/planar_mesh.json', 'r') as file:
        mesh_data = json.load(file)

    mesh = Mesh(vertices=mesh_data['vertices'], faces=mesh_data['faces'])
    test_mesh = Mesh(vertices=mesh_data['vertices'], faces=mesh_data['faces'])
    parts = split_disconnected_mesh(mesh)

    assert len(parts) == 1
    split_mesh = parts[0].coil_mesh

    assert compare(test_mesh.get_vertices(), split_mesh.get_vertices())
    assert compare(test_mesh.get_faces(), split_mesh.get_faces())
    compare(mesh.trimesh_obj.vertex_faces, split_mesh.trimesh_obj.vertex_faces)


if __name__ == "__main__":
    import logging
    # Set up logging
    log = logging.getLogger(__name__)
    logging.basicConfig(level=logging.DEBUG)

    # Run test
    test_split_disconnected_mesh()
