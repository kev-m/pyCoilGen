# System imports
from typing import List
import numpy as np
# Logging
import logging

# Local imports
from sub_functions.data_structures import CoilPart, Mesh

log = logging.getLogger(__name__)


def split_disconnected_mesh(coil_mesh_in: Mesh) -> List[CoilPart]:
    """
    Split the mesh and the stream function if there are disconnected pieces
    such as shielded gradients.

    Args:
        coil_mesh_in (Mesh): Input mesh to be split.

    Returns:
        List[CoilPart]: A list of CoilPart structures representing the split mesh parts.
    """
    coil_parts = []  # Initialize a list to store the split mesh parts

    vertices = coil_mesh_in.get_vertices()
    faces = coil_mesh_in.get_faces()

    def build_adjacency_list(vertices, faces):
        adjacency_list = {}
        for face in faces:
            for vertex_index in face:
                if vertex_index not in adjacency_list:
                    adjacency_list[vertex_index] = set()
            for i in range(len(face)):
                for j in range(i + 1, len(face)):
                    adjacency_list[face[i]].add(face[j])
                    adjacency_list[face[j]].add(face[i])
        return adjacency_list
    
    def dfs(node, adjacency_list, visited, component):
        visited[node] = True
        component.append(node)
        for neighbor in adjacency_list[node]:
            if not visited[neighbor]:
                dfs(neighbor, adjacency_list, visited, component)
    
    adjacency_list = build_adjacency_list(vertices, faces)
    visited = [False] * len(vertices)

    meshes = []    
    for vertex_index in range(len(vertices)):
        if not visited[vertex_index]:
            new_mesh = []
            dfs(vertex_index, adjacency_list, visited, new_mesh)
            meshes.append(new_mesh)
    
    for mesh_indices in meshes:
        coil_mesh = Mesh(vertices= [vertices[i] for i in mesh_indices],
                         faces= [[mesh_indices.index(idx) for idx in face] for face in faces if all(idx in mesh_indices for idx in face)]
        )
        part = CoilPart(coil_mesh=coil_mesh)
        coil_parts.append(part)

    return coil_parts
