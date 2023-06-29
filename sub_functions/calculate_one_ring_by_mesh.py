import numpy as np
from scipy.spatial import Delaunay

# Logging
import logging

# Local imports
from sub_functions.data_structures import CoilSolution

log = logging.getLogger(__name__)


def calculate_one_ring_by_mesh(coil_solution: CoilSolution, coil_parts, input, m_orl_debug):
    """
    Calculate the one-ring neighborhood of vertices in the coil mesh.

    Args:
        coil_parts (List[CoilPart]): List of coil parts.
        input: Input parameters.

    Returns:
        List[CoilPart]: Updated list of coil parts with calculated one ring information.
    """
    optimisation = coil_solution.optimisation  # Retrieve the solution optmisation parameters
    if not optimisation.use_preoptimization_temp:
        for part_ind in range(len(coil_parts)):
            # Extract the intermediate variables
            coil_part = coil_parts[part_ind]
            part_mesh = coil_part.coil_mesh  # Get the Mesh instance
            part_vertices = part_mesh.get_vertices()  # Get the vertices for the coil part.
            part_faces = part_mesh.get_faces()
            trimesh_obj = part_mesh.trimesh_obj  # Retrieve the encapsulated Trimesh instance
            optimisation = coil_solution.optimisation  # Retrieve the solution optmisation parameters
            # part_faces = part_faces.T  # Transpose the faces array


            # DEBUG: Remove this
            # dict_keys(['__header__', '__version__', '__globals__', 'node_triangles', 'node_triangles_corners', 'one_ring_list'])
            # log.debug("m_orl_debug: %s", m_orl_debug.keys())
            m_orl_node_triangles = m_orl_debug['node_triangles'] - 1
            m_orl_node_triangles_corners = m_orl_debug['node_triangles_corners']
            m_orl_one_ring_list = m_orl_debug['one_ring_list']

            num_nodes = part_vertices.shape[0]

            # TODO: one_ring_list is not calculated the same as MATLAB
            node_triangles = np.empty(num_nodes, dtype=object) #[]

            # TODO: Create function on Mesh class and delegate to trimesh_obj
            node_triangles_tri = trimesh_obj.vertex_faces
            # This creates an n,m array where m is padded with -1's. See 
            # https://trimsh.org/trimesh.base.html#trimesh.base.Trimesh.vertex_faces
            # Iterate and create reduced numpy arrays

            print(trimesh_obj.vertex_faces[0:10])
            print(trimesh_obj.vertices[0:10])

            node_triangles = np.array([row[row != -1] for row in node_triangles_tri], dtype=object)

            log.debug("node_triangles:\n%s", node_triangles[0:10])
            # print(trimesh_obj.vertex_degree)

            log.debug("m_orl_node_triangles:\n%s", m_orl_node_triangles)


            node_triangles_corners = [
                part_faces[x, :] for x in node_triangles
            ]  # Get triangle corners for each node

            one_ring_list = np.empty(num_nodes, dtype=object) #[]
            for node_ind in range(num_nodes):
                single_cell = [node_triangles_corners[node_ind][i, :]
                               for i in range(node_triangles_corners[node_ind].shape[0])]

                neighbor_faces = [
                    x[x != node_ind] for x in single_cell
                ]  # Remove the current node index from each triangle's corner indices
                one_ring_list[node_ind] = neighbor_faces

            # Make sure that the current orientation is uniform for all elements (old)
            for node_ind in range(num_nodes):
                for face_ind in range(len(one_ring_list[node_ind])):
                    point_aa = part_vertices[one_ring_list[node_ind][face_ind][0]]
                    point_bb = part_vertices[one_ring_list[node_ind][face_ind][1]]
                    point_cc = part_vertices[node_ind]
                    cross_vec = np.cross(point_bb - point_aa, point_aa - point_cc)

                    if np.sign(np.dot(part_mesh.n[node_ind], cross_vec)) > 0:
                        one_ring_list[node_ind][face_ind] = np.flipud(
                            one_ring_list[node_ind][face_ind]
                        )

            # Update the coil_part with one-ring neighborhood information
            node_triangle_mat = np.zeros((num_nodes, part_faces.shape[0]), dtype=int)
            for index in range(num_nodes):
                node_triangle_mat[index, node_triangles[index]] = 1
            
            coil_part.one_ring_list = np.array(one_ring_list, dtype=object)
            coil_part.node_triangles = np.array(node_triangles, dtype=object)
            coil_part.node_triangle_mat = node_triangle_mat
            # coil_part.coil_mesh.faces = part_faces.T

    else:
        raise Exception("Optimisation is not implemented!")
        # Use pre-optimized data from input object
        coil_part.one_ring_list = optimisation.one_ring_list
        coil_part.node_triangles = optimisation.node_triangles
        coil_part.node_triangle_mat = optimisation.node_triangle_mat

    return coil_parts

    """
            node_triangles_corners = [
                part_faces[node_triangles[node_ind]].tolist()
                for node_ind in range(len(node_triangles))
            ]

            one_ring_list = [[] for _ in range(num_nodes)]

            # Compute the one-ring neighborhood for each vertex
            for node_ind in range(num_nodes):
                triangles = vertex_to_triangles[node_ind]
                node_triangles_corners = part_faces[triangles]
                one_ring_list[node_ind] = node_triangles_corners[:, node_triangles_corners != node_ind].T

            # Ensure consistent orientation within the one-ring neighborhood
            for node_ind in range(num_nodes):
                for face_ind in range(one_ring_list[node_ind].shape[1]):
                    point_aa = part_vertices[:, one_ring_list[node_ind][0, face_ind]]
                    point_bb = part_vertices[:, one_ring_list[node_ind][1, face_ind]]
                    point_cc = part_vertices[:, node_ind]
                    cross_vec = np.cross(point_bb - point_aa, point_aa - point_cc)

                    if np.dot(part_mesh.n[:, node_ind], cross_vec) > 0:
                        one_ring_list[node_ind][:, face_ind] = np.flipud(one_ring_list[node_ind][:, face_ind])

            node_triangle_mat = np.zeros((num_nodes, part_faces.shape[0]), dtype=bool)

            # Update the coil_parts object
            coil_part.one_ring_list = one_ring_list
            coil_part.node_triangles = vertex_to_triangles
            coil_part.node_triangle_mat = node_triangle_mat
            part_faces = part_faces.T
    """
