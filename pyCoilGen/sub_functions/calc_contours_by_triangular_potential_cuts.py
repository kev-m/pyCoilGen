import numpy as np
from typing import List
from dataclasses import dataclass
import trimesh  # For face and vertex adjacency
import warnings

# Logging
import logging
import warnings

# Local imports
from .data_structures import CoilPart
from .constants import get_level, DEBUG_VERBOSE
from .data_structures import ContourLine, UnarrangedLoop, UnarrangedLoopContainer, UnsortedPoints, RawPart

log = logging.getLogger(__name__)


def _opposed_nodes(attached_triangles: np.ndarray, edge_nodes: np.ndarray) -> np.ndarray:
    """
    The node of each attached triangle that is not on the edge.

    Args:
        attached_triangles (ndarray): The two triangles (three node indices each) attached to each edge, (m, 2, 3).
        edge_nodes (ndarray): The two nodes of each edge, (m, 2).

    Returns:
        ndarray: The opposed node of each of the two triangles of each edge, (m, 2).
    """
    on_edge = ((attached_triangles == edge_nodes[:, np.newaxis, 0:1]) |
               (attached_triangles == edge_nodes[:, np.newaxis, 1:2]))
    opposed = np.zeros(edge_nodes.shape, dtype=edge_nodes.dtype)
    if not np.all(on_edge.sum(axis=2) == 2):
        # A triangle that does not contain both nodes of the edge, or with a repeated node: use the general set difference
        for x_ind in range(edge_nodes.shape[0]):
            for side in range(2):
                opposed[x_ind, side] = np.squeeze(np.setdiff1d(attached_triangles[x_ind, side], edge_nodes[x_ind]))
        return opposed
    return np.take_along_axis(attached_triangles, np.argmin(on_edge, axis=2)[:, :, np.newaxis], axis=2)[:, :, 0]


def _neighbour_search(edges: np.ndarray, opposed_nodes: np.ndarray):
    """
    A function that finds, for an edge of one potential level, the other edges of the level that continue the contour.

    An edge continues the contour of edge e if it contains one of the nodes opposed to e (in its two triangles), and one of
    the nodes of e. The edges are found with a lookup of the edges at each node, instead of a search through all edges.

    Args:
        edges (ndarray): The two nodes of each edge of the potential level, (m, 2).
        opposed_nodes (ndarray): The two nodes opposed to each edge, (m, 2).

    Returns:
        callable: neighbouring_edges(e) -> sorted list of the indices of the neighbouring edges.
    """
    edge_list = edges.tolist()
    opposed_list = opposed_nodes.tolist()
    edges_at_node = {}
    for edge_ind, (node_a, node_b) in enumerate(edge_list):
        edges_at_node.setdefault(node_a, []).append(edge_ind)
        edges_at_node.setdefault(node_b, []).append(edge_ind)

    def neighbouring_edges(edge_ind: int):
        node_a, node_b = edge_list[edge_ind]
        opposed_a, opposed_b = opposed_list[edge_ind]
        at_opposed = set(edges_at_node.get(opposed_a, ())) | set(edges_at_node.get(opposed_b, ()))
        at_edge = set(edges_at_node.get(node_a, ())) | set(edges_at_node.get(node_b, ()))
        return sorted(at_opposed & at_edge)

    return neighbouring_edges


def calc_contours_by_triangular_potential_cuts(coil_parts: List[CoilPart]):
    """
    Center the stream function potential around zero and add zeros around the periphery.

    Initialises the following properties of a CoilPart:
        - contour_lines
        - raw

    Depends on the following properties of the CoilParts:
        - coil_mesh
        - potential_level_list
        - stream_function

    Depends on the following input_args:
        - None

    Updates the following properties of a CoilPart:
        - None

    Args:
        coil_parts (List[CoilPart]): A list of CoilPart structures.

    Returns:
        coil_parts (list): Updated list of coil parts.
    """
    for part_ind in range(len(coil_parts)):
        part = coil_parts[part_ind]
        part_mesh = part.coil_mesh
        part_vertices = part_mesh.get_vertices()
        part_uv = part_mesh.uv
        part_faces = part_mesh.get_faces()

        # For the UV mesh, calculate:
        # 1. edge_nodes: The edges, i.e. the list of connected vertices (array m x 2)
        # 2. edge_attached_triangles: The attached triangles, i.e. the list of triangles that share an edge (array of n).

        # Compute edges and attached triangles
        mesh = trimesh.Trimesh(vertices=part_uv, faces=part_faces, process=False)
        # Returns the edges that are shared by the adjacent faces (index into faces array).
        edge_faces = mesh.face_adjacency
        # Returns the edges that are shared by the adjacent faces (index into vertices array).
        edge_nodes = mesh.face_adjacency_edges
        num_edges = edge_nodes.shape[0]

        if get_level() >= DEBUG_VERBOSE:
            log.debug(" -- edge_faces shape: %s, max(%d)", edge_faces.shape, np.max(edge_faces))  # 696,2: Max: 263
            log.debug(" -- edge_nodes shape: %s, max(%d)", edge_nodes.shape, np.max(edge_nodes))  # 696,2: Max: 263

        # NOTE: Vertex arrays order is reversed compared to MATLAB: [0,1,2] [0,7,1]	vs [1,8,2]	[1,2,3]
        # Must swap the order of the triangles, to correct index order
        edge_attached_triangles = np.stack((part_faces[edge_faces[:, 1]], part_faces[edge_faces[:, 0]]), axis=1)  # (m, 2, 3)

        # Take only the edge opposing nodes of these triangles: the one node of each triangle that is not on the edge
        edge_opposed_nodes = _opposed_nodes(edge_attached_triangles, edge_nodes)

        if get_level() >= DEBUG_VERBOSE:
            log.debug(" -- edge_opposed_nodes shape: %s, max(%d)", edge_opposed_nodes.shape, np.max(edge_opposed_nodes))

        # Test for all edges whether they cut one of the potential levels
        num_potentials = len(part.potential_level_list)
        rep_contour_level_list = np.tile(part.potential_level_list, (edge_nodes.shape[0], 1))
        edge_node_potentials = part.stream_function[edge_nodes]

        min_edge_potentials = np.min(edge_node_potentials, axis=1)
        max_edge_potentials = np.max(edge_node_potentials, axis=1)
        min_edge_potentials = np.tile(min_edge_potentials[:, np.newaxis], (1, num_potentials))
        max_edge_potentials = np.tile(max_edge_potentials[:, np.newaxis], (1, num_potentials))

        tri_below_pot_step = max_edge_potentials > rep_contour_level_list
        tri_above_pot_step = min_edge_potentials < rep_contour_level_list
        potential_cut_criteria = np.logical_and(tri_below_pot_step, tri_above_pot_step)
        potential_cut_criteria = potential_cut_criteria.astype(float)
        potential_cut_criteria[np.where(potential_cut_criteria == 0)] = np.nan

        # Calculate for each edge which cuts a potential the uv coordinates
        edge_lengths = np.linalg.norm(part_uv[edge_nodes[:, 1]] - part_uv[edge_nodes[:, 0]], axis=1)
        edge_lengths = np.tile(edge_lengths[:, np.newaxis], (1, num_potentials))
        edge_potential_span = edge_node_potentials[:, 1] - edge_node_potentials[:, 0]
        edge_potential_span = np.tile(edge_potential_span[:, np.newaxis], (1, num_potentials))

        warnings.filterwarnings('ignore')

        pot_dist_to_step = rep_contour_level_list - np.tile(edge_node_potentials[:, 0][:, np.newaxis],
                                                            (1, num_potentials))
        cut_point_distance_to_edge_node_1 = np.abs(pot_dist_to_step / edge_potential_span * edge_lengths)

        u_component_edge_vectors = part_uv[edge_nodes[:, 1], 0] - part_uv[edge_nodes[:, 0], 0]
        u_component_edge_vectors = np.tile(u_component_edge_vectors[:, np.newaxis], (1, num_potentials))
        v_component_edge_vectors = part_uv[edge_nodes[:, 1], 1] - part_uv[edge_nodes[:, 0], 1]
        v_component_edge_vectors = np.tile(v_component_edge_vectors[:, np.newaxis], (1, num_potentials))

        first_edge_node_u = np.tile(part_uv[edge_nodes[:, 0], 0][:, np.newaxis], (1, num_potentials))
        first_edge_node_v = np.tile(part_uv[edge_nodes[:, 0], 1][:, np.newaxis], (1, num_potentials))
        u_cut_point = potential_cut_criteria * (
            first_edge_node_u + u_component_edge_vectors * cut_point_distance_to_edge_node_1 / edge_lengths)
        v_cut_point = potential_cut_criteria * (
            first_edge_node_v + v_component_edge_vectors * cut_point_distance_to_edge_node_1 / edge_lengths)

        warnings.filterwarnings('default')

        # Create cell by sorting the cut points to the corresponding potential levels
        potential_sorted_cut_points = np.zeros((num_potentials), dtype=object)  # 20 x M x 3
        value3 = np.arange(edge_nodes.shape[0], dtype=int)
        for pot_ind in range(num_potentials):
            cut_points = np.column_stack(
                (u_cut_point[:, pot_ind], v_cut_point[:, pot_ind], value3))
            cut_points = cut_points[~np.isnan(cut_points[:, 0])]
            potential_sorted_cut_points[pot_ind] = cut_points
        # End of Part 1

        # Start of Part 2
        # NOTE: The 'raw' RawPart does not need to be part of a CoilPart.
        #       Consider renaming 'part.raw' to 'raw_part' and removing 'raw' from the CoilPart structure.
        # Create the unsorted points structure
        part.raw = RawPart()
        empty_potential_groups = [len(potential_sorted_cut_points[i]) ==
                                  0 for i in range(len(potential_sorted_cut_points))]
        num_false = len(empty_potential_groups) - sum(empty_potential_groups)
        part.raw.unsorted_points = [UnsortedPoints() for _ in range(num_false)]
        part.raw.unarranged_loops = [UnarrangedLoopContainer(loop=[]) for _ in range(num_false)]
        running_ind = 0

        for struct_ind in range(len(empty_potential_groups)):
            if not empty_potential_groups[struct_ind]:
                part.raw.unsorted_points[running_ind].potential = part.potential_level_list[struct_ind]
                part.raw.unsorted_points[running_ind].edge_ind = potential_sorted_cut_points[struct_ind][:, 2].astype(
                    int)
                part.raw.unsorted_points[running_ind].uv = potential_sorted_cut_points[struct_ind][:, :2]
                running_ind += 1

        # Separate loops within potential groups
        # Create loops
        for potential_group in range(len(part.raw.unsorted_points)):
            all_current_edges = edge_nodes[part.raw.unsorted_points[potential_group].edge_ind]
            all_current_opposed_nodes = edge_opposed_nodes[part.raw.unsorted_points[potential_group].edge_ind]
            all_current_uv_coords = part.raw.unsorted_points[potential_group].uv

            neighbouring_edges = _neighbour_search(all_current_edges, all_current_opposed_nodes)

            set_new_start = True
            num_build_loops = 0
            edge_already_used = np.zeros(all_current_edges.shape[0], dtype=int)

            group_unarranged_loop = part.raw.unarranged_loops[potential_group]

            while not np.all(edge_already_used):
                if set_new_start:
                    num_build_loops += 1
                    starting_edge = np.min(np.where(edge_already_used == 0)[0])

                    loop_item = UnarrangedLoop()
                    loop_item.add_uv(all_current_uv_coords[starting_edge])
                    loop_item.add_edge(all_current_edges[starting_edge])
                    group_unarranged_loop.loop.append(loop_item)
                    edge_already_used[starting_edge] = 1
                    current_edge = starting_edge

                    neighbouring_free_next_edges = neighbouring_edges(current_edge)

                    if len(neighbouring_free_next_edges) == 0:
                        break
                    elif len(neighbouring_free_next_edges) == 1:
                        next_edge = neighbouring_free_next_edges[0]
                    else:
                        if not edge_already_used[neighbouring_free_next_edges[0]]:
                            next_edge = neighbouring_free_next_edges[0]
                        else:
                            next_edge = neighbouring_free_next_edges[1]

                        set_new_start = False

                while next_edge != starting_edge:
                    edge_already_used[next_edge] = 1
                    loop_item = group_unarranged_loop.loop[num_build_loops - 1]
                    loop_item.add_uv(all_current_uv_coords[next_edge])
                    loop_item.add_edge(all_current_edges[next_edge])

                    current_edge = next_edge

                    possible_next_edges = [edge for edge in neighbouring_edges(current_edge)
                                           if not edge_already_used[edge]]

                    if len(possible_next_edges) == 0:
                        break
                    else:
                        if len(possible_next_edges) == 1:
                            next_edge = possible_next_edges[0]
                        else:
                            if not edge_already_used[possible_next_edges[0]]:
                                next_edge = possible_next_edges[0]
                            else:
                                next_edge = possible_next_edges[1]

                set_new_start = True
        # End of Part 2

        # Start of Part 3
        num_contours = 0
        # Evaluate current orientation for each loop
        for pot_ind in range(len(part.raw.unsorted_points)):
            center_segment_potential = part.raw.unsorted_points[pot_ind].potential
            potential_loop = part.raw.unarranged_loops[pot_ind]
            for loop_ind in range(len(potential_loop.loop)):
                potential_loop_item = potential_loop.loop[loop_ind]
                if len(potential_loop_item.edge_inds) > 2:
                    num_contours += 1
                    test_edge = int(np.floor(len(potential_loop_item.edge_inds) / 2))
                    first_edge = potential_loop_item.edge_inds[test_edge]
                    second_edge = potential_loop_item.edge_inds[test_edge + 1]
                    node_1 = np.intersect1d(first_edge, second_edge)
                    node_2 = np.setdiff1d(first_edge, node_1)
                    node_3 = np.setdiff1d(second_edge, node_1)
                    node_1_uv = coil_parts[part_ind].coil_mesh.uv[node_1, :][0]  # Change shape from (1,2) to (2,)
                    node_2_uv = coil_parts[part_ind].coil_mesh.uv[node_2, :][0]  # Change shape from (1,2) to (2,)
                    node_3_uv = coil_parts[part_ind].coil_mesh.uv[node_3, :][0]  # Change shape from (1,2) to (2,)
                    node_1_pot = coil_parts[part_ind].stream_function[node_1]
                    node_2_pot = coil_parts[part_ind].stream_function[node_2]
                    node_3_pot = coil_parts[part_ind].stream_function[node_3]

                    # Calculate the 2D gradient of the triangle
                    # list indices must be integers or slices, not tuple
                    center_segment_position = (potential_loop_item.uv[test_edge] +
                                               potential_loop_item.uv[test_edge + 1]) / 2

                    vec_center_node_1 = node_1_uv - center_segment_position
                    vec_center_node_2 = node_2_uv - center_segment_position
                    vec_center_node_3 = node_3_uv - center_segment_position

                    pot_diff_center_node_1 = node_1_pot - center_segment_potential
                    pot_diff_center_node_2 = node_2_pot - center_segment_potential
                    pot_diff_center_node_3 = node_3_pot - center_segment_potential

                    pot_gradient_vec = vec_center_node_1 * pot_diff_center_node_1 + \
                        vec_center_node_2 * pot_diff_center_node_2 + \
                        vec_center_node_3 * pot_diff_center_node_3

                    # Test the chirality of the segment on the potential gradient on that segment
                    segment_vec = potential_loop_item.uv[test_edge + 1] - potential_loop_item.uv[test_edge]

                    cross_vec = np.cross([segment_vec[0], segment_vec[1], 0],
                                         [pot_gradient_vec[0], pot_gradient_vec[1], 0])

                    potential_loop_item.current_orientation = int(np.sign(cross_vec[2]))

                    if potential_loop_item.current_orientation == -1:
                        potential_loop_item.uv = np.flipud(potential_loop_item.uv)
                        potential_loop_item.edge_inds = np.flipud(potential_loop_item.edge_inds)
                else:
                    # raise ValueError('Some loops are too small and contain only 2 points, therefore ill-defined')
                    log.error('Some loops are too small and contain only 2 points, therefore ill-defined')
        # End of Part 3

        # Start of Part 4
        # Build the contour lines
        raw_part = part.raw
        part.contour_lines = [ContourLine() for _ in range(num_contours)]
        contour_index = 0
        for pot_ind in range(len(raw_part.unsorted_points)):
            for loop_ind in range(len(raw_part.unarranged_loops[pot_ind].loop)):
                contour_line = part.contour_lines[contour_index]

                uv = raw_part.unarranged_loops[pot_ind].loop[loop_ind].uv
                contour_line.uv = uv.T

                potential = raw_part.unsorted_points[pot_ind].potential
                contour_line.potential = potential

                uv_center = np.mean(contour_line.uv, axis=1)
                uv_to_center_vecs = contour_line.uv[:, :-1] - uv_center[:, np.newaxis]
                uv_to_center_vecs = np.vstack((uv_to_center_vecs, np.zeros(uv_to_center_vecs.shape[1])))
                uv_vecs = contour_line.uv[:, 1:] - contour_line.uv[:, :-1]
                uv_vecs = np.vstack((uv_vecs, np.zeros(uv_vecs.shape[1])))
                rot_vecs = np.cross(uv_to_center_vecs.T, uv_vecs.T)  # Transpose
                track_orientation = int(np.sign(np.sum(rot_vecs[:, 2])))  # Swap, because vector is transposed
                contour_line.current_orientation = track_orientation

                contour_index += 1

        # End of Part 4

    return coil_parts
