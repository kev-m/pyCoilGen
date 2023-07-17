from dataclasses import dataclass
from typing import List
import numpy as np

# Logging
import logging

# Local imports
from sub_functions.data_structures import CoilPart

# Testing:
import trimesh

log = logging.getLogger(__name__)

###########################################################
# TODO: DEVELOPMENT: Move these to DataStructures


@dataclass
class Loop:
    loop = None


@dataclass
# Define the structure for unarranged loops
class UnarrangedLoop:
    edge_inds = None
    uv = None


@dataclass
# Define the structure for unsorted points
class UnsortedPoints:
    potential = None
    edge_ind = None
    uv = None


@dataclass
class RawPart:
    unsorted_points: List[UnsortedPoints] = None
    unarranged_loops: List[UnarrangedLoop] = None
#
##########################################################


# ChatGPT error, create this function
def numel(x):
    return len(x)


def calc_contours_by_triangular_potential_cuts(coil_parts: List[CoilPart]):
    """
    Center the stream function potential around zero and add zeros around the periphery.

    Args:
        coil_parts (list): List of coil parts.

    Returns:
        coil_parts (list): Updated list of coil parts.
    """

    for part_ind in range(len(coil_parts)):
        part = coil_parts[part_ind]
        part_mesh = part.coil_mesh
        part_vertices = part_mesh.get_vertices()  # .T  # Transpose vertices
        part_uv = part_mesh.uv  # .T  # Transpose UV coordinates
        part_faces = part_mesh.get_faces()  # .T  # Transpose faces

        # For the UV mesh, calculate:
        # 1. edge_nodes: The edges, i.e. the list of connected vertices (array m x 2)
        # 2. edge_attached_triangles: The attached triangles, i.e. the list of triangles that share an edge (array of n).
        # 3. num_attached_tris: The list of

        # Compute edges and attached triangles
        mesh = trimesh.Trimesh(vertices=part_uv,
                               faces=part_faces,
                               process=False)
        # Returns the edges that are shared by the adjacent faces (index into faces array).
        edge_faces = mesh.face_adjacency
        # Returns the edges that are shared by the adjacent faces (index into vertices array).
        edge_nodes = mesh.face_adjacency_edges
        num_edges = edge_nodes.shape[0]
        log.debug(" -- edge_faces shape: %s, max(%d)", edge_faces.shape, np.max(edge_faces))  # 696,2: Max: 263
        log.debug(" -- edge_nodes shape: %s, max(%d)", edge_nodes.shape, np.max(edge_nodes))  # 696,2: Max: 263

        # 1	2
        # 2	3
        # 2	4
        # 2	7
        # 2	8
        # 2	9
        # . .
        # edge_nodes

        # Goal: The pair of vertices of the faces that share an edge.
        # MATLAB: 696,2 Max: 264 (num vertices)
        # [1,8,2]	[1,2,3]
        # [4,3,2]	[1,2,3]
        # [4,2,10]	[4,3,2]
        # [7,2,8]	[2,7,9]
        # [1,8,2]	[7,2,8]
        # NOTE: Vertix arrays order is reversed compared to MATLAB: [0,1,2] [0,7,1]	vs [1,8,2]	[1,2,3]
        edge_attached_triangles = np.empty((num_edges, 2, 3), dtype=int)
        for index, edges in enumerate(edge_faces):
            edge_attached_triangles[index] = np.array((part_faces[edges[0]], part_faces[edges[1]]))

        # 696,2,3: Max: 263
        log.debug(" -- edge_attached_triangles shape: %s, max(%d)",
                  edge_attached_triangles.shape, np.max(edge_attached_triangles))

        # Take only the edge opposing nodes of these triangles
        # MATLAB: 696,2 Max: 264 (num vertices)
        # 8	3
        # 4	1
        # 10 3
        # 8	9
        # 1	7
        # 7	10
        # . .
        # NOTE: Index order is reversed compared to MATLAB
        edge_opposed_nodes = np.zeros_like(edge_nodes)
        for x_ind in range(edge_nodes.shape[0]):
            edge_opposed_nodes[x_ind, 0] = np.setdiff1d(edge_attached_triangles[x_ind, 0], edge_nodes[x_ind])
            edge_opposed_nodes[x_ind, 1] = np.setdiff1d(edge_attached_triangles[x_ind, 1], edge_nodes[x_ind])

        log.debug(" -- edge_opposed_nodes shape: %s, max(%d)", edge_opposed_nodes.shape, np.max(edge_opposed_nodes))

        # Test for all edges whether they cut one of the potential levels
        rep_contour_level_list = np.tile(part.potential_level_list, (edge_nodes.shape[0], 1))
        edge_node_potentials = part.stream_function[edge_nodes]
        min_edge_potentials = np.min(edge_node_potentials, axis=1)
        max_edge_potentials = np.max(edge_node_potentials, axis=1)
        min_edge_potentials = np.tile(min_edge_potentials[:, np.newaxis], (1, len(part.potential_level_list)))
        max_edge_potentials = np.tile(max_edge_potentials[:, np.newaxis], (1, len(part.potential_level_list)))

        tri_below_pot_step = max_edge_potentials > rep_contour_level_list
        tri_above_pot_step = min_edge_potentials < rep_contour_level_list
        potential_cut_criteria = np.logical_and(tri_below_pot_step, tri_above_pot_step)
        potential_cut_criteria = potential_cut_criteria.astype(float)
        potential_cut_criteria[np.where(potential_cut_criteria == 0)] = np.nan

        # Calculate for each edge which cuts a potential the uv coordinates
        edge_lengths = np.linalg.norm(part_uv[edge_nodes[:, 1]] - part_uv[edge_nodes[:, 0]], axis=1)
        edge_lengths = np.tile(edge_lengths[:, np.newaxis], (1, len(part.potential_level_list)))
        edge_potential_span = edge_node_potentials[:, 1] - edge_node_potentials[:, 0]
        edge_potential_span = np.tile(edge_potential_span[:, np.newaxis], (1, len(part.potential_level_list)))

        pot_dist_to_step = rep_contour_level_list - np.tile(edge_node_potentials[:, 0][:, np.newaxis],
                                                            (1, len(part.potential_level_list)))
        cut_point_distance_to_edge_node_1 = np.abs(pot_dist_to_step / edge_potential_span * edge_lengths)

        u_component_edge_vectors = part_uv[edge_nodes[:, 1], 0] - part_uv[edge_nodes[:, 0], 0]
        u_component_edge_vectors = np.tile(u_component_edge_vectors[:, np.newaxis], (1, len(part.potential_level_list)))
        v_component_edge_vectors = part_uv[edge_nodes[:, 1], 1] - part_uv[edge_nodes[:, 0], 1]
        v_component_edge_vectors = np.tile(v_component_edge_vectors[:, np.newaxis], (1, len(part.potential_level_list)))

        first_edge_node_u = np.tile(part_uv[edge_nodes[:, 0], 0][:, np.newaxis], (1, len(part.potential_level_list)))
        first_edge_node_v = np.tile(part_uv[edge_nodes[:, 0], 1][:, np.newaxis], (1, len(part.potential_level_list)))
        u_cut_point = potential_cut_criteria * (
            first_edge_node_u + u_component_edge_vectors * cut_point_distance_to_edge_node_1 / edge_lengths)
        v_cut_point = potential_cut_criteria * (
            first_edge_node_v + v_component_edge_vectors * cut_point_distance_to_edge_node_1 / edge_lengths)

        # Create cell by sorting the cut points to the corresponding potential levels
        potential_sorted_cut_points = []
        for pot_ind in range(len(part.potential_level_list)):
            cut_points = np.column_stack(
                (u_cut_point[:, pot_ind], v_cut_point[:, pot_ind], np.arange(edge_nodes.shape[0])))
            cut_points = cut_points[~np.isnan(cut_points[:, 0])]
            potential_sorted_cut_points.append(cut_points)

        # End of Part 1

        ###############################################################################
        # TODO: Verify: potential_cut_criteria, potential_sorted_cut_points

        #
        ###############################################################################

        # Start of Part 2
        # Create the unsorted points structure
        empty_potential_groups = [potential_sorted_cut_points[i] == [] for i in range(len(potential_sorted_cut_points))]
        part.raw = RawPart()
        num_false = len(empty_potential_groups) - sum(empty_potential_groups)
        part.raw.unsorted_points = [UnsortedPoints() for _ in range(num_false)]
        part.raw.unarranged_loops = [UnarrangedLoop() for _ in range(num_false)]
        running_ind = 0

        for struct_ind in range(len(empty_potential_groups)):
            if not empty_potential_groups[struct_ind]:
                part.raw.unsorted_points[running_ind].potential = part.potential_level_list[struct_ind]
                part.raw.unsorted_points[running_ind].edge_ind = potential_sorted_cut_points[struct_ind][:, 2]
                part.raw.unsorted_points[running_ind].uv = potential_sorted_cut_points[struct_ind][:, :2]
                running_ind += 1

        log.debug(" ---- here ---")
        # Separate loops within potential groups
        # part.raw.unarranged_loops.loop.edge_inds = []
        # part.raw.unarranged_loops.loop.uv = []

        # Create loops
        for potential_group in range(len(part.raw.unsorted_points)):
            all_current_edges = edge_nodes[part.raw.unsorted_points[potential_group].edge_ind]
            all_current_opposed_nodes = edge_opposed_nodes[part.raw.unsorted_points[potential_group].edge_ind]
            all_current_uv_coords = part.raw.unsorted_points[potential_group].uv

            set_new_start = True
            num_build_loops = 0
            edge_already_used = np.zeros(all_current_edges.shape[0], dtype=int)

            group_unarranged_loop = part.raw.unarranged_loops[potential_group]

            while not np.all(edge_already_used):
                if set_new_start:
                    num_build_loops += 1
                    starting_edge = np.min(np.where(edge_already_used == 0)[0])
                    loop = UnarrangedLoop()
                    loop.uv = [all_current_uv_coords[starting_edge]]
                    loop.edge_inds = [all_current_edges[starting_edge]]
                    group_unarranged_loop.loop.append(loop)
                    edge_already_used[starting_edge] = 1
                    current_edge = starting_edge

                    current_edge_nodes = all_current_edges[current_edge]
                    neighbouring_free_next_edges = np.where(
                        np.any(
                            np.logical_or(all_current_edges == all_current_opposed_nodes[current_edge, 0],
                                          all_current_edges == all_current_opposed_nodes[current_edge, 1]), axis=1) &
                        np.any(
                            np.logical_or(all_current_edges == current_edge_nodes[0],
                                          all_current_edges == current_edge_nodes[1]), axis=1)
                    )[0]

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
                    group_unarranged_loop.loop[num_build_loops - 1].uv.append(all_current_uv_coords[next_edge])
                    group_unarranged_loop.loop[num_build_loops - 1].edge_inds.append(all_current_edges[next_edge])
                    current_edge = next_edge

                    current_edge_nodes = all_current_edges[current_edge]
                    possible_next_edges = np.where(
                        np.any(
                            np.logical_or(all_current_edges == all_current_opposed_nodes[current_edge, 0],
                                          all_current_edges == all_current_opposed_nodes[current_edge, 1]), axis=1) &
                        np.any(
                            np.logical_or(all_current_edges == current_edge_nodes[0],
                                          all_current_edges == current_edge_nodes[1]), axis=1)
                    )[0]
                    possible_next_edges = np.setdiff1d(possible_next_edges, np.where(edge_already_used == 1)[0])

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
        # Evaluate current orientation for each loop
        part.raw.unarranged_loops[numel(part.raw.unsorted_points)].loop[0].current_orientation = []

        for pot_ind in range(numel(part.raw.unsorted_points)):
            center_segment_potential = part.raw.unsorted_points[pot_ind].potential

            for loop_ind in range(numel(part.raw.unarranged_loops[pot_ind].loop)):
                if numel(part.raw.unarranged_loops[pot_ind].loop[loop_ind].edge_inds) > 2:
                    test_edge = int(np.floor(part.raw.unarranged_loops[pot_ind].loop[loop_ind].edge_inds.shape[0] / 2))
                    first_edge = part.raw.unarranged_loops[pot_ind].loop[loop_ind].edge_inds[test_edge, :]
                    second_edge = part.raw.unarranged_loops[pot_ind].loop[loop_ind].edge_inds[test_edge + 1, :]
                    node_1 = np.intersect1d(first_edge, second_edge)
                    node_2 = np.setdiff1d(first_edge, node_1)
                    node_3 = np.setdiff1d(second_edge, node_1)
                    node_1_uv = coil_parts[part_ind].coil_mesh.uv[node_1, :]
                    node_2_uv = coil_parts[part_ind].coil_mesh.uv[node_2, :]
                    node_3_uv = coil_parts[part_ind].coil_mesh.uv[node_3, :]
                    node_1_pot = coil_parts[part_ind].stream_function[node_1]
                    node_2_pot = coil_parts[part_ind].stream_function[node_2]
                    node_3_pot = coil_parts[part_ind].stream_function[node_3]

                    # Calculate the 2D gradient of the triangle
                    center_segment_position = (part.raw.unarranged_loops[pot_ind].loop[loop_ind].uv[test_edge, :] +
                                               part.raw.unarranged_loops[pot_ind].loop[loop_ind].uv[test_edge + 1, :]) / 2

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
                    segment_vec = part.raw.unarranged_loops[pot_ind].loop[loop_ind].uv[test_edge + 1, :] - \
                        part.raw.unarranged_loops[pot_ind].loop[loop_ind].uv[test_edge, :]

                    cross_vec = np.cross([segment_vec[0], segment_vec[1], 0], [
                                         pot_gradient_vec[0], pot_gradient_vec[1], 0])

                    part.raw.unarranged_loops[pot_ind].loop[loop_ind].current_orientation = np.sign(cross_vec[2])

                    if part.raw.unarranged_loops[pot_ind].loop[loop_ind].current_orientation == -1:
                        part.raw.unarranged_loops[pot_ind].loop[loop_ind].uv = np.flipud(
                            part.raw.unarranged_loops[pot_ind].loop[loop_ind].uv)
                        part.raw.unarranged_loops[pot_ind].loop[loop_ind].edge_inds = np.flipud(
                            part.raw.unarranged_loops[pot_ind].loop[loop_ind].edge_inds)
                else:
                    raise ValueError('Some loops are too small and contain only 2 points, therefore ill-defined')
        # End of Part 3

        # Start of Part 4
        # Build the contour lines
        part.contour_lines.uv = []
        part.contour_lines.potential = []
        part.contour_lines.current_orientation = []
        build_ind = 1

        for pot_ind in range(numel(part.raw.unsorted_points)):
            for loop_ind in range(numel(part.raw.unarranged_loops[pot_ind].loop)):
                part.contour_lines[build_ind].uv = part.raw.unarranged_loops[pot_ind].loop[loop_ind].uv.T
                part.contour_lines[build_ind].potential = part.raw.unsorted_points[pot_ind].potential

                # Find the current orientation (for comparison with other loops)
                uv_center = np.mean(part.contour_lines[build_ind].uv, axis=1)
                uv_to_center_vecs = part.contour_lines[build_ind].uv[:, :-1] - np.expand_dims(uv_center, axis=1)
                uv_to_center_vecs = np.concatenate(
                    (uv_to_center_vecs, np.zeros((1, uv_to_center_vecs.shape[1]))), axis=0)
                uv_vecs = part.contour_lines[build_ind].uv[:, 1:] - part.contour_lines[build_ind].uv[:, :-1]
                uv_vecs = np.concatenate((uv_vecs, np.zeros((1, uv_vecs.shape[1]))), axis=0)
                rot_vecs = np.cross(uv_to_center_vecs, uv_vecs)
                track_orientation = np.sign(np.sum(rot_vecs[2, :]))
                part.contour_lines[build_ind].current_orientation = track_orientation

                build_ind += 1
        # End of Part 4

    return coil_parts


if __name__ == "__main__":
    pass
