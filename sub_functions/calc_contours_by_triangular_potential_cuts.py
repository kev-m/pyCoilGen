import numpy as np

from typing import List

# Logging
import logging

# Local imports
from sub_functions.data_structures import CoilPart

log = logging.getLogger(__name__)


def calc_contours_by_triangular_potential_cuts(coil_parts: List[CoilPart]):
    """
    Center the stream function potential around zero and add zeros around the periphery.

    Args:
        coil_parts (list): List of coil parts.

    Returns:
        coil_parts (list): Updated list of coil parts.
    """

    part = [None] * len(coil_parts)  # initialize container

    for part_ind in range(len(coil_parts)):
        coil_parts[part_ind]['coil_mesh']['vertices'] = np.transpose(
            coil_parts[part_ind]['coil_mesh']['vertices'])
        coil_parts[part_ind]['coil_mesh']['uv'] = np.transpose(
            coil_parts[part_ind]['coil_mesh']['uv'])
        coil_parts[part_ind]['coil_mesh']['faces'] = np.transpose(
            coil_parts[part_ind]['coil_mesh']['faces'])

        # build the edges of the mesh and for each edge the attached triangle
        tri = Delaunay(coil_parts[part_ind]['coil_mesh']['uv'])
        edge_nodes = tri.convex_hull
        edge_attached_triangles = tri.vertex_neighbor_vertices
        num_attached_tris = np.bincount(edge_attached_triangles[1])
        edge_nodes = edge_nodes[num_attached_tris == 2]
        edge_attached_triangles_inds = np.column_stack(
            (edge_attached_triangles[0], edge_attached_triangles[1]))

        # find the node indices of triangles for the corresponding edges
        edge_attached_triangles = []
        for x_ind, y_ind in edge_attached_triangles_inds:
            edge_attached_triangles.append(
                coil_parts[part_ind]['coil_mesh']['faces'][y_ind])

        coil_parts[part_ind]['edge_nodes'] = edge_nodes
        coil_parts[part_ind]['edge_attached_triangles'] = edge_attached_triangles

        # take only the edge opposing nodes of these triangles
        edge_opposed_nodes = np.zeros(edge_attached_triangles_inds.shape)
        for x_ind in range(edge_attached_triangles_inds.shape[0]):
            edge_opposed_nodes[x_ind, 0] = np.setdiff1d(
                edge_attached_triangles[x_ind, 0], edge_attached_triangles[x_ind, 1])
            edge_opposed_nodes[x_ind, 1] = np.setdiff1d(
                edge_attached_triangles[x_ind, 1], edge_attached_triangles[x_ind, 0])

        # test for all edges whether they cut one of the potential levels
        rep_contour_level_list = np.tile(
            coil_parts[part_ind]['potential_level_list'], (edge_nodes.shape[0], 1))
        edge_node_potentials = coil_parts[part_ind]['stream_function'](
            edge_nodes)
        min_edge_potentials = np.min(edge_node_potentials, axis=1)
        max_edge_potentials = np.max(edge_node_potentials, axis=1)
        min_edge_potentials = np.tile(
            min_edge_potentials, (1, coil_parts[part_ind]['potential_level_list'].size))
        max_edge_potentials = np.tile(
            max_edge_potentials, (1, coil_parts[part_ind]['potential_level_list'].size))

        tri_below_pot_step = max_edge_potentials > rep_contour_level_list
        tri_above_pot_step = min_edge_potentials < rep_contour_level_list
        potential_cut_criteria = tri_below_pot_step & tri_above_pot_step
        potential_cut_criteria = potential_cut_criteria.astype(float)
        potential_cut_criteria[potential_cut_criteria == 0] = np.nan

        coil_parts[part_ind]['edge_opposed_nodes'] = edge_opposed_nodes
        coil_parts[part_ind]['potential_cut_criteria'] = potential_cut_criteria

        # calculate for each edge which cuts a potential the uv coordinates
        edge_lengths = np.linalg.norm(coil_parts[part_ind]['coil_mesh']['uv'][edge_nodes[:, 1], :] -
                                      coil_parts[part_ind]['coil_mesh']['uv'][edge_nodes[:, 0], :], axis=1)
        edge_lengths = np.tile(
            edge_lengths, (1, coil_parts[part_ind]['potential_level_list'].size))
        edge_potential_span = edge_node_potentials[:,
                                                   1] - edge_node_potentials[:, 0]
        edge_potential_span = np.tile(
            edge_potential_span, (1, coil_parts[part_ind]['potential_level_list'].size))

        pot_dist_to_step = rep_contour_level_list - \
            np.tile(
                edge_node_potentials[:, 0], (1, coil_parts[part_ind]['potential_level_list'].size))
        cut_point_distance_to_edge_node_1 = np.abs(
            pot_dist_to_step / edge_potential_span * edge_lengths)

        u_component_edge_vectors = coil_parts[part_ind]['coil_mesh']['uv'][edge_nodes[:,
                                                                                      1], 0] - coil_parts[part_ind]['coil_mesh']['uv'][edge_nodes[:, 0], 0]
        u_component_edge_vectors = np.tile(
            u_component_edge_vectors, (1, coil_parts[part_ind]['potential_level_list'].size))
        v_component_edge_vectors = coil_parts[part_ind]['coil_mesh']['uv'][edge_nodes[:,
                                                                                      1], 1] - coil_parts[part_ind]['coil_mesh']['uv'][edge_nodes[:, 0], 1]
        v_component_edge_vectors = np.tile(
            v_component_edge_vectors, (1, coil_parts[part_ind]['potential_level_list'].size))

        first_edge_node_u = np.tile(coil_parts[part_ind]['coil_mesh']['uv'][edge_nodes[:, 0], 0], (
            1, coil_parts[part_ind]['potential_level_list'].size))
        first_edge_node_v = np.tile(coil_parts[part_ind]['coil_mesh']['uv'][edge_nodes[:, 0], 1], (
            1, coil_parts[part_ind]['potential_level_list'].size))
        u_cut_point = potential_cut_criteria * \
            (first_edge_node_u + u_component_edge_vectors *
             cut_point_distance_to_edge_node_1 / edge_lengths)
        v_cut_point = potential_cut_criteria * \
            (first_edge_node_v + v_component_edge_vectors *
             cut_point_distance_to_edge_node_1 / edge_lengths)

        coil_parts[part_ind]['u_cut_point'] = u_cut_point
        coil_parts[part_ind]['v_cut_point'] = v_cut_point

        # create cell by sorting the cut points to the corresponding potential levels
        potential_sorted_cut_points = [np.column_stack((u_cut_point[:, pot_ind], v_cut_point[:, pot_ind], np.arange(
            1, edge_nodes.shape[0] + 1))) for pot_ind in range(coil_parts[part_ind]['potential_level_list'].size)]
        potential_sorted_cut_points = [
            points[~np.isnan(points[:, 0]), :] for points in potential_sorted_cut_points]

        # create a struct with the unsorted points
        empty_potential_groups = [
            len(points) == 0 for points in potential_sorted_cut_points]
        part[part_ind]['raw']['unsorted_points'] = [{}
                                                    for _ in range(np.sum(~empty_potential_groups))]
        running_ind = 0
        for struct_ind in np.where(~empty_potential_groups)[0]:
            part[part_ind]['raw']['unsorted_points'][running_ind]['potential'] = coil_parts[part_ind]['potential_level_list'][struct_ind]
            part[part_ind]['raw']['unsorted_points'][running_ind]['edge_ind'] = potential_sorted_cut_points[struct_ind][:, 2]
            part[part_ind]['raw']['unsorted_points'][running_ind]['uv'] = potential_sorted_cut_points[struct_ind][:, :2]
            running_ind += 1

        # separate loops within potential groups
        # building loops by edge connectivity information

        # select the one of two possible triangles which has not been used yet
        part[part_ind]['raw']['unarranged_loops']['loop']['edge_inds'] = []
        part[part_ind]['raw']['unarranged_loops']['loop']['uv'] = []

        # create loops
        for potential_group in range(len(part[part_ind]['raw']['unsorted_points'])):
            all_current_edges = edge_nodes[part[part_ind]['raw']
                                           ['unsorted_points'][potential_group]['edge_ind'] - 1, :]
            all_current_opposed_nodes = edge_opposed_nodes[part[part_ind]
                                                           ['raw']['unsorted_points'][potential_group]['edge_ind'] - 1, :]
            all_current_uv_coords = part[part_ind]['raw']['unsorted_points'][potential_group]['uv']
            set_new_start = 1
            num_build_loops = 0
            edge_already_used = np.zeros(all_current_edges.shape[0], dtype=int)

            # begin to connect
            while not np.all(edge_already_used):
                if set_new_start == 1:  # set new start if loop is closed within one potential group or line of segments has ended
                    # initialize the starting position
                    num_build_loops += 1
                    starting_edge = np.min(np.where(edge_already_used == 0)[0])
                    part[part_ind]['raw']['unarranged_loops'][potential_group]['loop'][num_build_loops]['uv'] = [
                        all_current_uv_coords[starting_edge]]
                    part[part_ind]['raw']['unarranged_loops'][potential_group]['loop'][
                        num_build_loops]['edge_inds'] = all_current_edges[starting_edge, :]
                    # mark the start position as "included"
                    edge_already_used[starting_edge] = 1
                    current_edge = starting_edge
                    # find the next edge
                    # find the edges which contain the opposed nodes of the current edge as well as one of the nodes of the current edge
                    current_edge_nodes = all_current_edges[current_edge, :]
                    neighbouring_free_next_edges = np.where(np.any((all_current_edges == all_current_opposed_nodes[current_edge, 0]) | (
                        all_current_edges == all_current_opposed_nodes[current_edge, 1]), axis=1) & np.any((all_current_edges == current_edge_nodes[0]) | (all_current_edges == current_edge_nodes[1]), axis=1))[0]

                    if len(neighbouring_free_next_edges) == 0:
                        break
                    elif len(neighbouring_free_next_edges) == 1:
                        next_edge = neighbouring_free_next_edges[0]
                    else:
                        # select as a starting direction one of the two possible nodes
                        if not edge_already_used[neighbouring_free_next_edges[0]]:
                            next_edge = neighbouring_free_next_edges[0]
                        else:
                            next_edge = neighbouring_free_next_edges[1]
                        set_new_start = 0
                while next_edge != starting_edge:
                    # include the next point
                    edge_already_used[next_edge] = 1
                    part[part_ind]['raw']['unarranged_loops'][potential_group]['loop'][num_build_loops]['uv'].append(
                        all_current_uv_coords[next_edge])
                    part[part_ind]['raw']['unarranged_loops'][potential_group]['loop'][num_build_loops]['edge_inds'].append(
                        all_current_edges[next_edge])
                    current_edge = next_edge
                    # find the next edge
                    # find the edges which contain the opposed nodes of the current edge as well as one of the nodes of the current edge
                    current_edge_nodes = all_current_edges[current_edge, :]
                    possible_next_edges = np.where(np.any((all_current_edges == all_current_opposed_nodes[current_edge, 0]) | (all_current_edges == all_current_opposed_nodes[current_edge, 1]), axis=1) & np.any(
                        (all_current_edges == current_edge_nodes[0]) | (all_current_edges == current_edge_nodes[1]), axis=1))[0]
                    possible_next_edges = np.setdiff1d(
                        possible_next_edges, np.where(edge_already_used == 1)[0])
                    # check if the starting edge is under the possible next edges
                    if len(possible_next_edges) == 0:
                        break
                    else:
                        if len(possible_next_edges) == 1:
                            next_edge = possible_next_edges[0]
                        else:
                            # select as a starting direction one of the two possible nodes
                            if not edge_already_used[possible_next_edges[0]]:
                                next_edge = possible_next_edges[0]
                            else:
                                next_edge = possible_next_edges[1]
                set_new_start = 1

       # evaluate for each loop the current orientation
        part[part_ind]['raw']['unarranged_loops'][len(
            part[part_ind]['raw']['unsorted_points']) - 1]['loop'][0]['current_orientation'] = []
        for pot_ind in range(len(part[part_ind]['raw']['unsorted_points'])):
            center_segment_potential = part[part_ind]['raw']['unsorted_points'][pot_ind]['potential']
            for loop_ind in range(len(part[part_ind]['raw']['unarranged_loops'][pot_ind]['loop'])):
                if len(part[part_ind]['raw']['unarranged_loops'][pot_ind]['loop'][loop_ind]['edge_inds']) > 2:
                    test_edge = int(np.floor(len(
                        part[part_ind]['raw']['unarranged_loops'][pot_ind]['loop'][loop_ind]['edge_inds']) / 2))
                    first_edge = part[part_ind]['raw']['unarranged_loops'][pot_ind]['loop'][loop_ind]['edge_inds'][test_edge]
                    second_edge = part[part_ind]['raw']['unarranged_loops'][pot_ind]['loop'][loop_ind]['edge_inds'][test_edge + 1]
                    node_1 = np.intersect1d(first_edge, second_edge)
                    node_2 = np.setdiff1d(first_edge, node_1)
                    node_3 = np.setdiff1d(second_edge, node_1)
                    node_1_uv = coil_parts[part_ind]['coil_mesh']['uv'][node_1]
                    node_2_uv = coil_parts[part_ind]['coil_mesh']['uv'][node_2]
                    node_3_uv = coil_parts[part_ind]['coil_mesh']['uv'][node_3]
                    node_1_pot = coil_parts[part_ind]['stream_function'][node_1]
                    node_2_pot = coil_parts[part_ind]['stream_function'][node_2]
                    node_3_pot = coil_parts[part_ind]['stream_function'][node_3]
                    # calculate the 2D gradient of the triangle
                    center_segment_position = (part[part_ind]['raw']['unarranged_loops'][pot_ind]['loop'][loop_ind]['uv']
                                               [test_edge] + part[part_ind]['raw']['unarranged_loops'][pot_ind]['loop'][loop_ind]['uv'][test_edge + 1]) / 2
                    vec_center_node_1 = node_1_uv - center_segment_position
                    vec_center_node_2 = node_2_uv - center_segment_position
                    vec_center_node_3 = node_3_uv - center_segment_position
                    pot_diff_center_node_1 = node_1_pot - center_segment_potential
                    pot_diff_center_node_2 = node_2_pot - center_segment_potential
                    pot_diff_center_node_3 = node_3_pot - center_segment_potential
                    pot_gradient_vec = vec_center_node_1 * pot_diff_center_node_1 + vec_center_node_2 * \
                        pot_diff_center_node_2 + vec_center_node_3 * pot_diff_center_node_3
                    # test the chirality of the segment on the potential gradient on that segment
                    segment_vec = part[part_ind]['raw']['unarranged_loops'][pot_ind]['loop'][loop_ind]['uv'][test_edge +
                                                                                                             1] - part[part_ind]['raw']['unarranged_loops'][pot_ind]['loop'][loop_ind]['uv'][test_edge]
                    cross_vec = np.cross(np.concatenate(
                        (segment_vec, [0])), np.concatenate((pot_gradient_vec, [0])))
                    part[part_ind]['raw']['unarranged_loops'][pot_ind]['loop'][loop_ind]['current_orientation'] = np.sign(
                        cross_vec[2])

                    if part[part_ind]['raw']['unarranged_loops'][pot_ind]['loop'][loop_ind]['current_orientation'] == -1:
                        part[part_ind]['raw']['unarranged_loops'][pot_ind]['loop'][loop_ind]['uv'] = np.flipud(
                            part[part_ind]['raw']['unarranged_loops'][pot_ind]['loop'][loop_ind]['uv'])
                        part[part_ind]['raw']['unarranged_loops'][pot_ind]['loop'][loop_ind]['edge_inds'] = np.flipud(
                            part[part_ind]['raw']['unarranged_loops'][pot_ind]['loop'][loop_ind]['edge_inds'])
                else:
                    raise ValueError(
                        'Some loops are too small and contain only 2 points, therefore ill-defined')

        # build the contour lines
        coil_parts[part_ind]['contour_lines']['uv'] = []
        coil_parts[part_ind]['contour_lines']['potential'] = []
        coil_parts[part_ind]['contour_lines']['current_orientation'] = []
        build_ind = 1
        for pot_ind in range(len(part[part_ind]['raw']['unsorted_points'])):
            for loop_ind in range(len(part[part_ind]['raw']['unarranged_loops'][pot_ind]['loop'])):
                coil_parts[part_ind]['contour_lines'][build_ind]['uv'] = part[part_ind][
                    'raw']['unarranged_loops'][pot_ind]['loop'][loop_ind]['uv']
                coil_parts[part_ind]['contour_lines'][build_ind]['potential'] = part[part_ind]['raw']['unsorted_points'][pot_ind]['potential']
                # find the current orientation (for comparison with other loops)
                uv_center = np.mean(
                    coil_parts[part_ind]['contour_lines'][build_ind]['uv'], axis=1)
                uv_to_center_vecs = coil_parts[part_ind]['contour_lines'][build_ind]['uv'][:,
                                                                                           :-1] - uv_center[:, np.newaxis]
                uv_to_center_vecs = np.concatenate(
                    (uv_to_center_vecs, np.zeros((1, uv_to_center_vecs.shape[1]))), axis=0)
                uv_vecs = coil_parts[part_ind]['contour_lines'][build_ind]['uv'][:,
                                                                                 1:] - coil_parts[part_ind]['contour_lines'][build_ind]['uv'][:, :-1]
                uv_vecs = np.concatenate(
                    (uv_vecs, np.zeros((1, uv_vecs.shape[1]))), axis=0)
                rot_vecs = np.cross(uv_to_center_vecs, uv_vecs)
                track_orientation = np.sign(np.sum(rot_vecs[2, :]))
                coil_parts[part_ind]['contour_lines'][build_ind]['current_orientation'] = track_orientation
                build_ind += 1

        coil_parts[part_ind]['coil_mesh']['vertices'] = np.transpose(
            coil_parts[part_ind]['coil_mesh']['vertices'])
        coil_parts[part_ind]['coil_mesh']['uv'] = np.transpose(
            coil_parts[part_ind]['coil_mesh']['uv'])
        coil_parts[part_ind]['coil_mesh']['faces'] = np.transpose(
            coil_parts[part_ind]['coil_mesh']['faces'])

    return coil_parts


if __name__ == "__main__":
    pass
