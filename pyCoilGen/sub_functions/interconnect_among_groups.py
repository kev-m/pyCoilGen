import numpy as np

from typing import List

# Logging
import logging

# Local imports
from .data_structures import CoilPart, Shape3D, Cuts, DataStructure
from .open_loop_with_3d_sphere import open_loop_with_3d_sphere
from .remove_points_from_loop import remove_points_from_loop

log = logging.getLogger(__name__)


def interconnect_among_groups(coil_parts: List[CoilPart], input_args, m_c_part=None):
    """
    Interconnects groups to generate a single wire track.

    Initialises the following properties of a CoilPart:
        - opening_cuts_among_groups
        - wire_path

    Depends on the following properties of the CoilParts:
        - connected_group
        - level_positions
        - group_levels

    Depends on the following input_args:
        - interconnection_cut_width

    Updates the following properties of a CoilPart:
        - None

    Args:
        coil_parts (List[CoilPart]): List of CoilPart structures, each containing a coil_mesh.
        input_args (Any): The input argument (structure) used in the MATLAB function (not used in the Python function).

    Returns:
        coil_parts (List[CoilPart]): The updated CoilParts list.
    """
    # Local parameters
    additional_points_to_remove = 2

    for part_ind in range(len(coil_parts)):
        coil_part = coil_parts[part_ind]
        connected_group_buff = coil_part.connected_group

        # Gather all return points to dismiss them for the search of group cut points
        all_cut_points = np.concatenate([group.return_path.uv for group in connected_group_buff], axis=1)

        # Group fusions
        level_hierarchy = [len(coil_part.level_positions[i]) for i in range(len(coil_part.level_positions))]  # [Y]

        for level_ind in range(max(level_hierarchy), -1, -1):
            levels_to_process = [idx for idx, value in enumerate(level_hierarchy) if value == level_ind]

            # Interconnect one level into a single group
            for single_level_ind in range(len(levels_to_process)):  # 1 level

                current_level = levels_to_process[single_level_ind]
                groups_to_connect = coil_part.group_levels[current_level].tolist()

                # Select the current host group of the level
                is_enclosing = [0] * len(groups_to_connect)

                if coil_part.level_positions[current_level]:
                    current_top_group = coil_part.level_positions[current_level][-1]
                    groups_to_connect.append(current_top_group)
                    is_enclosing.append(1)
                group_len = len(groups_to_connect)

                # Make the n-1 interconnections in an optimal way resulting in one track for that level
                num_connections_to_do = group_len - 1

                if num_connections_to_do > 0:
                    coil_part.opening_cuts_among_groups = [Cuts() for _ in range(num_connections_to_do)]

                    for connect_ind in range(num_connections_to_do):  # 3
                        # Get the tracks to connect
                        grouptracks_to_connect = [connected_group_buff[group] for group in groups_to_connect]

                        # Remove the return_path for the search of mutual group cuts
                        # Create a copy so that edits do not affect connected_group_buff entries
                        grouptracks_to_connect_without_returns = [
                            connected_group_buff[group].copy() for group in groups_to_connect]

                        for group_ind in range(group_len):
                            grouptracks_to_connect_without_returns[group_ind].uv, grouptracks_to_connect_without_returns[group_ind].v = remove_points_from_loop(
                                grouptracks_to_connect[group_ind], all_cut_points, additional_points_to_remove
                            )
                            grouptracks_to_connect_without_returns[group_ind].unrolled_coords = np.array(
                                [np.arctan2(grouptracks_to_connect_without_returns[group_ind].v[1, :], grouptracks_to_connect_without_returns[group_ind].v[0, :]),
                                 grouptracks_to_connect_without_returns[group_ind].v[2, :]]
                            )

                        # Select the return paths of those interconnected groups for later
                        min_group_dists = np.zeros((group_len, group_len))
                        min_group_inds = np.zeros((group_len, group_len), dtype=int)
                        min_pos_group = DataStructure(v=[[None for _ in range(group_len)] for _ in range(group_len)],
                                                      uv=[[None for _ in range(group_len)] for _ in range(group_len)])

                        # Find the minimal distance positions between the groups and the points with minimal distance
                        for ind1 in range(group_len):
                            for ind2 in range(group_len):
                                if ind2 != ind1:
                                    group_a = grouptracks_to_connect_without_returns[ind1]
                                    group_b = grouptracks_to_connect_without_returns[ind2]
                                    near_ind = np.zeros(group_a.v.shape[1], dtype=int)
                                    near_dist = np.zeros(group_a.v.shape[1])

                                    for point_ind in range(group_a.v.shape[1]):
                                        # Calculate the distances between group_b.v and each point in group_a.v
                                        distances = np.sqrt((group_b.v[0, :] - group_a.v[0, point_ind])**2 +
                                                            (group_b.v[1, :] - group_a.v[1, point_ind])**2 +
                                                            (group_b.v[2, :] - group_a.v[2, point_ind])**2)
                                        # Find the minimum distance and its index
                                        near_dist[point_ind] = np.min(distances)
                                        near_ind[point_ind] = np.argmin(distances)

                                    total_min_dist, total_min_ind = np.min(near_dist), np.argmin(near_dist)
                                    min_group_inds[ind1, ind2] = total_min_ind

                                    min_pos_group.uv[ind1][ind2] = grouptracks_to_connect_without_returns[ind1].uv[:, total_min_ind]
                                    min_pos_group.v[ind1][ind2] = grouptracks_to_connect_without_returns[ind1].v[:, total_min_ind]
                                    min_group_dists[ind1, ind2] = total_min_dist
                                    min_group_dists[min_group_dists == 0] = np.inf

                        # Select the pair of groups with the shortest respective distance
                        min_dist_couple1 = min_group_dists == np.min(min_group_dists)
                        min_dist_couple = np.where(min_dist_couple1)
                        couple_group1 = min_dist_couple[1][0]
                        couple_group2 = min_dist_couple[0][0]

                        # Open the loop
                        target_point_p = min_pos_group.v[couple_group1][couple_group2]  # Python shape
                        target_point = [[target_point_p[0]], [target_point_p[1]], [target_point_p[2]]]  # MATLAB shape

                        opened_group_1, cut_shape_1, _ = open_loop_with_3d_sphere(
                            grouptracks_to_connect[couple_group1], target_point, input_args.interconnection_cut_width)

                        target_point_p = min_pos_group.v[couple_group2][couple_group1]
                        target_point = [[target_point_p[0]], [target_point_p[1]], [target_point_p[2]]]  # MATLAB shape
                        opened_group_2, cut_shape_2, _ = open_loop_with_3d_sphere(
                            grouptracks_to_connect[couple_group2], target_point, input_args.interconnection_cut_width)

                        # Save the cut shapes for later plotting
                        coil_part.opening_cuts_among_groups[connect_ind].cut1 = cut_shape_1
                        coil_part.opening_cuts_among_groups[connect_ind].cut2 = cut_shape_2

                        # Fuse both groups
                        # Check which fusing order is better:
                        track_combilength1 = np.concatenate([opened_group_1.v, opened_group_2.v], axis=1)
                        track_combilength2 = np.concatenate([opened_group_2.v, opened_group_1.v], axis=1)

                        track_combilength1 = np.sum(np.linalg.norm(
                            track_combilength1[:, 1:] - track_combilength1[:, :-1], axis=0))
                        track_combilength2 = np.sum(np.linalg.norm(
                            track_combilength2[:, 1:] - track_combilength2[:, :-1], axis=0))

                        if track_combilength1 < track_combilength2:
                            fused_group = Shape3D(v=np.concatenate([opened_group_1.v, opened_group_2.v], axis=1),
                                                  uv=np.concatenate([opened_group_1.uv, opened_group_2.uv], axis=1))
                        else:
                            fused_group = Shape3D(v=np.concatenate([opened_group_2.v, opened_group_1.v], axis=1),
                                                  uv=np.concatenate([opened_group_2.uv, opened_group_1.uv], axis=1))

                        # Overwrite fused track into both group point arrays
                        # Delete one of the connected groups to avoid redundancy
                        # Do not select the host level group here!
                        if is_enclosing[couple_group1]:
                            connected_group_buff[groups_to_connect[couple_group1]].uv = fused_group.uv
                            connected_group_buff[groups_to_connect[couple_group1]].v = fused_group.v
                            is_enclosing.pop(couple_group2)
                            groups_to_connect = np.delete(groups_to_connect, couple_group2)
                        else:
                            connected_group_buff[groups_to_connect[couple_group2]].uv = fused_group.uv
                            connected_group_buff[groups_to_connect[couple_group2]].v = fused_group.v
                            is_enclosing.pop(couple_group1)
                            groups_to_connect = np.delete(groups_to_connect, couple_group1)
                        # Update the group_len variable
                        group_len = len(groups_to_connect)

        # Select the full track as the final return
        arr1 = [(len(connected_group_buff[idx].v[0]), idx) for idx, group in enumerate(connected_group_buff)]
        _, is_final_ind = max(arr1)
        full_track = connected_group_buff[is_final_ind]
        # Shift the open ends to the boundaries of the coil
        min_ind = np.argmax(full_track.v[2, :])
        full_track.v = np.roll(full_track.v, -min_ind-1, axis=1)
        full_track.uv = np.roll(full_track.uv, -min_ind-1, axis=1)

        # Assign the outputs
        coil_part.wire_path = full_track

    return coil_parts


"""
Note: The code assumes that the functions remove_points_from_loop and open_loop_with_3d_sphere are already defined
elsewhere or imported. Additionally, some variable names have been slightly modified to conform to Python's naming
conventions. The logic and functionality of the MATLAB function have been retained in the Python equivalent.
"""
