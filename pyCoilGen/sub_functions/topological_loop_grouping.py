import numpy as np
from typing import List
import logging

# Local imports
from .data_structures import CoilPart, TopoGroup
from .check_mutual_loop_inclusion import check_mutual_loop_inclusion

log = logging.getLogger(__name__)


def topological_loop_grouping(coil_parts: List[CoilPart]):
    """
    Group the contour loops in topological order.

    Initialises the following properties of a CoilPart:
        - loop_groups:
        - group_levels: 
        - level_positions:
        - groups:

    Depends on the following properties of the CoilParts:
        - contour_lines

    Depends on the following input_args:
        - None

    Updates the following properties of a CoilPart:
        - contour_lines

    Args:
        coil_parts (List[CoilPart]): List of CoilPart structures, each containing contour_lines.

    Returns:
        List[CoilPart]: Updated list of CoilPart structures with contour loop groups.
    """
    for part_ind in range(len(coil_parts)):
        coil_part = coil_parts[part_ind]
        num_total_loops = len(coil_part.contour_lines)
        loop_in_loop_mat = np.zeros((num_total_loops, num_total_loops), dtype=int)

        # Enhancement: This can be parallelised.
        # Check for all loop enclosures of other loops
        for loop_to_test in range(num_total_loops):
            for loop_num in range(num_total_loops):
                if loop_to_test != loop_num:
                    loop_in_loop_mat[loop_to_test, loop_num] = check_mutual_loop_inclusion(
                        coil_part.contour_lines[loop_num].uv,
                        coil_part.contour_lines[loop_to_test].uv
                    )

        # Clear the diagonals from the loop_in_loop_mat
        mask = np.ones((num_total_loops, num_total_loops), dtype=int) - np.eye(num_total_loops, dtype=int)
        loop_in_loop_mat = loop_in_loop_mat * mask
        lower_loops = [list(np.nonzero(loop_in_loop_mat[:, loop_to_test])[0])
                       for loop_to_test in range(num_total_loops)]
        higher_loops = [list(np.nonzero(loop_in_loop_mat[loop_to_test, :])[0])
                        for loop_to_test in range(num_total_loops)]
        is_global_top_loop = [len(higher_loops[loop_num]) == num_total_loops - 1 for loop_num in range(num_total_loops)]

        # Assign '0' to loops that have no lower loops
        empty_cells = [ind for ind, lower_loop in enumerate(lower_loops) if not lower_loop]

        for empty_cell in empty_cells:
            lower_loops[empty_cell] = [0]

        # Convert the list of lower loops into parallel levels
        group_levels = [None] * num_total_loops

        for loop_to_test in range(num_total_loops):
            group_levels[loop_to_test] = [ind for ind, lower_loop in enumerate(lower_loops) if
                                          lower_loop == lower_loops[loop_to_test]]

        # Delete the repetition in the parallel levels and the singular levels
        multi_element_indices = [index for index, cell in enumerate(group_levels) if len(cell) != 1]
        new_group_levels = []
        for i in multi_element_indices:
            if not group_levels[i] in new_group_levels:
                new_group_levels.append(group_levels[i])

        # Remove levels with only one member unless it is the singular top level
        top_level_indices = [index for index, cell in enumerate(
            group_levels) if len(cell) == 1 and is_global_top_loop[cell[0]]]
        for i in top_level_indices:
            if not group_levels[i] in new_group_levels:
                new_group_levels.append(group_levels[i])

        group_levels = new_group_levels

        # Creating the loop groups (containing still the loops of the inner groups)
        overlapping_loop_groups_num = np.asarray([item for group_level in group_levels for item in group_level])
        # Horizontally concatenate the cell array elements
        overlapping_loop_groups = np.asarray([item for group_level in group_levels for item in group_level])

        # Horizontally concatenate the cell array elements and convert to a list of individual elements (cells)
        yyy = []
        for index1, group in enumerate(overlapping_loop_groups):
            xxx = [group]
            xxx += higher_loops[overlapping_loop_groups_num[index1]]
            yyy.append(xxx)
        overlapping_loop_groups = yyy

        # Build the group topology by checking the loop content of a certain group
        # to see if it is a superset of the loop content of another group
        group_in_group_mat = np.zeros((len(overlapping_loop_groups), len(overlapping_loop_groups)), dtype=int)

        for group_index_1 in range(len(overlapping_loop_groups)):
            for group_index_2 in range(len(overlapping_loop_groups)):
                group_in_group_mat[group_index_1, group_index_2] = all(
                    np.isin(overlapping_loop_groups[group_index_1], overlapping_loop_groups[group_index_2]))

            group_in_group_mat[group_index_1, group_index_1] = 0

        group_is_subgroup_of = [list(np.nonzero(group_in_group_mat[group_index, :])[0])
                                for group_index in range(len(overlapping_loop_groups))]
        group_contains_following_group = [list(np.nonzero(group_in_group_mat[:, group_index])[0])
                                          for group_index in range(len(overlapping_loop_groups))]

        # Remove loops from group if they also belong to a respective subgroup
        loop_groups = [group.copy() for group in overlapping_loop_groups]
        for index, loop_group in enumerate(overlapping_loop_groups):
            loop_groups[index] = np.array(loop_group, copy=True)

        for iiii in range(len(overlapping_loop_groups)):
            loops_to_remove = [loop for subgroup_index in group_contains_following_group[iiii]
                               for loop in overlapping_loop_groups[subgroup_index]]
            diff = set(loop_groups[iiii]) - set(loops_to_remove)
            # Order the loop_groups
            loop_groups[iiii] = np.array(sorted(list(diff)))

        # Order the groups based on the number of loops
        # Generating the list of group lengths
        # Sorting the indices based on the lengths in descending order
        group_lengths = [len(loop_groups[x]) for x in range(len(loop_groups))]
        sort_ind = sorted(range(len(loop_groups)), key=lambda x: group_lengths[x], reverse=True)

        coil_part.loop_groups = np.empty((len(loop_groups)), dtype=object)
        for i in range(len(loop_groups)):
            coil_part.loop_groups[i] = loop_groups[sort_ind[i]]

        loop_groups = coil_part.loop_groups  # Use the np.array from here on.

        # Renumber (=rename) the groups (also in the levels)
        renamed_group_levels = group_levels.copy()
        for iiii in range(len(group_levels)):
            for level_index in range(len(group_levels[iiii])):
                for renamed_index in range(len(loop_groups)):
                    if group_levels[iiii][level_index] in loop_groups[renamed_index]:
                        renamed_group_levels[iiii][level_index] = renamed_index
                        break

        # Re-sort parallel_levels to new group names
        sort_ind_level = sorted(range(len(renamed_group_levels)), key=lambda i: min(renamed_group_levels[i]))
        coil_part.group_levels = np.empty((len(group_levels)), dtype=object)
        for i in range(len(renamed_group_levels)):
            coil_part.group_levels[i] = np.asarray(renamed_group_levels[sort_ind_level[i]])

        # Find for each parallel level the groups that contain that level
        loops_per_level = []
        for group_level in coil_part.group_levels:
            loops = []
            for loop_idx in group_level:
                loops.extend(coil_part.loop_groups[loop_idx])
            loops_per_level.append(loops)

        for level_index in range(len(group_levels)):
            loops_per_level.append([])
            for level_loop in group_levels[level_index]:
                loops_per_level[-1] += loop_groups[level_loop].tolist()

        level_enclosed_by_loop = []
        for level_index in range(len(group_levels)):
            level_enclosed_by_loop.append([])
            for loop_index in range(len(loops_per_level[level_index])):
                for contour_index in range(len(coil_part.contour_lines)):
                    if loop_in_loop_mat[contour_index, loops_per_level[level_index][loop_index]] == 1:
                        level_enclosed_by_loop[-1].append(contour_index)

        level_positions = []
        for level_index in range(len(level_enclosed_by_loop)):
            level_positions.append([])
            for iiii in range(len(loop_groups)):
                if any(loop in level_enclosed_by_loop[level_index] for loop in loop_groups[iiii]):
                    level_positions[-1].append(iiii)

        for aaaa in range(len(level_positions)):
            setdiff = set(level_positions[aaaa]) - set(coil_part.group_levels[aaaa])
            level_positions[aaaa] = list(sorted(setdiff))

        # Sort the level_positions according to their rank
        rank_of_group = [0] * len(loop_groups)
        for aaaa in range(len(loop_groups)):
            x = [aaaa in x for x in coil_part.group_levels]
            x_i = [index for index, value in enumerate(x) if value]
            assert len(x_i) == 1
            rank_of_group[aaaa] = len(level_positions[x_i[0]])

        for level_index in range(len(group_levels)):
            level_positions[level_index] = sorted(level_positions[level_index], key=lambda x: rank_of_group[x])

        coil_part.level_positions = level_positions
        # Build the group container
        coil_part.groups = np.empty((len(loop_groups)), dtype=object)
        for iiii, loop_group in enumerate(loop_groups):
            # Sort the loops in each group according to the rank (the number of elements in higher_loops for each
            # group)
            content = [len(higher_loops[x]) for x in loop_group]
            sort_ind_loops = sorted(range(len(content)), key=lambda x: content[x], reverse=True)

            this_group = TopoGroup()    # Create Topological group container
            this_group.loops = []       # Assign loops member
            for jjjj in sort_ind_loops:
                loop_group_index = coil_part.loop_groups[iiii][jjjj]
                this_contour = coil_part.contour_lines[loop_group_index]
                this_group.loops.append(this_contour)

            coil_part.groups[iiii] = this_group

    return coil_parts
