import numpy as np
from typing import List
import logging

# Local imports
from sub_functions.data_structures import CoilPart, TopoGroup
from sub_functions.check_mutual_loop_inclusion import check_mutual_loop_inclusion

log = logging.getLogger(__name__)


def topological_loop_grouping(coil_parts: List[CoilPart], input_args):
    """
    Group the contour loops in topological order.

    Initialises the following properties of a CoilPart:
        - loop_groups: ?
        - group_levels: 
        - level_positions: ?
        - groups: ?

    Args:
        coil_parts (List[CoilPart]): List of CoilPart structures, each containing contour_lines.
        input_args: The input arguments (structure).

    Returns:
        List[CoilPart]: Updated list of CoilPart structures with contour loop groups.
    """
    for part_ind in range(len(coil_parts)):
        coil_part = coil_parts[part_ind]
        num_total_loops = len(coil_part.contour_lines)
        loop_in_loop_mat = np.zeros((num_total_loops, num_total_loops))

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
            group_levels[loop_to_test] = [ind for ind, lower_loop in enumerate(
                lower_loops) if lower_loop == lower_loops[loop_to_test]]

        # Delete the repetition in the parallel levels and the singular levels
        multi_element_indices = [index for index, cell in enumerate(group_levels) if len(cell) != 1]
        new_group_levels = []
        for i in multi_element_indices:
            if not group_levels[i] in new_group_levels:
                new_group_levels.append(group_levels[i])
        top_level_indices = [index for index, cell in enumerate(
            group_levels) if len(cell) == 1 and is_global_top_loop[cell[0]]]
        for i in top_level_indices:
            if not group_levels[i] in new_group_levels:
                new_group_levels.append(group_levels[i])

        group_levels = new_group_levels

        # Creating the loop groups (containing still the loops of the inner groups)
        overlapping_loop_groups_num = [loop_num for loop_num in group_levels[0]]
        # Horizontally concatenate the cell array elements
        overlapping_loop_groups = [[item] for item in group_levels[0]]
        # Horizontally concatenate the cell array elements and convert to a list of individual elements (cells)
        yyy = []
        for index1, group in enumerate(overlapping_loop_groups):
            xxx = [group[0]]
            xxx += higher_loops[overlapping_loop_groups_num[index1]]
            yyy.append(xxx)
        overlapping_loop_groups = yyy

        # Build the group topology by checking the loop content of a certain group
        # to see if it is a superset of the loop content of another group
        group_in_group_mat = np.zeros((len(overlapping_loop_groups), len(overlapping_loop_groups)))

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
            loop_groups[iiii] = list(diff)

        # Order the loop_groups
        # TODO: Check the MATLAB if the group sub-lists are sorted too.
        coil_part.loop_groups = np.empty((len(overlapping_loop_groups)), dtype=object)
        for index, sub_loop in enumerate(loop_groups):
            coil_part.loop_groups[index] = np.array(sorted(sub_loop))
        loop_groups = coil_part.loop_groups  # Use the np.array from here on.

        # Order the groups based on the number of loops
        len_array = [len(group) for group in loop_groups]
        if min(len_array) < max(len_array):
            sort_indices = np.argsort(len_array)[::-1]
            # Sort each group level
            for group_index, group_level in enumerate(group_levels):
                sorted_group_level = group_level.copy()
                for index in range(len(group_level)):
                    sorted_group_level[index] = group_level[sort_indices[index]]
                group_levels[group_index] = sorted_group_level

        # Renumber (=rename) the groups (also in the levels)
        renamed_group_levels = group_levels.copy()
        for iiii in range(len(group_levels)):
            for level_index in range(len(group_levels[iiii])):
                for renamed_index in range(len(loop_groups)):
                    if group_levels[iiii][level_index] in loop_groups[renamed_index]:
                        renamed_group_levels[iiii][level_index] = renamed_index
                        break

        # Re-sort parallel_levels to new group names
        renamed_group_levels = sorted(list(set(tuple(level) for level in renamed_group_levels)))
        group_levels = [list(level) for level in renamed_group_levels]

        coil_part.group_levels = np.array(group_levels)

        # Find for each parallel level the groups that contain that level
        loops_per_level = []

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

        for level_index in range(len(level_positions)):
            level_positions[level_index] = [group for group in level_positions[level_index]
                                            if group not in group_levels[level_index]]

        # Sort the level_positions according to their rank
        rank_of_group = [0] * len(loop_groups)

        for iiii in range(len(loop_groups)):
            rank_of_group[iiii] = len([level for level in level_positions if iiii in level])

        for level_index in range(len(group_levels)):
            level_positions[level_index] = sorted(level_positions[level_index], key=lambda x: rank_of_group[x])

        coil_part.level_positions = level_positions
        # Build the group container
        coil_part.groups = np.empty((len(loop_groups)), dtype=object)
        for iiii, loop_group in enumerate(loop_groups):
            # Sort the loops in each group according to the rank (the number of elements in higher_loops for each
            # group)
            # TODO: Check if higher_loops is correct!
            # unsorted1 = [len(higher_loops[x]) for x in loop_group]
            # sort_ind_loops1 = sorted(unsorted1, reverse=True)

            # Alternate:
            # Sort the loops in each group according to the ordered potentials
            potentials = [coil_part.contour_lines[x].current_orientation *
                          coil_part.contour_lines[x].potential for x in coil_part.loop_groups[iiii]]
            sort_ind_loops = sorted(range(len(potentials)), key=lambda i: potentials[i])

            this_group = TopoGroup()    # Create Topopological group container
            this_group.loops = []       # Assign loops member
            for jjjj in sort_ind_loops:
                loop_group_index = coil_part.loop_groups[iiii][jjjj]
                this_contour = coil_part.contour_lines[loop_group_index]
                this_group.loops.append(this_contour)

            coil_part.groups[iiii] = this_group

    return coil_parts
