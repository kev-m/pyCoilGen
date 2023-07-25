import numpy as np
from typing import List
import logging

# Local imports
from sub_functions.data_structures import CoilPart
from sub_functions.check_mutual_loop_inclusion import check_mutual_loop_inclusion

log = logging.getLogger(__name__)

def topological_loop_grouping(coil_parts: List[CoilPart], input_args):
    """
    Group the contour loops in topological order.

    Args:
        coil_parts (List[CoilPart]): List of CoilPart structures, each containing contour_lines.
        input_args: The input arguments (structure).

    Returns:
        List[CoilPart]: Updated list of CoilPart structures with contour loop groups.
    """
    for part_ind in range(len(coil_parts)):
        num_total_loops = len(coil_parts[part_ind].contour_lines)
        loop_in_loop_mat = np.zeros((num_total_loops, num_total_loops))

        # Check for all loop enclosures of other loops
        for loop_to_test in range(num_total_loops):
            for loop_num in range(num_total_loops):
                if loop_to_test != loop_num:
                    loop_in_loop_mat[loop_to_test, loop_num] = check_mutual_loop_inclusion(
                        coil_parts[part_ind].contour_lines[loop_num].uv,
                        coil_parts[part_ind].contour_lines[loop_to_test].uv
                    )

        # TypeError: ufunc 'invert' not supported for the input types, and the inputs could not be safely coerced to any supported types according to the casting rule ''safe''
        loop_in_loop_mat = loop_in_loop_mat * ~np.eye(num_total_loops)

        lower_loops = [list(np.nonzero(loop_in_loop_mat[:, loop_to_test])[0]) for loop_to_test in range(num_total_loops)]
        higher_loops = [list(np.nonzero(loop_in_loop_mat[loop_to_test, :])[0]) for loop_to_test in range(num_total_loops)]

        # Assign '0' to loops that have no lower loops
        empty_cells = [ind for ind, lower_loop in enumerate(lower_loops) if not lower_loop]

        for empty_cell in empty_cells:
            lower_loops[empty_cell] = [0]

        # Convert the list of lower loops into parallel levels
        group_levels = [None] * num_total_loops

        for loop_to_test in range(num_total_loops):
            group_levels[loop_to_test] = [ind for ind, lower_loop in enumerate(lower_loops) if lower_loop == lower_loops[loop_to_test]]

        # Check the possibility of the top level being composed out of a single group
        is_global_top_loop = [len(higher_loops[loop_num]) == num_total_loops - 1 for loop_num in range(num_total_loops)]

        # Delete the repetition in the parallel levels and the singular levels
        group_levels = sorted(list(set(tuple(level) for level in group_levels if len(level) > 1 or is_global_top_loop[level[0]])))

        # Creating the loop groups (containing still the loops of the inner groups)
        overlapping_loop_groups = [group_level for group_level in group_levels]
        overlapping_loop_groups_num = [loop_num for group_level in group_levels for loop_num in group_level]

        for overlap_index in range(len(overlapping_loop_groups)):
            overlapping_loop_groups[overlap_index] += higher_loops[overlapping_loop_groups_num[overlap_index]]

        # Build the group topology by checking the loop content of a certain group
        # to see if it is a superset of the loop content of another group
        group_in_group_mat = np.zeros((len(overlapping_loop_groups), len(overlapping_loop_groups)))

        for group_index_1 in range(len(overlapping_loop_groups)):
            for group_index_2 in range(len(overlapping_loop_groups)):
                group_in_group_mat[group_index_1, group_index_2] = all(np.isin(overlapping_loop_groups[group_index_1], overlapping_loop_groups[group_index_2]))

            group_in_group_mat[group_index_1, group_index_1] = 0

        group_is_subgroup_of = [list(np.nonzero(group_in_group_mat[group_index, :])[0]) for group_index in range(len(overlapping_loop_groups))]
        group_contains_following_group = [list(np.nonzero(group_in_group_mat[:, group_index])[0]) for group_index in range(len(overlapping_loop_groups))]

        # Remove loops from group if they also belong to a respective subgroup
        loop_groups = [group.copy() for group in overlapping_loop_groups]

        for group_index in range(len(overlapping_loop_groups)):
            loops_to_remove = [loop for subgroup_index in group_contains_following_group[group_index] for loop in overlapping_loop_groups[subgroup_index]]
            loop_groups[group_index] = list(set(loop_groups[group_index]) - set(loops_to_remove))

        # Order the groups based on the number of loops
        renamed_group_levels = group_levels.copy()
        group_levels = [group_levels[group_index] for group_index in np.argsort([len(group) for group in loop_groups])[::-1]]

        # Renumber (=rename) the groups (also in the levels)
        for group_index in range(len(group_levels)):
            for level_index in range(len(group_levels[group_index])):
                for renamed_index in range(len(loop_groups)):
                    if group_levels[group_index][level_index] in loop_groups[renamed_index]:
                        renamed_group_levels[group_index][level_index] = renamed_index

        # Resort parallel_levels to new group names
        renamed_group_levels = sorted(list(set(tuple(level) for level in renamed_group_levels)))
        group_levels = [list(level) for level in renamed_group_levels]

        # Find for each parallel level the groups that contain that level
        loops_per_level = []

        for level_index in range(len(group_levels)):
            loops_per_level.append([])
            for level_loop in group_levels[level_index]:
                loops_per_level[-1] += loop_groups[level_loop]

        level_enclosed_by_loop = []

        for level_index in range(len(group_levels)):
            level_enclosed_by_loop.append([])
            for loop_index in range(len(loops_per_level[level_index])):
                for contour_index in range(len(coil_parts[part_ind].contour_lines)):
                    if loop_in_loop_mat[contour_index, loops_per_level[level_index][loop_index]] == 1:
                        level_enclosed_by_loop[-1].append(contour_index)

        level_positions = []

        for level_index in range(len(level_enclosed_by_loop)):
            level_positions.append([])
            for group_index in range(len(loop_groups)):
                if any(loop in level_enclosed_by_loop[level_index] for loop in loop_groups[group_index]):
                    level_positions[-1].append(group_index)

        for level_index in range(len(level_positions)):
            level_positions[level_index] = [group for group in level_positions[level_index] if group not in group_levels[level_index]]

        # Sort the level_positions according to their rank
        rank_of_group = [0] * len(loop_groups)

        for group_index in range(len(loop_groups)):
            rank_of_group[group_index] = len([level for level in level_positions if group_index in level])

        for level_index in range(len(group_levels)):
            level_positions[level_index] = sorted(level_positions[level_index], key=lambda x: rank_of_group[x])

        # Build the group container
        for group_index in range(len(loop_groups)):
            # Sort the loops in each group according to the rank
            sorted_loops = sorted(
                loop_groups[group_index],
                key=lambda x: len(higher_loops[x]),
                reverse=True
            )

            group = []
            for loop_index in sorted_loops:
                loop_data = coil_parts[part_ind].contour_lines[loop_index]
                loop_entry = {
                    "number_points": loop_data.uv.shape[1],
                    "v": loop_data.v,
                    "uv": loop_data.uv,
                    "potential": loop_data.potential,
                    "current_orientation": loop_data.current_orientation
                }
                group.append(loop_entry)

            coil_parts[part_ind].groups.append(group)

    return coil_parts
