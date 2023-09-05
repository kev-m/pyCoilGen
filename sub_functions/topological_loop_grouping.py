import numpy as np
from typing import List
import logging

# Local imports
from sub_functions.data_structures import CoilPart, TopoGroup
from sub_functions.check_mutual_loop_inclusion import check_mutual_loop_inclusion

log = logging.getLogger(__name__)

# DEBUG
from helpers.visualisation import compare, compare_contains, passify_matlab
def topological_loop_grouping(coil_parts: List[CoilPart], input_args, m_c_parts=None) -> List[CoilPart]:

#def topological_loop_grouping(coil_parts: List[CoilPart], input_args) -> List[CoilPart]:
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
        # DEBUG
        if m_c_parts is not None:
            m_c_part = m_c_parts[part_ind]
            m_debug = m_c_part.topological_loop_grouping

        coil_part = coil_parts[part_ind]
        num_total_loops = len(coil_part.contour_lines)
        loop_in_loop_mat = np.zeros((num_total_loops, num_total_loops), dtype=int)

        # DEBUG
        if m_c_parts is not None:
            try:
                loop_in_loop_mat = np.load(f'debug/{input_args.project_name}_{part_ind}_topological_loop_grouping-loop_in_loop_mat.npy', allow_pickle=True)
            except FileNotFoundError:
                # Check for all loop enclosures of other loops
                for loop_to_test in range(num_total_loops):
                    for loop_num in range(num_total_loops):
                        if loop_to_test != loop_num:
                            loop_in_loop_mat[loop_to_test, loop_num] = check_mutual_loop_inclusion(
                                coil_part.contour_lines[loop_num].uv,
                                coil_part.contour_lines[loop_to_test].uv
                            )
        else:
            # Check for all loop enclosures of other loops
            for loop_to_test in range(num_total_loops):
                for loop_num in range(num_total_loops):
                    if loop_to_test != loop_num:
                        loop_in_loop_mat[loop_to_test, loop_num] = check_mutual_loop_inclusion(
                            coil_part.contour_lines[loop_num].uv,
                            coil_part.contour_lines[loop_to_test].uv
                        )

        # DEBUG
        if m_c_parts is not None:
            assert compare(loop_in_loop_mat, m_debug.loop_in_loop_mat1)
            np.save(f'debug/{input_args.project_name}_{part_ind}_topological_loop_grouping-loop_in_loop_mat.npy', loop_in_loop_mat)

        # Clear the diagonals from the loop_in_loop_mat
        mask = np.ones((num_total_loops, num_total_loops), dtype=int) - np.eye(num_total_loops, dtype=int)
        loop_in_loop_mat = loop_in_loop_mat * mask

        # DEBUG
        if m_c_parts is not None:
            assert compare(loop_in_loop_mat, m_debug.loop_in_loop_mat2)

        lower_loops = [list(np.nonzero(loop_in_loop_mat[:, loop_to_test])[0])
                       for loop_to_test in range(num_total_loops)]
        higher_loops = [list(np.nonzero(loop_in_loop_mat[loop_to_test, :])[0])
                        for loop_to_test in range(num_total_loops)]
        is_global_top_loop = [len(higher_loops[loop_num]) == num_total_loops - 1 for loop_num in range(num_total_loops)]

        # DEBUG
        if m_c_parts is not None:
            assert compare(lower_loops, m_debug.lower_loops1-1)
            assert compare(higher_loops, m_debug.higher_loops1-1)

        # Assign '0' to loops that have no lower loops
        empty_cells = [ind for ind, lower_loop in enumerate(lower_loops) if not lower_loop]

        for empty_cell in empty_cells:
            lower_loops[empty_cell] = [0]

        # Convert the list of lower loops into parallel levels
        group_levels = [None] * num_total_loops
        for loop_to_test in range(num_total_loops):
            group_levels[loop_to_test] = [ind for ind, lower_loop in enumerate(lower_loops) if 
                                          lower_loop == lower_loops[loop_to_test]]

        # DEBUG
        if m_c_parts is not None:
            assert compare(group_levels, m_debug.group_levels1-1)

        # Delete the repetition in the parallel levels and the singular levels
        multi_element_indices = [index for index, cell in enumerate(group_levels) if len(cell) != 1]
        new_group_levels = []
        for i in multi_element_indices:
            if not group_levels[i] in new_group_levels:
                new_group_levels.append(group_levels[i])

        # TODO: MATLAB has a way of converting to string which orders the groups by the first character,
        #       so 55 appears before 6. This has a weird effect on all subsequent results!?

        # Remove levels with only one member unless it is the singular top level
        top_level_indices = [index for index, cell in enumerate(
            group_levels) if len(cell) == 1 and is_global_top_loop[cell[0]]]
        for i in top_level_indices:
            if not group_levels[i] in new_group_levels:
                new_group_levels.append(group_levels[i])

        group_levels = new_group_levels


        # DEBUG
        if m_c_parts is not None:
            # MATLAB array/scalar, again!!
            if len(group_levels) == 1:
                # The MATLAB example will be a single leveled array!
                m_group_levels = m_debug.group_levels3.reshape(1,-1) - 1
            else:
                m_group_levels = m_debug.group_levels3-1
            assert compare_contains(group_levels, m_group_levels) # Ordering is different
            log.warning(" Using MATLAB group levels in %s, line 114.", __file__)
            ### group_levels = passify_matlab(m_debug.group_levels3-1, magic=2)
            m_group_levels = []
            for m_group_level in m_debug.group_levels3:
                m_group_levels.append((m_group_level-1).tolist())
            group_levels = m_group_levels


        # Creating the loop groups (containing still the loops of the inner groups)
        overlapping_loop_groups_num = np.asarray([item for group_level in group_levels for item in group_level])
        # Horizontally concatenate the cell array elements
        overlapping_loop_groups = np.asarray([item for group_level in group_levels for item in group_level])

        # DEBUG
        if m_c_parts is not None:
            #assert compare_contains(overlapping_loop_groups_num, m_debug.overlapping_loop_groups_num-1)
            #assert compare_contains(overlapping_loop_groups, np.array(m_debug.overlapping_loop_groups1-1, dtype=int))
            assert compare(overlapping_loop_groups_num, m_debug.overlapping_loop_groups_num-1)
            assert compare(overlapping_loop_groups, np.array(m_debug.overlapping_loop_groups1-1, dtype=int))

        # Horizontally concatenate the cell array elements and convert to a list of individual elements (cells)
        yyy = []
        for index1, group in enumerate(overlapping_loop_groups):
            xxx = [group]
            xxx += higher_loops[overlapping_loop_groups_num[index1]]
            yyy.append(xxx)
        overlapping_loop_groups = yyy

        # DEBUG
        if m_c_parts is not None:
            m_passified = np.empty((len(m_debug.overlapping_loop_groups2)), dtype=object)
            for i in range(len(m_debug.overlapping_loop_groups2)):
                m_passified[i] = passify_matlab(m_debug.overlapping_loop_groups2[i]-1)
            # assert compare_contains(overlapping_loop_groups, m_passified)
            assert compare(overlapping_loop_groups, m_passified)

        # Build the group topology by checking the loop content of a certain group
        # to see if it is a superset of the loop content of another group
        group_in_group_mat = np.zeros((len(overlapping_loop_groups), len(overlapping_loop_groups)), dtype=int)

        for group_index_1 in range(len(overlapping_loop_groups)):
            for group_index_2 in range(len(overlapping_loop_groups)):
                group_in_group_mat[group_index_1, group_index_2] = all(
                    np.isin(overlapping_loop_groups[group_index_1], overlapping_loop_groups[group_index_2]))

            group_in_group_mat[group_index_1, group_index_1] = 0

        # DEBUG
        if m_c_parts is not None:
            assert compare(group_in_group_mat, m_debug.group_in_group_mat)



        group_is_subgroup_of = [list(np.nonzero(group_in_group_mat[group_index, :])[0])
                                for group_index in range(len(overlapping_loop_groups))]

        # DEBUG
        if m_c_parts is not None:
            #assert compare_contains(group_is_subgroup_of, m_debug.group_is_subgroup_of-1)
            assert compare(group_is_subgroup_of, m_debug.group_is_subgroup_of-1)

        group_contains_following_group = [list(np.nonzero(group_in_group_mat[:, group_index])[0])
                                          for group_index in range(len(overlapping_loop_groups))]

        # DEBUG
        if m_c_parts is not None:
            # assert compare_contains(group_contains_following_group, m_debug.group_contains_following_group-1)
            assert compare(group_contains_following_group, m_debug.group_contains_following_group-1)

        # Remove loops from group if they also belong to a respective subgroup
        loop_groups = [group.copy() for group in overlapping_loop_groups]
        for index, loop_group in enumerate(overlapping_loop_groups):
            loop_groups[index] = np.array(loop_group, copy=True)

        # DEBUG
        if m_c_parts is not None:
            # assert compare_contains(group_contains_following_group, m_debug.group_contains_following_group-1)
            m_passified = np.empty((len(m_debug.loop_groups1)), dtype=object)
            for i in range(len(m_debug.loop_groups1)):
                m_passified[i] = passify_matlab(m_debug.loop_groups1[i]-1)

            assert len(loop_groups) == len(m_passified)
            assert compare(loop_groups, m_passified)

        for iiii in range(len(overlapping_loop_groups)):
            loops_to_remove = [loop for subgroup_index in group_contains_following_group[iiii]
                               for loop in overlapping_loop_groups[subgroup_index]]
            diff = set(loop_groups[iiii]) - set(loops_to_remove)
            # Order the loop_groups
            loop_groups[iiii] = np.array(sorted(list(diff)))

        # DEBUG
        if m_c_parts is not None:
            # assert compare_contains(group_contains_following_group, m_debug.group_contains_following_group-1)
            m_passified = np.empty((len(m_debug.loop_groups2)), dtype=object)
            for i in range(len(m_debug.loop_groups2)):
                m_passified[i] = passify_matlab(m_debug.loop_groups2[i]-1)

            assert len(loop_groups) == len(m_passified)
            assert compare(loop_groups, m_passified)

        # Order the groups based on the number of loops
        # Generating the list of group lengths
        # Sorting the indices based on the lengths in descending order
        #sort_ind = sorted(range(len(loop_groups)), key=lambda x: len(loop_groups[x]), reverse=True)
        group_lengths = [len(loop_groups[x]) for x in range(len(loop_groups))]
        sort_ind = sorted(range(len(loop_groups)), key=lambda x: group_lengths[x], reverse=True)
        #sort_ind = sorted(range(len(loop_groups)), key=lambda x: group_lengths[x])

        coil_part.loop_groups = np.empty((len(loop_groups)), dtype=object)
        for i in range(len(loop_groups)):
            coil_part.loop_groups[i] = loop_groups[sort_ind[i]]

        loop_groups = coil_part.loop_groups # Use the np.array from here on.

        # DEBUG
        if m_c_parts is not None:
            # Final sorted coil_parts(part_ind).loop_groups
            # assert compare_contains(group_contains_following_group, m_debug.group_contains_following_group-1)
            m_passified = np.empty((len(m_c_part.loop_groups)), dtype=object)
            for i in range(len(m_debug.loop_groups2)):
                m_passified[i] = passify_matlab(m_c_part.loop_groups[i]-1)

            assert len(loop_groups) == len(m_passified)
            assert compare(np.array(loop_groups, dtype=object), m_passified)


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


        # DEBUG
        if m_c_parts is not None:
            # MATLAB array/scalar, again!!
            if len(coil_part.group_levels) == 1:
                # The MATLAB example will be a single leveled array!
                m_group_levels = np.empty((1), dtype=object)
                m_group_levels[0] = m_debug.renamed_group_levels - 1
            else:
                m_group_levels = m_debug.renamed_group_levels-1

            assert len(coil_part.group_levels) == len(m_group_levels)
            assert compare(coil_part.group_levels, m_group_levels) # Pass


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

        # TODO: Not calculated correctly!
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
            #  content = cellfun(@(x) numel(higher_loops{x}), num2cell(coil_parts(part_ind).loop_groups{iiii}));
            #  [~, sort_ind_loops] = sort(content, 'descend');
            content = [len(higher_loops[x]) for x in loop_group]
            sort_ind_loops = sorted(range(len(content)), key=lambda x: content[x], reverse=True)

            # DEBUG
            if m_c_parts is not None:
                m_loop_vars = m_debug.loop_vars10[iiii]
                m_content = m_loop_vars.loop_debug.content
                m_sort_ind_loops1 = m_loop_vars.loop_debug.sort_ind_loops-1

            # DEBUG
            if m_c_parts is not None:
                assert np.allclose(sort_ind_loops, m_sort_ind_loops1)

            this_group = TopoGroup()    # Create Topopological group container
            this_group.loops = []       # Assign loops member
            for jjjj in sort_ind_loops:
                loop_group_index = coil_part.loop_groups[iiii][jjjj]
                this_contour = coil_part.contour_lines[loop_group_index]
                this_group.loops.append(this_contour)

            coil_part.groups[iiii] = this_group

    return coil_parts
