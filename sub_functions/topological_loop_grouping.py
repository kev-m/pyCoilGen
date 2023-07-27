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
        # M: lower_loops =
        #   1x40 cell array
        #   Columns 1 through 12
        #     {9x1 double}    {9x1 double}    {8x1 double}    {8x1 double}    {7x1 double}    {7x1 double}    {6x1 double}    {6x1 double}    {5x1 double}    {5x1 double}    {4x1 double}    {4x1 double}
        #   Columns 13 through 27
        #     {3x1 double}    {3x1 double}    {2x1 double}    {2x1 double}    {[19]}    {[20]}    {[0]}    {[0]}    {[0]}    {[0]}    {[21]}    {[22]}    {2x1 double}    {2x1 double}    {3x1 double}
        #   Columns 28 through 39
        #     {3x1 double}    {4x1 double}    {4x1 double}    {5x1 double}    {5x1 double}    {6x1 double}    {6x1 double}    {7x1 double}    {7x1 double}    {8x1 double}    {8x1 double}    {9x1 double}
        #   Column 40
        #     {9x1 double}
        lower_loops = [list(np.nonzero(loop_in_loop_mat[:, loop_to_test])[0])
                       for loop_to_test in range(num_total_loops)]
        # [3;5;7;9;11;13;15;18;20]	[4;6;8;10;12;14;16;17;19]	[5;7;9;11;13;15;18;20]	[6;8;10;12;14;16;17;19]	[7;9;11;13;15;18;20]	[8;10;12;14;16;17;19]	[9;11;13;15;18;20]	[10;12;14;16;17;19]	[11;13;15;18;20]	[12;14;16;17;19]	[13;15;18;20]	undefined	undefined	undefined	undefined	undefined	undefined	undefined	undefined	undefined	undefined	undefined	undefined	undefined	undefined	undefined	undefined	undefined	undefined	undefined	undefined	undefined	undefined	undefined	undefined	undefined	undefined	undefined	undefined	undefined

        # M: higher_loops =
        #   1x40 cell array
        #   Columns 1 through 13
        #     {1x0 double}    {1x0 double}    {[1]}    {[2]}    {[1 3]}    {[2 4]}    {[1 3 5]}    {[2 4 6]}    {[1 3 5 7]}    {[2 4 6 8]}    {[1 3 5 7 9]}    {[2 4 6 8 10]}    {[1 3 5 7 9 11]}
        #   Columns 14 through 21
        #     {[2 4 6 8 10 12]}    {[1 3 5 7 9 11 13]}    {[2 4 6 8 10 12 14]}    {[2 4 6 8 10 ... ]}    {[1 3 5 7 9 11 ... ]}    {[2 4 6 8 10 ... ]}    {[1 3 5 7 9 11 ... ]}    {[23 25 27 29 ... ]}
        #   Columns 22 through 29
        #     {[24 26 28 30 ... ]}    {[25 27 29 31 ... ]}    {[26 28 30 32 ... ]}    {[27 29 31 33 ... ]}    {[28 30 32 34 ... ]}    {[29 31 33 35 37 39]}    {[30 32 34 36 38 40]}    {[31 33 35 37 39]}
        #   Columns 30 through 40
        #     {[32 34 36 38 40]}    {[33 35 37 39]}    {[34 36 38 40]}    {[35 37 39]}    {[36 38 40]}    {[37 39]}    {[38 40]}    {[39]}    {[40]}    {1x0 double}    {1x0 double}        
        higher_loops = [list(np.nonzero(loop_in_loop_mat[loop_to_test, :])[0])
                        for loop_to_test in range(num_total_loops)]
        # []	[]	1	2	[1,3]	[2,4]	[1,3,5]	[2,4,6]	[1,3,5,7]	[2,4,6,8]	[1,3,5,7,9]	undefined	undefined	undefined	undefined	undefined	undefined	undefined	undefined	undefined	undefined	undefined	undefined	undefined	undefined	undefined	undefined	undefined	undefined	undefined	undefined	undefined	undefined	undefined	undefined	undefined	undefined	undefined	undefined	undefined        

        # Check the possibility of the top level being composed out of a single group
        # M: is_global_top_loop
        #   1x40 logical array
        #    0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
        is_global_top_loop = [len(higher_loops[loop_num]) == num_total_loops - 1 for loop_num in range(num_total_loops)]

        # Assign '0' to loops that have no lower loops
        empty_cells = [ind for ind, lower_loop in enumerate(lower_loops) if not lower_loop]

        for empty_cell in empty_cells:
            lower_loops[empty_cell] = [0]

        # Convert the list of lower loops into parallel levels
        group_levels = [None] * num_total_loops

        # M: group_levels [y]
        #   1x40 cell array
        #   Columns 1 through 19
        #     {[1]}    {[2]}    {[3]}    {[4]}    {[5]}    {[6]}    {[7]}    {[8]}    {[9]}    {[10]}    {[11]}    {[12]}    {[13]}    {[14]}    {[15]}    {[16]}    {[17]}    {[18]}    {[19 20 21 22]}
        #   Columns 20 through 35
        #     {[19 20 21 22]}    {[19 20 21 22]}    {[19 20 21 22]}    {[23]}    {[24]}    {[25]}    {[26]}    {[27]}    {[28]}    {[29]}    {[30]}    {[31]}    {[32]}    {[33]}    {[34]}    {[35]}
        #   Columns 36 through 40
        #     {[36]}    {[37]}    {[38]}    {[39]}    {[40]}
        # 42: coil_parts(part_ind).group_levels{aaaa}=find(cellfun(@(x) isequal(lower_loops{aaaa},x),lower_loops));
        for loop_to_test in range(num_total_loops):
            group_levels[loop_to_test] = [ind for ind, lower_loop in enumerate(
                lower_loops) if lower_loop == lower_loops[loop_to_test]]

        # Delete the repetition in the parallel levels and the singular levels
        # M: group_levels
        #   1x37 cell array
        #   Columns 1 through 18
        #     {[1]}    {[10]}    {[11]}    {[12]}    {[13]}    {[14]}    {[15]}    {[16]}    {[17]}    {[18]}    {[19 20 21 22]}    {[2]}    {[23]}    {[24]}    {[25]}    {[26]}    {[27]}    {[28]}
        #   Columns 19 through 37
        #     {[29]}    {[3]}    {[30]}    {[31]}    {[32]}    {[33]}    {[34]}    {[35]}    {[36]}    {[37]}    {[38]}    {[39]}    {[4]}    {[40]}    {[5]}    {[6]}    {[7]}    {[8]}    {[9]}
        ## % remove levels with only one member except it is the singular top level        
        multi_element_indices = [index for index, cell in enumerate(group_levels) if len(cell) != 1]
        new_group_levels = []
        for i in multi_element_indices:
            if not group_levels[i] in new_group_levels:
                new_group_levels.append(group_levels[i])
        top_level_indices = [index for index, cell in enumerate(group_levels) if len(cell) == 1 and is_global_top_loop[cell[0]]]
        for i in top_level_indices:
            if not group_levels[i] in new_group_levels:
                new_group_levels.append(group_levels[i])

        # M: group_levels
        #   1x1 cell array
        #     {[19 20 21 22]}
        # 49: coil_parts(part_ind).group_levels=coil_parts(part_ind).group_levels(cellfun(@numel,coil_parts(part_ind).group_levels)~=1 | arrayfun(@(x) is_global_top_loop(coil_parts(part_ind).group_levels{x}(1)),1:numel(coil_parts(part_ind).group_levels))==1); % remove levels with only one member except it is the singular top level
        group_levels = new_group_levels

        # Creating the loop groups (containing still the loops of the inner groups)
        # M: overlapping_loop_groups_num [y]
        # 19	20	21	22
        # L91: overlapping_loop_groups_num = horzcat(coil_parts(part_ind).group_levels{:});
        overlapping_loop_groups_num = [loop_num for loop_num in group_levels[0]]
        # M: overlapping_loop_groups [y]
        #   1x4 cell array
        #     {[19]}    {[20]}    {[21]}    {[22]}        
        # Horizontally concatenate the cell array elements
        overlapping_loop_groups = [[item] for item in group_levels[0]]
        # M: overlapping_loop_groups
        #   1x4 cell array
        #     {[19 2 4 6 8 10 12 14 16 17]}    {[20 1 3 5 7 9 11 13 15 18]}    {[21 23 25 27 29 31 33 35 37 39]}    {[22 24 26 28 30 32 34 36 38 40]}
        # Horizontally concatenate the cell array elements and convert to a list of individual elements (cells)
        yyy = []
        for index1, group in enumerate(overlapping_loop_groups):
            xxx = [group[0]]
            xxx += higher_loops[overlapping_loop_groups_num[index1]]
            yyy.append(xxx)
        overlapping_loop_groups = yyy

        # MATLAB cell indexing is confusing!!
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
        loop_groups = coil_part.loop_groups # Use the np.array from here on.

        # Order the groups based on the number of loops
        # M: sort_ind =
        #      1     2     3     4        
        sort_indices = np.argsort([len(group) for group in loop_groups])[::-1]
        # Sort each group level
        for group_index, group_level in enumerate(group_levels):
            group_levels[group_index] = [group_level[sort_idex] for sort_idex in sort_indices]

        # Renumber (=rename) the groups (also in the levels)
        renamed_group_levels = group_levels.copy()
        for iiii in range(len(group_levels)):
            for level_index in range(len(group_levels[iiii])):
                for renamed_index in range(len(loop_groups)):
                    if group_levels[iiii][level_index] in loop_groups[renamed_index]:
                        renamed_group_levels[iiii][level_index] = renamed_index

        # Resort parallel_levels to new group names
        renamed_group_levels = sorted(list(set(tuple(level) for level in renamed_group_levels)))
        group_levels = [list(level) for level in renamed_group_levels]

        assert len(group_levels) == 1 # MATLAB cell is always has 1 element
        coil_part.group_levels = group_levels[0]

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
        # M: coil_parts(part_ind).loop_groups
        # ans =
        #   1x4 cell array
        #     {[2 4 6 8 10 12 14 16 17 19]}    {[1 3 5 7 9 11 13 15 18 20]}    {[21 23 25 27 29 31 33 35 37 39]}    {[22 24 26 28 30 32 34 36 38 40]}
        for iiii, loop_group in enumerate(loop_groups):
            # Sort the loops in each group according to the rank

            # Sort the loop groups based on the number of elements in higher_loops for each group
            unsorted = [len(higher_loops[x]) for x in loop_group]
            # M: sort_ind_loops =
            #     10     9     8     7     6     5     4     3     2     1
            sort_ind_loops = sorted(unsorted, reverse=True)

            this_group = TopoGroup()    # Create Topopological group container
            this_group.loops = []       # Assign loops member
            #kkkk = 0
            for jjjj in sort_ind_loops:
                loop_group_index = coil_part.loop_groups[iiii][jjjj]
                this_contour = coil_part.contour_lines[loop_group_index]
                #this_group.loops(kkkk).number_points = size(this_contour.uv, 2);
                #this_group.loops(kkkk).v = this_contour.v
                #this_group.loops(kkkk).uv = this_contour.uv
                #this_group.loops(kkkk).potential = this_contour.potential
                #this_group.loops(kkkk).current_orientation = this_contour.current_orientation
                this_group.loops.append(this_contour)
                #kkkk += 1


            coil_part.groups[iiii] = this_group

    return coil_parts
