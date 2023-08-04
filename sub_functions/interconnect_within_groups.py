import numpy as np

from typing import List

# Logging
import logging

# Local imports
from sub_functions.data_structures import CoilPart, Shape2D, Shape3D, TopoGroup
from sub_functions.find_group_cut_position import find_group_cut_position
from sub_functions.open_loop_with_3d_sphere import open_loop_with_3d_sphere

log = logging.getLogger(__name__)

def interconnect_within_groups(coil_parts: List[CoilPart], input_args, m_c_part = None):
    """
    Interconnects the loops within each group for each coil part.

    Parameters:
        coil_parts (List[CoilPart]): List of CoilPart structures containing coil_mesh and other data.
        input_args (Any): Input arguments (Structure or any other type).

    Returns:
        List[CoilPart]: The updated list of CoilPart structures with interconnected loops and cutshapes.
    """

    cut_plane_definition = 'nearest'  # nearest or B0
    # cut_height_ratio = 1 / 2  # the ratio of height to width of the individual cut_shapes

    for part_ind in range(len(coil_parts)):
        coil_part = coil_parts[part_ind]
        num_part_groups = len(coil_part.groups)

        # Initialize fields in each CoilPart
        # Get references to planar and curved meshes
        coil_part.connected_group = [None] * num_part_groups

        # Take the cut selection if it is given in the input
        switch = len(input_args.force_cut_selection)
        if switch == 0:
            force_cut_selection = ['none'] * num_part_groups
        elif switch == 1:
            if input_args.force_cut_selection[0] == 'high':
                force_cut_selection = ['high'] * num_part_groups
            else:
                force_cut_selection = ['none'] * num_part_groups
        else:
            force_cut_selection = input_args.force_cut_selection

        # Generate cutshapes, open, and interconnect the loops within each group
        # Sort the force cut selection within their level according to their average z-position
        avg_z_value = np.zeros(len(coil_part.loop_groups))
        for group_ind in range(num_part_groups):
            part_group = coil_part.groups[group_ind]
            all_points = part_group.loops[0].v.copy()
            for loop_ind in range(0, len(part_group.loops)):
                all_points = np.hstack((all_points, part_group.loops[loop_ind].v))
            # Why multiply all_points by [0.05, 0, 1] ?
            avg_z_value[group_ind] = np.sum(all_points * np.array([[0.05], [0], [1]])) / all_points.shape[1]

        new_group_inds = np.argsort(avg_z_value)

        for group_ind in range(num_part_groups):
            part_group = coil_part.groups[group_ind]
            coil_part.connected_group[group_ind] = TopoGroup()
            part_connected_group = coil_part.connected_group[group_ind]

            if len(part_group.loops) == 1:
                # If the group consists of only one loop, it is not necessary to open it
                part_group.opened_loop = part_group.loops.uv
                part_group.cutshape.uv = np.array([np.nan, np.nan])
                part_connected_group.return_path.uv = np.array([np.nan, np.nan])
                part_connected_group.uv = part_group.loops.uv.copy()
                part_connected_group.v = part_group.loops.v.copy()
                part_connected_group.spiral_in.uv = part_group.loops.uv
                part_connected_group.spiral_in.v = part_group.loops.v
                part_connected_group.spiral_out.uv = part_group.loops.uv
                part_connected_group.spiral_out.v = part_group.loops.v
            else:
                part_group.opened_loop = [None] * len(part_group.loops)
                part_group.cutshape = [None] * len(part_group.loops)
                cut_positions = find_group_cut_position(
                    part_group,
                    coil_part.group_centers.v[:, group_ind],
                    coil_part.coil_mesh,
                    input_args.b_0_direction,
                    cut_plane_definition
                )
                # NOTE: cut_positions[x].cut_point.v is (2,3) (i.e. Python)
                # Choose either low or high cutshape
                force_cut_selection = [force_cut_selection[ind] for ind in new_group_inds]

                for loop_ind in range(len(part_group.loops)):
                    if force_cut_selection[group_ind] == 'high':
                        cut_position_used = cut_positions[loop_ind].high_cut.v
                    elif force_cut_selection[group_ind] == 'low':
                        cut_position_used = cut_positions[loop_ind].low_cut.v
                    else:
                        cut_position_used = cut_positions[loop_ind].high_cut.add_v

                    # NOTE: high_cut/low_cut.v are (n,3) whereas part_group.loops[] etc are (3,n)
                    # Temporary hack until all v and uv are changed from MATLAB (2,m) to Python (m,2)
                    cut_position_used = [[cut_position_used[0]], [cut_position_used[1]], [cut_position_used[2]]]
                    opened_loop, uv, _ = open_loop_with_3d_sphere(
                        part_group.loops[loop_ind],
                        cut_position_used,
                        input_args.interconnection_cut_width
                    )
                    part_group.cutshape[loop_ind] = Shape2D(uv=uv)
                    part_group.opened_loop[loop_ind] = Shape3D(uv=opened_loop.uv, v=opened_loop.v)

                # Build the interconnected group by adding the opened loops
                part_connected_group.spiral_in = Shape3D()
                part_connected_group.spiral_out = Shape3D()
                # for loop_ind = 1:numel(coil_parts(part_ind).groups(group_ind).loops)
                for loop_ind in range(0, len(part_group.loops)-1): # [Loop 1]
                    loop_item = part_group.opened_loop[loop_ind]

                    part_connected_group.add_uv(loop_item.uv)
                    part_connected_group.add_v(loop_item.v)

                    part_connected_group.spiral_in.add_uv(loop_item.uv)
                    part_connected_group.spiral_in.add_v(loop_item.v)
                    
                    other_item = part_group.opened_loop[len(part_group.loops) - loop_ind - 1]
                    part_connected_group.spiral_out.add_uv(other_item.uv)
                    part_connected_group.spiral_out.add_v(other_item.v)

                # Add the return path
                part_connected_group.return_path = Shape3D()
                for loop_ind in range(len(part_group.loops)-1, -1, -1): # Loop2
                    loop_item = part_group.opened_loop[loop_ind]
                    part_connected_group.return_path.add_uv(np.mean(loop_item.uv[:, [0, -1]], axis=1).reshape(-1, 1))
                    part_connected_group.return_path.add_v(np.mean(loop_item.v[:, [0, -1]], axis=1).reshape(-1, 1))

                part_connected_group.add_uv(part_connected_group.return_path.uv)
                part_connected_group.add_v(part_connected_group.return_path.v)
                # Close the connected group
                part_connected_group.add_uv(part_connected_group.uv[:, [0]])
                part_connected_group.add_v(part_connected_group.v[:, [0]])

    return coil_parts
