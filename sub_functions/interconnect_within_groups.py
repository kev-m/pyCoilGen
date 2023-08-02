import numpy as np

from typing import List

# Logging
import logging

# Local imports
from sub_functions.data_structures import CoilPart, Shape2D, Shape3D, TopoGroup
from sub_functions.find_group_cut_position import find_group_cut_position
from sub_functions.open_loop_with_3d_sphere import open_loop_with_3d_sphere

log = logging.getLogger(__name__)

# Debugging
from helpers.visualisation import get_linenumber

def interconnect_within_groups(coil_parts: List[CoilPart], input_args):
    """
    Interconnects the loops within each group for each coil part.

    Parameters:
        coil_parts (List[CoilPart]): List of CoilPart structures containing coil_mesh and other data.
        input_args (Any): Input arguments (Structure or any other type).

    Returns:
        List[CoilPart]: The updated list of CoilPart structures with interconnected loops and cutshapes.
    """

    cut_plane_definition = 'nearest'  # nearest or B0
    cut_height_ratio = 1 / 2  # the ratio of height to width of the individual cut_shapes

    for part_ind in range(len(coil_parts)):
        coil_part = coil_parts[part_ind]
        num_part_groups = len(coil_part.groups)

        # Initialize fields in each CoilPart
        coil_part.connected_group = []

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
        # M: avg_z_value	[-0.3618,0.3589,-0.3676,0.3567]	1x4	double [y]
        avg_z_value = np.zeros(len(coil_part.loop_groups))
        for group_ind in range(num_part_groups):
            part_group = coil_part.groups[group_ind]
            all_points = part_group.loops[0].v.copy()
            for loop_ind in range(1, len(part_group.loops)):
                all_points = np.hstack((all_points, part_group.loops[loop_ind].v))
            # Why multiply all_points by [0.05, 0, 1] ?
            avg_z_value[group_ind] = np.sum(all_points * np.array([[0.05], [0], [1]])) / all_points.shape[1]

        # M: new_group_inds	[3,1,4,2]	1x4	double [y]
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
                part_connected_group.uv = part_group.loops.uv
                part_connected_group.v = part_group.loops.v
                part_connected_group.spiral_in.uv = part_group.loops.uv
                part_connected_group.spiral_in.v = part_group.loops.v
                part_connected_group.spiral_out.uv = part_group.loops.uv
                part_connected_group.spiral_out.v = part_group.loops.v
            else:
                part_group.opened_loop = [None] * len(part_group.loops)
                part_group.cutshape = [None] * len(part_group.loops)

                # Generate the cutshapes for all the loops within the group
                log.debug("--- here ---: %s, line %d", __file__, get_linenumber())
                # M: cut_position(1).group(1).cut_point
                # v	[0.0011,0.0011;0.4999,0.4999;-0.0138,-0.7357]	3x2	double
                # uv	[-1.2860,-2.0066;-0.0017,0.0533]	2x2	double
                # segment_ind	[19,46]	1x2	double

                # M:  cut_position(group_ind).group.cut_point
                # ans = 
                #   struct with fields:
                #               v: [3x2 double]
                #              uv: [2x2 double]
                #     segment_ind: [19 46] [y]
                # ans = 
                #   struct with fields:
                #               v: [3x2 double]
                #              uv: [2x2 double]
                #     segment_ind: [18 44] [y]
                # ans = 
                #   struct with fields:
                #               v: [3x2 double]
                #              uv: [2x2 double]
                #     segment_ind: [16 44] [y]
                # ans = 
                #   struct with fields:
                #               v: [3x2 double]
                #              uv: [2x2 double]
                #     segment_ind: [15 39] [y]
                # ans = 
                #   struct with fields:
                #               v: [3x2 double]
                #              uv: [2x2 double]
                #     segment_ind: [13 35]
                # ans = 
                #   struct with fields:
                #               v: [3x2 double]
                #              uv: [2x2 double]
                #     segment_ind: [13 31]
                # ans = 
                #   struct with fields:
                #               v: [3x2 double]
                #              uv: [2x2 double]
                #     segment_ind: [12 27]
                # ans = 
                #   struct with fields:
                #               v: [3x2 double]
                #              uv: [2x2 double]
                #     segment_ind: [9 21]
                # ans = 
                #   struct with fields:
                #               v: [3x2 double]
                #              uv: [2x2 double]
                #     segment_ind: [8 16]
                # ans = 
                #   struct with fields:
                #               v: [3x2 double]
                #              uv: [2x2 double]
                #     segment_ind: [3 5]
                cut_positions = find_group_cut_position(
                    part_group,
                    coil_part.group_centers.v[:, group_ind],
                    coil_part.coil_mesh,
                    input_args.b_0_direction,
                    cut_plane_definition
                )
                # NOTE: cut_positions[x].cut_point.v is (2,3) (i.e. Python)
                # Choose either low or high cutshape
                # M: force_cut_selection =
                    #   1x4 cell array
                    #     {'high'}    {'high'}    {'high'}    {'high'}
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
                    # M: part_group.loops[loop_ind].uv
                    # -0.1389	-0.0605	-0.0722	-0.0605	-0.0694	-0.1058	-0.1135	-0.1175	-0.3352	-0.538	-0.6413	-0.6744	-0.9057	-1.0831	-1.1099	-1.2258	-1.24	-1.2444	-1.2863	-1.2474	-1.244	-1.1276	-1.1171	-1.0943	-0.9137	-0.674	-0.6478	-0.3802	-0.3382	-0.0879	-0.0611	-0.0571	-0.0503	-0.0627	-0.1398	-0.1361	-0.4727	-0.5672	-0.9804	-1.0389	-1.4183	-1.7057	-1.7505	-1.7704	-1.9512	-1.9569	-2.0071	-1.9252	-1.9171	-1.7139	-1.6922	-1.6469	-1.3529	-0.9833	-0.9265	-0.7483	-0.4453	-0.1389
                    # 1.8348	1.8087	1.6925	1.6612	1.5437	1.4898	1.3893	1.2891	1.2625	1.181	1.1143	1.0805	0.9077	0.6817	0.6423	0.3701	0.3334	0.2993	0.001	-0.3011	-0.3326	-0.6151	-0.6445	-0.6793	-0.9137	-1.0956	-1.1231	-1.2319	-1.2687	-1.4108	-1.4399	-1.5635	-1.6847	-1.7117	-1.7558	-1.8525	-1.9117	-1.9022	-1.7398	-1.7057	-1.4183	-1.0389	-0.986	-0.9403	-0.4802	-0.4327	0.0577	0.5392	0.5859	1.022	1.0654	1.114	1.4627	1.7204	1.7504	1.7877	1.8985	1.8348
                    # M: part_group.loops[loop_ind].v
                    # -0.4935	-0.4962	-0.4961	-0.4963	-0.4963	-0.4944	-0.4943	-0.4944	-0.483	-0.4497	-0.433	-0.4232	-0.3536	-0.2634	-0.25	-0.1422	-0.1294	-0.1175	0	0.1185	0.1294	0.24	0.25	0.2616	0.3536	0.4253	0.433	0.4763	0.483	0.4956	0.4971	0.4969	0.4968	0.4965	0.494	0.4939	0.483	0.4734	0.433	0.4222	0.3536	0.264	0.25	0.2381	0.1294	0.117	0	-0.117	-0.1294	-0.2381	-0.25	-0.264	-0.3536	-0.4222	-0.433	-0.4517	-0.483	-0.4935
                    # 0.0492	0.0292	0.0298	0.0279	0.028	0.0428	0.0433	0.0428	0.1294	0.2097	0.25	0.2628	0.3536	0.4227	0.433	0.4777	0.483	0.4845	0.5	0.4844	0.483	0.4372	0.433	0.4241	0.3536	0.2601	0.25	0.1454	0.1294	0.0336	0.0223	0.0236	0.0245	0.0263	0.0453	0.0467	0.1294	0.1524	0.25	0.264	0.3536	0.4222	0.433	0.438	0.483	0.4846	0.5	0.4846	0.483	0.438	0.433	0.4222	0.3536	0.264	0.25	0.2049	0.1294	0.0492
                    # -0.6	-0.5662	-0.45	-0.4176	-0.3	-0.2504	-0.15	-0.0496	-0.0499	-0.0501	-0.0261	-0.0185	-0.0193	-0.0194	-0.0159	-0.0159	-0.0143	-0.0138	-0.0138	-0.0127	-0.0131	-0.0124	-0.0138	-0.0168	-0.0157	-0.0146	-0.0206	-0.0199	-0.0385	-0.15	-0.1758	-0.3	-0.4217	-0.45	-0.5025	-0.6	-0.6949	-0.7213	-0.7213	-0.7297	-0.7297	-0.7297	-0.7334	-0.7351	-0.7351	-0.7357	-0.7357	-0.7357	-0.7351	-0.7351	-0.7334	-0.7297	-0.7297	-0.7297	-0.7211	-0.6939	-0.6939	-0.6
                    # M: potential	-297.9376	1x1	double
                    # M: current_orientation	-1	1x1	double

                    # Open the loop
                    # M: coil_parts(part_ind).groups(group_ind).cutshape(loop_ind).uv
                    # ans =
                    #   Columns 1 through 19
                    #    -1.5528   -1.5395   -1.5265   -1.5139   -1.5019   -1.4907   -1.4804   -1.4713   -1.4635   -1.4571   -1.4523   -1.4490   -1.4473   -1.4473   -1.4490   -1.4523   -1.4571   -1.4635   -1.4713
                    #     0.2284    0.2276    0.2251    0.2210    0.2154    0.2082    0.1998    0.1901    0.1794    0.1677    0.1554    0.1426    0.1294    0.1161    0.1030    0.0901    0.0778    0.0661    0.0554
                    #   Columns 20 through 38
                    #    -1.4804   -1.4907   -1.5019   -1.5139   -1.5265   -1.5395   -1.5528   -1.5660   -1.5790   -1.5917   -1.6037   -1.6149   -1.6251   -1.6342   -1.6420   -1.6484   -1.6533   -1.6566   -1.6582
                    #     0.0457    0.0373    0.0302    0.0245    0.0204    0.0179    0.0171    0.0179    0.0204    0.0245    0.0302    0.0373    0.0457    0.0554    0.0661    0.0778    0.0901    0.1030    0.1161
                    #   Columns 39 through 51
                    #    -1.6582   -1.6566   -1.6533   -1.6484   -1.6420   -1.6342   -1.6251   -1.6149   -1.6037   -1.5917   -1.5790   -1.5660   -1.5528
                    #     0.1294    0.1426    0.1554    0.1677    0.1794    0.1901    0.1998    0.2082    0.2154    0.2210    0.2251    0.2276    0.2284

                    # M: opened_loop.uv
                    # ans =
                    #    -1.5650   -1.5633   -1.5477   -1.6137   -1.6148   -1.5405
                    #     0.0178    0.0008   -0.1931    0.0182    0.0216    0.2277
                    # M: opened_loop.v
                    # ans =
                    #     0.0000    0.0059    0.0662    0.0011    0.0000   -0.0662
                    #     0.5000    0.4992    0.4913    0.4999    0.5000    0.4913
                    #    -0.2927   -0.2931   -0.3000   -0.3419   -0.3426   -0.3000
                    # M: sphere_diameter	0.1	1x1	double
                    opened_loop, uv, _ = open_loop_with_3d_sphere(
                        part_group.loops[loop_ind], # 3x58
                        cut_position_used,
                        input_args.interconnection_cut_width
                    )
                    # coil_parts(part_ind).groups(group_ind).loops(loop_ind).open_loop_with_3d_sphere = curve_points_in;

                    part_group.cutshape[loop_ind] = Shape2D(uv=uv)
                    part_group.opened_loop[loop_ind] = Shape3D(uv=opened_loop.uv, v=opened_loop.v)

                # Build the interconnected group by adding the opened loops
                part_connected_group.spiral_in = Shape3D()
                part_connected_group.spiral_out = Shape3D()
                for loop_ind in range(0, len(part_group.loops)):
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
                for loop_ind in range(len(part_group.loops)-1, -1, -1):
                    loop_item = part_group.opened_loop[loop_ind]
                    part_connected_group.return_path.add_uv(np.mean(loop_item.uv[:, [0, -1]], axis=1).reshape(-1, 1))
                    part_connected_group.return_path.add_v(np.mean(loop_item.v[:, [0, -1]], axis=1).reshape(-1, 1))

                part_connected_group.add_uv(part_connected_group.return_path.uv)
                part_connected_group.add_v(part_connected_group.return_path.v)

                # Close the connected group
                part_connected_group.add_uv(part_connected_group.uv[:, [0]])
                part_connected_group.add_v(part_connected_group.v[:, [0]])

    return coil_parts
