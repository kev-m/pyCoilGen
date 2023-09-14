import numpy as np
from typing import List

from .data_structures import CoilPart, PCBTrack, PCBLayer, GroupLayout, PCBPart, Polygon
from .calc_3d_rotation_matrix_by_vector import calc_3d_rotation_matrix_by_vector
from .find_segment_intersections import find_segment_intersections
from .smooth_track_by_folding import smooth_track_by_folding

import logging

log = logging.getLogger(__name__)


def generate_cylindrical_pcb_print(coil_parts: List[CoilPart], input_args):
    """
    Generate a 2D pattern that can be rolled around a cylinder.

    Initialises the following properties of a CoilPart:
        - pcb_tracks

    Depends on the following properties of the CoilParts:
        - wire_path
        - points_to_shift
        - connected_group

    Depends on the following input_args:
        - surface_is_cylinder_flag
        - make_cylindrical_pcb
        - conductor_cross_section_width
        - cylinder_mesh_parameter_list
        - pcb_interconnection_method
        - pcb_spiral_end_shift_factor

    Updates the following properties of a CoilPart:
        - None

    Args:
        coil_parts (List[CoilPart]): List of CoilPart structures.
        input_args: Input arguments structure.

    Returns:
        coil_parts (List[CoilPart]): The updated CoilParts list.
    """

    if input_args.surface_is_cylinder_flag and input_args.make_cylindrical_pcb:

        pcb_track_width = input_args.conductor_cross_section_width
        cylinder_radius = input_args.cylinder_mesh_parameter_list[1]
        rot_mat = calc_3d_rotation_matrix_by_vector(
            np.array([
                input_args.cylinder_mesh_parameter_list[4],
                input_args.cylinder_mesh_parameter_list[5],
                input_args.cylinder_mesh_parameter_list[6]
            ]),
            input_args.cylinder_mesh_parameter_list[7]
        )

        if input_args.pcb_interconnection_method != 'spiral_in_out':
            for part_ind, coil_part in enumerate(coil_parts):
                coil_part = coil_parts[part_ind]
                # Rotate the wire on the cylinder
                aligned_wire_path = np.dot(rot_mat, coil_part.wire_path.v)

                # Calculate phi angles for each point
                phi_coord = np.arctan2(aligned_wire_path[1, :], aligned_wire_path[0, :])
                layout_2d = np.array([
                    phi_coord * cylinder_radius,
                    aligned_wire_path[2, :]
                ])

                if layout_2d[0, 0] != layout_2d[0, -1] and layout_2d[1, 0] != layout_2d[1, -1]:
                    layout_2d = np.hstack((layout_2d, layout_2d[:, [0]]))

                segment_starts = np.sort(np.concatenate([
                    [1],
                    np.where(np.diff(coil_part.points_to_shift) == 1)[0] + 1,
                    np.where(np.diff(coil_part.points_to_shift) == -1)[0] + 2
                ]))

                num_segments = len(segment_starts)
                pcb_parts = [PCBPart()] * num_segments

                for wire_part_ind in range(num_segments):
                    if wire_part_ind < num_segments - 1:
                        wire_part_inds = slice(segment_starts[wire_part_ind] - 1, segment_starts[wire_part_ind + 1])
                    else:
                        wire_part_inds = slice(segment_starts[wire_part_ind] - 1, None)

                    segment_points = layout_2d[:, wire_part_inds]
                    long_vecs = segment_points[:, 1:] - segment_points[:, :-1]
                    long_vecs = np.hstack((long_vecs, long_vecs[:, [-1]]))
                    long_vecs = long_vecs / np.linalg.norm(long_vecs, axis=0)
                    ortho_vecs = np.array([
                        long_vecs[1, :],
                        long_vecs[0, :] * -1
                    ]) / np.linalg.norm(long_vecs, axis=0)

                    pcb_track_shape = np.hstack((
                        segment_points + ortho_vecs * (pcb_track_width / 2),
                        np.fliplr(segment_points) - np.fliplr(ortho_vecs) * (pcb_track_width / 2)
                    ))

                    pcb_parts[wire_part_ind].track_shape = pcb_track_shape

                coil_part.pcb_tracks = PCBTrack(upper_layer=PCBLayer(group_layouts=[GroupLayout(wire_parts=pcb_parts)]))

        else:  # Generate the pcb form the spiral in/out tracks
            # Initialise 'coil_part.pcb_tracks' elements (PCBTrack)

            for part_ind, coil_part in enumerate(coil_parts):
                coil_part = coil_parts[part_ind]

                upper_layer = PCBLayer()
                lower_layer = PCBLayer()

                upper_layer.group_layouts = []
                lower_layer.group_layouts = []
                for i in range(len(coil_part.connected_group)):
                    lgl = GroupLayout(wire_parts=None)
                    ugl = GroupLayout(wire_parts=None)
                    upper_layer.group_layouts.append(ugl)
                    lower_layer.group_layouts.append(lgl)

                for group_ind, connected_group in enumerate(coil_part.connected_group):
                    track_spiral_in = connected_group.spiral_in.v
                    track_spiral_out = connected_group.spiral_out.v
                    aligned_wire_path_spiral_in = np.dot(rot_mat, track_spiral_in)
                    aligned_wire_path_spiral_out = np.dot(rot_mat, track_spiral_out)

                    phi_coord_spiral_in = np.arctan2(
                        aligned_wire_path_spiral_in[1, :], aligned_wire_path_spiral_in[0, :])
                    phi_coord_spiral_out = np.arctan2(
                        aligned_wire_path_spiral_out[1, :], aligned_wire_path_spiral_out[0, :])
                    layout_2d_spiral_in = np.array([
                        phi_coord_spiral_in,
                        aligned_wire_path_spiral_in[2, :]
                    ])
                    layout_2d_spiral_out = np.array([
                        phi_coord_spiral_out,
                        aligned_wire_path_spiral_out[2, :]
                    ])

                    point_1 = (layout_2d_spiral_in[:, 0] + layout_2d_spiral_out[:, -1]) / 2
                    point_2 = (layout_2d_spiral_in[:, -1] + layout_2d_spiral_out[:, 0]) / 2
                    center_position = (layout_2d_spiral_in[:, 0] + layout_2d_spiral_in[:, -1] +
                                       layout_2d_spiral_out[:, 0] + layout_2d_spiral_out[:, -1]) / 4

                    point_1 = point_1 + (point_1 - center_position) * (input_args.pcb_spiral_end_shift_factor / 100)
                    point_2 = point_2 + (point_2 - center_position) * (input_args.pcb_spiral_end_shift_factor / 100)

                    layout_2d_spiral_in = np.column_stack((
                        point_1,
                        point_1,
                        layout_2d_spiral_in[:, 1:-1],
                        point_2,
                        point_2
                    ))

                    layout_2d_spiral_out = np.column_stack((
                        point_2,
                        point_2,
                        layout_2d_spiral_out[:, 1:-1],
                        point_1,
                        point_1
                    ))

                    for group_layer in ['upper', 'lower']:
                        if group_layer == 'upper':
                            layout_2d = layout_2d_spiral_in
                        else:
                            layout_2d = layout_2d_spiral_out
                        positive_wrap = np.where(np.diff(layout_2d[0, :]) > 1.75 * np.pi)[0]
                        negative_wrap = np.where(np.diff(layout_2d[0, :]) < (1.75 * np.pi) * (-1))[0]
                        positive_wrap = positive_wrap[positive_wrap != layout_2d.shape[1] - 1]
                        negative_wrap = negative_wrap[negative_wrap != layout_2d.shape[1] - 1]

                        full_wrap_spart_inds = np.sort(
                            np.concatenate(([0], [layout_2d.shape[1] - 1], positive_wrap + 1, negative_wrap + 1))
                        )

                        layout_2d[:, positive_wrap + 1] -= np.vstack((
                            np.ones(len(positive_wrap)) * 2 * np.pi,
                            np.zeros(len(positive_wrap))
                        ))

                        layout_2d[:, negative_wrap + 1] += np.vstack((
                            np.ones(len(negative_wrap)) * 2 * np.pi,
                            np.zeros(len(negative_wrap))
                        ))

                        cut_rectangle = np.array([
                            [-np.pi, np.pi, np.pi, -np.pi],
                            [
                                np.max(aligned_wire_path_spiral_in[2, :]) +
                                np.abs(np.max(aligned_wire_path_spiral_in[2, :])) * 0.1,
                                np.max(aligned_wire_path_spiral_in[2, :]) +
                                np.abs(np.max(aligned_wire_path_spiral_in[2, :])) * 0.1,
                                np.min(aligned_wire_path_spiral_in[2, :]) -
                                np.abs(np.max(aligned_wire_path_spiral_in[2, :])) * 0.1,
                                np.min(aligned_wire_path_spiral_in[2, :]) -
                                np.abs(np.max(aligned_wire_path_spiral_in[2, :])) * 0.1
                            ]
                        ])

                        cut_rectangle = np.column_stack((cut_rectangle, cut_rectangle[:, 0]))

                        layout_2d[0, :] *= cylinder_radius
                        cut_rectangle[0, :] *= cylinder_radius

                        pcb_parts = [None] * (len(full_wrap_spart_inds) - 1)
                        for point_ind in range(len(full_wrap_spart_inds) - 1):
                            pcb_part = PCBPart(
                                uv=layout_2d[:, full_wrap_spart_inds[point_ind] +
                                             1:full_wrap_spart_inds[point_ind + 1]+1],
                                ind1=full_wrap_spart_inds[point_ind] + 1,
                                ind2=full_wrap_spart_inds[point_ind + 1]
                            )
                            pcb_parts[point_ind] = pcb_part

                        np.warnings.filterwarnings('ignore')

                        for wrap_ind in range(len(pcb_parts)):
                            intersection_cut = find_segment_intersections(pcb_parts[wrap_ind].uv, cut_rectangle)
                            real_cuts = [np.any(~np.isnan(cut.segment_inds)) for cut in intersection_cut]
                            is_real_cut_ind = np.where(real_cuts)[0]

                            if len(is_real_cut_ind) > 0:
                                wire_part_points = pcb_parts[wrap_ind].uv
                                uv_point = intersection_cut[is_real_cut_ind[0]].uv
                                cut_segment_ind = intersection_cut[is_real_cut_ind[0]].segment_inds

                                # Experience shows that this should be true
                                assert len(cut_segment_ind) == 1
                                cut_segment_ind = cut_segment_ind[0]

                                if cut_segment_ind != 0:
                                    pcb_parts[wrap_ind].uv = np.hstack((
                                        wire_part_points[:, :cut_segment_ind+1],
                                        uv_point.reshape(-1, 1),
                                        wire_part_points[:, cut_segment_ind+1:-1]
                                    ))

                        np.warnings.filterwarnings('default')

                        for wrap_ind in range(1, len(pcb_parts)):
                            if pcb_parts[wrap_ind - 1].uv[0, -1] > 0:
                                pcb_parts[wrap_ind].uv = np.hstack((
                                    pcb_parts[wrap_ind - 1].uv[:, -
                                                               1].reshape(-1, 1) - np.array([2 * np.pi * cylinder_radius, 0]),
                                    pcb_parts[wrap_ind].uv
                                ))
                            else:
                                pcb_parts[wrap_ind].uv = np.hstack((
                                    pcb_parts[wrap_ind - 1].uv[:, -
                                                               1].reshape(-1, 1) + np.array([2 * np.pi * cylinder_radius, 0]),
                                    pcb_parts[wrap_ind].uv
                                ))

                        pcb_parts = [part for part in pcb_parts if part.uv.shape[1] >= 2]  # delete fragments

                        # Generate the track shapes for the individual wire parts
                        for wire_part_ind in range(len(pcb_parts)):
                            if pcb_parts[wire_part_ind].uv.shape[1] > 5:
                                arr1 = pcb_parts[wire_part_ind].uv[:, 0].reshape(-1, 1)
                                arr2 = smooth_track_by_folding(pcb_parts[wire_part_ind].uv[:, 1:-1], 3)
                                arr3 = pcb_parts[wire_part_ind].uv[:, -1].reshape(-1, 1)
                                smoothed_track = np.hstack((
                                    arr1,
                                    arr2,
                                    arr3
                                ))
                            else:
                                smoothed_track = pcb_parts[wire_part_ind].uv

                            long_vecs = smoothed_track[:, 1:] - smoothed_track[:, :-1]
                            long_vecs = np.hstack((long_vecs, long_vecs[:, -1].reshape(-1, 1)))
                            long_vecs = long_vecs / np.tile(np.linalg.norm(long_vecs, axis=0), (2, 1))

                            ortho_vecs = np.vstack((long_vecs[1, :], -long_vecs[0, :]))
                            ortho_vecs = ortho_vecs / np.tile(np.linalg.norm(ortho_vecs, axis=0), (2, 1))
                            arr4 = smoothed_track + ortho_vecs * (pcb_track_width / 2)
                            arr5 = np.fliplr(smoothed_track) - np.fliplr(ortho_vecs) * (pcb_track_width / 2)
                            pcb_parts[wire_part_ind].track_shape = np.hstack((
                                arr4,
                                arr5
                            ))

                            arr6 = pcb_parts[wire_part_ind].track_shape[:, [0]]
                            pcb_parts[wire_part_ind].track_shape = np.hstack((
                                pcb_parts[wire_part_ind].track_shape,
                                arr6
                            ))

                            pcb_parts[wire_part_ind].polygon_track = Polygon(data=pcb_parts[wire_part_ind].track_shape)

                        # Write the outputs
                        if group_layer == 'upper':
                            upper_layer.group_layouts[group_ind].wire_parts = pcb_parts
                            # np.savetxt(f"upper_layer_part{part_ind}_group{group_ind}_wire_part{wire_part_ind}.txt", wire_part.track_shape.T, fmt="%f")
                        else:
                            lower_layer.group_layouts[group_ind].wire_parts = pcb_parts
                            # np.savetxt(f"lower_layer_part{part_ind}_group{group_ind}_wire_part{wire_part_ind}.txt", wire_part.track_shape.T, fmt="%f")

                coil_part.pcb_tracks = PCBTrack(upper_layer=upper_layer, lower_layer=lower_layer)

                # Save the tracks as a vector file
                # coil_mesh = coil_part.coil_mesh
                # rot_cylinder_vertices = np.dot(rot_mat, coil_mesh.get_vertices().T) # Need to tranpose into MATLAB shape
                # phi_coords_mesh = np.arctan2(rot_cylinder_vertices[1, :], rot_cylinder_vertices[0, :])
                # unrolled_cylinder = np.array([
                #    phi_coords_mesh * cylinder_radius,
                #    rot_cylinder_vertices[2, :]
                # ])

                # save_pcb_tracks_as_svg(coil_part.pcb_tracks, input_args['field_shape_function'], 'pcb_layout', part_ind, unrolled_cylinder, input_args['output_directory'])
            # end
        # end
    # end
    return coil_parts
# end
