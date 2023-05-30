import numpy as np

from data_structures import LayoutGradient

def calculate_gradient(coil_parts, target_field, input):
    """
    Calculate the average of the local gradients to neighbors for each point.

    Args:
        coil_parts (list): List of CoilPart objects representing different parts of the coil.
        target_field: Target field.
        input: Input parameters.

    Returns:
        layout_gradient: Layout gradient object containing the calculated gradients.
    """
    # Determine field function based on input
    if input.field_shape_function == 'none':
        field_function = 'z'
    else:
        field_function = input.field_shape_function

    layout_gradient = LayoutGradient()
    layout_gradient.dBxdxyz = np.zeros(target_field.coords.shape)
    layout_gradient.dBydxyz = np.zeros(target_field.coords.shape)
    layout_gradient.dBzdxyz = np.zeros(target_field.coords.shape)
    DBxdxyz = np.zeros(target_field.coords.shape)
    DBydxyz = np.zeros(target_field.coords.shape)
    DBzdxyz = np.zeros(target_field.coords.shape)

    if hasattr(coil_parts[0], 'wire_path'):
        # Use wire paths
        for part in coil_parts:
            for i in range(len(part.wire_path.v)):
                [DBxdxyz, DBydxyz, DBzdxyz] = direct_biot_savart_gradient_calc(part.wire_path.v[i],
                                                                                target_field.coords)
                layout_gradient.dBxdxyz += DBxdxyz
                layout_gradient.dBydxyz += DBydxyz
                layout_gradient.dBzdxyz += DBzdxyz
    else:
        # Use contour lines
        for part in coil_parts:
            for contour_line in part.contour_lines:
                [DBxdxyz, DBydxyz, DBzdxyz] = direct_biot_savart_gradient_calc(contour_line.v,
                                                                                target_field.coords)
                layout_gradient.dBxdxyz += DBxdxyz
                layout_gradient.dBydxyz += DBydxyz
                layout_gradient.dBzdxyz += DBzdxyz

    my_fun = eval("@(x,y,z)" + field_function)
    norm_dir_x = my_fun(1, 0, 0)
    norm_dir_y = my_fun(0, 1, 0)
    norm_dir_z = my_fun(0, 0, 1)
    gradient_direction = np.array([norm_dir_x, norm_dir_y, norm_dir_z])
    gradient_direction /= np.linalg.norm(gradient_direction)

    # Project the gradient direction to the full set of cartesian gradients
    layout_gradient.gradient_in_target_direction = np.sqrt((gradient_direction[0] * layout_gradient.dBzdxyz[0])**2 +
                                                           (gradient_direction[1] * layout_gradient.dBzdxyz[1])**2 +
                                                           (gradient_direction[2] * layout_gradient.dBzdxyz[2])**2)
    layout_gradient.mean_gradient_in_target_direction = np.mean(layout_gradient.gradient_in_target_direction,
                                                                 axis=None, where=~np.isnan(layout_gradient.gradient_in_target_direction))
    layout_gradient.std_gradient_in_target_direction = np.std(layout_gradient.gradient_in_target_direction,
                                                               axis=None, where=~np.isnan(layout_gradient.gradient_in_target_direction))

    # Adjust to the unit [mT/m/A]
    layout_gradient.dBxdxyz *= 1000
    layout_gradient.dBydxyz *= 1000
    layout_gradient.dBzdxyz *= 1000
    layout_gradient.gradient_in_target_direction *= 1000
    layout_gradient.mean_gradient_in_target_direction *= 1000
    layout_gradient.std_gradient_in_target_direction *= 1000

    return layout_gradient


def direct_biot_savart_gradient_calc(wire_elements, target_coords):
    """
    Calculate the magnetic field gradient using Biot-Savart law for wire elements.

    Args:
        wire_elements: Wire elements represented as a sequence of coordinate points.
        target_coords: Target coordinates.

    Returns:
        DBxdxyz: Gradient of the x-component of the magnetic field.
        DBydxyz: Gradient of the y-component of the magnetic field.
        DBzdxyz: Gradient of the z-component of the magnetic field.
    """
    num_tp = target_coords.shape[1]
    track_part_length = 1000

    if wire_elements.shape[1] > track_part_length:
        track_part_inds = np.arange(1, wire_elements.shape[1], track_part_length)
        track_part_inds = np.append(track_part_inds, wire_elements.shape[1])
        if track_part_inds[-2] == track_part_inds[-1]:
            track_part_inds = track_part_inds[:-1]
        wire_parts = []
        for i in range(len(track_part_inds) - 1):
            wire_part = WirePart()
            wire_part.coord = wire_elements[:, track_part_inds[i]:track_part_inds[i + 1]]
            wire_part.seg_coords = (wire_part.coord[:, :-1] + wire_part.coord[:, 1:]) / 2
            wire_part.currents = wire_part.coord[:, 1:] - wire_part.coord[:, :-1]
            wire_parts.append(wire_part)
    else:
        wire_part = WirePart()
        wire_part.coord = wire_elements
        wire_part.seg_coords = (wire_part.coord[:, :-1] + wire_part.coord[:, 1:]) / 2
        wire_part.currents = wire_part.coord[:, 1:] - wire_part.coord[:, :-1]
        wire_parts = [wire_part]

    DBxdxyz = np.zeros((3, num_tp))
    DBydxyz = np.zeros((3, num_tp))
    DBzdxyz = np.zeros((3, num_tp))

    for wire_part in wire_parts:
        target_p = np.tile(target_coords, (1, 1, wire_part.seg_coords.shape[1]))
        target_p = np.transpose(target_p, (0, 2, 1))
        cur_pos = np.tile(wire_part.seg_coords[:, :, np.newaxis], (1, 1, num_tp))
        cur_dir = np.tile(wire_part.currents[:, :, np.newaxis], (1, 1, num_tp))

        x = target_p[0, :, :]
        y = target_p[1, :, :]
        z = target_p[2, :, :]
        l_x = cur_pos[0, :, :]
        l_y = cur_pos[1, :, :]
        l_z = cur_pos[2, :, :]
        d_l_x = cur_dir[0, :, :]
        d_l_y = cur_dir[1, :, :]
        d_l_z = cur_dir[2, :, :]

        E = ((x - l_x) ** 2 + (y - l_y) ** 2 + (z - l_z) ** 2) ** (3 / 2)
        dEdx = 3 * (x - l_x) * ((x - l_x) ** 2 + (y - l_y) ** 2 + (z - l_z) ** 2) ** (1 / 2)
        dEdy = 3 * (y - l_y) * ((x - l_x) ** 2 + (y - l_y) ** 2 + (z - l_z) ** 2) ** (1 / 2)
        dEdz = 3 * (z - l_z) * ((x - l_x) ** 2 + (y - l_y) ** 2 + (z - l_z) ** 2) ** (1 / 2)
        C = 10 ** (-7)

        phi_x = (d_l_y * (z - l_z) - d_l_z * (y - l_y))
        phi_y = (d_l_z * (x - l_x) - d_l_x * (z - l_z))
        phi_z = (d_l_x * (y - l_y) - d_l_y * (x - l_x))

        theta_factor = ((E * E) ** (-1)) * (-1) * C

        dBx_dx = theta_factor * dEdx * phi_x
        dBx_dy = theta_factor * dEdy * phi_x + (E ** (-1)) * C * (-1) * d_l_z
        dBx_dz = theta_factor * dEdz * phi_x + (E ** (-1)) * C * d_l_y

        dBy_dx = theta_factor * dEdx * phi_y + (E ** (-1)) * C * d_l_z
        dBy_dy = theta_factor * dEdy * phi_y
        dBy_dz = theta_factor * dEdz * phi_y + (E ** (-1)) * C * (-1) * d_l_x

        dBz_dx = theta_factor * dEdx * phi_z + (E ** (-1)) * C * (-1) * d_l_y
        dBz_dy = theta_factor * dEdy * phi_z + (E ** (-1)) * C * d_l_x
        dBz_dz = theta_factor * dEdz * phi_z

        DBxdxyz += np.array([np.sum(dBx_dx, axis=1), np.sum(dBx_dy, axis=1), np.sum(dBx_dz, axis=1)])
        DBydxyz += np.array([np.sum(dBy_dx, axis=1), np.sum(dBy_dy, axis=1), np.sum(dBy_dz, axis=1)])
        DBzdxyz += np.array([np.sum(dBz_dx, axis=1), np.sum(dBz_dy, axis=1), np.sum(dBz_dz, axis=1)])

    return DBxdxyz, DBydxyz, DBzdxyz
