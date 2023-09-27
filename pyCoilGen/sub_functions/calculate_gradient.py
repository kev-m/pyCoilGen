import numpy as np

from typing import List

from .data_structures import CoilPart, TargetField, LayoutGradient, WirePart


def calculate_gradient(coil_parts: List[CoilPart], input_args, target_field: TargetField) -> LayoutGradient:
    """
    Calculate local gradient.

    Args:
        coil_parts (list[CoilPart]): List of coil parts.
        target_field (TargetField): Target magnetic field.
        input_args (dict): Input arguments.

    Returns:
        LayoutGradient: Computed layout gradient.

    """
    # Determine field function based on input
    if input_args.field_shape_function == 'none':
        field_function = 'z'
    else:
        field_function = input_args.field_shape_function

    # Initialize gradient components
    # Assuming target_field.coords is (3,m) MATLAB shape
    target_field_coords_T = target_field.coords.T
    db_shape = target_field_coords_T.shape
    layout_gradient = LayoutGradient(
        dBxdxyz=np.zeros(db_shape),
        dBydxyz=np.zeros(db_shape),
        dBzdxyz=np.zeros(db_shape)
    )

    for coil_part in coil_parts:
        if coil_part.wire_path is not None:
            DBxdxyz, DBydxyz, DBzdxyz = direct_biot_savart_gradient_calc_3(
                coil_part.wire_path.v.T, target_field_coords_T)
        else:
            for contour in coil_part.contour_lines:
                DBxdxyz, DBydxyz, DBzdxyz = direct_biot_savart_gradient_calc_3(contour.v.T, target_field_coords_T)

        layout_gradient.dBxdxyz += DBxdxyz
        layout_gradient.dBydxyz += DBydxyz
        layout_gradient.dBzdxyz += DBzdxyz

    # Define field_function as a lambda function
    my_fun = eval('lambda x,y,z: ' + field_function)
    # def my_fun(x, y, z): return eval(field_function)

    norm_dir_x = my_fun(1., 0., 0.)
    norm_dir_y = my_fun(0., 1., 0.)
    norm_dir_z = my_fun(0., 0., 1.)
    gradient_direction = np.array([norm_dir_x, norm_dir_y, norm_dir_z])
    gradient_direction /= np.linalg.norm(gradient_direction)

    # Project the gradient direction to the full set of cartesian gradients
    layout_gradient.gradient_in_target_direction = np.sqrt((gradient_direction[0] * layout_gradient.dBzdxyz[:, 0])**2 +
                                                           (gradient_direction[1] * layout_gradient.dBzdxyz[:, 1])**2 +
                                                           (gradient_direction[2] * layout_gradient.dBzdxyz[:, 2])**2)
    layout_gradient.mean_gradient_in_target_direction = np.nanmean(layout_gradient.gradient_in_target_direction)
    layout_gradient.std_gradient_in_target_direction = np.nanstd(layout_gradient.gradient_in_target_direction)

    # Convert units to [mT/m/A]
    layout_gradient.dBxdxyz *= 1000.
    layout_gradient.dBydxyz *= 1000.
    layout_gradient.dBzdxyz *= 1000.
    layout_gradient.gradient_in_target_direction *= 1000.
    layout_gradient.mean_gradient_in_target_direction *= 1000.
    layout_gradient.std_gradient_in_target_direction *= 1000.

    return layout_gradient


def direct_biot_savart_gradient_calc_2(wire_elements, target_coords):
    """
    Calculate the gradient of magnetic field using Biot-Savart law.

    Args:
        wire_elements (numpy.ndarray): Array of wire element coordinates. Shape: (m, 3).
        target_coords (numpy.ndarray): Array of target coordinates. Shape: (n, 3).

    Returns:
        numpy.ndarray: Gradient components dBxdxyz, DBydxyz, DBzdxyz. Shape: (3, n).

    """
    num_tp = target_coords.shape[0]
    track_part_length = 1000

    if wire_elements.shape[0] > track_part_length:
        track_part_inds = np.arange(0, wire_elements.shape[0], track_part_length)
        track_part_inds = np.append(track_part_inds, wire_elements.shape[0])
        if track_part_inds[-2] == track_part_inds[-1]:
            track_part_inds = track_part_inds[:-1]

        wire_parts = []
        for i in range(len(track_part_inds) - 1):
            wire_part = WirePart()
            wire_part.coord = wire_elements[track_part_inds[i]:track_part_inds[i + 1], :]
            wire_part.seg_coords = (wire_part.coord[:-1, :] + wire_part.coord[1:, :]) / 2
            wire_part.currents = wire_part.coord[1:, :] - wire_part.coord[:-1, :]
            wire_parts.append(wire_part)
    else:
        wire_part = WirePart()
        wire_part.coord = wire_elements
        wire_part.seg_coords = (wire_part.coord[:-1, :] + wire_part.coord[1:, :]) / 2
        wire_part.currents = wire_part.coord[1:, :] - wire_part.coord[:-1, :]

    DBxdxyz = np.zeros((num_tp, 3))
    DBydxyz = np.zeros((num_tp, 3))
    DBzdxyz = np.zeros((num_tp, 3))
    for wire_part in wire_parts:
        target_p = np.repeat(target_coords[np.newaxis, :, :], wire_part.seg_coords.shape[0], axis=0)
        cur_pos = np.repeat(wire_part.seg_coords[:, np.newaxis, :], num_tp, axis=1)
        cur_dir = np.repeat(wire_part.currents[:, np.newaxis, :], num_tp, axis=1)

        x = target_p[:, :, 0]
        y = target_p[:, :, 1]
        z = target_p[:, :, 2]
        l_x = cur_pos[:, :, 0]
        l_y = cur_pos[:, :, 1]
        l_z = cur_pos[:, :, 2]
        d_l_x = cur_dir[:, :, 0]
        d_l_y = cur_dir[:, :, 1]
        d_l_z = cur_dir[:, :, 2]

        E = ((x - l_x)**2 + (y - l_y)**2 + (z - l_z)**2)**(3/2)
        dEdx = 3 * (x - l_x) * ((x - l_x)**2 + (y - l_y)**2 + (z - l_z)**2)**(1/2)
        dEdy = 3 * (y - l_y) * ((x - l_x)**2 + (y - l_y)**2 + (z - l_z)**2)**(1/2)
        dEdz = 3 * (z - l_z) * ((x - l_x)**2 + (y - l_y)**2 + (z - l_z)**2)**(1/2)
        C = 1e-7

        phi_x = d_l_y * (z - l_z) - d_l_z * (y - l_y)
        phi_y = d_l_z * (x - l_x) - d_l_x * (z - l_z)
        phi_z = d_l_x * (y - l_y) - d_l_y * (x - l_x)

        theta_factor = ((E**2)**(-1)) * (-1) * C

        dBx_dx = theta_factor * dEdx * phi_x
        dBx_dy = theta_factor * dEdy * phi_x + (E**(-1)) * C * (-1) * d_l_z
        dBx_dz = theta_factor * dEdz * phi_x + (E**(-1)) * C * d_l_y

        dBy_dx = theta_factor * dEdx * phi_y + (E**(-1)) * C * d_l_z
        dBy_dy = theta_factor * dEdy * phi_y
        dBy_dz = theta_factor * dEdz * phi_y + (E**(-1)) * C * (-1) * d_l_x

        dBz_dx = theta_factor * dEdx * phi_z + (E**(-1)) * C * (-1) * d_l_y
        dBz_dy = theta_factor * dEdy * phi_z + (E**(-1)) * C * d_l_x
        dBz_dz = theta_factor * dEdz * phi_z

        # Accumulate gradients
        DBxdxyz += np.column_stack((np.sum(dBx_dx, axis=0), np.sum(dBx_dy, axis=0), np.sum(dBx_dz, axis=0)))
        DBydxyz += np.column_stack((np.sum(dBy_dx, axis=0), np.sum(dBy_dy, axis=0), np.sum(dBy_dz, axis=0)))
        DBzdxyz += np.column_stack((np.sum(dBz_dx, axis=0), np.sum(dBz_dy, axis=0), np.sum(dBz_dz, axis=0)))

    return DBxdxyz, DBydxyz, DBzdxyz


def direct_biot_savart_gradient_calc_3(wire_elements, target_coords):
    """
    Calculate the gradient of magnetic field using Biot-Savart law.

    Args:
        wire_elements (numpy.ndarray): Array of wire element coordinates. Shape: (m, 3).
        target_coords (numpy.ndarray): Array of target coordinates. Shape: (n, 3).

    Returns:
        numpy.ndarray: Gradient components dBxdxyz, DBydxyz, DBzdxyz. Shape: (3, n).

    """
    num_tp = target_coords.shape[0]
    track_part_length = 1000

    if wire_elements.shape[0] > track_part_length:
        track_part_inds = np.arange(0, wire_elements.shape[0], track_part_length)
        track_part_inds = np.append(track_part_inds, wire_elements.shape[0])
        if track_part_inds[-2] == track_part_inds[-1]:
            track_part_inds = track_part_inds[:-1]

        wire_parts = []
        part_ind_start = track_part_inds[0]
        for i in range(len(track_part_inds) - 1):
            part_ind_end = track_part_inds[i + 1]
            wire_part = WirePart()
            wire_part.coord = wire_elements[part_ind_start:part_ind_end, :]
            wire_part.seg_coords = (wire_part.coord[:-1, :] + wire_part.coord[1:, :]) / 2
            wire_part.currents = wire_part.coord[1:, :] - wire_part.coord[:-1, :]
            wire_parts.append(wire_part)
            # Start next loop before the last point of the previous, since the last point is not used in the loop below
            part_ind_start = part_ind_end-1
    else:
        wire_part = WirePart()
        wire_part.coord = wire_elements
        wire_part.seg_coords = (wire_part.coord[:-1, :] + wire_part.coord[1:, :]) / 2
        wire_part.currents = wire_part.coord[1:, :] - wire_part.coord[:-1, :]
        wire_parts = [wire_part]

    DBxdxyz = np.zeros((num_tp, 3))
    DBydxyz = np.zeros((num_tp, 3))
    DBzdxyz = np.zeros((num_tp, 3))
    for wire_part in wire_parts:
        target_p = np.repeat(target_coords[np.newaxis, :, :], wire_part.seg_coords.shape[0], axis=0)
        cur_pos = np.repeat(wire_part.seg_coords[:, np.newaxis, :], num_tp, axis=1)
        cur_dir = np.repeat(wire_part.currents[:, np.newaxis, :], num_tp, axis=1)

        x = target_p[:, :, 0]
        y = target_p[:, :, 1]
        z = target_p[:, :, 2]
        l_x = cur_pos[:, :, 0]
        l_y = cur_pos[:, :, 1]
        l_z = cur_pos[:, :, 2]
        d_l_x = cur_dir[:, :, 0]
        d_l_y = cur_dir[:, :, 1]
        d_l_z = cur_dir[:, :, 2]

        """
        E = ((x - l_x)**2 + (y - l_y)**2 + (z - l_z)**2)**(3/2)
        dEdx = 3 * (x - l_x) * ((x - l_x)**2 + (y - l_y)**2 + (z - l_z)**2)**(1/2)
        dEdy = 3 * (y - l_y) * ((x - l_x)**2 + (y - l_y)**2 + (z - l_z)**2)**(1/2)
        dEdz = 3 * (z - l_z) * ((x - l_x)**2 + (y - l_y)**2 + (z - l_z)**2)**(1/2)
        """
        # Optimized
        squared_difference = (x - l_x)**2. + (y - l_y)**2. + (z - l_z)**2.
        E = squared_difference**(3./2.)
        sqrt_squared_difference = np.sqrt(squared_difference)
        dEdx = 3. * (x - l_x) * sqrt_squared_difference
        dEdy = 3. * (y - l_y) * sqrt_squared_difference
        dEdz = 3. * (z - l_z) * sqrt_squared_difference

        C = 1e-7

        phi_x = d_l_y * (z - l_z) - d_l_z * (y - l_y)
        phi_y = d_l_z * (x - l_x) - d_l_x * (z - l_z)
        phi_z = d_l_x * (y - l_y) - d_l_y * (x - l_x)

        theta_factor = ((E**2)**(-1.)) * (-1.) * C

        theta_dEdx = theta_factor * dEdx
        theta_dEdy = theta_factor * dEdy
        theta_dEdz = theta_factor * dEdz
        E_inv_C = (E**(-1.)) * C

        dBx_dx = theta_dEdx * phi_x
        dBx_dy = theta_dEdy * phi_x + E_inv_C * (-1.) * d_l_z
        dBx_dz = theta_dEdz * phi_x + E_inv_C * d_l_y

        dBy_dx = theta_dEdx * phi_y + E_inv_C * d_l_z
        dBy_dy = theta_dEdy * phi_y
        dBy_dz = theta_dEdz * phi_y + E_inv_C * (-1.) * d_l_x

        dBz_dx = theta_dEdx * phi_z + E_inv_C * (-1.) * d_l_y
        dBz_dy = theta_dEdy * phi_z + E_inv_C * d_l_x
        dBz_dz = theta_dEdz * phi_z

        # Accumulate gradients
        DBxdxyz += np.column_stack((np.sum(dBx_dx, axis=0), np.sum(dBx_dy, axis=0), np.sum(dBx_dz, axis=0)))
        DBydxyz += np.column_stack((np.sum(dBy_dx, axis=0), np.sum(dBy_dy, axis=0), np.sum(dBy_dz, axis=0)))
        DBzdxyz += np.column_stack((np.sum(dBz_dx, axis=0), np.sum(dBz_dy, axis=0), np.sum(dBz_dz, axis=0)))

    return DBxdxyz, DBydxyz, DBzdxyz
