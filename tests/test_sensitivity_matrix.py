"""
Compare the vectorised sensitivity matrices with the original, loop based, implementations.

The reference implementations below are the original code, which looped over nodes, triangles and Gauss points.
They are kept as the specification that the fast implementations must reproduce.
"""
# System
import contextlib
import copy
import io
import logging
import time
from typing import List

# Libraries
import numpy as np
import pytest

# Local imports
from pyCoilGen.pyCoilGen_release import pyCoilGen
from pyCoilGen.sub_functions.gauss_legendre_integration_points_triangle import gauss_legendre_integration_points_triangle
from pyCoilGen.sub_functions.calculate_sensitivity_matrix import calculate_sensitivity_matrix
from pyCoilGen.sub_functions.calculate_gradient_sensitivity_matrix import calculate_gradient_sensitivity_matrix
from pyCoilGen.sub_functions.data_structures import CoilPart


def reference_sensitivity_matrix(coil_parts: List[CoilPart], target_field, input_args) -> List[CoilPart]:
    """
    Calculate the sensitivity matrix.

    Initialises the following properties of a CoilPart:
        - sensitivity_matrix: (3, m, num vertices)

    Updates the following properties of a CoilPart:
        - None

    Args:
        coil_parts (List[CoilPart]): List of coil parts.
        target_field: The target field.

    Returns:
        List[CoilPart]: Updated list of coil parts with sensitivity matrix.

    """
    for part_ind in range(len(coil_parts)):
        coil_part = coil_parts[part_ind]

        target_points = target_field.coords
        gauss_order = input_args.gauss_order

        # Calculate the weights and the test point for the Gauss-Legendre integration on each triangle
        u_coord, v_coord, gauss_weight = gauss_legendre_integration_points_triangle(gauss_order)
        num_gauss_points = len(gauss_weight)
        biot_savart_coeff = 1e-7
        num_nodes = len(coil_part.basis_elements)
        num_target_points = target_points.shape[1]
        sensitivity_matrix = np.zeros((3, num_target_points, num_nodes))

        for node_ind in range(num_nodes):
            basis_element = coil_part.basis_elements[node_ind]

            dCx = np.zeros(num_target_points)
            dCy = np.zeros(num_target_points)
            dCz = np.zeros(num_target_points)

            for tri_ind in range(len(basis_element.area)):
                node_point = basis_element.triangle_points_ABC[tri_ind, :, 0]
                point_b = basis_element.triangle_points_ABC[tri_ind, :, 1]
                point_c = basis_element.triangle_points_ABC[tri_ind, :, 2]

                x1, y1, z1 = node_point
                x2, y2, z2 = point_b
                x3, y3, z3 = point_c

                vx, vy, vz = basis_element.current[tri_ind]

                for gauss_ind in range(num_gauss_points):
                    xgauss_in_uv = x1 * u_coord[gauss_ind] + x2 * v_coord[gauss_ind] + \
                        x3 * (1 - u_coord[gauss_ind] - v_coord[gauss_ind])
                    ygauss_in_uv = y1 * u_coord[gauss_ind] + y2 * v_coord[gauss_ind] + \
                        y3 * (1 - u_coord[gauss_ind] - v_coord[gauss_ind])
                    zgauss_in_uv = z1 * u_coord[gauss_ind] + z2 * v_coord[gauss_ind] + \
                        z3 * (1 - u_coord[gauss_ind] - v_coord[gauss_ind])

                    distance_norm = (
                        (xgauss_in_uv - target_points[0])**2 + (ygauss_in_uv - target_points[1])**2 + (zgauss_in_uv - target_points[2])**2)**(-3/2)

                    dCx += ((-1) * vz * (target_points[1] - ygauss_in_uv) + vy * (target_points[2] - zgauss_in_uv)
                            ) * distance_norm * 2 * basis_element.area[tri_ind] * gauss_weight[gauss_ind]
                    dCy += ((-1) * vx * (target_points[2] - zgauss_in_uv) + vz * (target_points[0] - xgauss_in_uv)
                            ) * distance_norm * 2 * basis_element.area[tri_ind] * gauss_weight[gauss_ind]
                    dCz += ((-1) * vy * (target_points[0] - xgauss_in_uv) + vx * (target_points[1] - ygauss_in_uv)
                            ) * distance_norm * 2 * basis_element.area[tri_ind] * gauss_weight[gauss_ind]

            sensitivity_matrix[:, :, node_ind] = np.array([dCx, dCy, dCz]) * biot_savart_coeff

        coil_part.sensitivity_matrix = sensitivity_matrix

    return coil_parts


def reference_gradient_sensitivity_matrix(coil_parts, target_field, input_args):
    """
    Calculate the gradient sensitivity matrix for coil parts.

    Initialises the following properties of the CoilParts:
        - gradient_sensitivity_matrix

    Depends on the following properties of the CoilParts:
        - basis_elements

    Depends on the following input_args:
        - gauss_order

    Updates the following properties of a CoilPart:
        - None

    Args:
        coil_parts (list): List of coil parts.
        target_field: The target field.
        input: The input parameters.

    Returns:
        list: List of coil parts with the gradient sensitivity matrix calculated.
    """

    target_points = target_field.coords
    gauss_order = input_args.gauss_order
    u_coord, v_coord, gauss_weight = gauss_legendre_integration_points_triangle(gauss_order)

    # Calculate the sensitivity matrix for each coil part
    for part_ind in range(len(coil_parts)):
        biot_savart_coeff = 10 ** (-7)
        plate_thickness = 0.001
        num_nodes = len(coil_parts[part_ind].basis_elements)
        num_target_points = target_points.shape[1]  # TODO: Check 1 or 0?
        gradient_sensitivity_matrix = np.zeros((3, num_target_points, num_nodes))  # TODO: Check shape for Python

        x = target_points[0, :]
        y = target_points[1, :]
        z = target_points[2, :]

        for node_ind in range(num_nodes):
            DBzdx = np.zeros(num_target_points)
            DBzdy = np.zeros(num_target_points)
            DBzdz = np.zeros(num_target_points)

            for tri_ind in range(coil_parts[part_ind].basis_elements[node_ind].triangle_points_ABC.shape[0]):
                node_point = coil_parts[part_ind].basis_elements[node_ind].triangle_points_ABC[tri_ind, :, 0]
                point_b = coil_parts[part_ind].basis_elements[node_ind].triangle_points_ABC[tri_ind, :, 1]
                point_c = coil_parts[part_ind].basis_elements[node_ind].triangle_points_ABC[tri_ind, :, 2]

                x1, y1, z1 = node_point
                x2, y2, z2 = point_b
                x3, y3, z3 = point_c

                d_l_x = coil_parts[part_ind].basis_elements[node_ind].current[tri_ind, 0]
                d_l_y = coil_parts[part_ind].basis_elements[node_ind].current[tri_ind, 1]
                d_l_z = coil_parts[part_ind].basis_elements[node_ind].current[tri_ind, 2]

                for gauss_ind in range(len(gauss_weight)):
                    l_x = x1 * u_coord[gauss_ind] + x2 * v_coord[gauss_ind] + \
                        x3 * (1 - u_coord[gauss_ind] - v_coord[gauss_ind])
                    l_y = y1 * u_coord[gauss_ind] + y2 * v_coord[gauss_ind] + \
                        y3 * (1 - u_coord[gauss_ind] - v_coord[gauss_ind])
                    l_z = z1 * u_coord[gauss_ind] + z2 * v_coord[gauss_ind] + \
                        z3 * (1 - u_coord[gauss_ind] - v_coord[gauss_ind])

                    E = ((x - l_x) ** 2 + (y - l_y) ** 2 + (z - l_z) ** 2) ** (3 / 2)
                    dEdx = 3 * (x - l_x) * ((x - l_x) ** 2 + (y - l_y) ** 2 + (z - l_z) ** 2) ** (1 / 2)
                    dEdy = 3 * (y - l_y) * ((x - l_x) ** 2 + (y - l_y) ** 2 + (z - l_z) ** 2) ** (1 / 2)
                    dEdz = 3 * (z - l_z) * ((x - l_x) ** 2 + (y - l_y) ** 2 + (z - l_z) ** 2) ** (1 / 2)
                    C = 10 ** (-7)
                    phi_x = (d_l_y * (z - l_z) - d_l_z * (y - l_y))
                    phi_y = (d_l_z * (x - l_x) - d_l_x * (z - l_z))
                    phi_z = (d_l_x * (y - l_y) - d_l_y * (x - l_x))
                    theta_factor = ((E * E) ** (-1)) * (-1) * C
                    dBz_dx = theta_factor * dEdx * phi_z + (E ** (-1)) * C * (-1) * d_l_y
                    dBz_dy = theta_factor * dEdy * phi_z + (E ** (-1)) * C * d_l_x
                    dBz_dz = theta_factor * dEdz * phi_z

                    DBzdx += dBz_dx * gauss_weight[gauss_ind]
                    DBzdy += dBz_dy * gauss_weight[gauss_ind]
                    DBzdz += dBz_dz * gauss_weight[gauss_ind]

            gradient_sensitivity_matrix[:, :, node_ind] = np.vstack(
                (DBzdx, DBzdy, DBzdz)) * biot_savart_coeff * plate_thickness

        coil_parts[part_ind].gradient_sensitivity_matrix = gradient_sensitivity_matrix

    return coil_parts


CASES = {
    'cylinder': {
        'coil_mesh': 'create cylinder mesh', 'cylinder_mesh_parameter_list': [0.8, 0.3, 14, 14, 1, 0, 0, 0],
        'surface_is_cylinder_flag': True, 'field_shape_function': 'y', 'interconnection_cut_width': 0.05,
        'normal_shift_length': 0.01, 'target_region_radius': 0.05},
    'biplanar': {
        'coil_mesh': 'create bi-planar mesh', 'biplanar_mesh_parameter_list': [0.25, 0.25, 10, 10, 1, 0, 0, 0, 0, 0, 0.2],
        'surface_is_cylinder_flag': False, 'field_shape_function': 'x', 'interconnection_cut_width': 0.01,
        'normal_shift_length': 0.005, 'target_region_radius': 0.05},
}


@pytest.fixture(scope='module', params=sorted(CASES))
def solution(request, tmp_path_factory):
    work_dir = str(tmp_path_factory.mktemp(request.param))
    arguments = dict(CASES[request.param], target_region_resolution=6, iteration_num_mesh_refinement=0,
                     set_roi_into_mesh_center=True, force_cut_selection=['high'], level_set_method='primary',
                     levels=6, skip_postprocessing=True, skip_inductance_calculation=True, exporter='none', debug=0,
                     project_name=request.param, output_directory=work_dir, persistence_dir=work_dir)
    with contextlib.redirect_stdout(io.StringIO()):
        return pyCoilGen(logging.getLogger('test_sensitivity'), arguments)


def _fresh_parts(solution):
    """Copy the coil parts, without the matrices that are under test."""
    parts = copy.deepcopy(solution.coil_parts)
    for part in parts:
        part.sensitivity_matrix = None
        part.gradient_sensitivity_matrix = None
    return parts


def test_sensitivity_matrix_matches_the_loop_implementation(solution):
    expected = reference_sensitivity_matrix(_fresh_parts(solution), solution.target_field, solution.input_args)
    actual = calculate_sensitivity_matrix(_fresh_parts(solution), solution.target_field, solution.input_args)
    for e, a in zip(expected, actual):
        assert a.sensitivity_matrix.shape == e.sensitivity_matrix.shape
        np.testing.assert_allclose(a.sensitivity_matrix, e.sensitivity_matrix, rtol=1e-10, atol=1e-18)


def test_gradient_sensitivity_matrix_matches_the_loop_implementation(solution):
    expected = reference_gradient_sensitivity_matrix(_fresh_parts(solution), solution.target_field, solution.input_args)
    actual = calculate_gradient_sensitivity_matrix(_fresh_parts(solution), solution.target_field, solution.input_args)
    for e, a in zip(expected, actual):
        assert a.gradient_sensitivity_matrix.shape == e.gradient_sensitivity_matrix.shape
        np.testing.assert_allclose(a.gradient_sensitivity_matrix, e.gradient_sensitivity_matrix, rtol=1e-10, atol=1e-22)


def test_sensitivity_matrices_do_not_depend_on_the_block_size(solution, monkeypatch):
    """Evaluating in many small blocks must give the same answer as one block."""
    from pyCoilGen.sub_functions import sensitivity_kernels
    expected = calculate_sensitivity_matrix(_fresh_parts(solution), solution.target_field, solution.input_args)
    monkeypatch.setattr(sensitivity_kernels, '_MAX_PAIRS_PER_BLOCK', 1)  # One triangle per block
    actual = calculate_sensitivity_matrix(_fresh_parts(solution), solution.target_field, solution.input_args)
    for e, a in zip(expected, actual):
        np.testing.assert_allclose(a.sensitivity_matrix, e.sensitivity_matrix, rtol=1e-12)
