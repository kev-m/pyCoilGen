import numpy as np
from typing import List

# Logging
import logging

# Local imports
from .data_structures import CoilSolution, BasisElement, CoilPart
from .gauss_legendre_integration_points_triangle import gauss_legendre_integration_points_triangle
from .sensitivity_kernels import stack_basis_elements, integrate_over_node_triangles

log = logging.getLogger(__name__)


def _biot_savart_kernel(rx, ry, rz, current, area, weight):
    """
    The Biot-Savart contribution of every Gauss point of every triangle: (I x r) / |r|^3, weighted with 2*area*weight.

    The arrays are modified in place, to avoid allocating temporary arrays.

    Args:
        rx, ry, rz (ndarray): The components of the vector from the sample point to the target point, each (num
            samples, num target points). These are overwritten.
        current (ndarray): The current density of the triangle of each sample, (num samples, 3).
        area (ndarray): The area of the triangle of each sample, (num samples,).
        weight (ndarray): The Gauss weight of each sample, (num samples,).

    Returns:
        tuple: The x, y and z contributions, each (num samples, num target points).
    """
    dist2 = np.multiply(rx, rx)
    dist2 += ry * ry
    dist2 += rz * rz
    scale = np.sqrt(dist2)
    scale *= dist2  # |r|^3
    np.reciprocal(scale, out=scale)
    scale *= (2 * area * weight)[:, np.newaxis]
    ix, iy, iz = current[:, 0:1], current[:, 1:2], current[:, 2:3]
    cx = iy * rz
    cx -= iz * ry
    cx *= scale
    cy = iz * rx
    cy -= ix * rz
    cy *= scale
    cz = ix * ry
    cz -= iy * rx
    cz *= scale
    return cx, cy, cz


def calculate_sensitivity_matrix(coil_parts: List[CoilPart], target_field, input_args) -> List[CoilPart]:
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
    target_points = target_field.coords
    gauss_order = input_args.gauss_order

    # Calculate the weights and the test point for the Gauss-Legendre integration on each triangle
    u_coord, v_coord, gauss_weight = gauss_legendre_integration_points_triangle(gauss_order)
    biot_savart_coeff = 1e-7

    for coil_part in coil_parts:
        triangle_points, current, area, counts = stack_basis_elements(coil_part)
        coil_part.sensitivity_matrix = biot_savart_coeff * integrate_over_node_triangles(
            triangle_points, current, area, counts, target_points, u_coord, v_coord, gauss_weight, _biot_savart_kernel,
            num_threads=getattr(input_args, "num_threads", 0), fast_mode=0)

    return coil_parts
