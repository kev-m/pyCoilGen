import numpy as np

from .gauss_legendre_integration_points_triangle import gauss_legendre_integration_points_triangle
from .sensitivity_kernels import stack_basis_elements, integrate_over_node_triangles


def _gradient_kernel(rx, ry, rz, current, area, weight):
    """
    The contribution of every Gauss point of every triangle to the gradient of Bz: (dBz/dx, dBz/dy, dBz/dz), weighted
    with the Gauss weight.

    Args:
        rx, ry, rz (ndarray): The components of the vector from the sample point to the target point, each (num
            samples, num target points).
        current (ndarray): The current density of the triangle of each sample, (num samples, 3).
        area (ndarray): The area of the triangle of each sample, (num samples,). Not used.
        weight (ndarray): The Gauss weight of each sample, (num samples,).

    Returns:
        tuple: The dBz/dx, dBz/dy and dBz/dz contributions, each (num samples, num target points).
    """
    C = 10 ** (-7)
    # With E = |r|^3: -C * dE/dx / E^2 = -3 * C * rx / |r|^5 and C / E = C / |r|^3
    inv_dist = rx * rx
    inv_dist += ry * ry
    inv_dist += rz * rz
    np.sqrt(inv_dist, out=inv_dist)
    np.reciprocal(inv_dist, out=inv_dist)
    inv_dist2 = inv_dist * inv_dist
    inv_E = inv_dist2 * inv_dist            # 1 / |r|^3
    inv_dist2 *= inv_dist2
    inv_dist2 *= inv_dist                   # 1 / |r|^5
    theta_factor = inv_dist2
    w = weight[:, np.newaxis]
    d_l_x, d_l_y = current[:, 0:1], current[:, 1:2]
    theta_phi = d_l_x * ry
    theta_phi -= d_l_y * rx                 # phi_z
    theta_phi *= theta_factor
    theta_phi *= -3 * C
    inv_E *= C
    dx = theta_phi * rx
    dx -= inv_E * d_l_y
    dx *= w
    dy = theta_phi * ry
    dy += inv_E * d_l_x
    dy *= w
    dz = theta_phi
    dz *= rz
    dz *= w
    return dx, dy, dz


def calculate_gradient_sensitivity_matrix(coil_parts, target_field, input_args):
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

    biot_savart_coeff = 10 ** (-7)
    plate_thickness = 0.001

    # Calculate the sensitivity matrix for each coil part
    for coil_part in coil_parts:
        triangle_points, current, area, counts = stack_basis_elements(coil_part)
        coil_part.gradient_sensitivity_matrix = biot_savart_coeff * plate_thickness * integrate_over_node_triangles(
            triangle_points, current, area, counts, target_points, u_coord, v_coord, gauss_weight, _gradient_kernel,
            num_threads=getattr(input_args, "num_threads", 0))

    return coil_parts
