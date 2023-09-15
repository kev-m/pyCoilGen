import numpy as np


def gauss_legendre_integration_points_triangle(n):
    """
    Calculate the coordinates and weighting coefficients for Gauss-Legendre integration on a triangle.

    Args:
        n (int): Order of the integration.

    Returns:
        u (ndarray): U coordinates of the integration points.
        v (ndarray): V coordinates of the integration points.
        ck (ndarray): Weighting coefficients.

    """

    eta, w = calc_weights_gauss(n)
    num_points = eta.shape[0] * eta.shape[0]
    u = np.zeros((num_points, 1))
    v = np.zeros((num_points, 1))
    ck = np.zeros((num_points, 1))

    k = 0
    for i in range(eta.shape[0]):
        for j in range(eta.shape[0]):
            u[k, 0] = (1 + eta[i]) / 2
            v[k, 0] = (1 - eta[i]) * (1 + eta[j]) / 4
            ck[k, 0] = ((1 - eta[i]) / 8) * w[i] * w[j]
            k += 1

    return u, v, ck


def calc_weights_gauss(n):
    """
    Generate the abscissa and weights for Gauss-Legendre quadrature.

    Args:
        n (int): Number of points.

    Returns:
        g_abscissa (ndarray): Abscissa values.
        g_weights (ndarray): Weighting coefficients.

    """

    g_abscissa = np.zeros(n)  # Preallocations.
    g_weights = np.zeros(n)
    m = int((n + 1) / 2)

    for ii in range(1, m + 1):
        z = np.cos(np.pi * (ii - 0.25) / (n + 0.5))  # Initial estimate.
        z1 = z + 1.0
        while abs(z - z1) > np.finfo(float).eps:
            p1 = 1
            p2 = 0

            for jj in range(1, n + 1):
                p3 = p2
                p2 = p1
                p1 = ((2 * jj - 1) * z * p2 - (jj - 1) * p3) / jj  # The Legendre polynomial.

            pp = n * (z * p1 - p2) / (z ** 2 - 1)  # The L.P. derivative.
            z1 = z
            z = z1 - p1 / pp

        g_abscissa[ii - 1] = -z  # Build up the abscissas.
        g_abscissa[n - ii] = z
        g_weights[ii - 1] = 2 / ((1 - z ** 2) * (pp ** 2))  # Build up the weights.
        g_weights[n - ii] = g_weights[ii - 1]

    return g_abscissa, g_weights
