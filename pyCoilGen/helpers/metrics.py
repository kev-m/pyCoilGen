import numpy as np
import matplotlib.pyplot as plt


# ==========================================================
# PRINT PERFORMANCE METRICS
# ==========================================================

def print_metrics(Bz, coords, axis = 'x'):
    """
    Print and visualize magnetic gradient field metrics.
    This function creates a 3D scatter plot of the magnetic field and computes
    gradient efficiency and linearity error metrics along the specified axis.
    Args:
        Bz (numpy.ndarray): Magnetic field values in Tesla. 1D array of field
            measurements at each coordinate point.
        coords (numpy.ndarray): Coordinate positions. 2D array of shape (N, 3)
            containing x, y, z coordinates in meters for each field measurement.
        axis (str, optional): Axis along which to evaluate gradient metrics.
            Must be one of 'x', 'y', or 'z'. Defaults to 'x'.
    Returns:
        None
    Raises:
        ValueError: If axis is not one of 'x', 'y', or 'z'.
    Prints:
        - Gradient efficiency (mT/m/A): Linear fit slope along specified axis.
        - Max linearity error within DSV (%): Maximum deviation from linear fit
            expressed as a percentage of the maximum fitted value.
    Note:
        The function assumes the measurement points within the designated
        sampling volume (DSV) are those within ±1 mm of the respective axis.
    """


    x = coords[:, 0]
    y = coords[:, 1]
    z = coords[:, 2]

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(x, y, z, c=Bz * 1e3, cmap='jet')
    plt.colorbar(sc, label='Gradient Field (mT)')
    ax.set_title('Optimized: ' + axis + ' Magnetic Gradient Field')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    plt.show()

    # Evaluate along X-axis
    if axis == 'x':
        axis_mask = (np.abs(y) < 1e-3) & (np.abs(z) < 1e-3)
        x_axis = x[axis_mask]
        g_axis = Bz[axis_mask]
        
        coeffs = np.polyfit(x_axis, g_axis, 1)
        slope = coeffs[0]

        print("\n===== X GRADIENT PERFORMANCE =====")
        print(f"Gradient efficiency (mT/m/A): {slope*1e3:.3f}")

        linear_fit = np.polyval(coeffs, x_axis)
        error = (g_axis - linear_fit) / np.max(np.abs(linear_fit)) * 100
        print(f"Max linearity error within DSV (%): {np.max(np.abs(error)):.2f}")

    elif axis == 'y':
        axis_mask = (np.abs(x) < 1e-3) & (np.abs(z) < 1e-3)
        y_axis = y[axis_mask]
        g_axis = Bz[axis_mask]
        
        coeffs = np.polyfit(y_axis, g_axis, 1)
        slope = coeffs[0]

        print("\n===== Y GRADIENT PERFORMANCE =====")
        print(f"Gradient efficiency (mT/m/A): {slope*1e3:.3f}")

        linear_fit = np.polyval(coeffs, y_axis)
        error = (g_axis - linear_fit) / np.max(np.abs(linear_fit)) * 100
        print(f"Max linearity error within DSV (%): {np.max(np.abs(error)):.2f}")

    elif axis == 'z':
        axis_mask = (np.abs(x) < 1e-3) & (np.abs(y) < 1e-3)
        z_axis = z[axis_mask]
        g_axis = Bz[axis_mask]
        
        coeffs = np.polyfit(z_axis, g_axis, 1)
        slope = coeffs[0]

        print("\n===== Z GRADIENT PERFORMANCE =====")
        print(f"Gradient efficiency (mT/m/A): {slope*1e3:.3f}")

        linear_fit = np.polyval(coeffs, z_axis)
        error = (g_axis - linear_fit) / np.max(np.abs(linear_fit)) * 100
        print(f"Max linearity error within DSV (%): {np.max(np.abs(error)):.2f}")