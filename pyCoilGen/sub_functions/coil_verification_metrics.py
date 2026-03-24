############################################################
# STEP 11: VERIFY COIL FIELD AND PERFORMANCE
############################################################

import numpy as np
import matplotlib.pyplot as plt
import magpylib as magpy


############################################################
# 11.1 Convert loops → current wires
############################################################

def loops_to_magpy(loops, current=1.0):

    wires = []

    for loop in loops:

        # Ensure closed loop
        if not np.allclose(loop[0], loop[-1]):
            loop = np.vstack([loop, loop[0]])

        wire = magpy.current.Line(
            current=current,
            vertices=loop
        )

        wires.append(wire)

    return wires


############################################################
# 11.2 Create sampling points inside DSV
############################################################

def create_dsv_points(radius=0.016, resolution=11):

    xs = np.linspace(-radius, radius, resolution)
    ys = np.linspace(-radius, radius, resolution)
    zs = np.linspace(-radius, radius, resolution)

    pts = []

    for x in xs:
        for y in ys:
            for z in zs:

                if x*x + y*y + z*z <= radius*radius:
                    pts.append([x,y,z])

    return np.array(pts)


############################################################
# 11.3 Compute magnetic field from wires
############################################################

def compute_field(wires, points):

    B = magpy.getB(wires, points)

    return B


############################################################
# 11.4 Fit gradient tensor
############################################################

def fit_gradient_tensor(points, B):

    x = points[:,0]
    y = points[:,1]
    z = points[:,2]

    Bz = B[:,2]

    A = np.column_stack([
        np.ones(len(points)),
        x,
        y,
        z
    ])

    coeff, *_ = np.linalg.lstsq(A, Bz, rcond=None)

    B0 = coeff[0]
    Gx = coeff[1]
    Gy = coeff[2]
    Gz = coeff[3]

    return B0, Gx, Gy, Gz


############################################################
# 11.5 Compute linearity error
############################################################

def compute_linearity(points, Bz, Gx):

    x = points[:,0]

    ideal = Gx * x

    error = Bz - ideal

    linearity = np.max(np.abs(error)) / np.max(np.abs(ideal))

    return linearity * 100


############################################################
# 11.6 Visualize gradient behavior
############################################################

def visualize_gradient(points, B):

    x = points[:,0]
    Bz = B[:,2]

    plt.figure()

    plt.scatter(x, Bz, s=10)

    plt.xlabel("x (m)")
    plt.ylabel("Bz (T)")
    plt.title("Generated Gradient Field")

    plt.grid()

    plt.show()


############################################################
# 11.7 Visualize 2D slice of field
############################################################

def visualize_field_slice(points, B):

    x = points[:,0]
    z = points[:,2]
    Bz = B[:,2]

    plt.figure()

    plt.scatter(x*1000, z*1000, c=Bz, s=40)

    plt.xlabel("x (mm)")
    plt.ylabel("z (mm)")

    plt.title("Bz field distribution")

    plt.colorbar(label="Tesla")

    plt.gca().set_aspect('equal')

    plt.show()


############################################################
# 11.8 Full evaluation routine
############################################################

def evaluate_coil_performance(loops):

    print("\nSTEP 11: Evaluating coil field performance\n")

    print("Creating current wires...")

    wires = loops_to_magpy(loops)

    print("Generating DSV sampling points...")

    points = create_dsv_points()

    print("Total sampling points:", len(points))

    print("Computing magnetic field...")

    B = compute_field(wires, points)

    print("Fitting gradient tensor...")

    B0, Gx, Gy, Gz = fit_gradient_tensor(points, B)

    Bz = B[:,2]

    linearity = compute_linearity(points, Bz, Gx)

    print("\n========= COIL PERFORMANCE =========")

    print("Uniform field offset B0:", B0, "T")

    print("Gradient components:")

    print("  Gx:", Gx, "T/m/A")
    print("  Gy:", Gy, "T/m/A")
    print("  Gz:", Gz, "T/m/A")

    print("\nGradient efficiency:")

    print("  ", Gx*1000, "mT/m/A")

    print("\nLinearity error inside DSV:")

    print("  ", linearity, "%")

    print("====================================\n")

    visualize_gradient(points, B)

    visualize_field_slice(points, B)

    performance = {
        "B0": B0,
        "Gx": Gx,   
        "Gy": Gy,
        "Gz": Gz,
        "linearity_error_percent": linearity
    }

    return {
        "performance": performance
    }


