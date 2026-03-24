import numpy as np
import trimesh
from scipy.interpolate import splprep, splev

# ------------------------
# LOAD + SMOOTH PATH
# ------------------------
data = np.loadtxt('images/biplanar_xgradient_Tenacity_x_plate0_loop0.txt')

x, y = data[:, 0], data[:, 1]

def smooth_path(x, y, smoothing=5.0, n_points=300):
    tck, _ = splprep([x, y], s=smoothing, per=True)
    u = np.linspace(0, 1, n_points)
    x_new, y_new = splev(u, tck)
    return np.array(x_new), np.array(y_new)

x, y = smooth_path(x, y)

points = np.column_stack([x, y, np.zeros_like(x)])

# ------------------------
# CREATE PLATE
# ------------------------
plate = trimesh.creation.cylinder(
    radius=76.2,
    height=3,
    sections=128
)

plate.apply_translation([0, 0, 1.5])

# ------------------------
# CREATE TUBE (ROBUST)
# ------------------------
radius = 1.6
segments = []

for i in range(len(points) - 1):
    p1 = points[i]
    p2 = points[i + 1]

    segment = trimesh.creation.cylinder(
        radius=radius,
        segment=[p1, p2]
    )

    segments.append(segment)

tube = trimesh.util.concatenate(segments)

# Embed into plate
tube.apply_translation([0, 0, 2])

# ------------------------
# BOOLEAN DIFFERENCE
# ------------------------
result = plate.difference(tube)

# ------------------------
# EXPORT
# ------------------------
result.export('plate_trimesh.stl')

print("✅ STL generated with Trimesh")


