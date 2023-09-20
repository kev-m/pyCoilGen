"""
Author: Kevin Meyer
Bela Pena s.p.
September 2023

Demonstrate using the plotting routines to visualise data from the s2_shim_coil example.
"""

from os import makedirs

import matplotlib.pyplot as plt
from pyCoilGen.helpers.persistence import load
import pyCoilGen.plotting as pcg_plt

solution = load('debug', 's2_shim_coil', 'final')
which = solution.input_args.project_name
save_dir = f'{solution.input_args.output_directory}'
makedirs(save_dir, exist_ok=True)

coil_solutions = [solution]

# Plot a multi-plot summary of the solution
pcg_plt.plot_various_error_metrics(coil_solutions, 0, f'{which}', save_dir=save_dir)

# Plot the 2D projection of stream function contour loops.
pcg_plt.plot_2D_contours_with_sf(coil_solutions, 0, f'{which} 2D', save_dir=save_dir)
pcg_plt.plot_3D_contours_with_sf(coil_solutions, 0, f'{which} 3D', save_dir=save_dir)

# Plot the vector fields
coords = solution.target_field.coords

# Plot the computed target field.
plot_title=f'{which} Target Field '
field = solution.solution_errors.combined_field_layout
pcg_plt.plot_vector_field_xy(coords, field, plot_title=plot_title, save_dir=save_dir)

# Plot the difference between the computed target field and the input target field.
plot_title=f'{which} Target Field Error '
field = solution.solution_errors.combined_field_layout - solution.target_field.b
pcg_plt.plot_vector_field_xy(coords, field, plot_title=plot_title, save_dir=save_dir)
