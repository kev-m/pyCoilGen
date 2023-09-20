import logging
import numpy as np
from typing import List
from os import makedirs

import matplotlib.pyplot as plt

from pyCoilGen.pyCoilGen_release import pyCoilGen
from pyCoilGen.sub_functions.constants import DEBUG_BASIC
from pyCoilGen.sub_functions.data_structures import CoilSolution, SolutionErrors, FieldErrors, TargetField
from pyCoilGen.helpers.persistence import load

# Plotting
import pyCoilGen.plotting as pcg_plt


log = logging.getLogger(__name__)
# logging.basicConfig(level=logging.DEBUG)
logging.basicConfig(level=logging.INFO)

# Change the default values to suit your application
arg_dict = {
    'field_shape_function': 'y',            # % definition of the target field ['x']
    'coil_mesh_file': 'cylinder_radius500mm_length1500mm.stl',
    'secondary_target_weight': 0.5,         # [1.0]
    'target_region_resolution': 10,        # MATLAB 10 is the default
    'levels': 20,                           # The number of potential steps, determines the number of windings [10]
    # a potential offset value for the minimal and maximal contour potential [0.5]
    'pot_offset_factor': 0.25,
    'interconnection_cut_width': 0.1,       # Width cut used when cutting and joining wire paths; in metres [0.01]
    # Displacement that overlapping return paths will be shifted along the surface normals; in meter [0.001]
    'normal_shift_length': 0.025,
    'iteration_num_mesh_refinement': 1,     # % the number of refinements for the mesh; [0]
    'set_roi_into_mesh_center': True,       # [False]
    'force_cut_selection': ['high'],        # []
    'make_cylindrical_pcb': True,           # [False]
    'conductor_cross_section_width': 0.015,  # [0.002]
    'tikhonov_reg_factor': 100,             # Tikhonov regularization factor for the SF optimization [1]

    'output_directory': 'images',           # [Current directory]
    'project_name': 'ygradient_cylinder',
    'fasthenry_bin': '../FastHenry2/bin/fasthenry',  # [/usr/bin/fasthenry']
    'persistence_dir': 'debug',             # [debug]
    # 'debug': DEBUG_VERBOSE,
    'debug': DEBUG_BASIC,                   # [0 = NONE]
}

# solution = pyCoilGen(log, arg_dict) # Calculate the solution
# which = arg_dict['project_name']
# Calculate the errors
# [loaded] = oad('debug', which, 'final')
# solution = load('debug', 'biplanar_xgradient', 'final')
# solution = load('debug', 'Preoptimzed_SVD_Coil', 'final')
# solution = load('debug', 'Preoptimzed_Breast_Coil', 'final')
solution = load('debug', 's2_shim_coil', 'final')
# solution = load('debug', 'shielded_ygradient_coil', 'final')
# print(solution.input_args)
which = solution.input_args.project_name
save_dir = f'{solution.input_args.output_directory}'
makedirs(save_dir, exist_ok=True)

coil_solutions = [solution]
# pcg_plt.plot_error_different_solutions(coil_solutions, [0], 'gradient study')
pcg_plt.plot_various_error_metrics(coil_solutions, 0, f'{which}', save_dir=save_dir)
pcg_plt.plot_2D_contours_with_sf(coil_solutions, 0, f'{which} 2D', save_dir=save_dir)
pcg_plt.plot_3D_contours_with_sf(coil_solutions, 0, f'{which} 3D', save_dir=save_dir)

# Plot vector fields
coords = solution.target_field.coords

plot_title=f'{which} Target Field '
field = solution.solution_errors.combined_field_layout
pcg_plt.plot_vector_field_xy(coords, field, plot_title=plot_title, save_dir=save_dir)
# pcg_plt.plot_vector_field_yz(coords, field, plot_title=plot_title, save_dir=save_dir)
# pcg_plt.plot_vector_field_xz(coords, field, plot_title=plot_title, save_dir=save_dir)

plot_title=f'{which} Target Field Error '
field = solution.solution_errors.combined_field_layout - solution.target_field.b
pcg_plt.plot_vector_field_xy(coords, field, plot_title=plot_title, save_dir=save_dir)
# pcg_plt.plot_vector_field_yz(coords, field, plot_title=plot_title, save_dir=save_dir)
# pcg_plt.plot_vector_field_xz(coords, field, plot_title=plot_title, save_dir=save_dir)
