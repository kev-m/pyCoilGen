# System imports
import sys
import numpy as np

# Logging
import logging


# Local imports
from pyCoilGen.pyCoilGen_release import pyCoilGen
from pyCoilGen.sub_functions.constants import DEBUG_BASIC, DEBUG_VERBOSE


if __name__ == '__main__':
    # Set up logging
    log = logging.getLogger(__name__)
    logging.basicConfig(level=logging.DEBUG)
    # logging.basicConfig(level=logging.INFO)

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
        'cross_sectional_points': np.array([np.sin(np.linspace(0, 2 * np.pi, 10)),
                                            np.cos(np.linspace(0, 2 * np.pi, 10))]) * 0.01,
        'tikhonov_reg_factor': 100,             # Tikhonov regularization factor for the SF optimization [1]

        'output_directory': 'images',           # [Current directory]
        'project_name': 'ygradient_coil',       # ['CoilGen']
        'fasthenry_bin': '../FastHenry2/bin/fasthenry',  # [/usr/bin/fasthenry']
        'persistence_dir': 'debug',             # [debug]
        # 'debug': DEBUG_VERBOSE,
        'debug': DEBUG_BASIC,                   # [0 = NONE]
    }

    result = pyCoilGen(log, arg_dict)

"""
Parse inputs:
Elapsed time is 0.019486 seconds.
Load geometry:
Elapsed time is 0.018189 seconds.
Split the mesh and the stream function into disconnected pieces.
Elapsed time is 0.014632 seconds.
Upsample the mesh by subdivision:
Elapsed time is 0.063404 seconds.
Parameterize the mesh:
Elapsed time is 0.033485 seconds.
Define the target field:
Elapsed time is 1.468534 seconds.
Evaluate the temp data:
Elapsed time is 0.019747 seconds.
Calculate mesh one ring:
Elapsed time is 0.124147 seconds.
Create the basis funtion container which represents the current density:
Elapsed time is 0.198522 seconds.
Calculate the sensitivity matrix:
Elapsed time is 1.810927 seconds.
Calculate the gradient sensitivity matrix:
Elapsed time is 3.663818 seconds.
Calculate the resistance matrix:
Elapsed time is 0.328228 seconds.
Optimize the stream function toward target field and secondary constraints:
Elapsed time is 0.406587 seconds.
Calculate the potential levels for the discretization:
Elapsed time is 0.004627 seconds.
Generate the contours:
Elapsed time is 0.289798 seconds.
Process contours: Evaluate loop significance
Elapsed time is 0.669709 seconds.
Find the minimal distance between the contour lines:
Elapsed time is 0.004533 seconds.
Group the contour loops in topological order:
Elapsed time is 0.619209 seconds.
Calculate center locations of groups:
Elapsed time is 0.024559 seconds.
Interconnect the single groups:
Elapsed time is 0.174792 seconds.
Interconnect the groups to a single wire path:

filename = 

    "debug/debug_single_level_ind_debugygradient_coil_1.mat"

Elapsed time is 0.416225 seconds.
Shift the return paths over the surface:
Elapsed time is 0.466666 seconds.
Create PCB Print:
Elapsed time is 0.212128 seconds.
Generate volumetric coil body:
Elapsed time is 1.049221 seconds.
Calculate the inductance with fast henry:
 FastHenry2 is not installed in the Folder- "Program Files (x86)" -", magnetic inductance will not be calculated.. 
Elapsed time is 0.059660 seconds.
Evaluate the result for the final wire track:
Elapsed time is 1.330877 seconds.
Calculate the resuting gradient field:
Elapsed time is 0.801548 seconds.

filename = 

    "debug/ygradient_coil_1_10.mat"

Elapsed time is 2.152556 seconds.
"""
