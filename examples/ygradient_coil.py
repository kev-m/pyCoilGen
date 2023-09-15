# System imports
import sys
from pathlib import Path
import numpy as np

# Logging
import logging

#######################################################################
# Add the sub_functions directory to the Python module search path
# Only required for the development environment
import sys
from pathlib import Path
sub_functions_path = Path(__file__).resolve().parent / '..'
sys.path.append(str(sub_functions_path))
#
#######################################################################


## Local imports
from pyCoilGen.pyCoilGen_release import pyCoilGen
from pyCoilGen.sub_functions.constants import DEBUG_BASIC, DEBUG_VERBOSE


if __name__ == '__main__':
    # Set up logging
    log = logging.getLogger(__name__)
    logging.basicConfig(level=logging.DEBUG)
    # logging.basicConfig(level=logging.INFO)

    arg_dict = {
        'field_shape_function': 'y',  # % definition of the target field
        'coil_mesh_file': 'cylinder_radius500mm_length1500mm.stl',
        'target_mesh_file': 'none',
        'secondary_target_mesh_file': 'none',
        'secondary_target_weight': 0.5,
        'target_region_radius': 0.15,  # ...  % in meter
        # 'target_region_resolution': 5,  # MATLAB 10 is the default
        'use_only_target_mesh_verts': False,
        'sf_source_file': 'none',
        # % the number of potential steps that determines the later number of windings (Stream function discretization)
        'levels': 20,
        'pot_offset_factor': 0.25,  # % a potential offset value for the minimal and maximal contour potential ; must be between 0 and 1
        'surface_is_cylinder_flag': True,
        'interconnection_cut_width': 0.1,  # % the width for the interconnections are interconnected; in meter
        'normal_shift_length': 0.025,  # % the length for which overlapping return paths will be shifted along the surface normals; in meter
        'iteration_num_mesh_refinement': 1,  # % the number of refinements for the mesh;
        'set_roi_into_mesh_center': True,
        'force_cut_selection': ['high'],  # ...
        'level_set_method': 'primary',  # ... %Specify one of the three ways the level sets are calculated: "primary","combined", or "independent"
        'interconnection_method': 'regular',
        'skip_postprocessing': False,
        'skip_inductance_calculation': False,
        'make_cylndrical_pcb': True,
        'conductor_cross_section_width': 0.015,
        'cross_sectional_points': np.array([np.sin(np.linspace(0, 2 * np.pi, 10)),
                                     np.cos(np.linspace(0, 2 * np.pi, 10))]) * 0.01,
        'sf_opt_method': 'tikhonov', # ...
        'fmincon_parameter': [1000.0, 10 ^ 10, 1.000000e-10, 1.000000e-10, 1.000000e-10],
        'tikhonov_reg_factor': 100,  # %Tikhonov regularization factor for the SF optimization

        'output_directory': 'images',
        'project_name': 'ygradient_coil',
        'fasthenry_bin': '../FastHenry2/bin/fasthenry',
        'persistence_dir': 'debug',
        #'debug': DEBUG_VERBOSE,
        'debug': DEBUG_BASIC,
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