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
        'persistence_dir': 'debug',             # [debug]
        # 'debug': DEBUG_VERBOSE,
        'debug': DEBUG_BASIC,                   # [0 = NONE]
    }

    result = pyCoilGen(log, arg_dict)
