import numpy as np

# System imports
import sys

# Logging
import logging

# Local imports
from pyCoilGen.pyCoilGen_release import pyCoilGen
from pyCoilGen.sub_functions.constants import DEBUG_BASIC, DEBUG_VERBOSE

if __name__ == '__main__':
    # Set up logging
    log = logging.getLogger(__name__)
    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
    # logging.basicConfig(level=logging.INFO)


    tikhonov_factor = 10000
    num_levels = 30
    pcb_width = 0.002
    cut_width = 0.025
    normal_shift = 0.006
    min_loop_significance = 3

    circular_resolution = 10
    conductor_width = 0.0015
    theta = np.linspace(0, 2*np.pi, circular_resolution)
    cross_sectional_points = np.array([np.sin(theta), np.cos(theta)]) * conductor_width
    normal_shift_smooth_factors = [5, 5, 5]

    # Define the parameters as a dictionary
    parameters = {
        'field_shape_function': '0.2500000000000001*x + 0.7694208842938134*y + 0.5877852522924731*z',
        'coil_mesh_file': 'create cylinder mesh',
        'cylinder_mesh_parameter_list': [0.4913, 0.154, 50, 50, 0, 1, 0, np.pi/2],
        'surface_is_cylinder_flag': True,
        'min_loop_significance': min_loop_significance,
        'target_region_radius': 0.1,
        'levels': num_levels,
        'pot_offset_factor': 0.25,
        'interconnection_cut_width': cut_width,
        'conductor_cross_section_width': pcb_width,
        'normal_shift_length': normal_shift,
        'skip_postprocessing': False,
        'make_cylindrical_pcb': True,
        'skip_inductance_calculation': False,
        'cross_sectional_points': cross_sectional_points,
        'normal_shift_smooth_factors': normal_shift_smooth_factors,
        # 'smooth_flag': True,
        'smooth_factor': 2,
        'save_stl_flag': True,
        'tikhonov_reg_factor': tikhonov_factor,

        'output_directory': 'images',  # [Current directory]
        'project_name': 'find_group_cut_test', # See https://github.com/kev-m/pyCoilGen/issues/60
        'persistence_dir': 'debug',
        'debug': DEBUG_BASIC,

        'sf_dest_file' : 'test_find_group_cut',
        'sf_source_file' : 'test_find_group_cut'

    }

    # Run the algorithm with the given parameters (CoilGen function is not provided here)
    result = pyCoilGen(log, parameters)