# System imports
import sys

# Logging
import logging

# Local imports
from pyCoilGen.pyCoilGen_release import pyCoilGen
from pyCoilGen.sub_functions.constants import DEBUG_BASIC, DEBUG_VERBOSE


"""
Author: Philipp Amrein, University Freiburg, Medical Center, Radiology,
Medical Physics
February 2022

This scripts generates a shielded y gradient coil
"""

if __name__ == '__main__':
    # Set up logging
    log = logging.getLogger(__name__)
    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
    # logging.basicConfig(level=logging.INFO)

    arg_dict = {
        'field_shape_function': 'y',  # definition of the target field
        'coil_mesh_file': 'Double_coaxial_open_cylinder_r1_400mm_r2_600_length_1500mm.stl',
        'target_mesh_file': 'none',
        'target_region_resolution': 10,  # MATLAB 10 is the default
        'secondary_target_mesh_file': 'Open_cylinder_r750mm_length_1500mm.stl',
        'secondary_target_weight': 0.5,
        'target_region_radius': 0.15,  # in meter
        'use_only_target_mesh_verts': False,
        'sf_source_file': 'none',
        # the number of potential steps that determines the later number of windings (Stream function discretization)
        'levels': 26,
        'pot_offset_factor': 0.25,           # a potential offset value for the minimal and maximal contour potential ; must be between 0 and 1
        'surface_is_cylinder_flag': True,
        'interconnection_cut_width': 0.05,   # the width for the interconnections are interconnected; in meter
        'normal_shift_length': 0.01,         # the length for which overlapping return paths will be shifted along the surface normals; in meter
        'iteration_num_mesh_refinement': 1,  # the number of refinements for the mesh (Was: 1)
        'set_roi_into_mesh_center': True,
        'force_cut_selection': ['high'],
        # Specify one of the three ways the level sets are calculated: "primary","combined", or "independent"
        'level_set_method': 'primary',
        'skip_postprocessing': False,
        'skip_inductance_calculation': False,
        'tikhonov_reg_factor': 10,           # Tikhonov regularization factor for the SF optimization

        'output_directory': 'images',        # [Current directory]
        'project_name': 'shielded_ygradient_coil',
        'persistence_dir': 'debug',
        'debug': DEBUG_BASIC,
    }

    result = pyCoilGen(log, arg_dict)
