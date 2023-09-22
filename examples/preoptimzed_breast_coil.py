# System imports
import sys

import numpy as np

# Logging
import logging


# Local imports
from pyCoilGen.pyCoilGen_release import pyCoilGen
from pyCoilGen.sub_functions.constants import DEBUG_BASIC, DEBUG_VERBOSE


"""
Author: Philipp Amrein, University Freiburg, Medical Center, Radiology,
Medical Physics
February 2022

This generates a diffusion weighting MR gradient coil for the female breast. An already optimized solution for the
stream function is loaded.

For the background of this project refer to: Jia F, Littin S, Amrein P, Yu H, Magill AW, Kuder TA, Bickelhaupt 
S, Laun F, Ladd ME, Zaitsev M. Design of a high-performance non-linear gradient coil for diffusion weighted MRI
of the breast. J Magn Reson. 2021 Oct;331:107052. doi: 10.1016/j.jmr.2021.107052. Epub 2021 Aug 14. PMID: 34478997.
"""

if __name__ == '__main__':
    # Set up logging
    log = logging.getLogger(__name__)
    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
    # logging.basicConfig(level=logging.INFO)

    theta = np.arange(0, 2 * np.pi + (2 * np.pi) / (10 - 1), (2 * np.pi) / (10 - 1))
    cross_section = np.vstack((np.sin(theta), np.cos(theta)))
    cross_section *= np.array([[0.002], [0.002]])

    arg_dict = {
        'field_shape_function': 'none',  # definition of the target field
        'coil_mesh_file': 'none',
        # 'min_loop_significance':1,
        'use_only_target_mesh_verts': False,
        'sf_source_file': 'source_data_breast_coil.npy',
        # the number of potential steps that determines the later number of windings (Stream function discretization)
        'levels': 14,
        'pot_offset_factor': 0.25,  # a potential offset value for the minimal and maximal contour potential ; must be between 0 and 1
        'surface_is_cylinder_flag': False,
        'interconnection_cut_width': 0.01,  # the width for the interconnections are interconnected; in meter
        'normal_shift_length': 0.01,  # the length for which overlapping return paths will be shifted along the surface normals; in meter
        'force_cut_selection': ['high'],
        'level_set_method': 'primary',  # Specify one of the three ways the level sets are calculated: "primary","combined", or "independent"
        'skip_postprocessing': False,
        'cross_sectional_points': cross_section,
        'skip_sweep': False,
        'skip_inductance_calculation': False,

        'project_name': 'Preoptimzed_Breast_Coil',
        'persistence_dir': 'debug',
        'output_directory': 'images', # [Current directory]
        'debug': DEBUG_BASIC,
    }

    result = pyCoilGen(log, arg_dict)
