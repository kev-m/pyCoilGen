# System imports
import numpy as np


# Logging
import logging

# Local imports
# Add the sub_functions directory to the Python module search path
import sys
from pathlib import Path
sub_functions_path = Path(__file__).resolve().parent / '..'
sys.path.append(str(sub_functions_path))
from CoilGen import CoilGen
from sub_functions.constants import DEBUG_BASIC, DEBUG_VERBOSE

"""
Autor: Philipp Amrein, University Freiburg, Medical Center, Radiology,
Medical Physics
February 2022

This genearets a targeted SVD coil for the human brain. An already optimized solution for the stream function is
loaded.

For the background of this project refer to: Design of a shim coil array matched to the human brain anatomy
Feng Jia, Hatem Elshatlawy, Ali Aghaeifar, Ying-Hua Chu, Yi-Cheng Hsu, Sebastian Littin, Stefan Kroboth, Huijun Yu, 
Philipp Amrein, Xiang Gao, Wenchao Yang, Pierre LeVan, Klaus Scheffler, Maxim Zaitsev

"""

if __name__ == '__main__':
    # Set up logging
    log = logging.getLogger(__name__)
    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
    # logging.basicConfig(level=logging.INFO)

    arg_dict = {
        'field_shape_function': 'none',  # definition of the target field
        'coil_mesh_file': 'none',
        'use_only_target_mesh_verts': False,
        'sf_source_file': 'source_data_SVD_coil.npy',
        # the number of potential steps that determines the later number of windings (Stream function discretization)
        'levels': 30,
        'min_loop_signifcance': 5,
        'pot_offset_factor': 0.25,  # a potential offset value for the minimal and maximal contour potential ; must be between 0 and 1
        'surface_is_cylinder_flag': True,
        'interconnection_cut_width': 0.01,  # the width for the interconnections are interconnected; in meter
        'normal_shift_length': 0,  # the length for which overlapping return paths will be shifted along the surface normals; in meter
        'level_set_method': 'primary',  # Specify one of the three ways the level sets are calculated: "primary","combined", or "independent"
        'interconnection_method': 'regular',
        'skip_postprocessing': False,
        'skip_inductance_calculation': False,

        'project_name': 'Preoptimzed_SVD_Coil',
        'fasthenry_bin': '../FastHenry2/bin/fasthenry',
        'persistence_dir': 'debug',
        'output_directory': 'images',
        'debug': DEBUG_BASIC,
    }

    result = CoilGen(log, arg_dict)
