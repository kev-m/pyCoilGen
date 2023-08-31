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
        'min_loop_significance': 5,
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

"""
Timing information (MATLAB online server):
Parse inputs:
Elapsed time is 0.016330 seconds.
Load preoptimized data:
Split the mesh and the stream function into disconnected pieces.
Elapsed time is 0.027680 seconds.
Parameterize the mesh:
Elapsed time is 0.048049 seconds.
Elapsed time is 0.048924 seconds.
Calculate the potential levels for the discretization:
Elapsed time is 0.004498 seconds.
Generate the contours:
Elapsed time is 1.082546 seconds.
Process contours: Evaluate loop significance
Elapsed time is 59.712511 seconds.
Find the minimal distance between the contour lines:
Elapsed time is 0.005178 seconds.
Group the contour loops in topological order:
Elapsed time is 9.266655 seconds.
Calculate center locations of groups:
Elapsed time is 0.042878 seconds.
Interconnect the single groups:
Elapsed time is 0.254066 seconds.
Interconnect the groups to a single wire path:
Elapsed time is 5.473644 seconds.
Shift the return paths over the surface:
Elapsed time is 6.134210 seconds.
Create PCB Print:
Elapsed time is 0.008580 seconds.
Generate volumetric coil body:
Elapsed time is 6.501566 seconds.

filename = 

    "debug/Preoptimzed_SVD_Coil.mat"

>> 
"""

"""

"""