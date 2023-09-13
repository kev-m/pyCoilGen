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
from pyCoilGen import pyCoilGen
#from CoilGen_develop import CoilGen
from sub_functions.constants import DEBUG_BASIC, DEBUG_VERBOSE

"""
Author: Philipp Amrein, University Freiburg, Medical Center, Radiology,
Medical Physics
February 2022

This generates a targeted SVD coil for the human brain. An already optimized solution for the stream function is
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
        'normal_shift_length': 0.01,  # the length for which overlapping return paths will be shifted along the surface normals; in meter
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

    result = pyCoilGen(log, arg_dict)

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
Elapsed time is 9.096230 seconds.
Calculate center locations of groups:
Elapsed time is 0.103303 seconds.
Interconnect the single groups:
Elapsed time is 0.410112 seconds.
Interconnect the groups to a single wire path:
Elapsed time is 5.595537 seconds.
Shift the return paths over the surface:
Elapsed time is 6.271574 seconds.
Create PCB Print:
Elapsed time is 0.028241 seconds.
Generate volumetric coil body:
Elapsed time is 6.916382 seconds.
Calculate the inductance with fast henry:
 FastHenry2 is not installed in the Folder- "Program Files (x86)" -", magnetic inductance will not be calculated.. 
Elapsed time is 0.092656 seconds.
Evaluate the result for the final wire track:
Elapsed time is 112.334284 seconds.
Calculate the resuting gradient field:
Elapsed time is 48.145139 seconds.

filename = 

    "debug/Preoptimzed_SVD_Coil_0_10.mat"

>> 
    
"""

"""
Load preoptimized data:
INFO:sub_functions.load_preoptimized_data:Split the mesh and the stream function into disconnected pieces.
DEBUG:helpers.timing:Total elapsed time: 0.044327 seconds
INFO:sub_functions.load_preoptimized_data:Parameterize the mesh:
DEBUG:sub_functions.parameterize_mesh: - processing 0, vertices shape: (4880, 3)
DEBUG:sub_functions.parameterize_mesh: - max_face_normal: [0.7071067811721325, 0.7071067811865472, 2.6563999738451888e-08], max_face_normal_std: 0.7071067811865472
DEBUG:helpers.timing:Total elapsed time: 0.009694 seconds
DEBUG:helpers.timing:Elapsed time: 0.064414 seconds
Calculate the potential levels for the discretization:
DEBUG:helpers.timing:Elapsed time: 0.000069 seconds
DEBUG:CoilGen:Saving solution to 'debug/Preoptimzed_SVD_Coil_0_10_09.npy'
Generate the contours:
DEBUG:helpers.timing:Elapsed time: 1.507967 seconds
DEBUG:CoilGen:Saving solution to 'debug/Preoptimzed_SVD_Coil_0_10_10.npy'
Process contours: Evaluate loop significance
DEBUG:helpers.timing:Elapsed time: 87.904885 seconds
DEBUG:CoilGen:Saving solution to 'debug/Preoptimzed_SVD_Coil_0_10_11.npy'
Find the minimal distance between the contour lines:
DEBUG:helpers.timing:Elapsed time: 0.000013 seconds
DEBUG:CoilGen:Saving solution to 'debug/Preoptimzed_SVD_Coil_0_10_12.npy'
Group the contour loops in topological order:
DEBUG:helpers.timing:Elapsed time: 29.020553 seconds
DEBUG:CoilGen:Saving solution to 'debug/Preoptimzed_SVD_Coil_0_10_13.npy'
Calculate center locations of groups:
DEBUG:helpers.timing:Elapsed time: 0.073859 seconds
DEBUG:CoilGen:Saving solution to 'debug/Preoptimzed_SVD_Coil_0_10_14.npy'
Interconnect the single groups:
DEBUG:helpers.timing:Elapsed time: 0.127210 seconds
DEBUG:CoilGen:Saving solution to 'debug/Preoptimzed_SVD_Coil_0_10_15.npy'
Interconnect the groups to a single wire path:
CoilGen-Python/env/lib/python3.7/site-packages/numpy/core/fromnumeric.py:3441: RuntimeWarning: Mean of empty slice.
  out=out, **kwargs)
CoilGen-Python/env/lib/python3.7/site-packages/numpy/core/_methods.py:189: RuntimeWarning: invalid value encountered in double_scalars
  ret = ret.dtype.type(ret / rcount)
DEBUG:helpers.timing:Elapsed time: 6.859388 seconds
DEBUG:CoilGen:Saving solution to 'debug/Preoptimzed_SVD_Coil_0_10_16.npy'
Shift the return paths over the surface:
DEBUG:helpers.timing:Elapsed time: 49.813668 seconds
DEBUG:CoilGen:Saving solution to 'debug/Preoptimzed_SVD_Coil_0_10_17.npy'
Create PCB Print:
DEBUG:helpers.timing:Elapsed time: 0.000138 seconds
DEBUG:CoilGen:Saving solution to 'debug/Preoptimzed_SVD_Coil_0_10_18.npy'
Create sweep along surface:
DEBUG:trimesh:Exporting 237654 faces as STL
DEBUG:trimesh:Exporting 9600 faces as STL
DEBUG:helpers.timing:Elapsed time: 33.366084 seconds
DEBUG:CoilGen:Saving solution to 'debug/Preoptimzed_SVD_Coil_0_10_19.npy'
Calculate the inductance by coil layout:
DEBUG:helpers.timing:Elapsed time: 0.590177 seconds
DEBUG:CoilGen:Saving solution to 'debug/Preoptimzed_SVD_Coil_0_10_20.npy'
Evaluate the field errors:
DEBUG:helpers.timing:Elapsed time: 112.239657 seconds
DEBUG:CoilGen:Saving solution to 'debug/Preoptimzed_SVD_Coil_0_10_21.npy'
DEBUG:helpers.timing:Total elapsed time: 326.756189 seconds
"""