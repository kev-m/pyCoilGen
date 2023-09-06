# System imports
import sys
from pathlib import Path

import numpy as np

# Logging
import logging

# Local imports
# Add the sub_functions directory to the Python module search path
sub_functions_path = Path(__file__).resolve().parent / '..'
sys.path.append(str(sub_functions_path))
from sub_functions.constants import DEBUG_BASIC, DEBUG_VERBOSE
from CoilGen import CoilGen


"""
Autor: Philipp Amrein, University Freiburg, Medical Center, Radiology,
Medical Physics
February 2022

This genearets a diffusion weighting MR gradient coil for the female breast. An already optimized solution for the
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
        'field_shape_function':'none', # definition of the target field
        'coil_mesh_file':'none',    
        # 'min_loop_significance':1,
        'use_only_target_mesh_verts':False,
        'sf_source_file':'source_data_breast_coil.npy',
        'levels':14, # the number of potential steps that determines the later number of windings (Stream function discretization)
        'pot_offset_factor':0.25, # a potential offset value for the minimal and maximal contour potential ; must be between 0 and 1
        'surface_is_cylinder_flag':False,
        'interconnection_cut_width':0.01, # the width for the interconnections are interconnected; in meter
        'normal_shift_length':0.01, # the length for which overlapping return paths will be shifted along the surface normals; in meter
        'force_cut_selection':['high'],
        'level_set_method':'primary',  #Specify one of the three ways the level sets are calculated: "primary","combined", or "independent"
        'interconnection_method':'regular',
        'skip_postprocessing':False,
        'cross_sectional_points': cross_section,
        'skip_sweep':False,
        'skip_inductance_calculation':False,

        'project_name': 'Preoptimzed_Breast_Coil',
        'fasthenry_bin': '../FastHenry2/bin/fasthenry',
        'persistence_dir': 'debug',
        'output_directory': 'images',
        'debug': DEBUG_BASIC,
    }

    result = CoilGen(log, arg_dict)


"""
Load preoptimized data:
INFO:sub_functions.load_preoptimized_data:Split the mesh and the stream function into disconnected pieces.
DEBUG:helpers.timing:Total elapsed time: 0.047636 seconds
INFO:sub_functions.load_preoptimized_data:Parameterize the mesh:
DEBUG:sub_functions.parameterize_mesh: - processing 0, vertices shape: (4343, 3)
DEBUG:sub_functions.parameterize_mesh: - max_face_normal: [0.5369551209704353, 0.28360071350272237, 0.5590848385540563], max_face_normal_std: 0.5590848385540563
CoilGen-Python/env/lib/python3.7/site-packages/scipy/sparse/linalg/dsolve/linsolve.py:145: SparseEfficiencyWarning: spsolve requires A be CSC or CSR matrix format
  SparseEfficiencyWarning)
DEBUG:helpers.timing:Total elapsed time: 2.358559 seconds
DEBUG:helpers.timing:Elapsed time: 2.409311 seconds
Calculate the potential levels for the discretization:
DEBUG:helpers.timing:Elapsed time: 0.000088 seconds
DEBUG:CoilGen:Saving solution to 'debug/Preoptimzed_Breast_Coil_09.npy'
Generate the contours:
DEBUG:helpers.timing:Elapsed time: 0.746055 seconds
DEBUG:CoilGen:Saving solution to 'debug/Preoptimzed_Breast_Coil_10.npy'
Process contours: Evaluate loop significance
DEBUG:helpers.timing:Elapsed time: 6.264593 seconds
DEBUG:CoilGen:Saving solution to 'debug/Preoptimzed_Breast_Coil_11.npy'
Find the minimal distance between the contour lines:
DEBUG:helpers.timing:Elapsed time: 0.000010 seconds
DEBUG:CoilGen:Saving solution to 'debug/Preoptimzed_Breast_Coil_12.npy'
Group the contour loops in topological order:
DEBUG:helpers.timing:Elapsed time: 0.003526 seconds
DEBUG:CoilGen:Saving solution to 'debug/Preoptimzed_Breast_Coil_13.npy'
Calculate center locations of groups:
DEBUG:helpers.timing:Elapsed time: 0.016229 seconds
DEBUG:CoilGen:Saving solution to 'debug/Preoptimzed_Breast_Coil_14.npy'
Interconnect the single groups:
DEBUG:helpers.timing:Elapsed time: 0.000604 seconds
DEBUG:CoilGen:Saving solution to 'debug/Preoptimzed_Breast_Coil_15.npy'
Interconnect the groups to a single wire path:
CoilGen-Python/env/lib/python3.7/site-packages/numpy/core/fromnumeric.py:3441: RuntimeWarning: Mean of empty slice.
  out=out, **kwargs)
CoilGen-Python/env/lib/python3.7/site-packages/numpy/core/_methods.py:189: RuntimeWarning: invalid value encountered in double_scalars
  ret = ret.dtype.type(ret / rcount)
DEBUG:helpers.timing:Elapsed time: 10.404342 seconds
DEBUG:CoilGen:Saving solution to 'debug/Preoptimzed_Breast_Coil_16.npy'
Shift the return paths over the surface:
DEBUG:helpers.timing:Elapsed time: 10.624149 seconds
DEBUG:CoilGen:Saving solution to 'debug/Preoptimzed_Breast_Coil_17.npy'
Create PCB Print:
DEBUG:helpers.timing:Elapsed time: 0.000068 seconds
DEBUG:CoilGen:Saving solution to 'debug/Preoptimzed_Breast_Coil_18.npy'
Create sweep along surface:
DEBUG:trimesh:Exporting 80190 faces as STL
DEBUG:trimesh:Exporting 8574 faces as STL
DEBUG:helpers.timing:Elapsed time: 10.179749 seconds
DEBUG:CoilGen:Saving solution to 'debug/Preoptimzed_Breast_Coil_19.npy'
Calculate the inductance by coil layout:
DEBUG:helpers.timing:Elapsed time: 0.169019 seconds
DEBUG:helpers.timing:Total elapsed time: 41.522746 seconds
"""