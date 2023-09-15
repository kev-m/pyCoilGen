# System imports
import sys
from pathlib import Path

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


"""
Author: Philipp Amrein, University Freiburg, Medical Center, Radiology,
Medical Physics
February 2022

This scripts generates a "S2" shimming coil on a cylindrical support with 
four rectangular openings
"""

if __name__ == '__main__':
    # Set up logging
    log = logging.getLogger(__name__)
    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
    # logging.basicConfig(level=logging.INFO)

    arg_dict = {
        'field_shape_function': '2*x*y',  # definition of the target field
        'coil_mesh_file': 'cylinder_radius500mm_length1500mm_regular_holes.stl',
        'target_mesh_file': 'none',
        'secondary_target_mesh_file': 'none',
        'secondary_target_weight': 0.5,
        'target_region_radius': 0.15,  # in meter
        'use_only_target_mesh_verts': False,
        'sf_source_file': 'none',
        # the number of potential steps that determines the later number of windings (Stream function discretization)
        'levels': 14,
        'pot_offset_factor': 0.25,  # a potential offset value for the minimal and maximal contour potential ; must be between 0 and 1
        'surface_is_cylinder_flag': True,
        'interconnection_cut_width': 0.05,  # the width for the interconnections are interconnected; in meter
        'normal_shift_length': 0.01,  # the length for which overlapping return paths will be shifted along the surface normals; in meter
        'iteration_num_mesh_refinement': 0,  # the number of refinements for the mesh;
        'set_roi_into_mesh_center': True,
        'skip_normal_shift': False,
        'force_cut_selection': ['high' 'high' 'high' 'high' 'low' 'low' 'low' 'low'],
        'level_set_method': 'primary',  # Specify one of the three ways the level sets are calculated: "primary","combined", or "independent"
        'interconnection_method': 'regular',
        'skip_postprocessing': False,
        'skip_inductance_calculation': False,
        'conductor_thickness': 0.01,
        'smooth_flag': False,
        'tikhonov_reg_factor': 10,  # Tikhonov regularization factor for the SF optimization


        'output_directory': 'images',
        'project_name': 'shielded_ygradient_coil',
        'fasthenry_bin': '../FastHenry2/bin/fasthenry',
        'persistence_dir': 'debug',
        'debug': DEBUG_BASIC,
    }

    result = pyCoilGen(log, arg_dict)


"""
Load geometry:
DEBUG:sub_functions.read_mesh:Loading mesh: cylinder_radius500mm_length1500mm_regular_holes.stl
DEBUG:sub_functions.read_mesh:Loading STL
DEBUG:trimesh:face_normals didn't match triangles, ignoring!
DEBUG:trimesh:loaded <trimesh.Trimesh(vertices.shape=(2196, 3), faces.shape=(4184, 3))> using `load_stl` in 0.0057s
WARNING:sub_functions.read_mesh: Loaded mesh from STL. Assuming shape representative normal is [0,0,1]!
Split the mesh and the stream function into disconnected pieces.
DEBUG:helpers.timing:Elapsed time: 0.025174 seconds
Upsample the mesh by subdivision:
DEBUG:sub_functions.refine_mesh: - iteration_num_mesh_refinement: 0
DEBUG:helpers.timing:Elapsed time: 0.000329 seconds
Parameterize the mesh:
DEBUG:helpers.timing:Elapsed time: 0.013936 seconds
Define the target field:
DEBUG:sub_functions.define_target_field: - dbzdx_fun: 2*y
DEBUG:sub_functions.define_target_field: - dbzdy_fun: 2*x
DEBUG:sub_functions.define_target_field: - dbzdz_fun: 0
DEBUG:helpers.timing:Elapsed time: 0.045650 seconds
Calculate mesh one ring:
DEBUG:helpers.timing:Elapsed time: 1.002726 seconds
Create the basis function container which represents the current density:
DEBUG:helpers.timing:Elapsed time: 2.269126 seconds
Calculate the sensitivity matrix:
DEBUG:helpers.timing:Elapsed time: 10.509686 seconds
Calculate the gradient sensitivity matrix:
DEBUG:helpers.timing:Elapsed time: 14.924493 seconds
Calculate the resistance matrix:
DEBUG:helpers.timing:Elapsed time: 1.232278 seconds
Optimize the stream function toward target field and secondary constraints:
DEBUG:helpers.timing:Elapsed time: 267.154361 seconds
Calculate the potential levels for the discretization:
DEBUG:helpers.timing:Elapsed time: 0.000179 seconds
Generate the contours:
DEBUG:helpers.timing:Elapsed time: 0.936881 seconds
Process contours: Evaluate loop significance
DEBUG:helpers.timing:Elapsed time: 6.898909 seconds
Find the minimal distance between the contour lines:
DEBUG:helpers.timing:Elapsed time: 0.000028 seconds
Group the contour loops in topological order:
DEBUG:helpers.timing:Elapsed time: 5.659241 seconds
Calculate center locations of groups:
DEBUG:helpers.timing:Elapsed time: 0.020656 seconds
Interconnect the single groups:
DEBUG:helpers.timing:Elapsed time: 0.105890 seconds
Interconnect the groups to a single wire path:
DEBUG:helpers.timing:Elapsed time: 2.374159 seconds
Shift the return paths over the surface:
DEBUG:helpers.timing:Elapsed time: 6.774546 seconds
Create PCB Print:
DEBUG:helpers.timing:Elapsed time: 0.000144 seconds
Create sweep along surface:
DEBUG:trimesh:Exporting 62082 faces as STL
DEBUG:trimesh:Exporting 4184 faces as STL
DEBUG:helpers.timing:Elapsed time: 7.518816 seconds
Calculate the inductance by coil layout:
DEBUG:helpers.timing:Elapsed time: 0.055387 seconds
Evaluate the field errors:
DEBUG:helpers.timing:Elapsed time: 2.129279 seconds
Calculate the gradient:
pyCoilGen/examples/../sub_functions/calculate_gradient.py:56: RuntimeWarning: invalid value encountered in true_divide
  gradient_direction /= np.linalg.norm(gradient_direction)
pyCoilGen/examples/../sub_functions/calculate_gradient.py:62: RuntimeWarning: Mean of empty slice
  layout_gradient.mean_gradient_in_target_direction = np.nanmean(layout_gradient.gradient_in_target_direction)
pyCoilGen/env/lib/python3.7/site-packages/numpy/lib/nanfunctions.py:1671: RuntimeWarning: Degrees of freedom <= 0 for slice.
  keepdims=keepdims)
DEBUG:helpers.timing:Elapsed time: 1.490430 seconds
DEBUG:helpers.timing:Total elapsed time: 331.554300 seconds
"""