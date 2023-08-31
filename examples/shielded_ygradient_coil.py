# System imports
import sys
from pathlib import Path

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
        'target_region_resolution': 9,  # MATLAB 10 is the default
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
        'iteration_num_mesh_refinement': 0,  # the number of refinements for the mesh (Was: 1)
        'set_roi_into_mesh_center': True,
        'force_cut_selection': ['high'],
        # Specify one of the three ways the level sets are calculated: "primary","combined", or "independent"
        'level_set_method': 'primary',
        'interconnection_method': 'regular',
        'skip_postprocessing': False,
        'skip_inductance_calculation': False,
        'tikonov_reg_factor': 10,           # Tikonov regularization factor for the SF optimization

        'project_name': 'shielded_ygradient_coil',
        'fasthenry_bin': '../FastHenry2/bin/fasthenry',
        'persistence_dir': 'debug',
        'debug': DEBUG_BASIC,
    }

    result = CoilGen(log, arg_dict)

    """
    Load geometry:
DEBUG:sub_functions.read_mesh:Loading mesh: Double_coaxial_open_cylinder_r1_400mm_r2_600_length_1500mm.stl
DEBUG:sub_functions.read_mesh:Loading STL
DEBUG:trimesh:loaded <trimesh.Trimesh(vertices.shape=(528, 3), faces.shape=(960, 3))> using `load_stl` in 0.0020s
WARNING:sub_functions.read_mesh: Loaded mesh from STL. Assuming shape representative normal is [0,0,1]!
DEBUG:trimesh:face_normals didn't match triangles, ignoring!
DEBUG:trimesh:loaded <trimesh.Trimesh(vertices.shape=(5248, 3), faces.shape=(10240, 3))> using `load_stl` in 0.0134s
Split the mesh and the stream function into disconnected pieces.
DEBUG:helpers.timing:Elapsed time: 0.007639 seconds
DEBUG:CoilGen:Saving solution to 'debug/shielded_ygradient_coil_00.npy'
Upsample the mesh by subdivision:
DEBUG:sub_functions.refine_mesh: - iteration_num_mesh_refinement: 0
DEBUG:helpers.timing:Elapsed time: 0.000121 seconds
DEBUG:CoilGen:Saving solution to 'debug/shielded_ygradient_coil_01.npy'
Parameterize the mesh:
DEBUG:sub_functions.parameterize_mesh: - processing 0, vertices shape: (220, 3)
DEBUG:sub_functions.parameterize_mesh: - max_face_normal: [0.7071067811865476, 0.7071067811865476, 0.0], max_face_normal_std: 0.7071067811865476
DEBUG:sub_functions.parameterize_mesh: - processing 1, vertices shape: (308, 3)
DEBUG:sub_functions.parameterize_mesh: - max_face_normal: [0.7071067811854254, 0.7071067811854254, 1.7815624485878228e-06], max_face_normal_std: 0.7071067811854254
DEBUG:helpers.timing:Elapsed time: 0.003877 seconds
DEBUG:CoilGen:Saving solution to 'debug/shielded_ygradient_coil_02.npy'
Define the target field:
DEBUG:sub_functions.define_target_field: - dbzdx_fun: 0
DEBUG:sub_functions.define_target_field: - dbzdy_fun: 1
DEBUG:sub_functions.define_target_field: - dbzdz_fun: 0
DEBUG:helpers.timing:Elapsed time: 0.015147 seconds
DEBUG:CoilGen:Saving solution to 'debug/shielded_ygradient_coil_02b.npy'
Calculate mesh one ring:
DEBUG:helpers.timing:Elapsed time: 0.082356 seconds
DEBUG:CoilGen:Saving solution to 'debug/shielded_ygradient_coil_03.npy'
Create the basis function container which represents the current density:
DEBUG:helpers.timing:Elapsed time: 0.234091 seconds
DEBUG:CoilGen:Saving solution to 'debug/shielded_ygradient_coil_04.npy'
Calculate the sensitivity matrix:
DEBUG:helpers.timing:Elapsed time: 5.054585 seconds
DEBUG:CoilGen:Saving solution to 'debug/shielded_ygradient_coil_05.npy'
Calculate the gradient sensitivity matrix:
DEBUG:helpers.timing:Elapsed time: 6.876736 seconds
DEBUG:CoilGen:Saving solution to 'debug/shielded_ygradient_coil_06.npy'
Calculate the resistance matrix:
DEBUG:helpers.timing:Elapsed time: 0.144778 seconds
DEBUG:CoilGen:Saving solution to 'debug/shielded_ygradient_coil_07.npy'
Optimize the stream function toward target field and secondary constraints:
DEBUG:helpers.timing:Elapsed time: 8.100640 seconds
DEBUG:CoilGen:Saving solution to 'debug/shielded_ygradient_coil_08.npy'
Calculate the potential levels for the discretization:
DEBUG:helpers.timing:Elapsed time: 0.000200 seconds
DEBUG:CoilGen:Saving solution to 'debug/shielded_ygradient_coil_09.npy'
Generate the contours:
DEBUG:helpers.timing:Elapsed time: 0.162524 seconds
DEBUG:CoilGen:Saving solution to 'debug/shielded_ygradient_coil_10.npy'
Process contours: Evaluate loop significance
DEBUG:helpers.timing:Elapsed time: 2.362569 seconds
DEBUG:CoilGen:Saving solution to 'debug/shielded_ygradient_coil_11.npy'
Find the minimal distance between the contour lines:
DEBUG:helpers.timing:Elapsed time: 0.000014 seconds
DEBUG:CoilGen:Saving solution to 'debug/shielded_ygradient_coil_12.npy'
Group the contour loops in topological order:
DEBUG:helpers.timing:Elapsed time: 0.026619 seconds
DEBUG:CoilGen:Saving solution to 'debug/shielded_ygradient_coil_13.npy'
Calculate center locations of groups:
DEBUG:helpers.timing:Elapsed time: 0.028737 seconds
DEBUG:CoilGen:Saving solution to 'debug/shielded_ygradient_coil_14.npy'
Interconnect the single groups:
DEBUG:helpers.timing:Elapsed time: 0.001446 seconds
DEBUG:CoilGen:Saving solution to 'debug/shielded_ygradient_coil_15.npy'
Interconnect the groups to a single wire path:
env/lib/python3.7/site-packages/numpy/core/fromnumeric.py:3441: RuntimeWarning: Mean of empty slice.
  out=out, **kwargs)
env/lib/python3.7/site-packages/numpy/core/_methods.py:189: RuntimeWarning: invalid value encountered in double_scalars
  ret = ret.dtype.type(ret / rcount)
DEBUG:helpers.timing:Elapsed time: 24.489258 seconds
DEBUG:CoilGen:Saving solution to 'debug/shielded_ygradient_coil_16.npy'
Shift the return paths over the surface:
DEBUG:helpers.timing:Elapsed time: 1.322775 seconds
DEBUG:CoilGen:Saving solution to 'debug/shielded_ygradient_coil_17.npy'
Create PCB Print:
DEBUG:helpers.timing:Elapsed time: 0.000052 seconds
DEBUG:CoilGen:Saving solution to 'debug/shielded_ygradient_coil_18.npy'
Create sweep along surface:
DEBUG:trimesh:Exporting 31482 faces as STL
DEBUG:trimesh:Exporting 400 faces as STL
DEBUG:trimesh:Exporting 3960 faces as STL
DEBUG:trimesh:Exporting 560 faces as STL
DEBUG:helpers.timing:Elapsed time: 1.464263 seconds
DEBUG:CoilGen:Saving solution to 'debug/shielded_ygradient_coil_19.npy'
Calculate the inductance by coil layout:
DEBUG:helpers.timing:Elapsed time: 0.067878 seconds
DEBUG:helpers.timing:Total elapsed time: 56.483047 seconds
"""
