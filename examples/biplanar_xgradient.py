# System imports
import sys

# Logging
import logging

## Local imports
from pyCoilGen.pyCoilGen_release import pyCoilGen
from pyCoilGen.sub_functions.constants import DEBUG_BASIC, DEBUG_VERBOSE

if __name__ == '__main__':
    # Set up logging
    log = logging.getLogger(__name__)
    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
    #logging.basicConfig(level=logging.INFO)

    arg_dict = {
        'field_shape_function': 'x',  # definition of the target field
        'coil_mesh_file': 'bi_planer_rectangles_width_1000mm_distance_500mm.stl',
        'target_mesh_file': 'none',
        'secondary_target_mesh_file': 'none',
        'secondary_target_weight': 0.5,
        'target_region_radius': 0.1,  # in meter
        # 'target_region_resolution': 10,  # MATLAB 10 is the default
        'use_only_target_mesh_verts': False,
        'sf_source_file': 'none',
        # the number of potential steps that determines the later number of windings (Stream function discretization)
        'levels': 14,
        # a potential offset value for the minimal and maximal contour potential ; must be between 0 and 1
        'pot_offset_factor': 0.25,
        'surface_is_cylinder_flag': True,
        # the width for the interconnections are interconnected; in meter
        'interconnection_cut_width': 0.05,
        # the length for which overlapping return paths will be shifted along the surface normals; in meter
        'normal_shift_length': 0.01,
        'iteration_num_mesh_refinement': 1,  # the number of refinements for the mesh;
        'set_roi_into_mesh_center': True,
        'force_cut_selection': ['high'],
        # Specify one of the three ways the level sets are calculated: "primary","combined", or "independent"
        'level_set_method': 'primary',
        'interconnection_method': 'regular',
        'skip_postprocessing': False,
        'skip_inductance_calculation': False,
        'tikhonov_reg_factor': 10,  # Tikhonov regularization factor for the SF optimization

        'output_directory': 'images',
        'project_name': 'biplanar_xgradient',
        'fasthenry_bin': '../FastHenry2/bin/fasthenry',
        'persistence_dir': 'debug',
        'debug': DEBUG_BASIC,
    }

    result = pyCoilGen(log, arg_dict)

"""
Load geometry:
DEBUG:sub_functions.read_mesh:Loading mesh: bi_planer_rectangles_width_1000mm_distance_500mm.stl
DEBUG:sub_functions.read_mesh:Loading STL
DEBUG:trimesh:face_normals didn't match triangles, ignoring!
DEBUG:trimesh:loaded <trimesh.Trimesh(vertices.shape=(578, 3), faces.shape=(1024, 3))> using `load_stl` in 0.0017s
WARNING:sub_functions.read_mesh: Loaded mesh from STL. Assuming shape representative normal is [0,0,1]!
Split the mesh and the stream function into disconnected pieces.
DEBUG:sub_functions.split_disconnected_mesh:Faces need be adjusted
DEBUG:sub_functions.split_disconnected_mesh:Faces need be adjusted
DEBUG:helpers.timing:Elapsed time: 0.178596 seconds
DEBUG:CoilGen:Saving solution to 'debug/biplanar_xgradient_00.npy'
Upsample the mesh by subdivision:
DEBUG:sub_functions.refine_mesh: - iteration_num_mesh_refinement: 1
DEBUG:sub_functions.refine_mesh: - Refining part 0
DEBUG:sub_functions.refine_mesh: - Refining part 1
DEBUG:helpers.timing:Elapsed time: 0.001296 seconds
DEBUG:CoilGen:Saving solution to 'debug/biplanar_xgradient_01.npy'
Parameterize the mesh:
DEBUG:sub_functions.parameterize_mesh: - processing 0, vertices shape: (1089, 3)
DEBUG:sub_functions.parameterize_mesh: - max_face_normal: [0.0, 3.226634572778584e-11, 1.8311193880071151e-07], max_face_normal_std: 1.8311193880071151e-07
DEBUG:sub_functions.parameterize_mesh: - 3D mesh is already planar
DEBUG:sub_functions.parameterize_mesh: - mesh_part.uv shape: (1089, 2)
DEBUG:sub_functions.parameterize_mesh: - processing 1, vertices shape: (1089, 3)
DEBUG:sub_functions.parameterize_mesh: - max_face_normal: [0.0, 4.893789561351208e-11, 2.7798767890609095e-07], max_face_normal_std: 2.7798767890609095e-07
DEBUG:sub_functions.parameterize_mesh: - 3D mesh is already planar
DEBUG:sub_functions.parameterize_mesh: - mesh_part.uv shape: (1089, 2)
DEBUG:helpers.timing:Elapsed time: 0.006930 seconds
DEBUG:CoilGen:Saving solution to 'debug/biplanar_xgradient_02.npy'
Define the target field:
DEBUG:sub_functions.define_target_field: - dbzdx_fun: 1
DEBUG:sub_functions.define_target_field: - dbzdy_fun: 0
DEBUG:sub_functions.define_target_field: - dbzdz_fun: 0
DEBUG:helpers.timing:Elapsed time: 0.021164 seconds
DEBUG:CoilGen:Saving solution to 'debug/biplanar_xgradient_02b.npy'
Calculate mesh one ring:
DEBUG:helpers.timing:Elapsed time: 0.335712 seconds
DEBUG:CoilGen:Saving solution to 'debug/biplanar_xgradient_03.npy'
Create the basis function container which represents the current density:
DEBUG:helpers.timing:Elapsed time: 1.039374 seconds
DEBUG:CoilGen:Saving solution to 'debug/biplanar_xgradient_04.npy'
Calculate the sensitivity matrix:
DEBUG:helpers.timing:Elapsed time: 9.846714 seconds
DEBUG:CoilGen:Saving solution to 'debug/biplanar_xgradient_05.npy'
Calculate the gradient sensitivity matrix:
DEBUG:helpers.timing:Elapsed time: 13.759277 seconds
DEBUG:CoilGen:Saving solution to 'debug/biplanar_xgradient_06.npy'
Calculate the resistance matrix:
DEBUG:helpers.timing:Elapsed time: 0.601073 seconds
DEBUG:CoilGen:Saving solution to 'debug/biplanar_xgradient_07.npy'
Optimize the stream function toward target field and secondary constraints:
DEBUG:helpers.timing:Elapsed time: 241.403066 seconds
DEBUG:CoilGen:Saving solution to 'debug/biplanar_xgradient_08.npy'
Calculate the potential levels for the discretization:
DEBUG:helpers.timing:Elapsed time: 0.000214 seconds
DEBUG:CoilGen:Saving solution to 'debug/biplanar_xgradient_09.npy'
Generate the contours:
CoilGen-Python/examples/../sub_functions/calc_contours_by_triangular_potential_cuts.py:97: RuntimeWarning: divide by zero encountered in true_divide
  cut_point_distance_to_edge_node_1 = np.abs(pot_dist_to_step / edge_potential_span * edge_lengths)
DEBUG:helpers.timing:Elapsed time: 0.454257 seconds
DEBUG:CoilGen:Saving solution to 'debug/biplanar_xgradient_10.npy'
Process contours: Evaluate loop significance
DEBUG:helpers.timing:Elapsed time: 3.512877 seconds
DEBUG:CoilGen:Saving solution to 'debug/biplanar_xgradient_11.npy'
Find the minimal distance between the contour lines:
DEBUG:helpers.timing:Elapsed time: 0.000013 seconds
DEBUG:CoilGen:Saving solution to 'debug/biplanar_xgradient_12.npy'
Group the contour loops in topological order:
DEBUG:helpers.timing:Elapsed time: 0.016125 seconds
DEBUG:CoilGen:Saving solution to 'debug/biplanar_xgradient_13.npy'
Calculate center locations of groups:
DEBUG:helpers.timing:Elapsed time: 0.032338 seconds
DEBUG:CoilGen:Saving solution to 'debug/biplanar_xgradient_14.npy'
Interconnect the single groups:
DEBUG:helpers.timing:Elapsed time: 0.001819 seconds
DEBUG:CoilGen:Saving solution to 'debug/biplanar_xgradient_15.npy'
Interconnect the groups to a single wire path:
DEBUG:helpers.timing:Elapsed time: 17.554831 seconds
DEBUG:CoilGen:Saving solution to 'debug/biplanar_xgradient_16.npy'
Shift the return paths over the surface:
DEBUG:helpers.timing:Elapsed time: 3.329586 seconds
DEBUG:CoilGen:Saving solution to 'debug/biplanar_xgradient_17.npy'
Create PCB Print:
DEBUG:helpers.timing:Elapsed time: 0.000083 seconds
DEBUG:CoilGen:Saving solution to 'debug/biplanar_xgradient_18.npy'
Create sweep along surface:
DEBUG:trimesh:Exporting 32292 faces as STL
DEBUG:trimesh:Exporting 2048 faces as STL
DEBUG:trimesh:Exporting 32940 faces as STL
DEBUG:trimesh:Exporting 2048 faces as STL
DEBUG:helpers.timing:Elapsed time: 3.734822 seconds
DEBUG:CoilGen:Saving solution to 'debug/biplanar_xgradient_19.npy'
Calculate the inductance by coil layout:
DEBUG:helpers.timing:Elapsed time: 0.047963 seconds
DEBUG:helpers.timing:Total elapsed time: 318.708471 seconds
"""