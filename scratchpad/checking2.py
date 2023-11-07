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

    arg_dict1 = {
        'field_shape_function': 'x**2 + y**2',  # definition of the target field
        'coil_mesh_file': 'bi_planer_rectangles_width_1000mm_distance_500mm.stl',
        'target_region_radius': 0.1,  # in meter
        # 'target_region_resolution': 10,  # MATLAB 10 is the default
        'use_only_target_mesh_verts': False,
        'sf_source_file': 'none',
        # the number of potential steps that determines the later number of windings (Stream function discretization)
        'levels': 30,
        # a potential offset value for the minimal and maximal contour potential ; must be between 0 and 1
        'pot_offset_factor': 0.25,
        'surface_is_cylinder_flag': True,
        # the width for the interconnections are interconnected; in meter
        'interconnection_cut_width': 0.05,
        # the length for which overlapping return paths will be shifted along the surface normals; in meter
        'normal_shift_length': 0.01,
        # 'iteration_num_mesh_refinement': 1,  # the number of refinements for the mesh;
        'set_roi_into_mesh_center': True,
        'force_cut_selection': ['high'],
        # Specify one of the three ways the level sets are calculated: "primary","combined", or "independent"
        'level_set_method': 'primary',
        'skip_postprocessing': False,
        'skip_inductance_calculation': False,
        'tikhonov_reg_factor': 10,  # Tikhonov regularization factor for the SF optimization

        'sf_dest_file': 'images/loop_opening_exc/solution',  # Save pre-optimised solution

        'output_directory': 'images/loop_opening_exc',  # [Current directory]
        'project_name': 'loop_opening_exception',
        'persistence_dir': 'debug',
        'debug': DEBUG_BASIC,
    }

    arg_dict = {
        'field_shape_function': 'x',  # definition of the target field ['x']
        'coil_mesh': 'create bi-planar mesh',
        'biplanar_mesh_parameter_list': [1, 1, 30, 30, 0, 1, 0, 0, 0, 0, 0.5],
        'min_loop_significance': 3,  # [1] Remove loops if they contribute less than 3% to the target field.
        'target_region_radius': 0.125,  # [0.15] in meter
        'pot_offset_factor': 0.25,  # [0.5] a potential offset value for the minimal and maximal contour potential
        'interconnection_cut_width': 0.005,  # [0.01] the width for the interconnections are interconnected; in meter
        # the length for which overlapping return paths will be shifted along the surface normals; in meter
        'surface_is_cylinder_flag': True,
        'normal_shift_length': 0.01,  # [0.001]
        'make_cylindrical_pcb': False,  # [False]
        'save_stl_flag': True,
        'smooth_factor': 1,

        # 'tikhonov_reg_factor': 1000,  # Tikhonov regularization factor for the SF optimization
        # 'cut_plane_definition' : 'B0',
        'skip_postprocessing' : True,

        'output_directory': 'images/loop_opening_exc',  # [Current directory]
        'project_name': 'loop_opening_exception',
        'persistence_dir': 'debug',
        'debug': DEBUG_BASIC,
    }
    result = pyCoilGen(log, arg_dict)
