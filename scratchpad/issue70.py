# System imports
import sys

# Logging
import logging

# Debugging imports
# Add the sub_functions directory to the Python module search path
from pathlib import Path
sub_functions_path = Path(__file__).resolve().parent / '..'
sys.path.append(str(sub_functions_path))

# Local imports
from pyCoilGen.pyCoilGen_release import pyCoilGen
from pyCoilGen.sub_functions.constants import DEBUG_BASIC, DEBUG_VERBOSE

if __name__ == '__main__':
    # Set up logging
    log = logging.getLogger(__name__)
    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
    # logging.basicConfig(level=logging.INFO)

    arg_dict = {
        'field_shape_function': 'x',  # definition of the target field

        # 'coil_mesh_file': 'bi_planer_rectangles_width_1000mm_distance_500mm.stl',
        'coil_mesh': 'create bi-planar mesh',
        'biplanar_mesh_parameter_list': [1.0, 1.0,   # planar_height, planar_width of the planar mesh.
                                         20, 20,     # num_lateral_divisions, num_longitudinal_divisions
                                         0.0, 1.0, 0.0,    # target_normal_x, target_normal_y, target_normal_z
                                         0, 0, 0,    # center_position_x, center_position_y, center_position_z
                                         0.5],       # plane_distance (`float`): Distance between the two planes.

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
        'iteration_num_mesh_refinement': 0, # 1,  # the number of refinements for the mesh;
        'set_roi_into_mesh_center': True,
        'force_cut_selection': ['high'],
        # Specify one of the three ways the level sets are calculated: "primary","combined", or "independent"
        'level_set_method': 'primary',
        'skip_postprocessing': False,
        'skip_inductance_calculation': False,
        'tikhonov_reg_factor': 10,  # Tikhonov regularization factor for the SF optimization

        'output_directory': 'images/issue70',  # [Current directory]
        # 'project_name': 'issue70',
        'project_name': 'biplanar_xgradient_Tj',
        'persistence_dir': 'debug',
        'debug': DEBUG_BASIC,
    }

    result = pyCoilGen(log, arg_dict)
