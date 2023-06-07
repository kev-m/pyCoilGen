# System imports
import sys
from pathlib import Path

# Logging
import logging

# Local imports
# Add the sub_functions directory to the Python module search path
sub_functions_path = Path(__file__).resolve().parent / '..'
print(sub_functions_path)
sys.path.append(str(sub_functions_path))

# Do not move import from here!
from CoilGen import CoilGen

if __name__ == "__main__":
    # Set up logging
    log = logging.getLogger(__name__)
    logging.basicConfig(level=logging.DEBUG)

    # planar_mesh_parameter_list
    planar_height = 1.0
    planar_width = 0.5
    num_lateral_divisions = 4
    num_longitudinal_divisions = 5
    rotation_vector_x = 1.0
    rotation_vector_y = 0.0
    rotation_vector_z = 0.0
    rotation_angle = 0.0
    center_position_x = 0.0
    center_position_y = 0.0
    center_position_z = 0.0

    # cylinder_mesh_parameter_list
    cylinder_height = 0.5
    cylinder_radius = 0.25
    num_circular_divisions = 4

    arg_dict = {
        'field_shape_function': 'x',  # definition of the target field
        # 'coil_mesh_file': 'create planar mesh',
        'planar_mesh_parameter_list': [planar_height, planar_width, num_lateral_divisions, num_longitudinal_divisions,
                                       rotation_vector_x, rotation_vector_y, rotation_vector_z, rotation_angle,
                                       center_position_x, center_position_y, center_position_z],
        'coil_mesh_file': 'create cylinder mesh',
        'cylinder_mesh_parameter_list': [cylinder_height,
                                         cylinder_radius,
                                         num_circular_divisions,
                                         num_longitudinal_divisions,
                                         rotation_vector_x,
                                         rotation_vector_y,
                                         rotation_vector_z,
                                         rotation_angle
                                         ],
        'target_mesh_file': 'none',
        'secondary_target_mesh_file': 'none',
        'secondary_target_weight': 0.5,
        'target_region_radius': 0.1,  # in meter
        'use_only_target_mesh_verts': False,
        'sf_source_file': 'none',
        # the number of potential steps that determines the later number of windings (Stream function discretization)
        'levels': 14,
        # a potential offset value for the minimal and maximal contour potential ; must be between 0 and 1
        'pot_offset_factor': '0.25',
        'surface_is_cylinder_flag': True,
        # the width for the interconnections are interconnected; in meter
        'interconnection_cut_width': 0.05,
        # the length for which overlapping return paths will be shifted along the surface normals; in meter
        'normal_shift_length': 0.01,
        'iteration_num_mesh_refinement': 0,  # the number of refinements for the mesh;
        'set_roi_into_mesh_center': True,
        'force_cut_selection': ['high'],
        # Specify one of the three ways the level sets are calculated: "primary","combined", or "independent"
        'level_set_method': 'primary',
        'interconnection_method': 'regular',
        'skip_postprocessing': False,
        'skip_inductance_calculation': False,
        'tikonov_reg_factor': 10  # Tikonov regularization factor for the SF optimization
    }

    coil_parts, combined_mesh, sf_b_field, target_field, coil_inductance, radial_lumped_inductance, axial_lumped_inductance, radial_sc_inductance, axial_sc_inductance, field_errors, coil_gradient, is_supressed_point = CoilGen(
        log, arg_dict)
