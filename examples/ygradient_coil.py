# System imports
import sys
from pathlib import Path
import numpy as np

# Logging
import logging

# Local imports
# Add the sub_functions directory to the Python module search path
sub_functions_path = Path(__file__).resolve().parent / '..'
print(sub_functions_path)
sys.path.append(str(sub_functions_path))
from CoilGen import CoilGen


if __name__ == "__main__":
    # Set up logging
    log = logging.getLogger(__name__)
    logging.basicConfig(level=logging.DEBUG)
    # logging.basicConfig(level=logging.INFO)

    arg_dict = {
        'field_shape_function': 'y',  # % definition of the target field
        'coil_mesh_file': 'cylinder_radius500mm_length1500mm.stl',
        'target_mesh_file': 'none',
        'secondary_target_mesh_file': 'none',
        'secondary_target_weight': 0.5,
        'target_region_radius': 0.15,  # ...  % in meter
        'use_only_target_mesh_verts': False,
        'sf_source_file': 'none',
        # % the number of potential steps that determines the later number of windings (Stream function discretization)
        'levels': 20,
        'pot_offset_factor': 0.25,  # % a potential offset value for the minimal and maximal contour potential ; must be between 0 and 1
        'surface_is_cylinder_flag': True,
        'interconnection_cut_width': 0.1,  # % the width for the interconnections are interconnected; in meter
        'normal_shift_length': 0.025,  # % the length for which overlapping return paths will be shifted along the surface normals; in meter
        'iteration_num_mesh_refinement': 1,  # % the number of refinements for the mesh;
        'set_roi_into_mesh_center': True,
        'force_cut_selection': {'high'},  # ...
        'level_set_method': 'primary',  # ... %Specify one of the three ways the level sets are calculated: "primary","combined", or "independent"
        'interconnection_method': 'regular',
        'skip_postprocessing': False,
        'skip_inductance_calculation': False,
        'make_cylndrical_pcb': True,
        'conductor_cross_section_width': 0.015,
        'cross_sectional_points': np.array([np.sin(np.linspace(0, 2 * np.pi, 10)),
                                     np.cos(np.linspace(0, 2 * np.pi, 10))]) * 0.01,
        'sf_opt_method': 'tikkonov', # ...
        'fmincon_parameter': [1000.0, 10 ^ 10, 1.000000e-10, 1.000000e-10, 1.000000e-10],
        'tikonov_reg_factor': 100,  # %Tikonov regularization factor for the SF optimization
        'debug' : 2,
    }

    result = CoilGen(log, arg_dict)
