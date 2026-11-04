"""Reference designs for the regression harness.

Each case is a small, deterministic coil design that uses only the built-in mesh builders, so no data files or optional
packages are needed. The cases are kept small, so that the whole set runs in under a minute, while still exercising
the main code paths: mesh builders, both solvers, planar, bi-planar and cylindrical surfaces, post-processing,
the conductor sweep and the field error calculation.

To add a case, add it to `CASES` and generate its golden values with `harness_generate.py`.
"""

# System
from typing import Dict

# Settings that are common to all cases. The output is written to the directories that the harness passes in.
_COMMON = {
    'target_region_radius': 0.05,
    'target_region_resolution': 6,
    'iteration_num_mesh_refinement': 0,
    'set_roi_into_mesh_center': True,
    'force_cut_selection': ['high'],
    'level_set_method': 'primary',
    'skip_postprocessing': False,
    'skip_inductance_calculation': True,   # Needs the external FastHenry2 program
    'exporter': 'none',
    'debug': 0,
}

CASES: Dict[str, dict] = {
    # Rectangular bi-planar plates, with the default Tikhonov solver
    'biplanar_x_tikhonov': {
        **_COMMON,
        'coil_mesh': 'create bi-planar mesh',
        'biplanar_mesh_parameter_list': [0.25, 0.25, 12, 12, 1, 0, 0, 0, 0, 0, 0.2],
        'surface_is_cylinder_flag': False,
        'field_shape_function': 'x',
        'levels': 8,
        'pot_offset_factor': 0.25,
        'interconnection_cut_width': 0.01,
        'normal_shift_length': 0.005,
        'sf_opt_method': 'tikhonov',
        'tikhonov_reg_factor': 10,
    },

    # The same design with the iterative (constrained minimisation) solver
    'biplanar_x_iterative': {
        **_COMMON,
        'coil_mesh': 'create bi-planar mesh',
        'biplanar_mesh_parameter_list': [0.25, 0.25, 8, 8, 1, 0, 0, 0, 0, 0, 0.2],
        'surface_is_cylinder_flag': False,
        'field_shape_function': 'x',
        'levels': 8,
        'pot_offset_factor': 0.25,
        'interconnection_cut_width': 0.01,
        'normal_shift_length': 0.005,
        'sf_opt_method': 'iterative',
        'tikhonov_reg_factor': 10,
        'minimize_method': 'SLSQP',
        'minimize_method_parameters': "{'tol': 1e-6}",
        'minimize_method_options': "{'disp': False, 'maxiter': 20}",
    },

    # A single planar plate, below the target region
    'planar_z_tikhonov': {
        **_COMMON,
        'coil_mesh': 'create planar mesh',
        'planar_mesh_parameter_list': [0.4, 0.4, 16, 16, 0, 0, 1, 0, 0, -0.12, 0],
        'set_roi_into_mesh_center': False,
        'surface_is_cylinder_flag': False,
        'field_shape_function': 'z',
        'levels': 8,
        'pot_offset_factor': 0.25,
        'interconnection_cut_width': 0.01,
        'normal_shift_length': 0.005,
        'sf_opt_method': 'tikhonov',
        'tikhonov_reg_factor': 10,
    },

    # A cylinder, which uses the cylindrical parameterisation
    'cylinder_y_tikhonov': {
        **_COMMON,
        'coil_mesh': 'create cylinder mesh',
        'cylinder_mesh_parameter_list': [0.8, 0.3, 16, 16, 1, 0, 0, 0],
        'surface_is_cylinder_flag': True,
        'field_shape_function': 'y',
        'levels': 8,
        'pot_offset_factor': 0.25,
        'interconnection_cut_width': 0.05,
        'normal_shift_length': 0.01,
        'sf_opt_method': 'tikhonov',
        'tikhonov_reg_factor': 10,
    },

    # Circular bi-planar plates, from the polar mesh builder
    'circular_biplanar_z_tikhonov': {
        **_COMMON,
        'coil_mesh': 'create circular bi-planar mesh',
        'circular_biplanar_mesh_parameter_list': [0.07, 10, 36, 0, 0, 1, 0, 0, 0, 0.0735],
        'surface_is_cylinder_flag': False,
        'field_shape_function': 'z',
        'target_region_radius': 0.016,
        'levels': 10,
        'pot_offset_factor': 0.35,
        'interconnection_cut_width': 0.0016,
        'normal_shift_length': 0.002,
        'conductor_thickness': 0.0015,
        'sf_opt_method': 'tikhonov',
        'tikhonov_reg_factor': 200,
    },
}
