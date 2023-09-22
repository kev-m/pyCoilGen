import logging

from pyCoilGen.sub_functions.constants import DEBUG_BASIC, DEBUG_VERBOSE
from pyCoilGen.pyCoilGen_release import pyCoilGen

if __name__ == "__main__":
    # Set up logging
    log = logging.getLogger(__name__)
    # logging.basicConfig(level=logging.DEBUG)
    logging.basicConfig(level=logging.INFO)

    # create cylinder mesh: 0.4, 0.1125, 50, 50, copy from Matlab

    # Examples/biplanar_xgradient.m
    arg_dict1 = {
        # "b_0_direction": [0, 0, 1],
        # "biplanar_mesh_parameter_list": [0.25, 0.25, 20, 20, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2],
        # "circular_diameter_factor": 1.0,  # was circular_diameter_factor_cylinder_parameterization
        # "circular_mesh_parameter_list": [0.25, 20.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "coil_mesh_file": "bi_planer_rectangles_width_1000mm_distance_500mm.stl",
        # "conductor_cross_section_height": 0.002,
        # "conductor_cross_section_width": 0.002,
        # "conductor_thickness": 0.005,
        # "cross_sectional_points": [0, 0],
        # "cylinder_mesh_parameter_list": [0.4, 0.1125, 50, 50, 0.0, 1.0, 0.0, 0.0],
        "field_shape_function": "x",
        "force_cut_selection": ['high'],
        # "gauss_order": 2,
        "interconnection_cut_width": 0.05,
        "iteration_num_mesh_refinement": 0,  # MATLAB 1 is default, but 0 is faster
        "level_set_method": "primary",
        "levels": 14,
        # "make_cylindrical_pcb": False,
        # "min_loop_significance": 1,
        "normal_shift_length": 0.01,
        # "normal_shift_smooth_factors": [2, 3, 2],
        # "pcb_interconnection_method": "spiral_in_out",
        # "pcb_spiral_end_shift_factor": 10,
        # "planar_mesh_parameter_list": [0.25, 0.25, 20.0, 20.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        # "plot_flag": True,
        "pot_offset_factor": 0.25,
        # "save_stl_flag": True,
        "secondary_target_mesh_file": "none",
        "secondary_target_weight": 0.5,
        "set_roi_into_mesh_center": True,
        "sf_opt_method": "tikhonov",  # "tikhonov"
        # "minimize_method": "SLSQP", # Only used when 'sf_opt_method' is not tikhonov
        # "minimize_method_parameters" : "{'tol':1.e-6}",
        # "minimize_method_options" : "{'disp': True, 'maxiter' : 1000}",
        "sf_source_file": "none",
        # "skip_calculation_min_winding_distance": True,  # Default: 1
        "skip_inductance_calculation": False,
        # "skip_normal_shift": False,
        "skip_postprocessing": False,
        # "skip_sweep": False,
        # "smooth_factor": 1,
        # "specific_conductivity_conductor": 1.8e-8,
        "surface_is_cylinder_flag": True,
        # "target_field_definition_field_name": "none",
        # "target_field_definition_file": "none",
        # "target_gradient_strength": 1,
        "target_mesh_file": "none",
        "target_region_radius": 0.1,    # GitHub
        "target_region_resolution": 5,  # MATLAB 10 is the default but 5 is faster
        "tikhonov_reg_factor": 10,
        "use_only_target_mesh_verts": False,

        "output_directory": "images",
        "project_name": 'biplanar_xgradient',
        "persistence_dir": 'debug',
        "debug": DEBUG_BASIC,
    }  # INFO:pyCoilGen.helpers.timing:Total elapsed time: 12.071821 seconds

    # cylinder_radius500mm_length1500mm
    arg_dict2 = {
        "b_0_direction": [0, 0, 1],
        "biplanar_mesh_parameter_list": [0.25, 0.25, 20.0, 20.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2],
        "circular_diameter_factor": 1.0,  # was circular_diameter_factor_cylinder_parameterization
        "coil_mesh_file": "cylinder_radius500mm_length1500mm.stl",
        "conductor_cross_section_height": 0.002,
        "conductor_cross_section_width": 0.015,
        "conductor_thickness": 0.005,
        "cross_sectional_points": [[0.0, 0.006427876096865392, 0.00984807753012208, 0.008660254037844387, 0.0034202014332566887, -0.0034202014332566865, -0.008660254037844388, -0.009848077530122082, -0.006427876096865396, -2.4492935982947064e-18], [0.01, 0.007660444431189781, 0.0017364817766693042, -0.0049999999999999975, -0.009396926207859084, -0.009396926207859084, -0.004999999999999997, 0.0017364817766692998, 0.007660444431189778, 0.01]],
        "cylinder_mesh_parameter_list": [0.8, 0.3, 20.0, 20.0, 1.0, 0.0, 0.0, 0.0],
        "field_shape_function": "y",
        "force_cut_selection": ['high'],
        "gauss_order": 2,
        # "geometry_source_path": "/MATLAB Drive/CoilGen/Geometry_Data",
        "interconnection_cut_width": 0.1,
        "iteration_num_mesh_refinement": 0,  # MATLAB 1 is default, but 0 is faster
        "level_set_method": "primary",
        "levels": 20,
        "make_cylindrical_pcb": True,
        "min_loop_significance": 1,  # Was 0.1, a bug?
        "normal_shift_length": 0.025,
        "normal_shift_smooth_factors": [2, 3, 2],
        "pcb_interconnection_method": "spiral_in_out",
        "pcb_spiral_end_shift_factor": 10,
        "planar_mesh_parameter_list": [0.25, 0.25, 20.0, 20.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "plot_flag": True,
        "pot_offset_factor": 0.25,
        "save_stl_flag": True,
        "secondary_target_mesh_file": "none",
        "secondary_target_weight": 0.5,
        "set_roi_into_mesh_center": True,
        "sf_opt_method": "tikhonov",
        # "minimize_method": "SLSQP", # Only used when 'sf_opt_method' is not tikhonov
        # "minimize_method_parameters" : "{'tol':1.e-6}",
        # "minimize_method_options" : "{'disp': True, 'ftol': 1e-6, 'maxiter' : 1000}",
        "sf_source_file": "none",
        "skip_calculation_min_winding_distance": True,  # Default 1
        "skip_inductance_calculation": False,
        "skip_normal_shift": False,
        "skip_postprocessing": False,
        "skip_sweep": False,
        "smooth_factor": 1,
        "specific_conductivity_conductor": 1.8e-08,
        "surface_is_cylinder_flag": True,
        "target_field_definition_field_name": "none",
        "target_field_definition_file": "none",
        "target_gradient_strength": 1,
        "target_mesh_file": "none",
        "target_region_radius": 0.15,
        "target_region_resolution": 5,  # MATLAB 10 is the default but 5 is faster
        "tikhonov_reg_factor": 100,
        "use_only_target_mesh_verts": False,

        "debug": DEBUG_BASIC,
        "output_directory": "images",
        "project_name": 'ygradient_coil',
        "persistence_dir": 'debug',
    }  # INFO:pyCoilGen.helpers.timing:Total elapsed time: 9.396004 seconds

    solution1 = pyCoilGen(log, arg_dict1)
    solution2 = pyCoilGen(log, arg_dict2)
