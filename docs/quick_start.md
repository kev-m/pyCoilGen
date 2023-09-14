# Quick Start
While the **pyCoilGen** application can be used as a command-line application, due to the [large number of complex parameters](./configuration.md), it is best used as part of a Python script. 

This allows users to use the application, or previously saved data, to manipulate and/or view coil design parameters.

## Calling the Application
```python
import logging 
from pyCoilGen.pyCoilGen_release import pyCoilGen
from pyCoilGen.sub_functions.constants import DEBUG_BASIC

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)
#logging.basicConfig(level=logging.INFO)

# Change the default values to suit your application
arg_dict = {
    'field_shape_function': 'y',  # % definition of the target field
    'coil_mesh_file': 'cylinder_radius500mm_length1500mm.stl',
    'target_region_radius': 0.15,  # ...  % in meter
    'use_only_target_mesh_verts': False,        
    'levels': 20, # the number of potential steps that determines the later number of windings
    'pot_offset_factor': 0.25,  # % a potential offset value for the minimal and maximal contour potential ; must be between 0 and 1
    'surface_is_cylinder_flag': True,
    'interconnection_cut_width': 0.1,  # % the width for the interconnections are interconnected; in meter
    'normal_shift_length': 0.025,  # % the length for which overlapping return paths will be shifted along the surface normals; in meter
    'set_roi_into_mesh_center': True,
    'force_cut_selection': ['high'],  # ...
    'level_set_method': 'primary',  # ... %Specify one of the three ways the level sets are calculated: "primary","combined", or "independent"
    'interconnection_method': 'regular',
    'skip_postprocessing': False,
    'make_cylndrical_pcb': True,
    'conductor_cross_section_width': 0.015,
    'sf_opt_method': 'tikhonov', # ...
    'tikhonov_reg_factor': 100,  # %Tikhonov regularization factor for the SF optimization

    'output_directory': 'output',   # Where to save
    'project_name': 'ygradient_cylinder_example',
    'fasthenry_bin': '../FastHenry2/bin/fasthenry',
    'persistence_dir': 'debug',
    'debug': DEBUG_BASIC,
}

solution = pyCoilGen(log, arg_dict) # Calculate the solution
```

## Viewing Results
