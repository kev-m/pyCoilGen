# Quick Start

While the **pyCoilGen** application can be used as a command-line application, due to the [large number of parameters](./configuration.md) to be configured, it is best used as part of a Python script. 

This allows users to use the application, or previously saved data, to manipulate and/or view coil design parameters, field errors, etc.

## Calling the Application

The following code snippet shows an example of a basic Python script that creates a dictionary of parameters that is then passed to the `pyCoilGen` function to determine the solution.


```python
import logging 
from pyCoilGen.pyCoilGen_release import pyCoilGen
from pyCoilGen.sub_functions.constants import DEBUG_BASIC

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Change the default values to suit your application
arg_dict = {
    'field_shape_function': 'y',            # % definition of the target field ['x']
    'coil_mesh_file': 'cylinder_radius500mm_length1500mm.stl',
    'secondary_target_weight': 0.5,         # [1.0]
    'levels': 20,                           # The number of potential steps, determines the number of windings [10]
    'pot_offset_factor': 0.25,              # a potential offset value for the minimal and maximal contour potential [0.5]
    'interconnection_cut_width': 0.1,       # Width cut used when cutting and joining wire paths; in metres [0.01]
    'normal_shift_length': 0.025,           # Displacement that overlapping return paths will be shifted along the surface normals; in meter [0.001]
    'iteration_num_mesh_refinement': 1,     # % the number of refinements for the mesh; [0]
    'set_roi_into_mesh_center': True,       # [False]
    'force_cut_selection': ['high'],        # []
    'make_cylindrical_pcb': True,           # [False]
    'conductor_cross_section_width': 0.015, # [0.002]
    'tikhonov_reg_factor': 100,             # Tikhonov regularization factor for the SF optimization [1]

    'output_directory': 'images',           # [Current directory]
    'project_name': 'ygradient_coil',
    'persistence_dir': 'debug',             # [debug]
    'debug': DEBUG_BASIC,                   # [0 = NONE]
}


solution = pyCoilGen(log, arg_dict) # Calculate the solution

# Calculate the errors

```

## Viewing Results

The following code snippet shows an example of a basic Python script to view the results.