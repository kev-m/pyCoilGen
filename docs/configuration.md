# Configuration

- [Configuration](#configuration)
  - [Basic Settings](#basic-settings)
  - [Specify Mesh Geometry](#specify-mesh-geometry)
    - [Mesh Files](#mesh-files)
      - [Built-in Meshes](#built-in-meshes)
    - [Subdividing the Mesh](#subdividing-the-mesh)
  - [Parameterise Mesh](#parameterise-mesh)
  - [Target Field and Gradient Field](#target-field-and-gradient-field)
    - [Specifying the Gradient Field Co-ordinates Using an .stl Mesh](#specifying-the-gradient-field-co-ordinates-using-an-stl-mesh)
    - [Specifying the Gradient Field Co-ordinates Using a Sphere](#specifying-the-gradient-field-co-ordinates-using-a-sphere)
    - [Specifying the Gradient Field Value](#specifying-the-gradient-field-value)
  - [Discretisation and Calculation of Field Variables](#discretisation-and-calculation-of-field-variables)
    - [Winding Coil Contribution and Gradient Field Sensitivity](#winding-coil-contribution-and-gradient-field-sensitivity)
    - [Winding Coil Resistance](#winding-coil-resistance)
    - [Stream Function](#stream-function)
  - [Loading Pre-calculated Mesh and Stream Function](#loading-pre-calculated-mesh-and-stream-function)
  - [Build Contour Lines](#build-contour-lines)
    - [Contour Parameters](#contour-parameters)
  - [Find Contour Topology](#find-contour-topology)
    - [Contour Topology Parameters](#contour-topology-parameters)
  - [Interconnect Contours / Build Wire Path](#interconnect-contours--build-wire-path)
    - [Interconnection Parameters](#interconnection-parameters)
  - [Manage Overlaps](#manage-overlaps)
    - [Overlap Management Parameters](#overlap-management-parameters)
  - [Evaluate Results](#evaluate-results)
    - [Evaluation Parameters](#evaluation-parameters)


This section outlines the command-line parameters for pyCoilGen.

**Note:** All values are specified in SI units, i.e. metres, Amperes, Tesla.

## Basic Settings

`--output_directory` (Type: str, Default: Current Working Directory)

Specify the output directory, where intermediate images and the final output will be written to.

`--project_name` (Type: str, Default: 'CoilGen')

Specify a name which is used to create output files.

`--persistence_dir` (Type: str, Default: 'debug')

Specify the directory where project snapshots are written. A snapshot of the internal state is automatically written to this location when any unhandled exception occurs.

`--debug` (Type: int, Default: 0)

Control the Debug verbosity level: 0 = None, 1 = Basic, 2 = Verbose. With 

`--fasthenry_bin` (Type: str, Default: '/usr/bin/fasthenry')

Specify the location of the FastHenry2 binary. 

In Windows, the default installation location is 'C:\Program Files (x86)\FastFieldSolvers\FastHenry2\FastHenry2.exe'.

`--geometry_source_path` (Type: str, Default: Current Working Directory + '/Geometry_Data')
The directory where .stl geometry files are located.

## Specify Mesh Geometry

The coil mesh geometry must be specified. It can either be loaded from a pre-optimised Numpy pickle file or specified in parts.

### Mesh Files
`--coil_mesh_file` (Type: str, Default: 'none')
Specify that defines the winding coil surface. 

Either specify the filename of an .stl file to be loaded from the `geometry_source_path`, or use one of the built-in mesh specifications.

When using a built-in mesh specification, the mesh parameters must also be specified.

#### Built-in Meshes

The winding coil surface can be specified using a subset of built-in types. 

To use one of the built-in types, set `--coil_mesh_file` to one of the following special names. The mesh parameters are then specified using a second parameter. 

- `create cylinder mesh`
Create a cylindrical mesh according to `--cylinder_mesh_parameter_list` (Type: list of float, Default: [0.8, 0.3, 20, 20, 1, 0, 0, 0])

        cylinder_height (float): Height of the cylinder.
        cylinder_radius (float): Radius of the cylinder.
        num_circular_divisions (int): Number of circular divisions.
        num_longitudinal_divisions (int): Number of longitudinal divisions.
        rotation_vector_x (float): X-component of the rotation vector.
        rotation_vector_y (float): Y-component of the rotation vector.
        rotation_vector_z (float): Z-component of the rotation vector.
        rotation_angle (float): Rotation angle.


- `create planar mesh` 
Create a planar mesh according to `--planar_mesh_parameter_list` (Type: list of float, Default: [0.25, 0.25, 20, 20, 1, 0, 0, 0, 0, 0, 0])

        planar_height (float): Height of the planar mesh.
        planar_width (float): Width of the planar mesh.
        num_lateral_divisions (int): Number of divisions in the lateral direction.
        num_longitudinal_divisions (int): Number of divisions in the longitudinal direction.
        rotation_vector_x (float): X component of the rotation vector.
        rotation_vector_y (float): Y component of the rotation vector.
        rotation_vector_z (float): Z component of the rotation vector.
        rotation_angle (float): Rotation angle in radians.
        center_position_x (float): X component of the center position.
        center_position_y (float): Y component of the center position.
        center_position_z (float): Z component of the center position.


- `create bi-planar mesh`
Create a bi-planar mesh according to `--biplanar_mesh_parameter_list` (Type: list of float, Default: [0.25, 0.25, 20, 20, 1, 0, 0, 0, 0, 0, 0.2])

        planar_height (float): Height of the planar mesh.
        planar_width (float): Width of the planar mesh.
        num_lateral_divisions (int): Number of divisions in the lateral direction.
        num_longitudinal_divisions (int): Number of divisions in the longitudinal direction.
        target_normal_x (float): X-component of the target normal vector.
        target_normal_y (float): Y-component of the target normal vector.
        target_normal_z (float): Z-component of the target normal vector.
        center_position_x (float): X-coordinate of the center position.
        center_position_y (float): Y-coordinate of the center position.
        center_position_z (float): Z-coordinate of the center position.
        plane_distance (float): Distance between the two planes.


<!--
`--double_cone_mesh_parameter_list` (Type: list of float, Default: [0.8, 0.3, 0.3, 0.1, 20, 20, 1, 0, 0, 0])
  Parameters for the generation of a double cone ('diabolo') shaped mesh.

`--circular_mesh_parameter_list` (Type: list of float, Default: [0.25, 20, 1, 0, 0, 0, 0, 0, 0])
  Parameters for the generation of the (default) circular mesh.
-->

### Subdividing the Mesh

Once the mesh has been loaded, the mesh resolution can be increased using mesh subdivision.

`--iteration_num_mesh_refinement` (Type: int, Default: 0)
Specify the number of refinement iterations of the mesh. At each iteration, every mesh face is further subdivided into four faces.


## Parameterise Mesh

The 3D mesh of the coil winding surface needs to be projected onto a 2D surface.

`--surface_is_cylinder_flag` (Type: bool, Default: True)
Provide a hint to the application that the 3D coil mesh can be projected onto 2D using a simple cylindrical projection.

If cylindrical projection is inappropriate then an iterative mesh parameterisation approach is used.

`--circular_diameter_factor` (Type: float, default: 1)
Circular diameter factor for projecting the 3D coil mesh to 2D.

## Target Field and Gradient Field

The target gradient field can be either loaded from a Numpy pickle file, or defined by a volume generated by a mesh loaded from an .stl file or simply as a sphere of a defined radius.

- Using a Numpy Pickle file
`--target_field_definition_file` (Type: str, Default: 'none')
Specify the name of the Numpy pickle file that contains the target field co-ordinates and field value. 

If used, the target field file is loaded from the `target_fields` directory.

`--target_field_definition_field_name` (Type: str, Default: 'none')
Specify the field name of the target field definition within the Numpy pickle file.

### Specifying the Gradient Field Co-ordinates Using an .stl Mesh

The mesh defines the boundary of the target field and these parameters fine-tune the target field point selection.

`--target_mesh_file` (Type: str, Default: 'none')
Specify the mesh used to define the target field. Further

`--secondary_target_mesh_file` (Type: str, Default: 'none')
File of the secondary target mesh.

`--target_region_resolution` (Type: int, Default: 10)
Defines how many target points to create per dimension within the target region defined by the mesh.

Only used if `--use_only_target_mesh_verts` is False.

`--use_only_target_mesh_verts` (Type: bool, Default: False)
If True, specifies that only the vertices of the mesh are to be used.

### Specifying the Gradient Field Co-ordinates Using a Sphere
When both `--target_field_definition_file` and `--target_field_definition_file` are `'none'` then the target field co-ordinates are specified using a spherical volume.

`--target_region_radius` (Type: float, Default: 0.15)
Specifies the radius of the spherical target field. 

The target field co-ordinates are then created by sub-dividing the radius using  `--target_region_resolution`, which specifies how many co-ordinates to create along each axis.

`--set_roi_into_mesh_center` (Type: bool, Default: False)
This flag is used to set the ROI into the geometric center of the mesh. If set, the centre of the target sphere is moved to the mean of the target field vertices.


### Specifying the Gradient Field Value

Once the target gradient field co-ordinates have been specified, then the gradient field vectors can be calculated.

`--field_shape_function` (Type: str, Default: 'x')
Specifies the spatial function that defines the field.

`--target_gradient_strength` (Type: float, Default: 1)
Specifies the target gradient field strength in T/m.


## Discretisation and Calculation of Field Variables

### Winding Coil Contribution and Gradient Field Sensitivity

The magnetic field contribution and the gradient field sensitivity is calculated at every corresponding co-ordinate.

`--gauss_order` (Type: int, Default: 2)
This parameter determines the Gauss integration order used in the winding magnetic field calculations.


### Winding Coil Resistance

The winding coil resistance affects the gradient magnetic field due to the winding coil.

`--specific_conductivity_conductor` (Type: float, Default: 0.018e-6)
Specify the conductivity of the winding coil.

`--conductor_thickness` (Type: float, Default: 0.005)
Specify the thickness of the sheet current density within the stream function representation.


### Stream Function

The stream function represents the relationship between the coil parts and the gradient field.

The CoilGen application performs an optimisation calculation of the stream function.

`--sf_opt_method` (Type: str, Default: 'tikhonov')
Determines the stream function optimization method.

`--tikhonov_reg_factor` (Type: float, Default: 1)
Tikhonov regularization factor for the stream function optimization

`--minimize_method` (Type: str, Default: 'SLSQP')

If the `--sf_opt_method` is not `'tikhonov'`, then the Numpy `minimize` function is used and this parameter specifies which minimisation method to use.

`--minimize_method_parameters` (Type:str, Default:"{'tol': 1e-6}")
Specify additional method parameters.

`--minimize_method_options` (Type: str, Default: "{'disp': True, 'maxiter' : 100}")
Specify additional method options, specific to the method.

Please refer to the [official documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html)
for detailed information on some optimization-related parameters.

## Loading Pre-calculated Mesh and Stream Function

A pre-existing mesh and optimised stream function solution can be loaded from persistence.

`--sf_source_file` (Type: str, Default: 'none')
Specify the filename of the Numpy pickle file of the already optimized stream function. The file is loaded from the `Pre_Optimized_Solutions` directory.



## Build Contour Lines
The 3D mesh of the coil winding surface needs to be projected onto a 2D surface.

### Contour Parameters
`--levels` (Type: int, Default: 10)
  Number of potential levels.

`--level_set_method` (Type: str, Default: 'primary')
Specify the method for calculating the level sets. One of 'primary', 'combined' or 'independent'.

The contour levels are calculated from the stream function with contributions from the different coil meshes (if more than one), according to the level set method.

Use `primary` to calculate the contour potentials from primary coil mesh only. 
Use `combined` to calculate the potentials from the combined mesh. 
Use `independent` to calculate the potentials for each coil mesh independently.

The best method depends on each application. Users can examine the final computed gradient field and computed errors to inform their decision.

`--pot_offset_factor` (Type: float, Default: 1/2)
This factor is used to control the contour level step, according to the stream function range.







## Find Contour Topology

### Contour Topology Parameters
`--min_point_loop_number` (Type: int, Default: 20)
  Minimal required number of points of a single loop; otherwise loops will be removed.

`--min_loop_significance` (Type: int, Default: 1)
  Minimal required field contribution (in percent) to the target field; loops that contribute less than that can be deleted.

`--area_perimeter_deletion_ratio` (Type: int, Default: 5)
  Additional loop removal criteria which relates to the perimeter to surface ratio of the loop.

`--max_allowed_angle_within_coil_track` (Type: int, Default: 120)
  Maximum allowed angle of the track of the contours.

## Interconnect Contours / Build Wire Path

### Interconnection Parameters
`--interconnection_method` (Type: str, Default: 'regular')
  Interconnection method: 'regular' or 'spiral' in/out.

`--group_interconnection_method` (Type: str, Default: 'crossed')
  Group interconnection method: 'straight' or 'crossed'.

`--interconnection_cut_width` (Type: float, Default: 0.01)
  Width in meter of the opening cut for the interconnection of the loops.

`--skip_calculation_min_winding_distance` (Type: bool, Default: True)
  Flag to skip calculation of minimal winding distance.

## Manage Overlaps

### Overlap Management Parameters
`--skip_normal_shift` (Type: bool, Default: False)
  Flag to skip the shifting of return paths.

`--normal_shift_smooth_factors` (Type: list of int, Default: [2, 3, 2])
  Smoothing parameters regarding the normal shift.

`--skip_sweep` (Type: bool, Default: False)
  Flag to skip the generation of a volumetric (3D) coil body.

## Evaluate Results

### Evaluation Parameters
`--skip_inductance_calculation` (Type: bool, Default: False)
  Flag to skip inductance calculation.

`--skip_postprocessing` (Type: bool, Default: False)
  Flag to skip post-processing.

`--make_cylindrical_pcb` (Type: bool, Default: False)
  Flag to generate a rectangular pcb pattern to wrap around a cylinder.

`--pcb_interconnection_method` (Type: str, Default: 'spiral_in_out')
  Interconnection method for PCB: 'spiral_in_out' or other options.

`--pcb_spiral_end_shift_factor` (Type: int, Default: 10)
  Factor of shifting the open ends of the spirals in order to avoid overlaps; in percent.

`--force_cut_selection` (Type: list, Default: [])
  Force cut selection.

`--track_width_factor` (Type: float, Default: 0.5)
  Track width factor for PCB layout.

`--conductor_cross_section_width` (Type: float, Default: 0.002)
  Cross-section width of the conductor (for the inductance calculation) in meter.

`--conductor_cross_section_height` (Type: float, Default: 0.002)
  Cross-section height of the conductor (for the inductance calculation) in meter.

`--cross_sectional_points` (Type: list of float, Default: [0, 0])
  2D edge points for direct definition of the cross section of the conductor (build circular cut shapes).

