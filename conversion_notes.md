# Installation
## SciPiy and Dependencies
SciPy might actually not be needed for the release version. To be checked!!
Note: Also need BLAS and gfortran to install scipy:
```bash
 $ sudo apt-get install libopenblas-dev
 $ sudo apt install gfortran
```

## Trimesh Dependencies
Need to manually install Trimesh dependencies.
1. rtree (for nearest.on_surface)

I needed to manually install libspatialindex library (for rtree).
```bash
$ sudo apt-get install libspatialindex-dev
```

## FastHenry2
The `FastHenry2` application needs to downloaded and installed.

### Windows
Go to the [download](https://www.fastfieldsolvers.com/download.htm) page, fill out the form, then download the
`FastFieldSolvers` bundle, e.g. FastFieldSolvers Software Bundle Version 5.2.0

Under Linux systems, the project should be cloned from [GitHub](https://github.com/ediloren/FastHenry2) and compiled.
### Linux
```bash
$ git clone https://github.com/ediloren/FastHenry2.git
$ cd FastHenry2/src
$ make
```

# Conversion Notes
## Indexing
Note that MATLAB uses base index of 1 for arrays, whereas NumPy uses 0. Adjust all functions that create `faces` arrays accordingly.

## Mesh Geometry
Confirm: Are mesh normals computed according to the right-hand rule? i.e. defined using the "counter-clockwise" or "anti-clockwise"
vertex ordering, where the vertices of the face are specified in a counter-clockwise direction when viewed from the outside of the mesh.

## Other Weirdnesses
### build_cylinder_mesh.py
I think there may be a fence-post bug in the original MATLAB code. The height of the resulting
cylinder does not match the cylinder_height parameter. This is especially noticable for small
values of num_longitudinal_divisions. See test_build_cylinder_mesh.py.

### define_target_field.py
In define_target_field, line 104, I have to swap y and z coords to match MATLAB:
```python
target_points = np.vstack((target_grid_x.ravel(), target_grid_z.ravel(), target_grid_y.ravel()))
```



## Notes per Sub_Function
### calc_gradient_along_vector(field, field_coords, target_endcoding_function)
The target_endcoding_function needs to be converted, too. The Python implementation uses `eval` 
whereas the original MATLAB uses `my_fun=str2func("@(x,y,z)"+target_endcoding_function);`.

TODO: Check caller implementations and convert appropriately.

# Runtime Errors
When target_region_resolution is 10.
``` bash
File "/home/kevin/Dev/CoilGen-Python/sub_functions/interconnect_within_groups.py", line 77, in interconnect_within_groups
part_group.opened_loop = part_group.loops.uv
AttributeError: 'list' object has no attribute 'uv'
```