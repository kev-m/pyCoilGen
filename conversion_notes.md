# Installation
## SciPiy and Dependencies
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

 # Conversion Notes
 ## Indexing
 Note that Matlab uses base index of 1 for arrays, whereas Numpy uses 0. Adjust all functions that create `faces` arrays accordingly.

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


 ## Data Structures
 ```python
from dataclasses import dataclass
from typing import List

@dataclass
class Coil_mesh: # Maybe coil_mesh?
    uv: ndarray
    v: ndarray 


@dataclass
class Coil_parts:
    coil_mesh: List[Coil_mesh]
    potential_level_list: np.ndarray
```
 
## Function Calls
```python
def calc_gradient_along_vector(field: np.ndarray, field_coords: np.ndarray, target_encoding_function: str) -> CalcGradientAlongVectorResult:
    """
    Calculates the mean gradient in a given direction and angle.

    Args:
        field: The field data.
        field_coords: The field coordinates.
        target_encoding_function: The target encoding function as a string.

    Returns:
        The result of the mean gradient calculation.
    """
    # Implementation goes here

def calc_3d_rotation_matrix(rot_vec: np.ndarray, rot_angle: float) -> CalcRotationMatrixResult:
    """
    Calculates the 3D rotation matrix around a rotation axis given by a vector and an angle.

    Args:
        rot_vec: The rotation vector.
        rot_angle: The rotation angle.

    Returns:
        The result of the rotation matrix calculation.
    """
    # Implementation goes here

def calc_local_opening_gab(loop: Any, point_1: int, point_2: int, opening_gab: float) -> CalcLocalOpeningGabResult:
    """
    Calculates the local opening gab.

    Args:
        loop: The loop data.
        point_1: The first point index.
        point_2: The second point index.
        opening_gab: The opening gab value.

    Returns:
        The result of the local opening gab calculation.
    """
    # Implementation goes here

```

## Notes per Sub_Function
### calc_gradient_along_vector(field, field_coords, target_endcoding_function)
The target_endcoding_function needs to be converted, too. The Python implementation uses `eval` 
whereas the original MatLab uses `my_fun=str2func("@(x,y,z)"+target_endcoding_function);`.

TODO: Check caller implementations and convert appropriately.
