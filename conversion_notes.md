# Installation
## SciPiy and Dependencies
Note: Also need BLAS and gfortran to install scipy:
```bash
 $ sudo apt-get install libopenblas-dev
 $ sudo apt install gfortran
 ```

 # Conversion Notes
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
 
 ## Notes per Sub_Function
 ### calc_gradient_along_vector(field, field_coords, target_endcoding_function)
 The target_endcoding_function needs to be converted, too. The Python implementation uses `eval` 
 whereas the original MatLab uses `my_fun=str2func("@(x,y,z)"+target_endcoding_function);`.

 TODO: Check caller implementations and convert appropriately.
