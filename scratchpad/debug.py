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

# Do not move import from here!
from CoilGen import CoilGen

def debug1():
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

    x = CoilGen(log, arg_dict)


# A Planar mesh with a hole in the middle
def debug2():
    from sub_functions.data_structures import DataStructure, Mesh
    from sub_functions.parameterize_mesh import parameterize_mesh

    input_params = DataStructure(
        surface_is_cylinder_flag=False, circular_diameter_factor=0.0)

    # Create small planar mesh with a hole
    # Define the mesh parameters
    x_min, x_max = -1.0, 1.0
    y_min, y_max = -1.0, 1.0
    hole_min, hole_max = -0.25, 0.25
    num_rows, num_cols = 8, 8

    # Calculate step sizes
    x_step = (x_max - x_min) / num_rows
    y_step = (y_max - y_min) / num_cols

    # Generate vertices
    vertices = []
    for i in range(num_rows + 1):
        for j in range(num_cols + 1):
            x = x_min + i * x_step
            y = y_min + j * y_step

            # Check if the vertex is inside the hole region
            # if hole_min <= x <= hole_max and hole_min <= y <= hole_max:
            #    continue  # Skip this vertex

            z = 0.0  # Z-coordinate is 0 for a planar mesh
            vertices.append([x, y, z])

    vertices = np.array(vertices)

    # Generate faces
    faces = []
    for i in range(num_rows):
        for j in range(num_cols):
            v1 = i * (num_cols + 1) + j
            v2 = v1 + 1
            v3 = v1 + num_cols + 2
            v4 = v1 + num_cols + 1

            # Check if any vertex is inside the hole region
            if (hole_min <= vertices[v1, 0] <= hole_max and hole_min <= vertices[v1, 1] <= hole_max) or \
                    (hole_min <= vertices[v2, 0] <= hole_max and hole_min <= vertices[v2, 1] <= hole_max) or \
                    (hole_min <= vertices[v3, 0] <= hole_max and hole_min <= vertices[v3, 1] <= hole_max) or \
                    (hole_min <= vertices[v4, 0] <= hole_max and hole_min <= vertices[v4, 1] <= hole_max):
                continue  # Skip this face

            faces.append([v3, v2, v1])
            faces.append([v4, v3, v1])

    faces = np.array(faces)
    # Print the vertices and faces arrays
    # print("Vertices:\n", vertices)
    # print("Faces:\n", faces)

    mesh = Mesh(vertices=vertices, faces=faces)

    mesh.display()

    from sub_functions.data_structures import DataStructure
    parts = [DataStructure(coil_mesh=mesh)]
    result = parameterize_mesh(parts, input_params)


# Planar mesh from a file
def debug3():
    arg_dict = {
        'coil_mesh_file': 'dental_gradient_ccs_single_low.stl',
        'iteration_num_mesh_refinement': 0,  # the number of refinements for the mesh;
    }
    x = CoilGen(log, arg_dict)


if __name__ == "__main__":
    # Set up logging
    log = logging.getLogger(__name__)
    logging.basicConfig(level=logging.DEBUG)

    debug1() # Cylindrical mesh
    # debug2() # Planar mesh with a hole
    # debug3() #
