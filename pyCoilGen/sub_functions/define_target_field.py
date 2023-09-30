# System
import numpy as np
import os
from sympy import symbols, diff, lambdify

# Logging
import logging

# Local imports
from pyCoilGen.helpers.common import find_file
from .data_structures import TargetField
from .constants import *

log = logging.getLogger(__name__)


def define_target_field(coil_parts, target_mesh, secondary_target_mesh, input_args) -> TargetField:
    """
    Define the target field.

    Initialises the following properties of a CoilPart:
        - None

    Updates the following properties of a CoilPart:
        - None

    Depends on the following input_args:
        - target_field_definition_file
        - target_field_definition_field_name
        - use_only_target_mesh_verts
        - target_region_resolution
        - target_region_radius
        - set_roi_into_mesh_center
        - field_shape_function
        - target_gradient_strength
        - secondary_target_weight

    Args:
        coil_parts (list): List of coil parts.
        target_mesh (Mesh): The target mesh.
        secondary_target_mesh (Mesh): The secondary target mesh.
        input (object): Input parameters.

    Returns:
        target_field_out (TargetField): Target field information (3 x m).
        is_suppressed_point (ndarray): Array indicating whether a point is suppressed or not (m).

    Raises:
        FileNotFoundError if the file target_field_definition_field_name can not be found.
        KeyError if the field target_field_definition_field_name can not be found in the loaded data.
        ValueError if the shape of the data in the target field file is inconsistent.
    """
    target_field_out = TargetField()

    # Define the target field
    if input_args.target_field_definition_file != 'none':
        # Load target field definition file
        target_field_definition_file = find_file('target_fields', input_args.target_field_definition_file)
        [loaded_target_field] = np.load(target_field_definition_file, allow_pickle=True)

        target_field_def = input_args.target_field_definition_field_name

        if target_field_def in loaded_target_field:
            loaded_field = loaded_target_field[target_field_def]

            target_field_out.coords = loaded_target_field['coords']

            if len(loaded_field.shape) == 1:
                if loaded_field.shape[0] != target_field_out.coords.shape[1]:
                    raise ValueError("Target field '%s' length (%s) does not match 'coords' shape (%s)",
                                     target_field_def, loaded_field.shape, target_field_out.coords.shape)
                target_field_out.b = np.vstack((np.zeros_like(loaded_field), np.zeros_like(loaded_field), loaded_field))
            else:
                if loaded_field.shape != target_field_out.coords.shape:
                    raise ValueError("Target field '%s' shape (%s) does not match 'coords' shape (%s)",
                                     target_field_def, loaded_field.shape, target_field_out.coords.shape)
                target_field_out.b = loaded_field

            is_suppressed_point = np.zeros(target_field_out.b.shape[1])
            target_field_out.weights = np.ones_like(target_field_out.b)
            target_field_out.target_field_group_inds = np.ones(target_field_out.b.shape[1])
        else:
            raise KeyError(f"The target field with name '{target_field_def}' does not exist in the provided file.")

    else:
        if target_mesh is not None:  # Create evenly distributed points within the surface of the "target mesh"
            if not input_args.use_only_target_mesh_verts:
                target_mesh_x_bounds = [np.min(target_mesh.get_vertices()[:, 0]),
                                        np.max(target_mesh.get_vertices()[:, 0])]
                target_mesh_y_bounds = [np.min(target_mesh.get_vertices()[:, 1]),
                                        np.max(target_mesh.get_vertices()[:, 1])]
                target_mesh_z_bounds = [np.min(target_mesh.get_vertices()[:, 2]),
                                        np.max(target_mesh.get_vertices()[:, 2])]

                x_size = target_mesh_x_bounds[1] - target_mesh_x_bounds[0]
                y_size = target_mesh_y_bounds[1] - target_mesh_y_bounds[0]
                z_size = target_mesh_z_bounds[1] - target_mesh_z_bounds[0]

                num_points_per_x_dim = input_args.target_region_resolution
                num_points_per_y_dim = input_args.target_region_resolution
                num_points_per_z_dim = input_args.target_region_resolution

                target_x_coords = np.linspace(target_mesh_x_bounds[0], target_mesh_x_bounds[1], num_points_per_x_dim)
                target_y_coords = np.linspace(target_mesh_y_bounds[0], target_mesh_y_bounds[1], num_points_per_y_dim)
                target_z_coords = np.linspace(target_mesh_z_bounds[0], target_mesh_z_bounds[1], num_points_per_z_dim)

                target_grid_x, target_grid_y, target_grid_z = np.meshgrid(
                    target_x_coords, target_y_coords, target_z_coords)
                target_points = np.vstack((target_grid_x.ravel(), target_grid_y.ravel(), target_grid_z.ravel()))

                # Remove points not inside the target surface
                in_indices = target_mesh.contains(target_points.T)
                target_points = target_points[:, in_indices]
                # Add surface vertices from target mesh
                target_points3 = np.hstack((target_mesh.get_vertices().T, target_points))  
            else:
                target_points3 = target_mesh.get_vertices().T
        else:  # Define target point coordinates as points inside a sphere of a given radius
            num_points_per_dim = input_args.target_region_resolution
            target_x_coords = np.linspace(-2, 2, 4*(num_points_per_dim-1)+1) * input_args.target_region_radius
            target_y_coords = np.linspace(-2, 2, 4*(num_points_per_dim-1)+1) * input_args.target_region_radius
            target_z_coords = np.linspace(-2, 2, 4*(num_points_per_dim-1)+1) * input_args.target_region_radius
            target_grid_x, target_grid_y, target_grid_z = np.meshgrid(target_x_coords, target_y_coords, target_z_coords)

            # For some unknown reason I need to swap y and z coords to match MATLAB
            # target_points = np.vstack((target_grid_x.ravel(), target_grid_y.ravel(), target_grid_z.ravel()))
            target_points = np.vstack((target_grid_x.ravel(), target_grid_z.ravel(), target_grid_y.ravel()))

            # Select points inside a sphere
            # Calculate the Euclidean distance for each point
            distances = np.sqrt(np.sum(target_points ** 2, axis=0))

            # Filter out points outside the target region radius
            target_points2 = target_points[:, distances <= input_args.target_region_radius]

            if input_args.set_roi_into_mesh_center:
                all_verts = np.vstack([part.coil_mesh.get_vertices() for part in coil_parts])
                mean_pos = np.mean(all_verts, axis=0, keepdims=True)
                target_points3 = target_points2 - mean_pos.T
            else:
                target_points3 = target_points2

        # Remove identical points
        _, unique_inds = np.unique(target_points3, axis=1, return_index=True)
        target_points = target_points3[:, sorted(unique_inds)]
        # target_points = target_points3

        # Define the target field shape
        def field_func(x, y, z): return eval(input_args.field_shape_function)
        target_field = np.zeros_like(target_points)
        target_field[2, :] = field_func(target_points[0, :], target_points[1, :], target_points[2, :])

        # Add points where the magnetic field should be suppressed (=>0)
        if secondary_target_mesh is not None:
            num_suppressed_points = secondary_target_mesh.get_vertices().shape[0]
            target_points = np.hstack((target_points, secondary_target_mesh.get_vertices().T))
            target_field = np.hstack((target_field, np.zeros((target_field.shape[0], num_suppressed_points))))
            is_suppressed_point = np.zeros(target_points.shape[1], dtype=bool)
            is_suppressed_point[-num_suppressed_points:] = True
        else:
            is_suppressed_point = np.zeros(target_points.shape[1], dtype=bool)

        # Scale the fields to a targeted strength
        max_field_point_ind = np.argmax(target_field[2, :])
        min_field_point_ind = np.argmin(target_field[2, :])
        max_target_distance = np.linalg.norm(
            target_points[:, max_field_point_ind] - target_points[:, min_field_point_ind])
        max_field_difference = np.max(target_field[2, :]) - np.min(target_field[2, :])

        if abs(max_field_difference) > 10**(-10):
            target_field = target_field / (max_field_difference / max_target_distance) * \
                input_args.target_gradient_strength
        else:
            target_field = target_field * input_args.target_gradient_strength

        # Define weightings from 0 to 1 that weights the significance of target points
        target_field_weighting = np.ones(target_field.shape[1])
        target_field_weighting[is_suppressed_point] = input_args.secondary_target_weight
        target_field_group_inds = np.ones(target_field.shape[1])
        target_field_group_inds[is_suppressed_point] = 2

        # Calculate the gradients from the symbolic definition of the target field
        target_dbzbx, target_dbzby, target_dbzbz = symbolic_calculation_of_gradient(input_args, target_field)

        if get_level() > DEBUG_BASIC:
            log.debug(" Final target_field:\n%s", target_field[:, :5])
            log.debug(" Final target_points:\n%s", target_points[:, :5])
        target_field_out.b = target_field
        target_field_out.coords = target_points
        target_field_out.weights = target_field_weighting
        target_field_out.target_field_group_inds = target_field_group_inds
        target_field_out.target_gradient_dbdxyz = np.array([target_dbzbx, target_dbzby, target_dbzbz])

    return target_field_out, is_suppressed_point


def symbolic_calculation_of_gradient(input_args, target_field):
    """
    Calculate the gradients from the symbolic definition of the target field.

    Args:
        input_args (DataStructure): Input parameters.
        target_field (ndarray): (3,m) Target field values.

    Depends on the following input_args:
        - field_shape_function

    Returns:
        target_dbzbx (ndarray): Gradient in x-direction.
        target_dbzby (ndarray): Gradient in y-direction.
        target_dbzbz (ndarray): Gradient in z-direction.
    """
    try:

        x, y, z = symbols('x y z')
        dbzdx_expr = diff(input_args.field_shape_function, x)
        dbzdy_expr = diff(input_args.field_shape_function, y)
        dbzdz_expr = diff(input_args.field_shape_function, z)

        # Convert expressions to string representations
        dbzdx_str = str(dbzdx_expr)
        dbzdy_str = str(dbzdy_expr)
        dbzdz_str = str(dbzdz_expr)

        # DEBUG
        if input_args.debug >= DEBUG_BASIC:
            log.debug(' - dbzdx_fun: %s', dbzdx_str)
            log.debug(' - dbzdy_fun: %s', dbzdy_str)
            log.debug(' - dbzdz_fun: %s', dbzdz_str)

        # Define lambdify functions for array-wise operations
        dbzdx_fun = lambdify((x, y, z), dbzdx_str)
        dbzdy_fun = lambdify((x, y, z), dbzdy_str)
        dbzdz_fun = lambdify((x, y, z), dbzdz_str)

        # Evaluate the lambdify functions
        target_dbzbx = dbzdx_fun(target_field[0, :], target_field[1, :], target_field[2, :])
        target_dbzby = dbzdy_fun(target_field[0, :], target_field[1, :], target_field[2, :])
        target_dbzbz = dbzdz_fun(target_field[0, :], target_field[1, :], target_field[2, :])

        if is_multivalued(target_dbzbx) == False:
            target_dbzbx = np.repeat(target_dbzbx, target_field.shape[1])

        if is_multivalued(target_dbzby) == False:
            target_dbzby = np.repeat(target_dbzby, target_field.shape[1])

        if is_multivalued(target_dbzbz) == False:
            target_dbzbz = np.repeat(target_dbzbz, target_field.shape[1])

    except Exception as e:
        log.error(' Exception: %s', e)
        target_dbzbx = np.zeros_like(target_field[0, :])
        target_dbzby = np.zeros_like(target_field[1, :])
        target_dbzbz = np.zeros_like(target_field[2, :])
        log.error('Gradient Calculation from Symbolic Target failed')

    return target_dbzbx, target_dbzby, target_dbzbz


def is_multivalued(variable):
    """
    Checks if the given variable is multi-valued.

    Args:
        variable: The variable to be checked.

    Returns:
        bool: True if the variable is multi-valued, False otherwise.
    """
    return isinstance(variable, (list, tuple, set, dict, np.ndarray))
