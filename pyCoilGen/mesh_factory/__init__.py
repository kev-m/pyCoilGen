"""Initialize the mesh_factory package.

This module provides functions to dynamically load plugins for mesh creation.
"""

import os
import importlib

def load_mesh_factory_plugins():
    """Load all available mesh creation plugins.

    This function dynamically discovers and imports all Python files in the 
    mesh_factory directory (excluding this file), treating them as plugins.
    It returns a list of imported modules.

    Every plugin must be a module that exposes the following functions:

    - get_name()-> str       : Return the name of the mesh builder instruction.
    - get_parameters()->list : Return a list of tuples of the parameter names and default values.
    - register_args(parser)  : Called to register any required parameters with ArgParse.

    In addition, it must also provide a creator function that matches the value returned by `get_name()`, e.g.:
    - create_planar_mesh(input_args: argparse.Namespace) : Mesh or DataStructure(vertices, faces, normal)

    Returns:
        list: A list of imported plugin modules.

    """
    plugins = []

    # Load all .py files in the mesh_factory directory
    for file_name in os.listdir(os.path.dirname(__file__)):
        if file_name.endswith(".py") and file_name != "__init__.py":
            module_name = f"pyCoilGen.mesh_factory.{file_name[:-3]}"
            module = importlib.import_module(module_name)
            plugins.append(module)

    return plugins
