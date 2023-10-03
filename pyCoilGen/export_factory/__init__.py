"""Initialize the export_factory package.

This module provides functions to dynamically load plugins for exporting pyCoilGen artifacts.
"""

import os
import importlib

__exporter_plugins__ = []


def load_exporter_plugins():
    """Load all available export plugins.

    This function dynamically discovers and imports all Python files in the 
    export_factory directory (excluding this file), treating them as plugins.
    It returns a list of imported modules.

    Every plugin must be a module that exposes the following functions:

    - get_name()-> str       : Return the name of the mesh builder instruction.
    - get_parameters()->list : Return a list of tuples of the parameter names and default values.
    - register_args(parser)  : Called to register any required parameters with ArgParse.

    In addition, it must also provide an export function that matches the value returned by `get_name()`, e.g.:
    - export_stl_mesh(solution)

    Returns:
        list: A list of imported plugin modules.

    """
    if len(__exporter_plugins__) == 0:
        # Load all .py files in the export_factory directory
        for file_name in os.listdir(os.path.dirname(__file__)):
            if file_name.endswith(".py") and file_name != "__init__.py":
                module_name = f"pyCoilGen.export_factory.{file_name[:-3]}"
                module = importlib.import_module(module_name)
                __exporter_plugins__.append(module)

    return __exporter_plugins__
