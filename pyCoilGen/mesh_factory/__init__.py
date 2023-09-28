"""Initialize the mesh_factory package.

This module provides functions to dynamically load plugins for mesh creation.
"""

import os
import importlib

def load_plugins():
    """Load all available mesh creation plugins.

    This function dynamically discovers and imports all Python files in the 
    mesh_factory directory (excluding this file), treating them as plugins.
    It returns a list of imported modules.

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
