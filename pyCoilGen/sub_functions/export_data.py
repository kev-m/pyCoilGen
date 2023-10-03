# System imports
import os
from argparse import Namespace

# Logging
import logging

# Local imports
from pyCoilGen.export_factory import load_exporter_plugins

from .data_structures import CoilSolution

log = logging.getLogger(__name__)


def check_exporter_help(input_args: Namespace, print_function=print):
    """
    Print the help, if the exporter is 'help'.

    Args:
        input_args (Namespace): The project input arguments.
        print_function(function): The function used to write out the help.

    Returns:
        bool: True if help was requested.
    """
    plugin_name = input_args.exporter
    if plugin_name == 'help':
        print_function('Available exporter plugins are:')
        exporter_plugins = load_exporter_plugins()
        for plugin in exporter_plugins:
            name_function = getattr(plugin, 'get_name', None)
            parameters_function = getattr(plugin, 'get_parameters', None)
            if name_function:
                name = name_function()
                if parameters_function:
                    parameters = parameters_function()
                    parameter_name, default_value = parameters[0]
                    print_function(f"'{name}', Parameter: '{parameter_name}', Default values: {default_value}")
                    for i in range(1, len(parameters)):
                        print(f"\t\tParameter: '{parameter_name}', Default values: '{default_value}'")

                else:
                    print_function(f"'{name}', no parameters")
        return True
    return False


def export_data(solution: CoilSolution):
    """
    Use an exporter to save export data from the solution. 

    The exporter is specified using the `exporter` parameter.

    Args:
        solution (CoilSolution): The current coil solution.

    Returns:
        None

    Raises:
        ValueError if the export function can not be found.
    """
    input_args = solution.input_args

    # Read the list of exporters
    exporter_plugins = load_exporter_plugins()

    exporter = input_args.exporter

    plugin_name = exporter.replace(' ', '_').replace('-', '_')

    # Exit early if no exporter is specified.
    if plugin_name == 'none':
        log.debug("Exporter is 'none', exiting...")
        return

    print("Using exporter plugin: ", plugin_name)

    found = False
    for plugin in exporter_plugins:
        exporter_function = getattr(plugin, plugin_name, None)
        if exporter_function:
            log.debug("Calling exporter: %s", plugin_name)
            exporter_function(solution)
            found = True
            break

    if found == False:
        raise ValueError(f"Function {plugin_name} was not found in {input_args.exporter}")
