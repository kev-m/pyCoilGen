# Local import
from pyCoilGen.sub_functions.data_structures import DataStructure

# Function under test
from pyCoilGen.mesh_factory import load_mesh_factory_plugins


def test_build_planar_mesh():
    input_args = DataStructure(planar_mesh_parameter_list=[0.25, 0.25, 20, 20, 1, 0, 0, 0, 0, 0, 0])
    plugin_name = 'create planar mesh'.replace(' ', '_').replace('-', '_')
    plugins = load_mesh_factory_plugins()
    found = False
    for plugin in plugins:
        mesh_creation_function = getattr(plugin, plugin_name, None)
        if mesh_creation_function:
            coil_mesh = mesh_creation_function(input_args)
            found = True
            break

    assert found
