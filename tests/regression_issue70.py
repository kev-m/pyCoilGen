import numpy as np

# Test support
from pytest import approx

# Local import
from pyCoilGen.sub_functions.data_structures import DataStructure

# Function under test
from pyCoilGen.mesh_factory import load_mesh_factory_plugins


def test_issue70():

    planar_height = 0.75
    planar_width = 0.85
    num_lateral_divisions = 20
    num_longitudinal_divisions = 25
    target_normal_x = 0.0
    target_normal_y = 1.0
    target_normal_z = 0.0
    center_position_x = 0
    center_position_y = 0
    center_position_z = 0
    plane_distance = 0.5


    input_args = DataStructure(coil_mesh='create bi-planar mesh',
        biplanar_mesh_parameter_list=[planar_height, planar_width,
                                num_lateral_divisions, num_longitudinal_divisions,
                                target_normal_x, target_normal_y, target_normal_z,
                                center_position_x, center_position_y, center_position_z,
                                plane_distance],
    )

    print(f"input_args.coil_mesh => {input_args.coil_mesh}")
    plugin_name = input_args.coil_mesh.replace(' ', '_').replace('-', '_')
    plugins = load_mesh_factory_plugins()
    found = False
    for plugin in plugins:
        mesh_creation_function = getattr(plugin, plugin_name, None)
        if mesh_creation_function:
            coil_mesh = mesh_creation_function(input_args)
            found = True
            break

    assert found

    # Min x is -0.5 width
    assert np.min(coil_mesh.v[:, 0]) == approx(-planar_width/2.0)
    # Max x is 0.5 width
    assert np.max(coil_mesh.v[:, 0]) == approx(planar_width/2.0)

    # Min y is -0.5 height
    assert np.min(coil_mesh.v[:, 1]) == approx(-plane_distance/2.0)
    # Max y is 0.5 height
    assert np.max(coil_mesh.v[:, 1]) == approx(plane_distance/2.0)

    # Min z is 0
    assert np.min(coil_mesh.v[:, 2]) == -planar_height/2.0
    # Max z is 0
    assert np.max(coil_mesh.v[:, 2]) == planar_height/2.0
