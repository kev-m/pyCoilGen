from pyCoilGen.sub_functions.read_mesh import read_mesh
from pyCoilGen.sub_functions.data_structures import DataStructure


def test_help():
    input_args = DataStructure(coil_mesh='help')
    coil, target, shield = read_mesh(input_args)

    assert coil is None


def test_coil_planar():
    input_args = DataStructure(coil_mesh='create planar mesh', 
                               planar_mesh_parameter_list=[0.25, 0.25, 20, 20, 1, 0, 0, 0, 0, 0, 0.2],
                               target_mesh='none', target_mesh_file='none',
                               shield_mesh='none', secondary_target_mesh_file='none')
    coil, target, shield = read_mesh(input_args)

    assert coil is not None
    assert target is None
    assert shield is None

def test_target_planar():
    input_args = DataStructure(coil_mesh='create cylinder mesh',
                               cylinder_mesh_parameter_list=[0.8, 0.3, 20, 20, 1, 0, 0, 0],
                               target_mesh='create planar mesh', target_mesh_file='none',
                               planar_mesh_parameter_list=[0.25, 0.25, 20, 20, 1, 0, 0, 0, 0, 0, 0.2],
                               shield_mesh='none', secondary_target_mesh_file='none')
    coil, target, shield = read_mesh(input_args)

    assert coil is not None
    assert target is not None
    assert shield is None

def test_target_circular():
    input_args = DataStructure(coil_mesh='create bi-planar mesh',
                               biplanar_mesh_parameter_list=[0.25, 0.25, 20, 20, 1, 0, 0, 0, 0, 0, 0.2],
                               target_mesh='create circular mesh', target_mesh_file='none',
                               circular_mesh_parameter_list=[0.25, 20, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,],
                               shield_mesh='create planar mesh', secondary_target_mesh_file='none',
                               planar_mesh_parameter_list=[0.25, 0.25, 20, 20, 1, 0, 0, 0, 0, 0, 0.2],
                               )
    coil, target, shield = read_mesh(input_args)

    assert coil is not None
    assert target is not None
    assert shield is not None


if __name__ == '__main__':
    test_target_circular()
