#system imports
import numpy as np
from os import makedirs, path

# Test support
from pyCoilGen.sub_functions.data_structures import DataStructure, Mesh, CoilPart
from pyCoilGen.sub_functions.build_biplanar_mesh import build_biplanar_mesh
from pyCoilGen.sub_functions.parameterize_mesh import parameterize_mesh
from pyCoilGen.sub_functions.split_disconnected_mesh import split_disconnected_mesh
from pyCoilGen.sub_functions.load_preoptimized_data import load_preoptimized_data
from pyCoilGen.helpers.visualisation import compare

# Code under test
from pyCoilGen.helpers.persistence import save_preoptimised_data


def test_save_preoptimised_data():
    combined_mesh = build_biplanar_mesh(0.5, 0.5, 3, 3, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.2)
    parts = split_disconnected_mesh(Mesh(vertices=combined_mesh.vertices, faces=combined_mesh.faces))

    # Depends on the following properties of the CoilSolution:
    #    - target_field.coords, target_field.b
    #    - coil_parts[n].stream_function
    #    - combined_mesh.vertices, combined_mesh.faces
    #    - input_args.sf_dest_file

    fake_target_field = DataStructure(coords=np.ones((10, 3)), b=np.ones((10, 3)))
    fake_input_args = DataStructure(sf_dest_file='test_save_preoptimised_data')
    fake_solution = DataStructure(combined_mesh=combined_mesh, target_field=fake_target_field,
                                  coil_parts=parts, input_args=fake_input_args)

    # Fake up a stream function
    for coil_part in parts:
        coil_part.stream_function = np.ones((len(coil_part.coil_mesh.get_vertices())))

    ##################################################
    # Function under test
    filename1 = save_preoptimised_data(fake_solution, 'debug') # Save to default directory
    ##################################################

    assert 'debug' in filename1

    # Simplify test, use load_preoptimized_data to cross check
    crosscheck_input_args = DataStructure(sf_source_file=fake_input_args.sf_dest_file, 
                                          surface_is_cylinder_flag=True, circular_diameter_factor=1.0, debug=0)
    solution = load_preoptimized_data(crosscheck_input_args, 'debug')

    assert len(solution.coil_parts) == len(parts)
    assert compare(solution.combined_mesh.vertices, combined_mesh.vertices)
    # assert compare(solution.combined_mesh.faces, combined_mesh.faces) # Faces are in a different order
    for index, coil_part in enumerate(solution.coil_parts):
        t_part : CoilPart = parts[index]
        t_mesh : Mesh = t_part.coil_mesh

        # Verify the Mesh
        assert compare(t_mesh.get_vertices(), coil_part.coil_mesh.get_vertices())
        assert compare(t_mesh.get_faces(), coil_part.coil_mesh.get_faces())

        # Verify the stream_function
        assert compare(t_part.stream_function, coil_part.stream_function)

    # Verify the target_field (coords, b)
    target_field = solution.target_field
    assert compare(fake_target_field.coords, target_field.coords)
    assert compare(fake_target_field.b, target_field.b)

    # Test case 2: Writing to user-specified directory
    save_dir = path.join('debug', 'test')
    makedirs(save_dir, exist_ok=True)
    fake_solution.input_args.sf_dest_file=path.join(save_dir, 'test_save_preoptimised_data')

    ##################################################
    # Function under test
    filename2 = save_preoptimised_data(fake_solution) # Override default directory
    ##################################################
    assert 'Pre_Optimized_Solutions' not in filename2 
    assert path.exists(filename2)
    assert filename2.startswith(save_dir)

    crosscheck_input_args.sf_source_file = fake_solution.input_args.sf_dest_file
    ##################################################
    # Function under test
    solution2 = load_preoptimized_data(crosscheck_input_args) # Override default directory
    ##################################################


if __name__ == "__main__":
    import logging
    # Set up logging
    log = logging.getLogger(__name__)
    logging.basicConfig(level=logging.DEBUG)

    test_save_preoptimised_data()
