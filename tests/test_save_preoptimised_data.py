import numpy as np

# Test support
from pyCoilGen.sub_functions.data_structures import DataStructure, Mesh
from pyCoilGen.sub_functions.build_biplanar_mesh import build_biplanar_mesh
from pyCoilGen.sub_functions.parameterize_mesh import parameterize_mesh
from pyCoilGen.sub_functions.split_disconnected_mesh import split_disconnected_mesh

# Code under test
from pyCoilGen.helpers.persistence import save_preoptimised_data


def test_save_preoptimised_data():
    combined_mesh = build_biplanar_mesh(0.5, 0.5, 3, 3, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.2)
    parts = split_disconnected_mesh(Mesh(vertices=combined_mesh.vertices, faces=combined_mesh.faces))

    # Depends on the following properties of the CoilSolution:
    #    - target_field.coords, target_field.b
    #    - coil_parts[n].stream_function
    #    - combined_mesh.vertices, combined_mesh.faces
    #    - input_args.coords, target_field.b
    #    - input_args.persistence_dir, input_args.project_name

    fake_target_field = DataStructure(coords=np.ones((10, 3)), b=np.ones((10, 3)))
    fake_input_args = DataStructure(persistence_dir='debug', project_name='test_save_preoptimised_data')
    fake_solution = DataStructure(combined_mesh=combined_mesh, target_field=fake_target_field,
                                  coil_parts=parts, input_args=fake_input_args)

    # Fake up a stream function
    for coil_part in parts:
        coil_part.stream_function = np.ones((25))

    ##################################################
    # Function under test
    save_preoptimised_data(fake_solution)
    ##################################################


if __name__ == "__main__":
    import logging
    # Set up logging
    log = logging.getLogger(__name__)
    logging.basicConfig(level=logging.DEBUG)

    test_save_preoptimised_data()
