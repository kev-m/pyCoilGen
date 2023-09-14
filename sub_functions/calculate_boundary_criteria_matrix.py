import numpy as np


def calculate_boundary_criteria_matrix(coil_mesh, basis_elements):
    """
    Calculate the boundary_criteria_matrix which enforces a constant stream function on the boundary nodes.

    Args:
        coil_mesh: The mesh representing the coil geometry.
        basis_elements: The basis elements containing information about the triangles and current densities.

    Returns:
        boundary_criteria_matrix: The matrix that enforces the boundary conditions.
    """
    num_nodes = coil_mesh.vertices.shape[1]
    boundary_criteria_matrix = np.zeros((num_nodes, num_nodes))

    # Mark the nodes that are boundary nodes
    boundary_nodes = []
    for boundary_ind in range(len(coil_mesh.boundary)):
        boundary_nodes.extend(coil_mesh.boundary[boundary_ind])
    is_boundary = np.isin(np.arange(1, num_nodes+1), boundary_nodes)

    # Define the criteria that currents do not leave the surface
    boundary_criteria = []
    for boundary_ind in range(len(coil_mesh.boundary)):
        boundary_criteria.extend(zip(coil_mesh.boundary[boundary_ind][1:], coil_mesh.boundary[boundary_ind][:-1]))

    for hhhh in range(len(boundary_criteria)):
        boundary_criteria_matrix[hhhh, boundary_criteria[hhhh][0]-1] = 1
        boundary_criteria_matrix[hhhh, boundary_criteria[hhhh][1]-1] = -1

    return boundary_criteria_matrix
