def calculate_resistance_matrix(coil_parts, input):
    """
    Calculate the resistance matrix for coil parts.

    Args:
        coil_parts (list): List of coil parts.
        input (dict): Input parameters.

    Returns:
        list: Updated coil parts with resistance matrix.

    """

    gauss_order = input['gauss_order']
    conductor_thickness = input['conductor_thickness']
    specific_conductivity_copper = input['specific_conductivity_conductor']
    material_factor = specific_conductivity_copper / conductor_thickness

    for part_ind in range(len(coil_parts)):
        if not input['temp_evalution']['use_preoptimization_temp']:
            num_nodes = len(coil_parts[part_ind]['basis_elements'])

            # Calculate node adjacency matrix
            node_adjacency_mat = np.zeros((num_nodes, num_nodes), dtype=bool)
            faces = coil_parts[part_ind]['coil_mesh']['faces']
            for tri_ind in range(faces.shape[1]):
                node_adjacency_mat[faces[0, tri_ind], faces[1, tri_ind]] = True
                node_adjacency_mat[faces[1, tri_ind], faces[2, tri_ind]] = True
                node_adjacency_mat[faces[2, tri_ind], faces[0, tri_ind]] = True

            vert1, vert2 = np.where(node_adjacency_mat)
            mesh_edges = np.column_stack((vert1, vert2))
            mesh_edges_non_unique = np.column_stack((np.arange(num_nodes), mesh_edges[:, 0]))
            mesh_edges_non_unique = np.vstack((mesh_edges_non_unique, np.column_stack((np.arange(num_nodes), mesh_edges[:, 1]))))

            node_adjacency_mat = np.logical_or(node_adjacency_mat, node_adjacency_mat.T)
            coil_parts[part_ind]['node_adjacency_mat'] = node_adjacency_mat

            # Calculate resistance matrix
            resistance_matrix = np.zeros((num_nodes, num_nodes))
            basis_elements = coil_parts[part_ind]['basis_elements']
            for edge_ind in range(mesh_edges_non_unique.shape[0]):
                node_ind1 = mesh_edges_non_unique[edge_ind, 0]
                node_ind2 = mesh_edges_non_unique[edge_ind, 1]
                overlapping_triangles = np.intersect1d(basis_elements[node_ind1]['triangles'], basis_elements[node_ind2]['triangles'])
                resistance_sum = 0

                if len(overlapping_triangles) > 0:
                    for overlapp_tri_ind in overlapping_triangles:
                        first_node_triangle_positon = np.where(basis_elements[node_ind1]['triangles'] == overlapp_tri_ind)[0]
                        second_node_triangle_positon = np.where(basis_elements[node_ind2]['triangles'] == overlapp_tri_ind)[0]
                        triangle_area = basis_elements[node_ind1]['area'][first_node_triangle_positon]
                        primary_current = basis_elements[node_ind1]['current'][first_node_triangle_positon]
                        secondary_current = basis_elements[node_ind2]['current'][second_node_triangle_positon]
                        resistance_sum += np.dot(primary_current, secondary_current) * (triangle_area)**2

                    resistance_matrix[node_ind1, node_ind2] = resistance_sum

            resistance_matrix += resistance_matrix.T
            resistance_matrix *= material_factor

            coil_parts[part_ind]['resistance_matrix'] = resistance_matrix

        else:
            coil_parts[part_ind]['resistance_matrix'] = input['temp']['coil_parts'][part_ind]['resistance_matrix']

    return coil_parts
