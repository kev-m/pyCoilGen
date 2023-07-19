from dataclasses import dataclass
from typing import List
import numpy as np

# Logging
import logging

# Local imports
from sub_functions.data_structures import CoilPart
from sub_functions.constants import get_level, DEBUG_VERBOSE

# Testing:
import trimesh

log = logging.getLogger(__name__)

###########################################################
# TODO: Debugging - remove this when verified
from helpers.extraction import load_matlab, print_structure
from helpers.visualisation import compare, compare_contains, visualize_connections, visualize_vertex_connections
#
###########################################################


###########################################################
# TODO: DEVELOPMENT: Move these to DataStructures


@dataclass
class Loop:
    loop = None


# Define the structure for unarranged loops
class UnarrangedLoop:
    def __init__(self):
        self.edge_inds = []
        self.uv = np.zeros((1,2), dtype=float)
        self.current_orientation : float = None
    
    def add_edge(self, edge):
        self.edge_inds.append(edge)
        
    def add_uv(self, uv):
        self.uv = np.vstack((self.uv, [uv]))

class UnarrangedLoopContainer:
    def __init__(self):
        self.loop: List[UnarrangedLoop] = []

@dataclass
class ContourLine:
    uv = None
    potential : float = None
    current_orientation : List[float] = None

@dataclass
# Define the structure for unsorted points
class UnsortedPoints:
    potential : float = None
    edge_ind = None
    uv = None


@dataclass
class RawPart:
    unsorted_points: List[UnsortedPoints] = None
    unarranged_loops: List[UnarrangedLoop] = None
#
##########################################################


# ChatGPT error, create this function
def numel(x):
    return len(x)


def calc_contours_by_triangular_potential_cuts(coil_parts: List[CoilPart]):
    """
    Center the stream function potential around zero and add zeros around the periphery.

    Args:
        coil_parts (list): List of coil parts.

    Returns:
        coil_parts (list): Updated list of coil parts.
    """


    ###############################################################################
    # MATLAB comparison data
    # TODO: Remove this
    matlab_data = load_matlab('debug/ygradient_coil_calc_contours')
    data = matlab_data['data']
    print_structure(data)

    m_edge_nodes1 = data.edge_nodes1 - 1 # Fix base 1 indexing offsets
    m_edge_nodes2 = data.edge_nodes2 - 1 # Fix base 1 indexing offsets
    m_edge_attached_triangles_inds = data.edge_attached_triangles_inds
    m_edge_attached_triangles = data.edge_attached_triangles - 1 # Fix base 1 indexing offsets
    m_edge_opposed_nodes = data.edge_opposed_nodes - 1 # Fix base 1 indexing offsets
    m_min_edge_potentials1 = data.min_edge_potentials1
    m_min_edge_potentials2 = data.min_edge_potentials2
    m_max_edge_potentials1 = data.max_edge_potentials1
    m_max_edge_potentials2 = data.max_edge_potentials2
    m_rep_contour_level_list = data.rep_contour_level_list
    m_edge_node_potentials = data.edge_node_potentials
    m_potential_cut_criteria1 = data.potential_cut_criteria1
    m_potential_cut_criteria2 = data.potential_cut_criteria2
    m_potential_cut_criteria3 = data.potential_cut_criteria3
    m_tri_below_pot_step = data.tri_below_pot_step
    m_tri_above_pot_step = data.tri_above_pot_step
    m_u_cut_point = data.u_cut_point
    m_v_cut_point = data.v_cut_point
    m_u_component_edge_vectors = data.u_component_edge_vectors2
    m_v_component_edge_vectors = data.v_component_edge_vectors2
    m_potential_sorted_cut_points = data.potential_sorted_cut_points2
    log.debug(" -- m_potential_sorted_cut_points.shape: %s", m_potential_sorted_cut_points.shape)
    #m_unsorted_points = data.unsorted_points # ???
    #m_unarranged_loops = data.unarranged_loops # ???
    m_raw = data.raw
    #
    ###############################################################################


    for part_ind in range(len(coil_parts)):
        part = coil_parts[part_ind]
        part_mesh = part.coil_mesh
        part_vertices = part_mesh.get_vertices()  # .T  # Transpose vertices
        part_uv = part_mesh.uv  # .T  # Transpose UV coordinates
        part_faces = part_mesh.get_faces()  # .T  # Transpose faces

        # For the UV mesh, calculate:
        # 1. edge_nodes: The edges, i.e. the list of connected vertices (array m x 2)
        # 2. edge_attached_triangles: The attached triangles, i.e. the list of triangles that share an edge (array of n).
        # 3. num_attached_tris: The list of

        # Compute edges and attached triangles
        mesh = trimesh.Trimesh(vertices=part_uv,
                               faces=part_faces,
                               process=False)
        # Returns the edges that are shared by the adjacent faces (index into faces array).
        edge_faces = mesh.face_adjacency
        # Returns the edges that are shared by the adjacent faces (index into vertices array).
        edge_nodes = mesh.face_adjacency_edges
        num_edges = edge_nodes.shape[0]

        if get_level() >= DEBUG_VERBOSE:
            log.debug(" -- edge_faces shape: %s, max(%d)", edge_faces.shape, np.max(edge_faces))  # 696,2: Max: 263
            log.debug(" -- edge_nodes shape: %s, max(%d)", edge_nodes.shape, np.max(edge_nodes))  # 696,2: Max: 263

        if get_level() > DEBUG_VERBOSE:
            visualize_vertex_connections(part_uv, 800, 'images/edge_nodes_p.png', edge_nodes)
            visualize_vertex_connections(part_uv, 800, 'images/edge_nodes_m.png', m_edge_nodes2)


        # 1	2
        # 2	3
        # 2	4
        # 2	7
        # 2	8
        # 2	9
        # . .
        # edge_nodes

        # Goal: The pair of vertices of the faces that share an edge.
        # MATLAB: 696,2 Max: 264 (num vertices)
        # [1,8,2]	[1,2,3]
        # [4,3,2]	[1,2,3]
        # [4,2,10]	[4,3,2]
        # [7,2,8]	[2,7,9]
        # [1,8,2]	[7,2,8]
        # NOTE: Vertix arrays order is reversed compared to MATLAB: [0,1,2] [0,7,1]	vs [1,8,2]	[1,2,3]
        edge_attached_triangles = np.empty((num_edges, 2, 3), dtype=int)
        for index, edges in enumerate(edge_faces):
            # Must swap node indices, to correct index order
            #edge_attached_triangles[index] = np.array((part_faces[edges[0]], part_faces[edges[1]]))
            edge_attached_triangles[index] = np.array((part_faces[edges[1]], part_faces[edges[0]]))

        # 696,2,3: Max: 263
        log.debug(" -- edge_attached_triangles shape: %s, max(%d)",
                  edge_attached_triangles.shape, np.max(edge_attached_triangles))

        # Take only the edge opposing nodes of these triangles
        # MATLAB: 696,2 Max: 264 (num vertices)
        # 8	3
        # 4	1
        # 10 3
        # 8	9
        # 1	7
        # 7	10
        # . .
        edge_opposed_nodes = np.zeros_like(edge_nodes)
        for x_ind in range(edge_nodes.shape[0]):
            edge_opposed_nodes[x_ind, 0] = np.setdiff1d(edge_attached_triangles[x_ind, 0], edge_nodes[x_ind])
            edge_opposed_nodes[x_ind, 1] = np.setdiff1d(edge_attached_triangles[x_ind, 1], edge_nodes[x_ind])

        log.debug(" -- edge_opposed_nodes shape: %s, max(%d)", edge_opposed_nodes.shape, np.max(edge_opposed_nodes))

        ###############################################################################
        # TODO: Verify: edge_nodes, edge_opposed_nodes
        assert compare_contains(edge_nodes, m_edge_nodes2) # PASS
        # assert compare_contains(edge_opposed_nodes, m_edge_opposed_nodes) # FAIL: Index 3 is reversed compared to MATLAB index 7
        #
        ###############################################################################


        # Test for all edges whether they cut one of the potential levels
        num_potentials = len(part.potential_level_list)
        rep_contour_level_list = np.tile(part.potential_level_list, (edge_nodes.shape[0], 1))
        edge_node_potentials = part.stream_function[edge_nodes]
        min_edge_potentials = np.min(edge_node_potentials, axis=1)
        max_edge_potentials = np.max(edge_node_potentials, axis=1)
        ###############################################################################
        # TODO: Verify: min_edge_potentials, max_edge_potentials Part 1
        assert compare_contains(min_edge_potentials, m_min_edge_potentials1) # PASS
        assert compare_contains(max_edge_potentials, m_max_edge_potentials1) # PASS
        #
        ###############################################################################
        min_edge_potentials = np.tile(min_edge_potentials[:, np.newaxis], (1, num_potentials))
        max_edge_potentials = np.tile(max_edge_potentials[:, np.newaxis], (1, num_potentials))
        ###############################################################################
        # TODO: Verify: min_edge_potentials, max_edge_potentials Part 2
        assert compare_contains(min_edge_potentials, m_min_edge_potentials2) # PASS
        assert compare_contains(max_edge_potentials, m_max_edge_potentials2) # PASS
        #
        ###############################################################################


        tri_below_pot_step = max_edge_potentials > rep_contour_level_list
        tri_above_pot_step = min_edge_potentials < rep_contour_level_list
        potential_cut_criteria = np.logical_and(tri_below_pot_step, tri_above_pot_step)
        potential_cut_criteria = potential_cut_criteria.astype(float)
        potential_cut_criteria[np.where(potential_cut_criteria == 0)] = np.nan

        # Calculate for each edge which cuts a potential the uv coordinates
        edge_lengths = np.linalg.norm(part_uv[edge_nodes[:, 1]] - part_uv[edge_nodes[:, 0]], axis=1)
        edge_lengths = np.tile(edge_lengths[:, np.newaxis], (1, num_potentials))
        edge_potential_span = edge_node_potentials[:, 1] - edge_node_potentials[:, 0]
        edge_potential_span = np.tile(edge_potential_span[:, np.newaxis], (1, num_potentials))

        pot_dist_to_step = rep_contour_level_list - np.tile(edge_node_potentials[:, 0][:, np.newaxis],
                                                            (1, num_potentials))
        cut_point_distance_to_edge_node_1 = np.abs(pot_dist_to_step / edge_potential_span * edge_lengths)

        u_component_edge_vectors = part_uv[edge_nodes[:, 1], 0] - part_uv[edge_nodes[:, 0], 0]
        u_component_edge_vectors = np.tile(u_component_edge_vectors[:, np.newaxis], (1, num_potentials))
        v_component_edge_vectors = part_uv[edge_nodes[:, 1], 1] - part_uv[edge_nodes[:, 0], 1]
        v_component_edge_vectors = np.tile(v_component_edge_vectors[:, np.newaxis], (1, num_potentials))

        first_edge_node_u = np.tile(part_uv[edge_nodes[:, 0], 0][:, np.newaxis], (1, num_potentials))
        first_edge_node_v = np.tile(part_uv[edge_nodes[:, 0], 1][:, np.newaxis], (1, num_potentials))
        u_cut_point = potential_cut_criteria * (
            first_edge_node_u + u_component_edge_vectors * cut_point_distance_to_edge_node_1 / edge_lengths)
        v_cut_point = potential_cut_criteria * (
            first_edge_node_v + v_component_edge_vectors * cut_point_distance_to_edge_node_1 / edge_lengths)

        # Create cell by sorting the cut points to the corresponding potential levels
        potential_sorted_cut_points = np.zeros((num_potentials), dtype=object) # 20 x M x 3
        value3 = np.arange(edge_nodes.shape[0], dtype=int)
        for pot_ind in range(num_potentials):
            cut_points = np.column_stack(
                (u_cut_point[:, pot_ind], v_cut_point[:, pot_ind], value3))
            cut_points = cut_points[~np.isnan(cut_points[:, 0])]
            potential_sorted_cut_points[pot_ind] = cut_points

        # End of Part 1

        ###############################################################################
        # TODO: Verify: potential_cut_criteria, potential_sorted_cut_points

        assert compare(rep_contour_level_list, m_rep_contour_level_list) # PASS
        assert compare_contains(min_edge_potentials, m_min_edge_potentials2) # PASS
        assert compare_contains(max_edge_potentials, m_max_edge_potentials2) # PASS

        assert compare_contains(potential_cut_criteria, m_potential_cut_criteria3)  # PASS
        # assert compare_contains(edge_opposed_nodes, m_edge_opposed_nodes) # FAIL: Index 3 is reversed compared to MATLAB index 7

        # FAIL: The edge nodes indices are different?
        # assert compare(potential_sorted_cut_points, m_potential_sorted_cut_points)

        #
        ###############################################################################

        # Start of Part 2
        # Create the unsorted points structure
        part.raw = RawPart()
        empty_potential_groups = [potential_sorted_cut_points[i] == [] for i in range(len(potential_sorted_cut_points))]
        num_false = len(empty_potential_groups) - sum(empty_potential_groups)
        part.raw.unsorted_points = [UnsortedPoints() for _ in range(num_false)]
        part.raw.unarranged_loops = [UnarrangedLoopContainer() for _ in range(num_false)]
        running_ind = 0

        for struct_ind in range(len(empty_potential_groups)):
            if not empty_potential_groups[struct_ind]:
                part.raw.unsorted_points[running_ind].potential = part.potential_level_list[struct_ind]
                part.raw.unsorted_points[running_ind].edge_ind = potential_sorted_cut_points[struct_ind][:, 2].astype(int)
                part.raw.unsorted_points[running_ind].uv = potential_sorted_cut_points[struct_ind][:, :2]
                running_ind += 1

        # Separate loops within potential groups

        # Create loops
        for potential_group in range(len(part.raw.unsorted_points)):
            all_current_edges = edge_nodes[part.raw.unsorted_points[potential_group].edge_ind]
            all_current_opposed_nodes = edge_opposed_nodes[part.raw.unsorted_points[potential_group].edge_ind]
            all_current_uv_coords = part.raw.unsorted_points[potential_group].uv

            set_new_start = True
            num_build_loops = 0
            edge_already_used = np.zeros(all_current_edges.shape[0], dtype=int)

            group_unarranged_loop = part.raw.unarranged_loops[potential_group]

            while not np.all(edge_already_used):
                if set_new_start:
                    num_build_loops += 1
                    starting_edge = np.min(np.where(edge_already_used == 0)[0])

                    loop_item = UnarrangedLoop()
                    loop_item.add_uv(all_current_uv_coords[starting_edge])
                    loop_item.add_edge(all_current_edges[starting_edge])
                    group_unarranged_loop.loop.append(loop_item)
                    edge_already_used[starting_edge] = 1
                    current_edge = starting_edge

                    current_edge_nodes = all_current_edges[current_edge]
                    neighbouring_free_next_edges = np.where(
                        np.any(
                            np.logical_or(all_current_edges == all_current_opposed_nodes[current_edge, 0],
                                          all_current_edges == all_current_opposed_nodes[current_edge, 1]), axis=1) &
                        np.any(
                            np.logical_or(all_current_edges == current_edge_nodes[0],
                                          all_current_edges == current_edge_nodes[1]), axis=1)
                    )[0]

                    if len(neighbouring_free_next_edges) == 0:
                        break
                    elif len(neighbouring_free_next_edges) == 1:
                        next_edge = neighbouring_free_next_edges[0]
                    else:
                        if not edge_already_used[neighbouring_free_next_edges[0]]:
                            next_edge = neighbouring_free_next_edges[0]
                        else:
                            next_edge = neighbouring_free_next_edges[1]

                        set_new_start = False

                while next_edge != starting_edge:
                    edge_already_used[next_edge] = 1
                    loop_item = group_unarranged_loop.loop[num_build_loops - 1]
                    loop_item.add_uv(all_current_uv_coords[next_edge])
                    loop_item.add_edge(all_current_edges[next_edge])

                    current_edge = next_edge

                    current_edge_nodes = all_current_edges[current_edge]
                    possible_next_edges = np.where(
                        np.any(
                            np.logical_or(all_current_edges == all_current_opposed_nodes[current_edge, 0],
                                          all_current_edges == all_current_opposed_nodes[current_edge, 1]), axis=1) &
                        np.any(
                            np.logical_or(all_current_edges == current_edge_nodes[0],
                                          all_current_edges == current_edge_nodes[1]), axis=1)
                    )[0]
                    possible_next_edges = np.setdiff1d(possible_next_edges, np.where(edge_already_used == 1)[0])

                    if len(possible_next_edges) == 0:
                        break
                    else:
                        if len(possible_next_edges) == 1:
                            next_edge = possible_next_edges[0]
                        else:
                            if not edge_already_used[possible_next_edges[0]]:
                                next_edge = possible_next_edges[0]
                            else:
                                next_edge = possible_next_edges[1]

                set_new_start = True

        # End of Part 2

        # Start of Part 3
        # Evaluate current orientation for each loop
        ## part.raw.unarranged_loops[numel(part.raw.unsorted_points)-1].loop[0].current_orientation = []

        for pot_ind in range(numel(part.raw.unsorted_points)):
            center_segment_potential = part.raw.unsorted_points[pot_ind].potential
            potential_loop = part.raw.unarranged_loops[pot_ind]

            for loop_ind in range(numel(potential_loop.loop)):
                potential_loop_item = potential_loop.loop[loop_ind]
                if numel(potential_loop_item.edge_inds) > 2:
                    test_edge = int(np.floor(len(potential_loop_item.edge_inds) / 2))
                    first_edge = potential_loop_item.edge_inds[test_edge]
                    second_edge = potential_loop_item.edge_inds[test_edge + 1]
                    node_1 = np.intersect1d(first_edge, second_edge)
                    node_2 = np.setdiff1d(first_edge, node_1)
                    node_3 = np.setdiff1d(second_edge, node_1)
                    node_1_uv = coil_parts[part_ind].coil_mesh.uv[node_1, :][0] # Change shape from (1,2) to (2,)
                    node_2_uv = coil_parts[part_ind].coil_mesh.uv[node_2, :][0] # Change shape from (1,2) to (2,)
                    node_3_uv = coil_parts[part_ind].coil_mesh.uv[node_3, :][0] # Change shape from (1,2) to (2,)
                    node_1_pot = coil_parts[part_ind].stream_function[node_1]
                    node_2_pot = coil_parts[part_ind].stream_function[node_2]
                    node_3_pot = coil_parts[part_ind].stream_function[node_3]

                    # Calculate the 2D gradient of the triangle
                    # list indices must be integers or slices, not tuple
                    center_segment_position = (potential_loop_item.uv[test_edge] +
                                               potential_loop_item.uv[test_edge + 1]) / 2

                    vec_center_node_1 = node_1_uv - center_segment_position
                    vec_center_node_2 = node_2_uv - center_segment_position
                    vec_center_node_3 = node_3_uv - center_segment_position

                    pot_diff_center_node_1 = node_1_pot - center_segment_potential
                    pot_diff_center_node_2 = node_2_pot - center_segment_potential
                    pot_diff_center_node_3 = node_3_pot - center_segment_potential

                    pot_gradient_vec = vec_center_node_1 * pot_diff_center_node_1 + \
                        vec_center_node_2 * pot_diff_center_node_2 + \
                        vec_center_node_3 * pot_diff_center_node_3

                    # Test the chirality of the segment on the potential gradient on that segment
                    segment_vec = potential_loop_item.uv[test_edge + 1] - \
                        potential_loop_item.uv[test_edge]

                    cross_vec = np.cross([segment_vec[0], segment_vec[1], 0], [
                                         pot_gradient_vec[0], pot_gradient_vec[1], 0])

                    potential_loop_item.current_orientation = np.sign(cross_vec[2])

                    if potential_loop_item.current_orientation == -1:
                        potential_loop_item.uv = np.flipud(
                            potential_loop_item.uv)
                        potential_loop_item.edge_inds = np.flipud(
                            potential_loop_item.edge_inds)
                else:
                    raise ValueError('Some loops are too small and contain only 2 points, therefore ill-defined')
        # End of Part 3

        # Start of Part 4
        # Build the contour lines
        part.contour_lines = []
        for pot_ind in range(numel(part.raw.unsorted_points)):
            potential_loop = part.raw.unarranged_loops[pot_ind]
            for loop_ind in range(numel(potential_loop.loop)):
                potential_loop_item = potential_loop.loop[loop_ind]
                contour_line = ContourLine()
                # MATLAB: 7x2
                #     0.9614    0.0091
                #     0.9648   -0.0107
                #     0.9616   -0.0310
                #     0.9287   -0.1977
                #     0.8945   -0.0122
                #     0.9078    0.0432
                #     0.9322    0.1542
                contour_line.uv = potential_loop_item.uv.T # uv: (2,8) | M: 7x2 -> 2x7
                # MATLAB: -5.4430e+03
                contour_line.potential = part.raw.unsorted_points[pot_ind].potential

                # Find the current orientation (for comparison with other loops)
                # MATLAB: 0.939, -0.0065
                uv_center = np.mean(contour_line.uv, axis=1) # (2,)
                # MATLAB: (3x6)
                # 0.0255	0.029	0.0258	-0.0071	-0.0414	-0.0281
                # 0.0156	-0.0043	-0.0245	-0.1913	-0.0058	0.0496
                # 0	0	0	0	0	0
                uv_to_center_vecs = contour_line.uv[:, :-1] - np.expand_dims(uv_center, axis=1) # (2,5) | M: 2x6
                uv_to_center_vecs = np.concatenate(
                    (uv_to_center_vecs, np.zeros((1, uv_to_center_vecs.shape[1]))), axis=0) # (3,5) | M: 3x6
                # MATLAB: (3x6)
                # 0.0034	-0.0032	-0.0329	-0.0342	0.0133	0.0244
                # -0.0199	-0.0203	-0.1667	0.1855	0.0554	0.111
                # 0	0	0	0	0	0
                uv_vecs = contour_line.uv[:, 1:] - contour_line.uv[:, :-1] # (2,5)
                uv_vecs = np.concatenate((uv_vecs, np.zeros((1, uv_vecs.shape[1]))), axis=0) # (3,5)
                log.debug(" ---- here ---")
                # incompatible dimensions for cross product
                # (dimension must be 2 or 3)
                # MATLAB: (3,6) x (3,6) = (3x6)
                # 0	0	0	0	0	0
                # 0	0	0	0	0	0
                # -0.0006	-0.0006	-0.0051	-0.0079	-0.0022	-0.0043
                rot_vecs = np.cross(uv_to_center_vecs.T, uv_vecs.T) # (3,5) x (3,5) | M: (3x6) x (3x6)
                track_orientation = np.sign(np.sum(rot_vecs[:,2])) # MATLAB: -1
                contour_line.current_orientation = track_orientation

                part.contour_lines.append(contour_line)
        # End of Part 4

    return coil_parts


if __name__ == "__main__":
    pass
