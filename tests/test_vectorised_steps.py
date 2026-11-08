"""
Compare the vectorised pipeline steps with the original loop implementations, which are kept here as the specification.

The steps are: calculate_one_ring_by_mesh, calculate_basis_functions, calculate_resistance_matrix and
create_sweep_along_surface. A small bi-planar design is run once, and the inputs of each step are recorded and given to
both implementations.
"""
# System
import copy
import logging
import sys
from pathlib import Path

# Libraries
import numpy as np
import pytest

# Local imports
import pyCoilGen.pyCoilGen_release as release
from pyCoilGen.sub_functions import calc_contours_by_triangular_potential_cuts as contours
from pyCoilGen.sub_functions import mesh_parameterization_iterative as parameterization
from pyCoilGen.sub_functions.calculate_basis_functions import calculate_basis_functions
from pyCoilGen.sub_functions.calculate_one_ring_by_mesh import calculate_one_ring_by_mesh
from pyCoilGen.sub_functions.calculate_resistance_matrix import calculate_resistance_matrix
from pyCoilGen.sub_functions.create_sweep_along_surface import create_sweep_along_surface
from pyCoilGen.sub_functions.data_structures import Mesh

# The harness modules are not a package: import them from their folder, as the harness tests do
sys.path.insert(0, str(Path(__file__).parent / 'regression'))
from harness_cases import CASES  # noqa: E402


# ---------------------------------------------------------------------------------------------------------------------
# The original implementations
# ---------------------------------------------------------------------------------------------------------------------

def reference_one_ring(coil_parts):
    for coil_part in coil_parts:
        part_mesh = coil_part.coil_mesh
        part_vertices = part_mesh.get_vertices()
        part_faces = part_mesh.get_faces()
        num_vertices = part_vertices.shape[0]
        node_triangles = part_mesh.vertex_faces()
        node_triangles_corners = [part_faces[x, :] for x in node_triangles]
        one_ring_list = np.empty(num_vertices, dtype=object)
        for node_ind in range(num_vertices):
            single_cell = node_triangles_corners[node_ind]
            one_ring_list[node_ind] = np.asarray([x[x != node_ind] for x in single_cell])
        for node_ind in range(num_vertices):
            for face_ind in range(len(one_ring_list[node_ind])):
                point_aa = part_vertices[one_ring_list[node_ind][face_ind][0]]
                point_bb = part_vertices[one_ring_list[node_ind][face_ind][1]]
                point_cc = part_vertices[node_ind]
                cross_vec = np.cross(point_bb - point_aa, point_aa - point_cc)
                if np.sign(np.dot(part_mesh.n[node_ind], cross_vec)) > 0:
                    one_ring_list[node_ind][face_ind] = np.flipud(one_ring_list[node_ind][face_ind])
        coil_part.one_ring_list = one_ring_list
        coil_part.node_triangles = node_triangles
    return coil_parts


def reference_basis_functions(coil_parts):
    """Returns, for each part, the matrices and the per-node arrays of the original calculate_basis_functions."""
    results = []
    for coil_part in coil_parts:
        part_mesh = coil_part.coil_mesh
        part_vertices = part_mesh.get_vertices()
        part_faces = part_mesh.get_faces()
        num_vertices, num_faces = part_vertices.shape[0], part_faces.shape[0]
        current_density_mat = np.zeros((num_vertices, num_faces, 3))
        elements = []
        max_count = 0
        for v in range(num_vertices):
            ring = coil_part.one_ring_list[v]
            count = len(ring)
            max_count = max(max_count, count)
            node_point = part_vertices[v]
            area, normal = np.zeros(count), np.zeros((count, 3))
            points, current = np.zeros((count, 3, 3)), np.zeros((count, 3))
            for t in range(count):
                point_b, point_c = part_vertices[ring[t][0]], part_vertices[ring[t][1]]
                area[t] = np.linalg.norm(np.cross(point_c - node_point, point_b - node_point)) / 2
                normal[t] = np.cross(point_c - node_point, point_c - point_b) / \
                    np.linalg.norm(np.cross(point_c - node_point, point_c - point_b))
                points[t] = np.asarray([node_point, point_b, point_c]).T
                current[t] = (point_c - point_b) / (2 * area[t])
                current_density_mat[v, coil_part.node_triangles[v][t]] = current[t]
            elements.append((area, normal, points, current))
        is_real = np.zeros((num_vertices, max_count), dtype=int)
        corner_mat = np.zeros((num_vertices, max_count, 3, 3))
        normal_mat = np.zeros((num_vertices, max_count, 3))
        current_mat = np.zeros((num_vertices, max_count, 3))
        area_mat = np.zeros((num_vertices, max_count))
        for v, (area, normal, points, current) in enumerate(elements):
            count = len(area)
            is_real[v, :count] = True
            corner_mat[v, is_real[v] == 1] = points
            current_mat[v, is_real[v] == 1] = current
            area_mat[v, is_real[v] == 1] = area
            normal_mat[v, is_real[v] == 1] = normal
        results.append(dict(elements=elements, is_real_triangle_mat=is_real, triangle_corner_coord_mat=corner_mat,
                            current_mat=current_mat, area_mat=area_mat, face_normal_mat=normal_mat,
                            current_density_mat=current_density_mat))
    return results


def reference_resistance_matrix(coil_parts, input_args):
    material_factor = input_args.specific_resistivity_conductor / input_args.conductor_thickness
    results = []
    for coil_part in coil_parts:
        faces = coil_part.coil_mesh.get_faces()
        num_nodes = len(coil_part.basis_elements)
        adjacency = np.zeros((num_nodes, num_nodes), dtype=bool)
        for t in range(faces.shape[0]):
            adjacency[faces[t, 1], faces[t, 0]] = True
            adjacency[faces[t, 2], faces[t, 1]] = True
            adjacency[faces[t, 0], faces[t, 2]] = True
        rows, cols = np.where(adjacency)
        mesh_edges = np.column_stack((cols, rows))
        edges = np.vstack((np.hstack((np.arange(num_nodes), mesh_edges[:, 0])),
                           np.hstack((np.arange(num_nodes), mesh_edges[:, 1]))))
        adjacency = np.logical_or(adjacency, adjacency.T)
        resistance = np.zeros((num_nodes, num_nodes))
        elements = coil_part.basis_elements
        for e in range(edges.shape[1]):
            n1, n2 = edges[0, e], edges[1, e]
            shared = np.intersect1d(elements[n1].triangles, elements[n2].triangles)
            total = 0
            if len(shared) > 0:
                for tri in shared:
                    p1 = np.where(elements[n1].triangles == tri)[0]
                    p2 = np.where(elements[n2].triangles == tri)[0]
                    total += np.dot(elements[n1].current[p1], elements[n2].current[p2].T) * (elements[n1].area[p1])**2
                resistance[n1, n2] = np.squeeze(total)
        resistance += resistance.T
        resistance *= material_factor
        results.append((resistance, adjacency))
    return results


def reference_sweep_mesh(wire_v, wire_uv_unused, path_directions, normals, cross_section_points):
    """The vertices and faces of the original sweep, from the path points, directions, normals and cross section."""
    num_points, num_corners = wire_v.shape[1], cross_section_points.shape[1]
    face_points = np.zeros((num_points, 3, num_corners))
    all_nodes = np.zeros((num_points * num_corners, 3))
    run = 0
    for p in range(num_points):
        e1, e2 = path_directions[:, p], normals[:, p]
        e3 = np.cross(e1, e2) / np.linalg.norm(np.cross(e1, e2))
        e2 = np.cross(e1, e3) / np.linalg.norm(np.cross(e1, e3))
        for a in range(num_corners):
            face_points[p, :, a] = np.dot(np.column_stack((e2, e3, e1)), np.hstack((cross_section_points[:, a], 0)))
        for a in range(num_corners):
            face_points[p, :, a] += wire_v[:, p]
            all_nodes[run, :] = face_points[p, :, a]
            run += 1
    triangles = np.zeros(((num_points - 1) * num_corners * 2, 3), dtype=int)
    run = 0
    for t in range(num_points - 1):
        for e in range(num_corners):
            a_, b_ = t * num_corners + e, t * num_corners + (e + 1) % num_corners
            c_, d_ = (t + 1) * num_corners + e, (t + 1) * num_corners + (e + 1) % num_corners
            triangles[run] = [a_, b_, d_]
            triangles[run + 1] = [d_, c_, a_]
            run += 2
    return all_nodes, triangles


# ---------------------------------------------------------------------------------------------------------------------
# The recorded inputs
# ---------------------------------------------------------------------------------------------------------------------

@pytest.fixture(scope='module')
def recorded(tmp_path_factory):
    """Run a small design, recording the inputs of the steps."""
    work_dir = tmp_path_factory.mktemp('vectorised')
    recorded_inputs = {}
    originals = {}

    def record(name):
        function = getattr(release, name)
        originals[name] = function

        def wrapper(*args):
            recorded_inputs[name] = copy.deepcopy(args)
            return function(*args)
        setattr(release, name, wrapper)

    names = ['calculate_one_ring_by_mesh', 'calculate_basis_functions', 'calculate_resistance_matrix',
             'create_sweep_along_surface']
    for name in names:
        record(name)
    try:
        arguments = dict(CASES['biplanar_x_tikhonov'], project_name='vectorised', output_directory=str(work_dir),
                         persistence_dir=str(work_dir))
        release.pyCoilGen(logging.getLogger(__name__), arguments)
    finally:
        for name, function in originals.items():
            setattr(release, name, function)
    return recorded_inputs


def test_one_ring_matches_the_loops(recorded):
    (parts,) = recorded['calculate_one_ring_by_mesh']
    new = calculate_one_ring_by_mesh(copy.deepcopy(parts))
    reference = reference_one_ring(copy.deepcopy(parts))
    for new_part, reference_part in zip(new, reference):
        for ring_new, ring_ref in zip(new_part.one_ring_list, reference_part.one_ring_list):
            assert ring_new.dtype == ring_ref.dtype and np.array_equal(ring_new, ring_ref)
        for faces_new, faces_ref in zip(new_part.node_triangles, reference_part.node_triangles):
            assert np.array_equal(faces_new, faces_ref)


def test_basis_functions_match_the_loops(recorded):
    (parts,) = recorded['calculate_basis_functions']
    new = calculate_basis_functions(copy.deepcopy(parts))
    for new_part, reference in zip(new, reference_basis_functions(copy.deepcopy(parts))):
        for name in ('is_real_triangle_mat', 'triangle_corner_coord_mat', 'current_mat', 'area_mat', 'face_normal_mat',
                     'current_density_mat'):
            assert getattr(new_part, name).shape == reference[name].shape
            assert np.allclose(getattr(new_part, name), reference[name], rtol=1e-13, atol=1e-15), name
        for element, (area, normal, points, current) in zip(new_part.basis_elements, reference['elements']):
            assert np.allclose(element.area, area, rtol=1e-13, atol=0)
            assert np.allclose(element.face_normal, normal, rtol=1e-13, atol=1e-15)
            assert np.allclose(element.triangle_points_ABC, points, rtol=0, atol=0)
            assert np.allclose(element.current, current, rtol=1e-13, atol=1e-15)


def test_resistance_matrix_matches_the_loops(recorded):
    parts, input_args = recorded['calculate_resistance_matrix']
    new = calculate_resistance_matrix(copy.deepcopy(parts), input_args)
    for new_part, (resistance, adjacency) in zip(new, reference_resistance_matrix(copy.deepcopy(parts), input_args)):
        assert np.array_equal(new_part.node_adjacency_mat, adjacency)
        assert np.allclose(new_part.resistance_matrix, resistance, rtol=1e-12, atol=1e-12 * np.abs(resistance).max())


def test_sweep_matches_the_loops(recorded):
    parts, input_args = recorded['create_sweep_along_surface']
    new = create_sweep_along_surface(copy.deepcopy(parts), input_args)
    for part_before, part_new in zip(parts, new):
        # Rebuild the inputs of the sweep part of the original, from the (cleaned) wire path of the new implementation
        wire_v = part_new.wire_path.v
        wire_uv = part_new.wire_path.uv
        mesh_2d = Mesh(vertices=part_before.coil_mesh.uv, faces=part_before.coil_mesh.get_faces())
        face_inds, _ = mesh_2d.get_face_indices(wire_uv.T)
        normals = np.zeros((3, wire_v.shape[1]))
        found = face_inds != -1
        normals[:, found] = part_before.coil_mesh.fn[face_inds[found]].T
        for point_ind in np.nonzero(~found)[0]:
            normals[:, point_ind] = 0 if point_ind == 0 else normals[:, point_ind - 1]
        directions = wire_v[:, 1:] - wire_v[:, :-1]
        directions = np.column_stack((directions, directions[:, -1]))
        directions = directions / np.linalg.norm(directions, axis=0)
        directions[:, 1:-1] = (directions[:, :-2] + directions[:, 2:]) / 2
        directions = directions / np.linalg.norm(directions, axis=0)

        theta = np.linspace(0, 2 * np.pi, 10, endpoint=False)
        cross_section = np.vstack((np.sin(theta), np.cos(theta))) * input_args.conductor_thickness
        cross_section = cross_section - np.mean(cross_section, axis=1, keepdims=True)
        cross_section = cross_section[:, 0:-1]

        nodes, triangles = reference_sweep_mesh(wire_v, wire_uv, directions, normals, cross_section)
        mesh = part_new.layout_surface_mesh
        assert np.allclose(mesh.get_vertices(), nodes, rtol=0, atol=1e-15, equal_nan=True)
        assert np.array_equal(mesh.get_faces()[:len(triangles)], triangles)


# ---------------------------------------------------------------------------------------------------------------------
# Helpers of the contour calculation, and the mesh parameterisation
# ---------------------------------------------------------------------------------------------------------------------

def test_opposed_nodes_are_the_nodes_off_the_edge():
    rng = np.random.default_rng(2)
    triangles = np.stack([rng.permutation(10)[:3] for _ in range(40)])
    edge =np.stack([np.sort(rng.permutation(triangles[i])[:2]) for i in range(40)])
    attached = np.stack((triangles, triangles[:, ::-1]), axis=1)
    expected = np.array([[np.squeeze(np.setdiff1d(attached[i, side], edge[i])) for side in range(2)]
                         for i in range(40)])
    assert np.array_equal(contours._opposed_nodes(attached, edge), expected)


def test_neighbour_search_matches_the_array_search():
    rng = np.random.default_rng(4)
    edges = np.sort(rng.integers(0, 12, size=(60, 2)), axis=1)
    edges = edges[edges[:, 0] != edges[:, 1]]
    opposed = rng.integers(0, 12, size=edges.shape)
    search = contours._neighbour_search(edges, opposed)
    for e in range(len(edges)):
        reference = np.where(
            np.any(np.logical_or(edges == opposed[e, 0], edges == opposed[e, 1]), axis=1) &
            np.any(np.logical_or(edges == edges[e, 0], edges == edges[e, 1]), axis=1))[0]
        assert search(e) == reference.tolist()


def test_vectorised_cotangent_weights_match_the_loop():
    # A curved mesh patch with a boundary, through the parameterisation itself
    n = 6
    xs, ys = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n))
    vertices = np.column_stack((xs.ravel(), ys.ravel(), 0.2 * np.sin(3 * xs.ravel()) * ys.ravel()))
    faces = []
    for i in range(n - 1):
        for j in range(n - 1):
            a, b, c, d = i * n + j, i * n + j + 1, (i + 1) * n + j, (i + 1) * n + j + 1
            faces += [[a, b, d], [a, d, c]]
    faces = np.array(faces)

    def parameterise(use_loop):
        mesh = Mesh(vertices=vertices.copy(), faces=faces.copy())
        mesh.unique_vert_inds = np.arange(len(vertices))
        original = parameterization.cotanWeights
        if use_loop:
            parameterization.cotanWeights = parameterization._cotanWeights_loop
        try:
            return parameterization.mesh_parameterization_iterative(mesh).uv
        finally:
            parameterization.cotanWeights = original

    assert np.allclose(parameterise(False), parameterise(True), rtol=1e-12, atol=1e-12)
