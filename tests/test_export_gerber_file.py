import re
import numpy as np
import pytest
from pytest import approx

# Local imports
from pyCoilGen.sub_functions.data_structures import CoilSolution, CoilPart, DataStructure, Mesh, Shape3D
from pyCoilGen.helpers.gerber import (gerber_coordinate, create_gerber_tracks, create_gerber_complement)
from pyCoilGen.mesh_factory.build_circular_biplanar_mesh import simple_circular_mesh
from pyCoilGen.export_factory import load_exporter_plugins

# Code under test
from pyCoilGen.export_factory.export_gerber_file import (
    export_gerber_file, get_viewing_basis, project_to_plane, get_part_outlines, check_planar)

_COORDINATE = re.compile(r'^X(-?\d+)Y(-?\d+)(D0[123])\*$')


def parse_gerber(text):
    """Minimal Gerber reader: returns the commands, and the (x, y, op) tuples in millimetres."""
    lines = text.strip().split('\n')
    coordinates = []
    for line in lines:
        match = _COORDINATE.match(line)
        if match:
            coordinates.append((int(match.group(1)) / 1e6, int(match.group(2)) / 1e6, match.group(3)))
    return lines, coordinates


# ---------------------------------------------------------------------------
# helpers.gerber
# ---------------------------------------------------------------------------

def test_gerber_coordinate_format():
    assert gerber_coordinate(1.5, -2.25, 'D01') == 'X1500000Y-2250000D01*'
    assert gerber_coordinate(0.0, 0.0, 'D02') == 'X0Y0D02*'


def test_gerber_coordinate_invalid():
    with pytest.raises(ValueError):
        gerber_coordinate(0.0, 0.0, 'D99')
    with pytest.raises(ValueError):
        gerber_coordinate(10000.0, 0.0, 'D01')


def test_create_gerber_tracks():
    path = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 5.5], [10.0, 5.5]])  # Last point repeated
    text = create_gerber_tracks([path], 2.0)
    lines, coordinates = parse_gerber(text)

    # Header, aperture and footer
    assert lines[1] == '%MOMM*%'
    assert lines[2] == '%FSLAX46Y46*%'
    assert '%ADD10C,2.000000*%' in lines
    assert 'G01*' in lines
    assert lines[-1] == 'M02*'

    # One move, two draws; the repeated point is dropped
    assert [op for _, _, op in coordinates] == ['D02', 'D01', 'D01']
    assert coordinates[2][:2] == (10.0, 5.5)


def test_create_gerber_tracks_invalid_width():
    with pytest.raises(ValueError):
        create_gerber_tracks([np.array([[0.0, 0.0], [1.0, 0.0]])], 0.0)


def test_create_gerber_complement_structure():
    outline = np.array([[0.0, 0.0], [20.0, 0.0], [20.0, 20.0], [0.0, 20.0]])
    hole = np.array([[8.0, 8.0], [12.0, 8.0], [12.0, 12.0], [8.0, 12.0]])
    path = np.array([[2.0, 2.0], [18.0, 2.0]])
    text = create_gerber_complement(outline, [path], 1.0, [hole])
    lines, coordinates = parse_gerber(text)

    # Dark region, then clear polarity: the hole region, then the track
    assert lines.index('%LPD*%') < lines.index('G36*') < lines.index('G37*') < lines.index('%LPC*%')
    assert lines.count('G36*') == 2 and lines.count('G37*') == 2
    assert lines.index('%LPC*%') < len(lines) - 1 - lines[::-1].index('G36*')
    # The regions are closed: each ends at its start point
    assert coordinates[0][:2] == coordinates[4][:2] == (0.0, 0.0)
    assert lines[-1] == 'M02*'


def test_create_gerber_complement_degenerate_outline():
    with pytest.raises(ValueError):
        create_gerber_complement(np.array([[0.0, 0.0], [1.0, 0.0]]), [], 1.0)


# ---------------------------------------------------------------------------
# export_gerber_file
# ---------------------------------------------------------------------------

def make_plate(radius, z, outward_normal_z, nr=4, nt=16):
    """Create a CoilPart whose coil mesh faces outward_normal_z, with a spiral wire path with a normal shift."""
    vertices, faces = simple_circular_mesh(radius, nr, nt, normal_winding_direction=outward_normal_z > 0)
    vertices[:, 2] = z
    theta = np.linspace(0, 2 * np.pi, 50)
    spiral = np.vstack(((0.4 * radius + 0.1 * radius * theta / (2 * np.pi)) * np.cos(theta),
                        (0.4 * radius + 0.1 * radius * theta / (2 * np.pi)) * np.sin(theta),
                        np.full_like(theta, z) + np.where(theta > np.pi, 0.002, 0.0)))  # Shifted return path
    return CoilPart(coil_mesh=Mesh(vertices=vertices, faces=faces), wire_path=Shape3D(v=spiral))


def make_solution(tmp_path, **kwargs):
    input_args = DataStructure(output_directory=str(tmp_path), project_name='test', field_shape_function='z',
                               conductor_cross_section_width=0.002,
                               gerber_filename='{project}_{part_index}_{field_function}_{layer}.gbr',
                               gerber_layers=['tracks', 'complement'], **kwargs)
    parts = [make_plate(0.07, -0.0375, -1), make_plate(0.07, 0.0375, +1)]
    return CoilSolution(input_args=input_args, coil_parts=parts)


def test_viewing_basis_is_right_handed_and_outward():
    lower = make_plate(0.07, -0.0375, -1)
    upper = make_plate(0.07, 0.0375, +1)

    u, w, n = get_viewing_basis(upper)
    assert np.allclose(n, [0, 0, 1]) and np.allclose(u, [1, 0, 0]) and np.allclose(w, [0, 1, 0])

    # Seen from below, the y axis is flipped, so the view is not a mirror image
    u, w, n = get_viewing_basis(lower)
    assert np.allclose(n, [0, 0, -1]) and np.allclose(u, [1, 0, 0]) and np.allclose(w, [0, -1, 0])
    assert np.allclose(np.cross(u, w), n)


def test_viewing_basis_normal_along_x():
    vertices, faces = simple_circular_mesh(0.07, 3, 12)
    rotated = vertices[:, [2, 0, 1]]  # z -> x : normal along +x
    part = CoilPart(coil_mesh=Mesh(vertices=rotated, faces=faces))
    u, w, n = get_viewing_basis(part)
    assert np.allclose(n, [1, 0, 0]) and np.allclose(u, [0, 1, 0]) and np.allclose(np.cross(u, w), n)


def test_project_to_plane_accepts_both_layouts():
    points = np.array([[0.001, 0.002, 0.5], [0.003, 0.004, 0.5]])  # (n,3), in metres
    u, w = np.array([1.0, 0, 0]), np.array([0, 1.0, 0])
    expected = [[1.0, 2.0], [3.0, 4.0]]
    assert np.allclose(project_to_plane(points, u, w), expected)
    assert np.allclose(project_to_plane(points.T, u, w), expected)


def test_get_part_outlines():
    part = make_plate(0.07, 0.0, +1)
    u, w, _ = get_viewing_basis(part)
    outline, holes = get_part_outlines(part, u, w)
    assert holes == []
    assert np.max(np.linalg.norm(outline, axis=1)) == approx(70.0)


def test_check_planar_rejects_curved_parts():
    normal = np.array([0.0, 0.0, 1.0])
    check_planar(make_plate(0.07, 0.0, +1), normal)

    vertices, faces = simple_circular_mesh(0.07, 4, 16)
    vertices[:, 2] = vertices[:, 0]  # Tilted by 45 degrees, relative to z
    with pytest.raises(ValueError, match="planar"):
        check_planar(CoilPart(coil_mesh=Mesh(vertices=vertices, faces=faces)), normal)


def test_export_gerber_file_writes_all_layers(tmp_path):
    solution = make_solution(tmp_path)
    written = export_gerber_file(solution)

    assert len(written) == 4
    for index in (0, 1):
        for layer in ('tracks', 'complement'):
            assert (tmp_path / f'test_{index}_z_{layer}.gbr').exists()

    # Tracks: width from conductor_cross_section_width, all points in millimetres
    lines, coordinates = parse_gerber((tmp_path / 'test_1_z_tracks.gbr').read_text())
    assert '%ADD10C,2.000000*%' in lines
    assert len(coordinates) == 50
    assert max(np.hypot(x, y) for x, y, _ in coordinates) < 70.0

    # Complement: dark outline region then cleared tracks
    lines, _ = parse_gerber((tmp_path / 'test_1_z_complement.gbr').read_text())
    assert lines.index('%LPD*%') < lines.index('G36*') < lines.index('%LPC*%')


def test_export_gerber_file_bottom_plate_is_viewed_from_outside(tmp_path):
    solution = make_solution(tmp_path)
    export_gerber_file(solution)

    for index in (0, 1):
        _, coordinates = parse_gerber((tmp_path / f'test_{index}_z_tracks.gbr').read_text())
        sign = -1.0 if index == 0 else 1.0   # The lower plate (normal -z) has its y axis flipped
        original = solution.coil_parts[index].wire_path.v
        assert np.allclose([x for x, _, _ in coordinates], original[0] * 1000.0, atol=1e-5)
        assert np.allclose([y for _, y, _ in coordinates], sign * original[1] * 1000.0, atol=1e-5)


def test_export_gerber_file_single_layer_without_template(tmp_path):
    solution = make_solution(tmp_path)
    solution.input_args.gerber_layers = ['tracks']
    solution.input_args.gerber_filename = 'coil_{part_index}.gbr'
    assert len(export_gerber_file(solution)) == 2
    assert (tmp_path / 'coil_0.gbr').exists()

    # Several layers need the {layer} placeholder, or they would overwrite each other
    solution.input_args.gerber_layers = ['tracks', 'complement']
    with pytest.raises(ValueError, match="layer"):
        export_gerber_file(solution)


def test_export_gerber_file_errors(tmp_path):
    solution = make_solution(tmp_path)
    solution.input_args.gerber_layers = ['silkscreen']
    with pytest.raises(ValueError, match="Unsupported"):
        export_gerber_file(solution)

    solution = make_solution(tmp_path)
    solution.coil_parts[1].wire_path = None
    with pytest.raises(ValueError, match="no wire path"):
        export_gerber_file(solution)


def test_gerber_exporter_plugin_registered():
    plugins = [p for p in load_exporter_plugins() if getattr(p, 'get_name', lambda: None)() == 'export gerber file']
    assert len(plugins) == 1
    assert callable(getattr(plugins[0], 'export_gerber_file'))
    assert [name for name, _ in plugins[0].get_parameters()] == ['gerber_filename', 'gerber_layers']
