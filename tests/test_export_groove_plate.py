import numpy as np
import trimesh
import pytest
from pytest import approx

# Test support
pytest.importorskip("manifold3d")

# Local imports
from pyCoilGen.sub_functions.data_structures import CoilSolution, CoilPart, DataStructure, Mesh
from pyCoilGen.mesh_factory.build_circular_biplanar_mesh import simple_circular_mesh
from pyCoilGen.export_factory import load_exporter_plugins
from pyCoilGen.helpers.solid_boolean import solid_difference

# Code under test
from pyCoilGen.export_factory.export_groove_plate import create_groove_plate, export_groove_plate

RADIUS = 0.07       # metres
SURFACE_Z = 0.0375  # metres
POLYGON_AREA = 0.5 * 90 * np.sin(2 * np.pi / 90) * (RADIUS * 1000.0) ** 2   # mm^2, 90 angular divisions
GROOVE_VOLUME = 120.0 * 4.0 * 2.0   # mm^3: the default conductor, centred on the surface, 2 mm deep


def make_part(z, outward_normal_z, conductor_centre=(0.0, 0.0, 0.0), conductor_size=(0.12, 0.004, 0.004)):
    """Create a plate at height z, with a box shaped conductor, centred at the offset from the surface (metres)."""
    vertices, faces = simple_circular_mesh(RADIUS, 6, 90, normal_winding_direction=outward_normal_z > 0)
    vertices[:, 2] = z
    box = trimesh.creation.box(conductor_size)
    box.apply_translation(np.array(conductor_centre) + [0.0, 0.0, z])
    return CoilPart(coil_mesh=Mesh(vertices=vertices, faces=faces),
                    layout_surface_mesh=Mesh(vertices=box.vertices, faces=box.faces))


def test_solid_difference():
    box = trimesh.creation.box((10, 10, 10))
    hole = trimesh.creation.cylinder(radius=2, height=20, sections=64)
    result = solid_difference(box, hole)
    assert result.is_watertight
    assert result.volume == approx(1000.0 - 0.5 * 64 * np.sin(2 * np.pi / 64) * 4.0 * 10.0, rel=1e-5)


def test_solid_difference_invalid_solid():
    open_mesh = trimesh.Trimesh(vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]], faces=[[0, 1, 2]])
    with pytest.raises(ValueError, match="not a valid closed solid"):
        solid_difference(open_mesh, trimesh.creation.box((1, 1, 1)))


def test_create_groove_plate_volume_and_extent():
    thickness = 0.01
    plate = create_groove_plate(make_part(SURFACE_Z, +1), thickness)

    assert plate.is_watertight
    # The plate is laid flat: bottom at z=0, top at the thickness, in millimetres
    assert plate.bounds[0, 2] == approx(0.0, abs=1e-4) and plate.bounds[1, 2] == approx(10.0, abs=1e-4)
    assert np.max(np.abs(plate.bounds[:, :2])) == approx(70.0)
    # The 4 mm conductor, centred on the surface, cuts a 2 mm deep, 4 mm wide, 120 mm long groove
    assert plate.volume == approx(POLYGON_AREA * 10.0 - GROOVE_VOLUME, abs=1e-2)


def test_create_groove_plate_orientation():
    # A conductor displaced to world y=+20 mm lies at y=+20 mm for a plate facing +z, but at y=-20 mm for a plate
    # facing -z, which is viewed from the other side. In both cases the groove is 2 mm deep and 40 mm long.
    offset = (0.0, 0.02, 0.0)
    upper = create_groove_plate(make_part(SURFACE_Z, +1, offset, (0.04, 0.004, 0.004)), 0.01)
    lower = create_groove_plate(make_part(-SURFACE_Z, -1, offset, (0.04, 0.004, 0.004)), 0.01)

    def groove_floor_y(plate):
        # The groove floor is 2 mm below the top face (z=10 mm); no other vertices lie at that height
        floor = plate.vertices[np.isclose(plate.vertices[:, 2], 8.0, atol=1e-3)]
        assert len(floor) > 0
        return np.mean(floor[:, 1])

    assert groove_floor_y(upper) == approx(20.0, abs=1e-3)
    assert groove_floor_y(lower) == approx(-20.0, abs=1e-3)
    # Both have the same groove volume
    assert upper.volume == approx(lower.volume, rel=1e-6)


def test_create_groove_plate_conductor_above_surface():
    # Conductor lifted 3 mm above the surface: its underside is 1 mm above the surface, so there is no groove
    part = make_part(SURFACE_Z, +1, conductor_centre=(0.0, 0.0, 0.003))
    with pytest.raises(ValueError, match="no groove"):
        create_groove_plate(part, 0.01)


def test_create_groove_plate_cut_through():
    with pytest.raises(ValueError, match="cuts through"):
        create_groove_plate(make_part(SURFACE_Z, +1), 0.002)


def test_create_groove_plate_without_conductor():
    part = make_part(SURFACE_Z, +1)
    part.layout_surface_mesh = None
    with pytest.raises(ValueError, match="skip_sweep"):
        create_groove_plate(part, 0.01)


def test_export_groove_plate(tmp_path):
    input_args = DataStructure(output_directory=str(tmp_path), project_name='test', field_shape_function='z',
                               groove_plate_filename='{project}_{part_index}_{field_function}_plate.stl',
                               groove_plate_thickness=0.01)
    solution = CoilSolution(input_args=input_args,
                            coil_parts=[make_part(-SURFACE_Z, -1), make_part(SURFACE_Z, +1)])
    written = export_groove_plate(solution)

    assert len(written) == 2
    for index in (0, 1):
        plate = trimesh.load(tmp_path / f'test_{index}_z_plate.stl')
        assert plate.is_watertight
        assert plate.volume == approx(POLYGON_AREA * 10.0 - GROOVE_VOLUME, abs=1e-2)


def test_groove_plate_exporter_plugin_registered():
    plugins = [p for p in load_exporter_plugins() if getattr(p, 'get_name', lambda: None)() == 'export groove plate']
    assert len(plugins) == 1
    assert callable(getattr(plugins[0], 'export_groove_plate'))
    assert [name for name, _ in plugins[0].get_parameters()] == ['groove_plate_filename', 'groove_plate_thickness']
