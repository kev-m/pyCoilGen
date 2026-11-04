"""Boolean operations on solid meshes.

The operations use the optional `manifold3d` package. Install it with `pip install pycoilgen[fabrication]`.
"""

# System
import numpy as np

# Third-party
import trimesh


def _to_manifold(manifold3d, mesh: trimesh.Trimesh, name: str):
    """
    Convert a trimesh mesh to a manifold3d solid.

    Args:
        manifold3d (module): The imported manifold3d module.
        mesh (trimesh.Trimesh): The mesh to convert.
        name (str): The name of the mesh, used in error messages.

    Returns:
        manifold3d.Manifold: The solid.

    Raises:
        ValueError: If the mesh is not a valid, closed solid.
    """
    solid = manifold3d.Manifold(manifold3d.Mesh(vert_properties=np.asarray(mesh.vertices, dtype=np.float32),
                                                tri_verts=np.asarray(mesh.faces, dtype=np.uint32)))
    status = solid.status()
    if status != manifold3d.Error.NoError:
        raise ValueError(f"The {name} mesh is not a valid closed solid ({status}).")
    return solid


def solid_difference(base: trimesh.Trimesh, cutter: trimesh.Trimesh) -> trimesh.Trimesh:
    """
    Subtract one solid mesh from another.

    Both meshes must be closed (watertight) and consistently oriented. The calculation is performed in single
    precision, so the meshes should be expressed in units that give co-ordinate magnitudes of the order of 1 to 1000,
    e.g. millimetres.

    Args:
        base (trimesh.Trimesh): The solid to cut into.
        cutter (trimesh.Trimesh): The solid to remove from `base`.

    Returns:
        trimesh.Trimesh: The solid `base` without `cutter`. The mesh topology is exactly as calculated, so vertices
            that coincide, but belong to different sheets of the surface, are not merged.

    Raises:
        ImportError: If `manifold3d` is not installed.
        ValueError: If either mesh is not a valid closed solid, or if the result is empty.

    Example:
        >>> box = trimesh.creation.box((10, 10, 10))
        >>> hole = trimesh.creation.cylinder(radius=2, height=20)
        >>> result = solid_difference(box, hole)
    """
    try:
        import manifold3d
    except ImportError as exc:
        raise ImportError("This feature requires the optional 'manifold3d' package. "
                          "Install it with: pip install pycoilgen[fabrication]") from exc

    result = (_to_manifold(manifold3d, base, 'base') - _to_manifold(manifold3d, cutter, 'cutter')).to_mesh()
    vertices = np.asarray(result.vert_properties)[:, :3]
    faces = np.asarray(result.tri_verts)
    if len(faces) == 0:
        raise ValueError("The result of the difference is empty.")
    # Preserve the topology exactly as calculated: merging vertices (the default) can pinch the surface where
    # coincident vertices belong to different sheets, which would then no longer be a valid solid.
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
