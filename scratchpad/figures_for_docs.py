"""Utility class that generates figures for the documentation."""

#from pyCoilGen.mesh_factory import build_planar_mesh, build_biplanar_mesh, build_circular_mesh, build_cylinder_mesh
import importlib

from pyCoilGen.sub_functions.data_structures import Mesh

for which in ['build_planar_mesh', 'build_biplanar_mesh', 'build_circular_mesh', 'build_cylinder_mesh']:
    module_name = f'pyCoilGen.mesh_factory.{which}'

    # Dynamically import the module
    module = importlib.import_module(module_name)

    # Call the builder with the default value
    built = getattr(module, which)(*getattr(module, '__default_value__'))
    mesh = Mesh(vertices=built.vertices, faces=built.faces)
    mesh.export(f'images/figures/{which}.stl')