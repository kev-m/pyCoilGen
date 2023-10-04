# Changelog

## 0.1.1 (2023-10-04)

#### Fixes

* (meshes): Fix bi-planar mesh so that the normals point outwards.

Full set of changes: [`0.1.0...0.1.1`](https://github.com/kev-m/pyCoilGen/compare/0.1.0...0.1.1)

## 0.1.0 (2023-10-03)

#### New Features

* (exporter): Extending the list of supported export types.
* (meshes): Using the mesh factory for coil, target and shield meshes.
* (meshes): Adding 'create circular mesh' to the mesh factory.
* (meshes): Using auto-discovery to discover mesh builders. ([#55](https://github.com/kev-m/pyCoilGen/issues/55))
#### Fixes

* Bugfix with trying to access invalid function.
* (meshes): Using stl_mesh_filename and coil_mesh_file.
* (meshes): Supporting int and float parameters.

Full set of changes: [`0.0.11...0.1.0`](https://github.com/kev-m/pyCoilGen/compare/0.0.11...0.1.0)

## 0.0.11 (2023-09-28)

#### Fixes

* Fixing exception when skip_inductance_calculation is True.
* Fixing exception when skip_postprocessing is True.
* (build_cylinder_mesh): Generated cylinder mesh exactly matches input dimensions.
#### Docs

* Moving the release procedure to its own file. ([#51](https://github.com/kev-m/pyCoilGen/issues/51))

Full set of changes: [`0.0.10...0.0.11`](https://github.com/kev-m/pyCoilGen/compare/0.0.10...0.0.11)

## 0.0.10 (2023-09-26)

#### Docs

* Fixing URL in pyproject.toml

Full set of changes: [`0.0.9...0.0.10`](https://github.com/kev-m/pyCoilGen/compare/0.0.9...0.0.10)

## 0.0.9 (2023-09-26)

#### Docs

* Updating release procedure.
#### Others

* Globally reformatted sources.

Full set of changes: [`0.0.8...0.0.9`](https://github.com/kev-m/pyCoilGen/compare/0.0.8...0.0.9)

## 0.0.8 (2023-09-25)

#### New Features

* Initial release candidate.
