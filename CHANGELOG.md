# Changelog

## 0.5.0 (2026-11-08)

#### Performance improvements

* Calculate the sensitivity matrices on several threads, with a num_threads parameter
* Solve the Tikhonov normal equations with Cholesky instead of the pseudo-inverse
#### Others

* bump the version to 0.5.0

Full set of changes: [`0.4.1...0.5.0`](https://github.com/kev-m/CoilGen-Python/compare/0.4.1...0.5.0)

## 0.4.1 (2026-11-07)

#### New Features

* (regression): judge the gradient deviation, and keep within-tolerance designs within 5%
* (field_verification): measure the deviation of the gradient over space
#### Fixes

* (process_raw_loops): remove loops smaller than the interconnection cut
* (process_raw_loops): close contour loops with a tolerance, not by exact comparison
* (examples): repair the pre-optimised and quick_check examples, and the cylinder flag of the bi-planar one
* (mesh_parameterization_iterative): assign the opposite vertex as a scalar, for numpy 2.5
* (define_target_field): evaluate the symbolic target gradient at the target points and scale it
#### Performance improvements

* (biot_savart): blocked Biot-Savart field and gradient kernels, vectorised loop inclusion test
* (uv_to_xyz): locate points in the planar mesh faces in one vectorised batch
* (sensitivity): vectorise the sensitivity and gradient sensitivity matrices
#### Others

* (release): Adding a version helper to the release procedure.
* (release): Update the version comment.
* bump the version to 0.4.1
* re-save the pickled test data without the deprecated scipy mat_struct path
* (regression): add a stable, quick bi-planar case with 16 levels
* (regression): update the golden values of biplanar_x_tikhonov for the closed contour loops

Full set of changes: [`0.4.0...0.4.1`](https://github.com/kev-m/CoilGen-Python/compare/0.4.0...0.4.1)

## 0.4.0 (2026-11-05)

#### Fixes

* (create_sweep_along_surface): compute ohmian_resistance as rho * L / A
#### Refactorings

* (parse_input): rename specific_conductivity_conductor to specific_resistivity_conductor
#### Docs

* use a pytest marker expression that survives PowerShell to run all tests
* document the renamed specific_resistivity_conductor parameter
* document the ohmian_resistance units fix on a bug fixes page
#### Others

* (deps): drop the networkx dependency

Full set of changes: [`0.3.0...0.4.0`](https://github.com/kev-m/CoilGen-Python/compare/0.3.0...0.4.0)

## 0.3.0 (2026-11-05)

#### New Features

* (verification): add an independent Biot-Savart check of the wire path
* (export): add 'export groove plate' exporter for 3D-printable plates
* (export): add 'export gerber file' exporter for planar coil parts
* (mesh_factory): add 'create circular bi-planar mesh' builder
#### Fixes

* (interconnect_among_groups): choose the wire's open end independent of rounding noise
#### Refactorings

* (helpers): share the planar layer projection helpers
#### Docs

* credit PR [#82](https://github.com/kev-m/CoilGen-Python/issues/82) by Sairam Geethanath for the ideas behind the 0.3.0 features
* update the tested environments for trimesh 4.4
* refer to release 0.3.0 by its number, and document pillow and the harness rules
* remove the unused html_static_path setting
* describe the new features in the overview, glossary and contributing guide
* add the compatibility page to the index, and to the release checklist
* add a compatibility table of releases, Python and library versions
#### Others

* (deps): require trimesh 4.4 or newer, and explain the version limits
* (deps): allow current pillow versions
* (docs): add a ReadTheDocs configuration
* (docs): update Sphinx and myst-parser
* (deps): require Python 3.11 and allow numpy 2 and newer scipy
* (deps): declare networkx and allow trimesh 5
* (release): bump the version to 0.3.0
* skip the regression harness by default, and run it explicitly for a release
* (regression): tolerate library-dependent cases outside the golden environment
* (regression): judge the physical quality of each case, not just its fingerprint
* (regression): add a regression harness for complete coil designs

Full set of changes: [`0.2.4...0.3.0`](https://github.com/kev-m/CoilGen-Python/compare/0.2.4...0.3.0)

## 0.2.4 (2026-02-15)

#### Fixes

* Merging PR to address issue [#78](https://github.com/kev-m/CoilGen-Python/issues/78)

Full set of changes: [`0.2.3...0.2.4`](https://github.com/kev-m/CoilGen-Python/compare/0.2.3...0.2.4)

## 0.2.3 (2025-01-16)

#### Fixes

* (docs): Improving documentation
* (docs): Adding additional text to the documentation to improve chances of outward normals in rendered wirepath, as reported in issue [#75](https://github.com/kev-m/CoilGen-Python/issues/75)

Full set of changes: [`0.2.2...0.2.3`](https://github.com/kev-m/CoilGen-Python/compare/0.2.2...0.2.3)

## 0.2.2 (2024-06-24)

#### Fixes

* fixes the bug reported in issue [#70](https://github.com/kev-m/CoilGen-Python/issues/70) ([#71](https://github.com/kev-m/CoilGen-Python/issues/71))

Full set of changes: [`0.2.1...0.2.2`](https://github.com/kev-m/CoilGen-Python/compare/0.2.1...0.2.2)

## 0.2.1 (2024-03-18)

#### Fixes

* Fix runtime error due to numpy deprecation. ([#69](https://github.com/kev-m/CoilGen-Python/issues/69))
* Fix runtime errors due to deprecated np.warnings

Full set of changes: [`0.2.0...0.2.1`](https://github.com/kev-m/CoilGen-Python/compare/0.2.0...0.2.1)

## 0.2.0 (2023-11-07)

#### New Features

* (topology): Expose cut_plane_definition to CLI.
#### Fixes

* (postprocessing): Do not calculate inductance if skip_postprocessing is True.

Full set of changes: [`0.1.3...0.2.0`](https://github.com/kev-m/CoilGen-Python/compare/0.1.3...0.2.0)

## 0.1.3 (2023-10-11)

#### Fixes

* Implemented bugfix from CoilGen

Full set of changes: [`0.1.2...0.1.3`](https://github.com/kev-m/CoilGen-Python/compare/0.1.2...0.1.3)

## 0.1.2 (2023-10-09)

#### Fixes

* (installation): Relaxing dependencies.

Full set of changes: [`0.1.1...0.1.2`](https://github.com/kev-m/CoilGen-Python/compare/0.1.1...0.1.2)

## 0.1.1 (2023-10-04)

#### Fixes

* (meshes): Fix bi-planar mesh so that the normals point outwards.

Full set of changes: [`0.1.0...0.1.1`](https://github.com/kev-m/CoilGen-Python/compare/0.1.0...0.1.1)

## 0.1.0 (2023-10-03)

#### New Features

* (exporter): Extending the list of supported export types.
* (meshes): Using the mesh factory for coil, target and shield meshes.
* (meshes): Adding 'create circular mesh' to the mesh factory.
* (meshes): Using auto-discovery to discover mesh builders. ([#55](https://github.com/kev-m/CoilGen-Python/issues/55))
#### Fixes

* Bugfix with trying to access invalid function.
* (meshes): Using stl_mesh_filename and coil_mesh_file.
* (meshes): Supporting int and float parameters.

Full set of changes: [`0.0.11...0.1.0`](https://github.com/kev-m/CoilGen-Python/compare/0.0.11...0.1.0)

## 0.0.11 (2023-09-28)

#### Fixes

* Fixing exception when skip_inductance_calculation is True.
* Fixing exception when skip_postprocessing is True.
* (build_cylinder_mesh): Generated cylinder mesh exactly matches input dimensions.
#### Docs

* Moving the release procedure to its own file. ([#51](https://github.com/kev-m/CoilGen-Python/issues/51))

Full set of changes: [`0.0.10...0.0.11`](https://github.com/kev-m/CoilGen-Python/compare/0.0.10...0.0.11)

## 0.0.10 (2023-09-26)

#### Docs

* Fixing URL in pyproject.toml

Full set of changes: [`0.0.9...0.0.10`](https://github.com/kev-m/CoilGen-Python/compare/0.0.9...0.0.10)

## 0.0.9 (2023-09-26)

#### Docs

* Updating release procedure.
#### Others

* Globally reformatted sources.

Full set of changes: [`0.0.8...0.0.9`](https://github.com/kev-m/CoilGen-Python/compare/0.0.8...0.0.9)

## 0.0.8 (2023-09-25)

#### New Features

* Initial release candidate.
