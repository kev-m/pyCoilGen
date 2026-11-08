# AGENTS.md: guide for AI coding assistants

This file orients any AI coding assistant (and any new human contributor) to **pyCoilGen**. It is deliberately short.
It links to the authoritative documents instead of repeating them. If this file and the code disagree, the code wins:
fix this file.

## What the project is

**pyCoilGen** is a Python port of the MATLAB [CoilGen](https://github.com/Philipp-MR/CoilGen) (Amrein et al., 2022). It
designs coil winding layouts, mainly MRI/NMR gradient and shim coils, on 3D support surfaces with a boundary element
method. Input: a surface mesh and a target field. Output: interconnected, non-overlapping wire paths, field-error
metrics, and optional CAD, Gerber or groove-plate exports.

- License GPLv3+; published to PyPI as `pycoilgen`; Python >= 3.11 (tested on 3.11 and 3.13).
- Core dependencies: numpy, scipy, trimesh, sympy, pillow, matplotlib. Optional: `manifold3d` (groove plates),
  `magpylib` (field verification cross-check), FastHenry2 (inductance and resistance, an external program).
- Units are SI throughout (metres, tesla, amperes, ohms).

## Repository map

| Path | Contents |
|------|----------|
| `pyCoilGen/pyCoilGen_release.py` | **The pipeline entry point**, `pyCoilGen(log, input_args)`. Start here. |
| `pyCoilGen/pyCoilGen_develop.py` | Older development variant of the pipeline. Not the supported entry point. |
| `pyCoilGen/__main__.py` | Command-line entry (`python -m pyCoilGen`), arguments from `parse_input.py`. |
| `pyCoilGen/sub_functions/` | One module per pipeline step, normally named after the MATLAB function. |
| `pyCoilGen/sub_functions/data_structures.py` | `Mesh`, `CoilPart`, `TargetField`, `CoilSolution`, `ContourLine`, `WirePart`, etc. |
| `pyCoilGen/sub_functions/parse_input.py` | **All input parameters**, their defaults and help text. |
| `pyCoilGen/sub_functions/constants.py` | Debug levels (`DEBUG_NONE`=0, `DEBUG_BASIC`=1, `DEBUG_VERBOSE`=2). |
| `pyCoilGen/mesh_factory/` | Built-in mesh builders (planar, bi-planar, circular, cylinder, cut shapes) and STL loading. |
| `pyCoilGen/export_factory/` | Exporters: CAD, Gerber, groove plate. |
| `pyCoilGen/helpers/` | Utilities: persistence, Biot-Savart `field_verification`, triangulation, visualisation, timing. |
| `pyCoilGen/plotting/` | Plot functions. |
| `examples/` | Runnable scripts, each a parameter dict passed to `pyCoilGen`. Copy one to start. |
| `tests/` | Unit tests (`test_*.py`) and the regression harness (`tests/regression/`). |
| `data/` | Separate package `pycoilgen_data`: STL meshes, target fields, pre-optimised solutions. |
| `utilities/` | Separate package `pycoilgen_utils` (for example `stl_asc2bin`). |
| `docs/source/` | Sphinx and MyST Markdown docs (published to ReadTheDocs). |
| `debug/`, `images/`, `scratchpad/`, `prompts/` | Local output, scratch and historic material. Mostly git-ignored; do not rely on them. |

## How the pipeline works

`pyCoilGen()` runs numbered stages in order. The number is the "runpoint" in comments and persisted data, and matches the
MATLAB original.

1. **Setup**: parse inputs (a dict goes through `create_input`, otherwise the command line), set the debug level,
   create `persistence_dir` and `output_directory`.
2. **Stream function** (00-08): `read_mesh`, `split_disconnected_mesh`, `refine_mesh`, `parameterize_mesh`,
   `define_target_field`, `calculate_one_ring_by_mesh`, `calculate_basis_functions`, `calculate_sensitivity_matrix`,
   `calculate_gradient_sensitivity_matrix`, `calculate_resistance_matrix`, `stream_function_optimization`
   (`sf_opt_method`: `tikhonov` by default, or `iterative`). If `sf_source_file` is set, this stage is skipped and a
   pre-optimised solution is loaded. If `sf_dest_file` is set, the result is saved.
3. **Contours** (09-11): `calc_potential_levels`, `calc_contours_by_triangular_potential_cuts`, `process_raw_loops`.
4. **Wire path** (12-18): `find_minimal_contour_distance`, `topological_loop_grouping`, `calculate_group_centers`,
   `interconnect_within_groups`, `interconnect_among_groups`, `shift_return_paths`, `generate_cylindrical_pcb_print`.
   Skipped when `skip_postprocessing` is true.
5. **Evaluation and output**: conductor sweep (`create_sweep_along_surface`), inductance (needs FastHenry2, can be
   skipped with `skip_inductance_calculation`), `evaluate_field_errors`, `calculate_gradient`, `export_data`, and
   saving of the final `CoilSolution`.

The function returns a `CoilSolution` that holds `coil_parts`, `target_field`, `solution_errors`, `coil_gradient`, and so
on. Parts are lists of `CoilPart` objects, one per disconnected mesh piece.

Minimal use:

```python
import logging
from pyCoilGen.pyCoilGen_release import pyCoilGen

log = logging.getLogger(__name__)
solution = pyCoilGen(log, {'coil_mesh': 'create planar mesh', 'field_shape_function': 'z',
                           'planar_mesh_parameter_list': [0.4, 0.4, 16, 16, 0, 0, 1, 0, 0, -0.12, 0],
                           'project_name': 'demo', 'output_directory': 'images', 'persistence_dir': 'debug'})
```

Look at `examples/` for realistic parameter dicts, and at `parse_input.py` and
[docs/source/configuration.md](docs/source/configuration.md) for every parameter. Mesh builders are plugins in `mesh_factory/`, selected by name with `coil_mesh` (for example `'create planar mesh'`,
`'create bi-planar mesh'`, `'create cylinder mesh'`); an STL file is selected with `coil_mesh_file`. Each builder
registers its own parameters, so check its module for them.

## Set up and run

```bash
git clone https://github.com/kev-m/pyCoilGen
cd pyCoilGen
python -m venv .venv            # then activate it
pip install -r requirements.txt -r requirements-dev.txt
```

Linux may also need `libopenblas-dev gfortran` (SciPy) and `libspatialindex-dev` (trimesh `rtree`). FastHenry2 is
optional and external: see [CONTRIBUTING.md](CONTRIBUTING.md).

```bash
pytest                                        # default: unit tests, skips the slow regression cases
pytest -m regression                          # regression harness only (about 1 minute)
pytest -m "regression or not regression"      # everything, as for a release
python tests/regression/harness_generate.py --check   # compare to golden values, with a summary of differences
python examples/ygradient_coil.py             # run an example
```

`pytest.ini` sets `pythonpath = .` and by default runs with `-m "not regression"`.

## Conventions

- **Branches and PRs**: branch from `master`, open the PR against `master`.
- **Commits**: [Conventional Commits](https://www.conventionalcommits.org/) (`feat:`, `fix:`, `docs:`, `test:`,
  `refactor:`, `perf:`, `chore:`, `build:`; `!` or a `BREAKING CHANGE:` footer for breaking changes). The CHANGELOG is
  generated from these with `auto-changelog`, so write messages that are worth announcing. Details in
  [CONTRIBUTING.md](CONTRIBUTING.md).
- **Style**: PEP 8 (autopep8) and Google-style docstrings. Prefer small, focused commits.
- **Tests**: add or update a unit test in `tests/` for a behaviour change. Test files are named `test_<topic>.py`.
- **Docs**: Markdown in `docs/source/`. A new page must be added by hand to `docs/source/index.rst`. Build with
  `cd docs && make clean html`. Update the docs when you change a parameter or its behaviour.
- **Version**: single source of truth is `__version__` in `pyCoilGen/__init__.py`. Only change it as part of a release.
- **Release**: follow [RELEASE.md](RELEASE.md) (tests, docs, version, tag, changelog, `flit build` and `flit publish`).

## The regression harness: read before changing numerics

`tests/regression/` runs a small set of complete designs (planar, bi-planar, circular bi-planar, cylinder; both
solvers) and compares compact fingerprints and physical metrics to the golden values in `tests/regression/golden/`.

- A difference tells you *that* something changed. The physical metrics tell you whether it got *better* or *worse*.
  The gradient deviation and efficiency are the primary metrics. A "worse" primary metric fails the check.
- If the change is intended, regenerate the golden values (`python tests/regression/harness_generate.py [case ...]`)
  and commit them **together with the code change**, so a reviewer sees the difference.
- Small fingerprint differences across Python, numpy or scipy versions can be acceptable if the physics is not worse.
- To add a case, edit `CASES` in `tests/regression/harness_cases.py`. Use only the built-in mesh builders.

## Pitfalls and non-obvious facts

- **MATLAB heritage**: names, step numbers and comments follow the MATLAB code. Arrays are 0-based here but 1-based
  in MATLAB, and many arrays are stored `(3, N)` rather than `(N, 3)`. Check the shape comments before indexing.
  `conversion_notes.md` has the porting notes. Some modules (`matlab_internal.py`) emulate MATLAB behaviour.
- **Parameters are plain strings and dicts** validated in `parse_input.py`. A misspelled key in a dict is easy to make:
  check the parser. Deprecated names (for example `specific_conductivity_conductor`, now
  `specific_resistivity_conductor`) are kept for compatibility.
- **Result units**: resistance was wrong before 0.4.0 (see [docs/source/bug_fixes.md](docs/source/bug_fixes.md)).
- **The mesh limits accuracy**. A low node count caps how good the field can be. Mesh refinement
  (`iteration_num_mesh_refinement`) multiplies the cost.
- **The pipeline is slow and global**: a full design takes seconds to minutes. Prefer small meshes and low
  `target_region_resolution` in tests, and use the regression cases as the template.
- **Debug output**: `debug` (0/1/2) controls logging and plotting. Plots open windows at levels above 0, so keep tests at 0.
- **Output location**: results go to `output_directory` (default current directory) and `persistence_dir`
  (default `debug`). Run from a scratch directory, and never commit generated `.npy`, images, or Gerber files.
- **Optional dependencies**: code and tests that need `manifold3d`, `magpylib` or FastHenry2 must degrade or skip
  cleanly when they are missing.
- **Two packages share this repo** (`data/`, `utilities/`), each with its own `pyproject.toml` and release step.

## Where to read more

| Topic | Document |
|-------|----------|
| Install, quick start, configuration | [docs/source/installation.md](docs/source/installation.md), [quick_start.md](docs/source/quick_start.md), [configuration.md](docs/source/configuration.md) |
| Algorithm, beginner and expert | [how_it_works_beginners.md](docs/source/how_it_works_beginners.md), [how_it_works_experts.md](docs/source/how_it_works_experts.md) |
| Terms | [glossary.md](docs/source/glossary.md) |
| Python and numpy support, tested environments | [compatibility.md](docs/source/compatibility.md) |
| Known bugs and corrections | [bug_fixes.md](docs/source/bug_fixes.md) |
| Contributing, commits | [CONTRIBUTING.md](CONTRIBUTING.md) |
| Releasing | [RELEASE.md](RELEASE.md) |
| Docs build | [docs/README.md](docs/README.md) |
| Utilities package | [utilities/README.md](utilities/README.md) |

## Guidance for assistants

- Read the relevant sub_function and its test before editing. Keep changes minimal and match the existing style.
- Run the unit tests after a change. Run the regression harness for anything that touches numerics, meshes, contours or
  wire paths, and report the output of `--check` honestly, including "worse" metrics.
- Do not regenerate golden values, bump the version, tag, or publish unless asked.
- Do not commit or push unless asked. Use the Conventional Commit format when you do.
- Do not invent parameters or mesh-builder names. Look them up in `parse_input.py` and `mesh_factory/`.
