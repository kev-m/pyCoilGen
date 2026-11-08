# Release Procedure
The basic procedure for releasing a new version of **pyCoilGen** consists of:
- Running the unit tests.
- Checking the documentation
- Create a tag and update the change log
- Build and Publish the project

## Check the Unit Tests

Run the unit tests from the project top level directory. A plain `pytest` skips the slow regression harness
(see [pytest.ini](pytest.ini)), so run everything explicitly, including the harness:
```bash
pytest -m "regression or not regression"
```

The compiled kernels (the optional `fast` extra, `numba`) are a second code path, so run the tests and the harness
both with and without them in each environment. Without Numba installed, or with `PYCOILGEN_NUMBA=off`, the numpy
kernels run; with Numba installed (`pip install numba`), the tests of the compiled kernels also run, and
`PYCOILGEN_NUMBA=force` makes even the small regression designs use them:
```bash
PYCOILGEN_NUMBA=off pytest -m "regression or not regression"
PYCOILGEN_NUMBA=force pytest -m "regression or not regression"
```
(PowerShell: `$env:PYCOILGEN_NUMBA = "off"`, then run `pytest`, and remove the variable afterwards.)

Also run the regression harness in each environment that is listed in the tested environments table of
[docs/source/compatibility.md](docs/source/compatibility.md), and update that table (including the
Numba version) and the release table (also in the [README](README.md)) for the new version:
```bash
python tests/regression/harness_generate.py --check
```

## Check the Documentation

Build and check the documentation:
```bash
cd docs
make clean html
```

Load the `docs/build/html/index.html`.

## Update the Version

The project version is maintained in `pyCoilGen\__init__.py`. Edit the version to match the changes, according
to the commits, to match the semantic versioning and the generated CHANGELOG.md (see below).

## Create a Tag

**pyCoilGen** uses semantic versioning. Update the version number in [pyCoilGen/__init__.py](pyCoilGen/__init__.py) according to changes since the previous tag.

Create a tag with only the current number, e.g. `0.0.9`. Read the number from `pyCoilGen/__init__.py`, so that the tag
always matches the version in the package.

Run it from the project top level directory, with the virtual environment active, so that the source is imported.

PowerShell:
```powershell
$version = python -c "import pyCoilGen; print(pyCoilGen.__version__)"
git tag $version
```

Linux (bash, sh, etc.):
```bash
version=$(python -c 'import pyCoilGen; print(pyCoilGen.__version__)')
git tag "$version"
```

## Update the ChangeLog

**pyCoilGen** uses `auto-changelog` to parse git commit messages and generate the `CHANGELOG.md`.

```bash
auto-changelog
git add CHANGELOG.md
git commit -m "Updating CHANGELOG"
git push
git push --tags
```

## Building the Package

The sources are published as two packages using `flit` to build and publish the artifacts. The optional extras
(`fabrication`, `verification` and `fast`) are read from `pyproject.toml`, so `pip install pycoilgen[fast]` needs no
extra step. If you change the minimum version of an extra, check that it installs in the oldest tested environment.

The project details are defined in the `pyproject.toml` files. The version and description are defined in the top-level `__init__.py` file for each package.

This project uses [semantic versioning](https://semver.org/).

Build and publish the main artifact:
```bash
$ flit build
$ flit publish
```

Build and publish the data artifact if it has changed:
```bash
$ cd data
$ flit build
$ flit publish
```
## Make a GitHub Release

Go to the GitHub project administration page and [publish a release](https://github.com/kev-m/pyCoilGen/releases/new) using the tag created, above.

Update the `release` branch:
```bash
git checkout release
git rebase master
git push
# Return to master
git checkout master
```