# Contributing to pyCoilGen

Thank you for considering contributing to pyCoilGen! We welcome your contributions to help make this project even better. Before you get started, take a moment to review the following guidelines.

## Getting Started

### Communication

We recommend that interested contributors start by visiting our [GitHub Discussions page](discussions). Here, you can engage with the community, discuss ideas, and coordinate efforts.

### Branching and Development

- Development should be done on new branches created from the `master` branch.
- When you're ready to submit your changes, create a pull request (PR) targeting the `master` branch. 

### Code Style and Documentation

- Adhere to the [Google DocString formatting](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html) conventions when documenting your code.
- Follow PEP 8 guidelines for code style.

## Submitting Changes

1. Fork the repository and create a new branch for your feature or bug fix.
2. Make your changes and ensure that the code is properly documented.
3. Write appropriate tests if applicable.
4. Submit a pull request with a clear title and description outlining your changes.

## Reporting Issues

If you find a bug, have a feature request, or would like to suggest an improvement, open an issue on the [GitHub Issues page](issues).

## Code of Conduct

Read and adhere to our [Code of Conduct](CODE_OF_CONDUCT.md). Treat all contributors and users with respect and kindness.

## License

By contributing, you agree that your contributions will be licensed under the [LICENSE.txt](LICENSE.txt) file.

We appreciate your interest in contributing to pyCoilGen and look forward to working with you!

## Setting Up the Development Environment

### Cloning the Repository

You can clone the project from GitHub using the following command:

```bash
git clone https://github.com/kev-m/pyCoilGen
```

### Installing Dependencies

Once you have cloned the repository, navigate to the project directory and install the required dependencies using `pip`. It's recommended to use a virtual environment to manage dependencies.

```bash
cd pyCoilGen
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

The `requirements.txt` file contains the main dependencies, while `requirements-dev.txt` includes additional packages for development and testing. These commands will ensure you have the necessary environment set up for contributing to pyCoilGen.

## SciPiy and Dependencies

You may need to also manually install BLAS. On some Linux systems, BLAS also depends on gfortran.
```bash
 $ sudo apt-get install libopenblas-dev gfortran
```

## FastHenry2
The `FastHenry2` application is optionally used to calculate the resistance and inductance of the coil winding. 

This application needs to downloaded and installed.

### Windows
Go to the [download](https://www.fastfieldsolvers.com/download.htm) page, fill out the form, then download the
`FastFieldSolvers` bundle, e.g. FastFieldSolvers Software Bundle Version 5.2.0

Under Linux systems, the project should be cloned from [GitHub](https://github.com/ediloren/FastHenry2) and compiled.
### Linux
```bash
$ git clone https://github.com/ediloren/FastHenry2.git
$ cd FastHenry2/src
$ make
```

## Packaging and Publishing

The sources are published as two packages using `flit` to build and publish the artifacts.

The project details are defined in the `pyproject.toml` files. The version and description are defined in the top-level `__init__.py` file for each package.

This package uses [semantic versioning](https://semver.org/).

Build and publish the main artifact (temporary: to the `testpypi` server):

```bash
$ flit build
$ flit publish --repository testpypi
```

Build and publish the data artifact (temporary: to the `testpypi` server):

```bash
$ cd data
$ flit build
$ flit publish --repository testpypi
```