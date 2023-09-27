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

## Commit Messages

In order to support ChangeLog generation, this project uses [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/).

This means that there must be at least one commit message for any change worth announcing.

The commit messages are of the form:
```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

### Message Structure
The commit contains the following structural elements, to communicate intent to users:
 1. `fix:`: a commit of the type `fix:` patches a bug in the codebase (this correlates with `PATCH` in Semantic Versioning).
 2. `feat:`: a commit of the type `feat:` introduces a new feature to the codebase (this correlates with `MINOR` in Semantic Versioning).
 3. `BREAKING CHANGE`: a commit that has a footer `BREAKING CHANGE:`, or appends a `!` after the type/scope, introduces a breaking API change (correlating with 
 4. `MAJOR` in Semantic Versioning). A BREAKING CHANGE can be part of commits of any type.
 5. Other types are typically one of `build:`, `chore:`, `ci:`, `docs:`, `style:`, `refactor:`, `perf:`, `test:`, etc.
 6. footers other than `BREAKING CHANGE: <description>` may be provided and follow a convention similar to git trailer format.

### Examples
Commit message with description and breaking change footer
```
feat: allow provided config object to extend other configs

BREAKING CHANGE: `extends` key in config file is now used for extending other config files
```

Commit message with ! to draw attention to breaking change
```
feat!: send an email to the customer when a product is shipped
```

Commit message with scope and ! to draw attention to breaking change
```
feat(api)!: send an email to the customer when a product is shipped
```

Commit message with both ! and BREAKING CHANGE footer
```
chore!: drop support for Node 6

BREAKING CHANGE: use JavaScript features not available in Node 6.
```

Commit message with no body
```
docs: correct spelling of CHANGELOG
```

Commit message with scope
```
feat(lang): add Polish language
```

Commit message with multi-paragraph body and multiple footers
```
fix: prevent racing of requests

Introduce a request id and a reference to latest request. Dismiss
incoming responses other than from latest request.

Remove timeouts which were used to mitigate the racing issue but are
obsolete now.

Reviewed-by: Z
Refs: #123
```

## Project Release Procedure

The **pyCoilGen** release procedure is documented in the [Release Procedure](./RELEASE.md).