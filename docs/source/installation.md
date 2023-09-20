# Installation

> **Note:** Ensure that you have the necessary permissions to install packages on your system. Consider using a [Python virtual environment](https://docs.python.org/3/library/venv.html) to manage your dependencies if necessary.

To use **pyCoilGen**, you need to:

* [install a supported version of Python](install-python)
* [install the `pycoilgen` package](install-pycoilgen)
* [install other dependencies](other-dependencies-blas-and-gfortran), if these are not already on your system.

Optionally, you can install:

* the [**pyCoilGen** data package](install-pycoilgen-data-package-optional)
* [FastHenry2](fasthenry2)


## Install Python

**pyCoilGen** depends on Python >= 3.6. 

Please follow the instructions for your operating system to [install Python](https://www.python.org/downloads/).

## Install pyCoilGen

To install **pyCoilGen**, you can use `pip`, the Python package manager. 

```bash
$ pip install pycoilgen
```

## Other Dependencies - BLAS and gfortran

You may also need to manually install [BLAS](https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms). On some Linux systems, BLAS also depends on gfortran.

```bash
$ sudo apt-get install libopenblas-dev gfortran
```

## Optional packages

### Install pyCoilGen Data Package

There is an optional data package for **pyCoilGen** that provides coil mesh surface `.stl` files, pre-calculated target fields and solutions. This package can be installed with `pip`.

```bash
$ pip install pycoilgen_data 
```

These files will be automatically detected by **pyCoilGen**.


### FastHenry2

The `FastHenry2` application is used to calculate the resistance and inductance of the coil winding. This application needs to be downloaded and installed.

> **Note:** If this package is not installed, then the ... is not calculated. 

#### Windows

Go to the [download](https://www.fastfieldsolvers.com/download.htm) page, fill out the form, and then download the
`FastFieldSolvers` bundle, e.g. FastFieldSolvers Software Bundle Version 5.2.0

#### Linux

Clone the `FastHenry2` repository from [GitHub](https://github.com/ediloren/FastHenry2) and compile it:

```bash
$ git clone https://github.com/ediloren/FastHenry2.git
$ cd FastHenry2/src
$ make
```