# Installation

To use **pyCoilGen**, it needs to installed.

## Install Python

**pyCoilGen** depends on Python >= 3.6. It has been tested with Python 3.7.3 and 3.9.2.

Please follow the instructions for your operating system to [install Python](https://www.python.org/downloads/).

## Installing pyCoilGen

**Note:** Ensure that you have the necessary permissions to install packages on your system. Consider using a [virtual environment](https://docs.python.org/3/library/venv.html) to manage your dependencies if necessary.

To install **pyCoilGen**, you can use `pip`, the Python package manager. 

```bash
$ pip install pycoilgen
```

## Optional Extras
There is an optional data package for **pyCoilGen** that provides coil mesh surface `.stl` files, pre-calculated target fields and solutions that can also be installed using `pip`.

```bash
$ pip install pycoilgen_data 
```

This will install the specified extras along with the main package. These files will automatically by detected by **pyCoilGen** after the **pyCoilGenData** package has been installed.


## SciPiy and Dependencies

You may need to also manually install BLAS. On some Linux systems, BLAS also depends on gfortran.
```bash
$ sudo apt-get install libopenblas-dev gfortran
```

## FastHenry2
The `FastHenry2` application is optionally used to calculate the resistance and inductance of the coil winding. This application needs to downloaded and installed.

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