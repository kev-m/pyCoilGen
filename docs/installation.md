# Installation
## pyCoilGen

To install **pyCoilGen**, you can use `pip`, the Python package manager. Additionally, there are optional extras available for enhanced functionality.

## Install Python

**pyCoilGen** depends on Python >= 3.6. It has been tested with Python 3.7.3 and 3.9.2.

Please follow the instructions for your operating system to install Python.

## Installing pyCoilGen

```bash
$ pip install pyCoilGen
```

## Optional Extras
There are two optional extras that can be installed.

```bash
$ pip install pyCoilGen[geometry,solutions]
```

### Installing Geometry Data

To install additional geometry data, use the `Geometry_Data` extra:

```bash
$ pip install pyCoilGen[Geometry_Data]
```

### Installing Pre-Optimized Solutions

For pre-optimized solutions, use the `Pre_Optimized_Solutions` extra:

```bash
$ pip install pyCoilGen[Pre_Optimized_Solutions]
```

This will install the specified extras along with the main package. You can combine extras by separating them with commas in the square brackets, e.g., `pip install pyCoilGen[Geometry_Data,Pre_Optimized_Solutions]`.


**Note:** Ensure that you have the necessary permissions to install packages on your system. Consider using a virtual environment to manage your dependencies if necessary.


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