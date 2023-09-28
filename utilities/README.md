# pyCoilGen Utilities

[pyCoilGen](https://github.com/kev-m/pyCoilGen) is an application for generating coil layouts within the MRI/NMR environment. 

This package provides optional extra utilities that users may find useful:
- stl_asc2bin: Convert [ASCII STL](https://en.wikipedia.org/wiki/STL_(file_format)#ASCII) files to binary.

## Installation

Install **pyCoilGen Utilities** using pip:
```bash
$ pip install pycoilgen_utils
```

## Usage

### stl_asc2bin
```bash
usage: stl_asc2bin [-h] input_file output_file

Convert ASCII STL to Binary STL

positional arguments:
  input_file   Path to the input ASCII STL file
  output_file  Path for the output binary STL file

optional arguments:
  -h, --help   show this help message and exit
```

## License

See [`LICENSE.txt`](https://github.com/kev-m/pyCoilGen/blob/master/LICENSE.txt) for more information.

