# pyCoilGen
This Python project is a conversion of the [CoilGen MatLab project](https://github.com/Philipp-MR/CoilGen) developed by Philipp Amrein, a community-based tool for the generation of gradient field coil layouts within the MRI/NMR environment. It is based on a boundary element method and generates an interconnected non-overlapping wire-tracks on 3D support structures.

The user specifies a target field (e.g., `bz(x,y,z)=y`, to indicate a constant gradient in the y-direction) and a surface mesh geometry (e.g., a cylinder defined in an .stl file). The code then generates a coil layout in the form of a non-overlapping, interconnected wire trace to achieve the desired field.

It is a community-based tool designed for generating [gradient field coil](https://mriquestions.com/gradient-coils.html) layouts within the
[MRI](https://en.wikipedia.org/wiki/Magnetic_resonance_imaging)/NMR environment.

The tool is based on a boundary element method and creates interconnected, non-overlapping wire-tracks on 3D support structures.

## Features

- Specify a target field (e.g., `bz(x,y,z)=y`) and a surface mesh geometry.
- Supports built-in surface mesh geometries or importing meshes from .stl files.
- Generates a coil layout in the form of a non-overlapping, interconnected wire trace to achieve the desired field, exported as an .stl file.

For a detailed description of the algorithm, refer to the publication: [Link to Publication](https://onlinelibrary.wiley.com/doi/10.1002/mrm.29294)

## Examples

The `examples` directory contains several usage examples for pyCoilGen. These examples demonstrate different scenarios and configurations for generating coil layouts.

## Acknowledgements

This conversion from MatLab to Python was facilitated using [ChatGPT, May 24 through August 3 Version](https://chat.openai.com) with manual corrections. 

Additional cross-checking was done using the [online MATLAB](https://matlab.mathworks.com/) provided by MathWorks.

## Installation

Please refer to the [Installation Guide](docs/installation.md) for detailed instructions on how to install and set up pyCoilGen.

## Getting Started

To quickly get started with pyCoilGen, refer to the [Quick Start Guide](docs/quick_start.md) for a basic example and an overview of main components.

## Usage

For detailed information on how to use pyCoilGen, consult the [Usage Documentation](docs/usage.md). This includes code snippets, examples, and usage patterns.

## Configuration

Refer to the [Configuration Guide](docs/configuration.md) for information on how to configure pyCoilGen to suit specific needs. This includes details on available settings, options, and their default values.

## Advanced Usage

For advanced features, techniques, or use cases, consult the [Advanced Usage Documentation](docs/advanced_usage.md).

## API Reference

The [API Reference](docs/api_reference.md) provides detailed documentation for all classes, methods, and functions within pyCoilGen.

## Troubleshooting

If you encounter any issues, refer to the [Troubleshooting Guide](docs/troubleshooting.md) for common solutions.

## Contributing

If you'd like to contribute to pyCoilGen, please follow the guidelines outlined in the [Contributing Guide](CONTRIBUTING.md).

## License

See `LICENSE.txt` for more information.

## Contact

For inquiries, please contact Philipp Amrein at philipp.amrein@uniklinik-freiburg.de.

Project Link: [https://github.com/Philipp-MR/pyCoilGen](https://github.com/Philipp-MR/pyCoilGen)

## Issues

For issues related to this Python implementation, please visit the ([Issues](https://github.com/kev-m/pyCoilGen/issues) page.

## Citation

For citation of this work, please refer to the following publication:
- [Link to Publication](https://onlinelibrary.wiley.com/doi/10.1002/mrm.29294)
- [DOI: 10.1002/mrm.29294](https://doi.org/10.1002/mrm.29294)
