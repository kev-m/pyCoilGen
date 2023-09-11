# pyCoilGen
This Python project is a conversion of the [CoilGen MatLab project](https://github.com/Philipp-MR/CoilGen) developed by Philipp Amrein, a community-based tool for the generation of gradient field coil layouts within the MRI/NMR environment. It is based on a boundary element method and generates an interconnected non-overlapping wire-tracks on 3D support structures.

The user specifies a target field (e.g., `bz(x,y,z)=y`, to indicate a constant gradient in the y-direction) and a surface mesh geometry (e.g., a cylinder defined in an .stl file). The code then generates a coil layout in the form of a non-overlapping, interconnected wire trace to achieve the desired field.

A full description is given in the following publication: https://onlinelibrary.wiley.com/doi/10.1002/mrm.29294

### Acknowledgements
This conversion from MatLab to Python was done using [ChatGPT, May 24 through August 3 Version](https://chat.openai.com) and manual corrections.
Additional cross-checking was done using the [online MATLAB](https://matlab.mathworks.com/) provided by MathWorks.

<!-- LICENSE -->
## License

 See `LICENSE.txt` for more information.

<!-- CONTACT -->
## Contact

Philipp Amrein, philipp.amrein@uniklinik-freiburg.de

Project Link: [https://github.com/Philipp-MR/CoilGen](https://github.com/Philipp-MR/CoilGen])


## Citation

For citation of this work, please refer to the following publication:
https://onlinelibrary.wiley.com/doi/10.1002/mrm.29294
https://doi.org/10.1002/mrm.29294
