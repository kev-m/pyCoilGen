"""Magnetic Field Coil Generator for Python."""
# Semantic Versioning according to https://semver.org/spec/v2.0.0.html
__version__ = "0.5.0" # Minor: new num_threads parameter and PYCOILGEN_THREADS (backwards compatible); calculate the sensitivity matrices on up to 4 threads (default) and solve the Tikhonov system with Cholesky, for 25-40% faster designs from a mesh, with unchanged results
