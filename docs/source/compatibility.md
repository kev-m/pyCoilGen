# Compatibility

This page lists which Python and library versions each release of **pyCoilGen** works with. Use it to choose a release if you are bound to a particular version of Python.

pip honours the `requires-python` setting of each release. If you install **pyCoilGen** on a Python that is too old for the newest release, pip installs the newest release that does support it. To choose a release yourself, name it:

```bash
$ pip install pycoilgen==0.2.4
```

## Release Table

*Declared* is what the release states in its `pyproject.toml`. *Tested* is what has been run: for the current series, the whole test suite and the [regression harness](https://github.com/kev-m/pyCoilGen/blob/master/CONTRIBUTING.md) were run in the environments listed below. For the older releases, only the declared values are given, since they were not retested.

| Release | Date | Python (declared) | numpy | scipy | trimesh | Remarks |
|---------|------|-------------------|-------|-------|---------|---------|
| 0.0.8 to 0.1.1 | 2023-09 to 2023-10 | >= 3.6 | ==1.21.6 | ==1.7.3 | ==3.21.7 | The exact numpy and scipy versions limit these releases to the Python versions that they were built for, in practice 3.7 to 3.10. |
| 0.1.2 and 0.1.3 | 2023-10 | >= 3.6 | 1.* | 1.* | 3.* | |
| 0.2.0 to 0.2.4 | 2023-11 to 2026-02 | >= 3.6 | 1.* | 1.* | 3.* | numpy 1.x has no packages for Python 3.13 and later, so these releases cannot be installed there. In practice, Python 3.6 to 3.12. |
| next release (planned **0.3.0**) | | >= 3.11 | >= 1.26, < 3 | >= 1.11, < 2 | >= 3.23.5, < 6 | Tested, see below. |

Notes:

* Releases before 0.3.0 declare `>= 3.6` but were not tested against current Python versions. The Python versions given under "Remarks" follow from the numpy and scipy versions that they require.
* 0.0.8 is the oldest release that is tagged in the repository.

## Tested Environments (0.3.0 series)

The regression harness compares the complete output of five reference designs with stored values, and judges the physical quality of each design with an independent Biot-Savart calculation of the wire path (see [Independent Field Verification](./results.md#independent-field-verification)).

| Python | numpy | scipy | trimesh | Result |
|--------|-------|-------|---------|--------|
| 3.11 | 1.26.4 | 1.11.4 | 3.23.5 | Identical to the stored values. |
| 3.11 | 1.26.4 | 1.15.3 | 3.23.5 | Identical to the stored values (the environment that produced them). |
| 3.11 | 2.0.2 | 1.13.1 | 3.23.5 | All Tikhonov designs identical. The iterative design differs slightly, see below. |
| 3.11 | 2.4.6 | 1.15.3 | 3.23.5 | As above. |
| 3.13 | 2.5.4 | 1.18.1 | 5.2.0 | As above. |

Python 3.12 was not tested separately; it lies between the tested versions. The versions of networkx (2.8 and 3.6), matplotlib and trimesh (3.23.5 and 5.2.0) that were tried made no difference to the results.

### Results that depend on the library versions

* **The iterative solver** (`sf_opt_method='iterative'`) minimises with `scipy.optimize.minimize`. If it stops at its iteration limit, instead of converging, its result depends on the versions of numpy and scipy. The error values of such a design vary by a few percent between environments.
* **Ill-conditioned designs**, for example a single planar plate far from the target region, amplify rounding differences into small differences of the error values of the unconnected contours. The wire path is not affected.

Earlier results of the Tikhonov designs also differed between numpy 1 and numpy 2. That was not caused by numpy: a choice of the open end of the wire path was decided by rounding errors of about 1e-12 when several points had the same height. This is corrected in 0.3.0, which can change the start of the wire path, and with it the field error, of symmetric designs compared with 0.2.x.

## Optional Packages

| Extra | Package | Used for |
|-------|---------|----------|
| `fabrication` | manifold3d >= 3.0 | The [`export groove plate`](./configuration.md#data-exporters) exporter |
| `verification` | magpylib >= 5.0 | Cross-checking the [field verification](./results.md#independent-field-verification) |
