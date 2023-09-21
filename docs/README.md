# pyCoilGen Documentation

The documentation is written in [MarkDown](https://commonmark.org/help/) and built with Sphinx and the
[`myst-parser` extension](https://myst-parser.readthedocs.io/en/latest/index.html).

The documentation structure is maintained in [source/index.rst](source/index.rst). If a new file is added, it must be
manually added to the index in the appropriate place.

## Installation

To build the documentation locally, install the documentation dependencies using pip:
```bash
$ pip install -r requirements.txt
```

## Building

Build the documentation using `make`:
```bash
$ make clean html
```

The documentation can then be previewed by loading [build/html/index.html](build/html/index.html).

## Publishing

The documentation is automatically published to [ReadTheDocs](https://pycoilgen.readthedocs.io/) when the `master` branch is updated.