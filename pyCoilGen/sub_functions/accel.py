"""
Optional acceleration with Numba.

The hot paths of the pipeline (Biot-Savart sums over wire segments and over mesh triangles) have two implementations: the
numpy kernels in `wire_kernels` and `sensitivity_kernels`, which need nothing beyond the core dependencies, and the
compiled kernels in `numba_kernels`, which are 10 to 30 times faster on large problems. The compiled kernels are used
when Numba is installed (`pip install pycoilgen[fast]`) and the problem is large enough to pay for loading them.

The numpy kernels remain the reference: they are used when Numba is missing or fails to load, and the tests compare the
compiled kernels against them.

Control with the environment variable PYCOILGEN_NUMBA:
    - unset or "auto": use Numba when it is installed and the problem has at least `MIN_PAIRS` (source, target) pairs, or
      the problems so far have `MIN_TOTAL_PAIRS` pairs together.
    - "0", "off", "false" or "no": never use Numba.
    - "1", "on", "true", "yes" or "force": use Numba for every problem size (when it is installed).
"""

# System
from typing import Optional

# Logging
import logging

# Libraries
import os

log = logging.getLogger(__name__)

# In "auto" mode, Numba is used for a problem of at least `MIN_PAIRS` (source, target) pairs. Importing Numba and loading
# the compiled kernels takes about 0.5 s (and a few seconds to compile the first time on a machine, which is cached), and
# the numpy kernels need about 50 ns per pair, so a design must have a few million pairs to gain from it. Many smaller
# problems also add up: Numba is used once the problems seen so far have `MIN_TOTAL_PAIRS` pairs together (about 1 s).
MIN_PAIRS = 5_000_000
MIN_TOTAL_PAIRS = 20_000_000

_ENV_NAME = 'PYCOILGEN_NUMBA'
_OFF = ('0', 'off', 'false', 'no')
_FORCE = ('1', 'on', 'true', 'yes', 'force')

# The loaded `numba_kernels` module, None if it failed to load, and whether loading was attempted.
_kernels = None
_tried = False
# The number of pairs of the problems that were solved with numpy so far
_pairs_seen = 0


def mode() -> str:
    """The mode selected with PYCOILGEN_NUMBA: "off", "force" or "auto"."""
    value = os.environ.get(_ENV_NAME, '').strip().lower()
    if value in _OFF:
        return 'off'
    if value in _FORCE:
        return 'force'
    return 'auto'


def load():
    """
    Import the compiled kernels, once.

    Returns:
        module or None: `numba_kernels`, or None if Numba is not installed or cannot be used.
    """
    global _kernels, _tried
    if not _tried:
        _tried = True
        try:
            from . import numba_kernels
            _kernels = numba_kernels
        except Exception as error:  # ImportError, or a broken installation: either way, use numpy
            log.debug("Numba is not available (%s: %s). Using the numpy kernels.", type(error).__name__, error)
            _kernels = None
    return _kernels


def is_available() -> bool:
    """True if Numba is installed and usable (regardless of PYCOILGEN_NUMBA)."""
    return load() is not None


def kernels_for(num_pairs: int):
    """
    The compiled kernels if they should be used for a problem of this size, otherwise None.

    Args:
        num_pairs (int): The number of (source, target) pairs of the problem.

    Returns:
        module or None: `numba_kernels` or None.
    """
    global _pairs_seen
    selected = mode()
    if selected == 'off':
        return None
    if selected == 'auto' and num_pairs < MIN_PAIRS:
        _pairs_seen += num_pairs
        if _pairs_seen < MIN_TOTAL_PAIRS:
            return None
    return load()


def reset_pairs_seen():
    """Forget the problems seen so far (for tests)."""
    global _pairs_seen
    _pairs_seen = 0


def set_threads(requested: int = 0) -> int:
    """
    Set the number of threads that the compiled kernels use.

    Args:
        requested (int): The `num_threads` input parameter. A positive value is used (limited to the number of threads
            that Numba started with). 0 selects the environment variable PYCOILGEN_THREADS if it is set, otherwise all
            threads of Numba. Unlike the numpy kernels, the compiled kernels are limited by arithmetic, not memory
            bandwidth, so they scale with the number of cores.

    Returns:
        int: The number of threads in use.
    """
    import numba
    maximum = numba.config.NUMBA_NUM_THREADS
    wanted: Optional[int] = None
    if requested and requested > 0:
        wanted = int(requested)
    else:
        env = os.environ.get('PYCOILGEN_THREADS', '').strip()
        if env.isdigit() and int(env) > 0:
            wanted = int(env)
    threads = maximum if wanted is None else max(1, min(wanted, maximum))
    numba.set_num_threads(threads)
    return threads
