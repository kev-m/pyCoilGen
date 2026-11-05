"""Running the regression cases, and reading and writing their golden values."""

# System
import contextlib
import io
import json
import logging
from pathlib import Path
from typing import Optional

# Local imports
from pyCoilGen.pyCoilGen_release import pyCoilGen
from harness_cases import CASES
from harness_fingerprint import fingerprint_solution, environment_info
from pyCoilGen.helpers.field_verification import verify_solution

GOLDEN_DIR = Path(__file__).resolve().parent / 'golden'


def run_case(name: str, work_dir) -> tuple:
    """
    Run a regression case and return the fingerprint of its solution, and its physical quality metrics.

    The progress output of pyCoilGen is suppressed.

    Args:
        name (str): The name of the case, in `CASES`.
        work_dir (str or Path): The directory for the output and the intermediate files of the case.

    Returns:
        tuple: The fingerprint of the solution (dict), and the metrics of the independent Biot-Savart check of the
            final wire path (dict), see `pyCoilGen.helpers.field_verification.verify_solution`.

    Raises:
        KeyError: If there is no such case.
    """
    arguments = dict(CASES[name])
    arguments.update(project_name=name, output_directory=str(work_dir), persistence_dir=str(work_dir))
    with contextlib.redirect_stdout(io.StringIO()):
        solution = pyCoilGen(logging.getLogger(f'regression.{name}'), arguments)
    return fingerprint_solution(solution), verify_solution(solution)


def golden_path(name: str) -> Path:
    """
    Locate the golden values file of a case.

    Args:
        name (str): The name of the case.

    Returns:
        Path: The path of the JSON file.
    """
    return GOLDEN_DIR / f'{name}.json'


def load_golden(name: str) -> Optional[dict]:
    """
    Load the golden values of a case.

    Args:
        name (str): The name of the case.

    Returns:
        dict or None: The golden record, with `environment`, `fingerprint` and `physical` entries, or None if it does not exist.
    """
    path = golden_path(name)
    if not path.exists():
        return None
    with open(path) as file:
        return json.load(file)


def save_golden(name: str, fingerprint: dict, physical: dict) -> Path:
    """
    Save the golden values of a case, with a record of the environment that produced them.

    Args:
        name (str): The name of the case.
        fingerprint (dict): The fingerprint to save.
        physical (dict): The physical quality metrics to save.

    Returns:
        Path: The path of the JSON file that was written.
    """
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    path = golden_path(name)
    with open(path, 'w') as file:
        json.dump({'environment': environment_info(), 'fingerprint': fingerprint, 'physical': physical}, file, indent=1, sort_keys=True)
        file.write('\n')
    return path
