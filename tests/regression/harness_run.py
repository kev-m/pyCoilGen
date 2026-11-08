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
from harness_cases import CASES, LIBRARY_SENSITIVE, FINGERPRINT_RTOL
from harness_fingerprint import fingerprint_solution, environment_info, compare_fingerprints, PRIMARY_METRICS
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


def is_library_sensitive_here(name: str, golden: dict) -> bool:
    """
    Whether a case in `LIBRARY_SENSITIVE` is run with other numpy or scipy versions than its golden values.

    Args:
        name (str): The name of the case.
        golden (dict): The golden record, as returned by `load_golden`.

    Returns:
        bool: True if the result of the case is expected to vary in this environment.
    """
    environment, golden_environment = environment_info(), golden['environment']
    return name in LIBRARY_SENSITIVE and any(environment[key] != golden_environment[key] for key in ('numpy', 'scipy'))


def judged_verdicts(name: str, golden: dict, verdicts: dict) -> dict:
    """
    Select the physical verdicts that are meaningful to report for a case in this environment.

    A case in `LIBRARY_SENSITIVE` does not converge, so in an environment with other numpy or scipy versions than its
    golden values, only its primary metrics are judged. The others (for example the current consistency) vary from run
    to run and would only be reported as noise.

    Args:
        name (str): The name of the case.
        golden (dict): The golden record, as returned by `load_golden`.
        verdicts (dict): The verdicts of `compare_physical`.

    Returns:
        dict: The verdicts to report.
    """
    if is_library_sensitive_here(name, golden):
        return {metric: values for metric, values in verdicts.items() if metric in PRIMARY_METRICS}
    return verdicts


def compare_case(name: str, golden: dict, fingerprint: dict, rtol: float = 1e-6) -> tuple:
    """
    Compare the fingerprint of a run with the golden one, applying the rules of the case.

    A case in `LIBRARY_SENSITIVE` is only compared exactly if numpy and scipy have the golden versions. A case in
    `FINGERPRINT_RTOL` uses its own relative tolerance, if that is larger than `rtol`.

    Args:
        name (str): The name of the case.
        golden (dict): The golden record, as returned by `load_golden`.
        fingerprint (dict): The fingerprint of the current run.
        rtol (float): The default relative tolerance.

    Returns:
        tuple: The list of differences (list of str), and a note (str) that explains why the comparison was skipped,
            or an empty string.
    """
    golden_environment = golden['environment']
    if is_library_sensitive_here(name, golden):
        return [], (f"exact comparison skipped, numpy/scipy differ from the golden {golden_environment['numpy']}/"
                    f"{golden_environment['scipy']}: {LIBRARY_SENSITIVE[name]}")
    return compare_fingerprints(golden['fingerprint'], fingerprint, rtol=max(rtol, FINGERPRINT_RTOL.get(name, 0))), ''


def physical_rtol(name: str, golden: dict) -> float:
    """
    Choose the relative tolerance for judging the physical quality of a case.

    The default is 5%. A case in `LIBRARY_SENSITIVE`, in an environment with other numpy or scipy versions than the
    golden values, varies more, so 25% is used.

    Args:
        name (str): The name of the case.
        golden (dict): The golden record, as returned by `load_golden`.

    Returns:
        float: The relative tolerance.
    """
    if is_library_sensitive_here(name, golden):
        return 0.25
    return 0.05


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
