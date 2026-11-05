"""Regression harness: compare the results of the reference designs with their golden values.

These tests run complete coil designs, which takes about a minute in total. To skip them, use
`pytest -m "not regression"`; to run only them, use `pytest -m regression`.

If a test fails, the message lists every value that changed. If the change is intended, regenerate the golden values:
`python tests/regression/harness_generate.py`, and review the difference.
"""

import pytest

from harness_cases import CASES
from harness_run import run_case, load_golden, compare_case, physical_rtol
from harness_fingerprint import compare_fingerprints, compare_physical, environment_info, summarise_array

import numpy as np

# The relative tolerance. Results are deterministic on one machine; this allows for differences in the numerical
# libraries between platforms.
RTOL = 1e-6


@pytest.mark.regression
@pytest.mark.parametrize('name', list(CASES))
def test_case_matches_golden_values(name, tmp_path):
    golden = load_golden(name)
    assert golden is not None, f"No golden values for '{name}'. Run tests/regression/harness_generate.py {name}"

    fingerprint, _ = run_case(name, tmp_path)

    differences, note = compare_case(name, golden, fingerprint, rtol=RTOL)
    if note:
        pytest.skip(note)
    assert not differences, (
        f"'{name}' differs from the golden values ({len(differences)} differences).\n"
        f"Golden environment: {golden['environment']}\nCurrent environment: {environment_info()}\n"
        + "\n".join(differences[:30]))


@pytest.mark.regression
@pytest.mark.parametrize('name', list(CASES))
def test_case_physical_quality_is_not_worse(name, tmp_path):
    golden = load_golden(name)
    assert golden is not None and 'physical' in golden, f"No golden values for '{name}'."

    _, physical = run_case(name, tmp_path)

    verdicts = compare_physical(golden['physical'], physical, rtol=physical_rtol(name, golden))
    worse = {metric: values for metric, values in verdicts.items() if values[0] == 'worse'}
    assert not worse, (f"The physical quality of '{name}' is worse than the golden values:\n"
                       f"Golden environment: {golden['environment']}\nCurrent environment: {environment_info()}\n"
                       + "\n".join(f"{m}: {v[1]:.4g} -> {v[2]:.4g}" for m, v in worse.items()))


def test_compare_physical_judges_metrics():
    golden = {'rms_rel_error': 2.0, 'efficiency': 1e-4, 'current_consistency': 1.02, 'best_current': 5.0}
    same = compare_physical(golden, dict(golden))
    assert {v[0] for v in same.values()} == {'equal'} and 'best_current' not in same
    changed = compare_physical(golden, {'rms_rel_error': 1.0, 'efficiency': 0.8e-4, 'current_consistency': 1.2})
    assert changed['rms_rel_error'][0] == 'better'
    assert changed['efficiency'][0] == 'worse'
    assert changed['current_consistency'][0] == 'worse'
    # Small changes are within the tolerance
    assert compare_physical({'rms_rel_error': 2.0}, {'rms_rel_error': 2.05})['rms_rel_error'][0] == 'equal'


def test_compare_fingerprints_detects_differences():
    expected = {'a': 1, 'b': [1.0, 2.0], 'c': {'d': 'x', 'e': None}, 'f': float('nan')}
    assert compare_fingerprints(expected, expected) == []

    # Floating point tolerance
    assert compare_fingerprints({'v': 1.0}, {'v': 1.0 + 1e-9}, rtol=1e-6) == []
    assert len(compare_fingerprints({'v': 1.0}, {'v': 1.001}, rtol=1e-6)) == 1
    # Integers, strings and booleans must match exactly
    assert len(compare_fingerprints({'n': 10}, {'n': 11})) == 1
    assert len(compare_fingerprints({'w': True}, {'w': False})) == 1
    # Structure
    assert len(compare_fingerprints({'a': 1}, {'a': 1, 'extra': 2})) == 1
    assert len(compare_fingerprints({'a': 1, 'gone': 2}, {'a': 1})) == 1
    assert len(compare_fingerprints({'l': [1, 2]}, {'l': [1, 2, 3]})) == 1
    # NaN only matches NaN
    assert compare_fingerprints({'x': float('nan')}, {'x': float('nan')}) == []
    assert len(compare_fingerprints({'x': float('nan')}, {'x': 1.0})) == 1
    # Messages locate the difference
    assert 'coil_parts[1]/stream_function/norm' in compare_fingerprints(
        {'coil_parts': [{}, {'stream_function': {'norm': 1.0}}]},
        {'coil_parts': [{}, {'stream_function': {'norm': 2.0}}]})[0]


def test_summarise_array():
    summary = summarise_array(np.array([[3.0, 4.0], [0.0, 0.0]]))
    assert summary == {'shape': [2, 2], 'norm': 5.0, 'sum': 7.0}
    assert summarise_array(None) is None
