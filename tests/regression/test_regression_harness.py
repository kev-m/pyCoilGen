"""Regression harness: compare the results of the reference designs with their golden values.

These tests run complete coil designs, which takes about a minute in total, so a plain `pytest` skips them (see
pytest.ini). To run only them, use `pytest -m regression`; to run everything, use `pytest -m "regression or not regression"`.

If a test fails, the message lists every value that changed. If the change is intended, regenerate the golden values:
`python tests/regression/harness_generate.py`, and review the difference.
"""

import pytest

from harness_cases import CASES, GRADIENT_SPEC_PERCENT, LIBRARY_SENSITIVE
from harness_run import run_case, load_golden, compare_case, physical_rtol, judged_verdicts
from harness_fingerprint import (compare_fingerprints, compare_physical, environment_info, summarise_array,
                                 worse_metrics, spec_margin, PRIMARY_METRICS)
import warnings

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

    verdicts = judged_verdicts(name, golden, compare_physical(golden['physical'], physical, rtol=physical_rtol(name, golden)))
    for metric, values in verdicts.items():
        if metric not in PRIMARY_METRICS and values[0] == 'worse':
            warnings.warn(f"'{name}': {metric} is worse, {values[1]:.4g} -> {values[2]:.4g} (not a primary metric)")
    worse = worse_metrics(verdicts)
    assert not worse, (f"The physical quality of '{name}' is worse than the golden values:\n"
                       f"Golden environment: {golden['environment']}\nCurrent environment: {environment_info()}\n"
                       + "\n".join(f"{m}: {v[1]:.4g} -> {v[2]:.4g}" for m, v in worse.items()))


@pytest.mark.regression
@pytest.mark.parametrize('name', list(CASES))
def test_case_stays_within_the_gradient_tolerance(name, tmp_path):
    golden = load_golden(name)
    assert golden is not None and 'physical' in golden, f"No golden values for '{name}'."

    _, physical = run_case(name, tmp_path)

    margin = spec_margin(golden['physical'], physical, GRADIENT_SPEC_PERCENT)
    if margin is None:
        pytest.skip(f"'{name}' was outside the {GRADIENT_SPEC_PERCENT}% gradient tolerance already")
    assert margin >= 0, (f"'{name}' left the {GRADIENT_SPEC_PERCENT}% gradient tolerance: the maximum gradient "
                         f"deviation is {physical['gradient_max_dev']:.3g}%, "
                         f"it was {golden['physical']['gradient_max_dev']:.3g}%.")


def test_gradient_deviation_decides_not_the_field_error():
    golden = {'rms_rel_error': 2.0, 'gradient_max_dev': 2.0, 'gradient_rms_dev': 1.0, 'efficiency': 1e-4}
    # A field offset: the field error triples, the gradient is the same. A warning at most, not worse.
    offset = compare_physical(golden, {**golden, 'rms_rel_error': 6.0})
    assert offset['rms_rel_error'][0] == 'worse' and worse_metrics(offset) == {}
    # Less than half a percentage point more gradient deviation is within the tolerance, two points are not
    assert compare_physical(golden, {**golden, 'gradient_max_dev': 2.4})['gradient_max_dev'][0] == 'equal'
    distorted = compare_physical(golden, {**golden, 'gradient_max_dev': 4.0})
    assert list(worse_metrics(distorted)) == ['gradient_max_dev']


def test_spec_margin():
    assert spec_margin({'gradient_max_dev': 2.0}, {'gradient_max_dev': 4.5}, 5.0) == pytest.approx(0.5)
    assert spec_margin({'gradient_max_dev': 2.0}, {'gradient_max_dev': 5.5}, 5.0) < 0
    # Already outside the tolerance, or no metric: no tolerance to keep
    assert spec_margin({'gradient_max_dev': 15.0}, {'gradient_max_dev': 4.0}, 5.0) is None
    assert spec_margin({}, {'gradient_max_dev': 4.0}, 5.0) is None


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


def test_library_sensitive_cases_are_only_judged_by_their_primary_metrics_in_other_environments():
    verdicts = {'efficiency': ('equal', 1.0, 1.0), 'current_consistency': ('worse', 1.29, 1.6),
                'rms_rel_error': ('worse', 7.7, 9.0)}
    sensitive = next(iter(LIBRARY_SENSITIVE))
    other = {'environment': {**environment_info(), 'numpy': 'other'}}
    same = {'environment': environment_info()}

    assert judged_verdicts(sensitive, other, verdicts) == {'efficiency': verdicts['efficiency']}
    assert judged_verdicts(sensitive, same, verdicts) == verdicts                                  # The golden libraries
    insensitive = next(name for name in CASES if name not in LIBRARY_SENSITIVE)
    assert judged_verdicts(insensitive, other, verdicts) == verdicts                              # Other cases: always
