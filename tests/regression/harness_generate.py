"""Generate, or check, the golden values of the regression harness.

Usage, from the project root:

    python tests/regression/harness_generate.py                # Regenerate the golden values of all cases
    python tests/regression/harness_generate.py CASE [CASE...] # Regenerate only the named cases
    python tests/regression/harness_generate.py --check        # Compare the current environment, without writing

With --check, the exact fingerprint is compared, and the physical quality of the final wire path (an independent
Biot-Savart evaluation: field error, gradient efficiency, linearity, current) is judged as better, equal or worse.
A 'worse' physical metric counts as a failure.

Only regenerate the golden values when a change of the results is intended, for example after a deliberate
improvement of an algorithm. Review the difference in the golden files, as part of the commit.
"""

# System
import argparse
import sys
import tempfile
import time
from pathlib import Path

# Allow running as a script from the project root
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from harness_cases import CASES  # noqa: E402
from harness_run import run_case, load_golden, save_golden, compare_case, physical_rtol  # noqa: E402
from harness_fingerprint import compare_fingerprints, compare_physical, environment_info  # noqa: E402


def main(argv=None) -> int:
    """
    Run the requested cases, and write or check their golden values.

    Args:
        argv (list of str, optional): The command line arguments. Defaults to `sys.argv`.

    Returns:
        int: The exit code: 0 on success, 1 if `--check` found differences, 2 for invalid arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('cases', nargs='*', help="The cases to run. Default: all.")
    parser.add_argument('--check', action='store_true', help="Compare with the golden values, do not write them.")
    parser.add_argument('--rtol', type=float, default=1e-6, help="Relative tolerance for --check. Default: 1e-6")
    args = parser.parse_args(argv)

    names = args.cases or list(CASES)
    unknown = [name for name in names if name not in CASES]
    if unknown:
        print(f"Unknown cases: {unknown}. Available: {list(CASES)}")
        return 2

    print(f"Environment: {environment_info()}")
    failures = 0
    for name in names:
        with tempfile.TemporaryDirectory() as work_dir:
            start = time.time()
            fingerprint, physical = run_case(name, work_dir)
            elapsed = time.time() - start
        if args.check:
            golden = load_golden(name)
            if golden is None:
                print(f"{name}: no golden values ({elapsed:.1f} s)")
                failures += 1
                continue
            differences, note = compare_case(name, golden, fingerprint, rtol=args.rtol)
            print(f"{name}: {'OK' if not differences else f'{len(differences)} differences'} ({elapsed:.1f} s)")
            if note:
                print(f"    {note}")
            for difference in differences[:20]:
                print(f"    {difference}")
            verdicts = compare_physical(golden.get('physical', {}), physical, rtol=physical_rtol(name, golden))
            for metric, (verdict, old, new) in verdicts.items():
                print(f"    physical {metric}: {verdict} ({old:.4g} -> {new:.4g})")
            failures += bool(differences) or any(v[0] == 'worse' for v in verdicts.values())
        else:
            print(f"{name}: wrote {save_golden(name, fingerprint, physical)} ({elapsed:.1f} s)")
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
