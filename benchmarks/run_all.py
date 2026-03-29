"""
Run all 4 benchmarks x 3 constraint modes = 12 configurations.
Saves full raw output per benchmark to benchmarks/raw_outputs/ and
writes a combined results file.

Usage:
    python benchmarks/run_all.py
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

BENCHMARKS = [
    ("quantum_tomography", "quantum_tomography.py", []),
    ("kspace_reconstruction", "kspace_reconstruction.py", []),
    ("mnist_compression", "mnist_compression.py", ["--n-seeds", "1", "--epochs", "15"]),
    ("ood_detection", "ood_detection.py", ["--epochs", "20"]),
]

CONSTRAINTS = ["general", "unitary", "doubly_stochastic"]

BENCH_DIR = Path(__file__).parent
RAW_DIR = BENCH_DIR / "raw_outputs"
COMBINED_FILE = BENCH_DIR / "all_results.txt"


def run_one(name: str, script: str, extra_args: list[str], constraint: str) -> tuple[str, float, int]:
    """Run a single benchmark, stream to console AND save to file. Returns (label, elapsed, returncode)."""
    label = f"{name}__{constraint}"
    out_file = RAW_DIR / f"{label}.txt"

    cmd = [
        sys.executable, str(BENCH_DIR / script),
        "--constraint", constraint,
    ] + extra_args

    header = (
        f"\n{'#' * 70}\n"
        f"# {name} [{constraint}]\n"
        f"# CMD: {' '.join(cmd)}\n"
        f"{'#' * 70}\n"
    )
    print(header, flush=True)

    t0 = time.time()
    # Use Popen so we can stream AND capture
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(BENCH_DIR.parent),
        env=env,
    )

    lines: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
        lines.append(line)

    proc.wait()
    elapsed = time.time() - t0

    # Save raw output
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(header)
        f.writelines(lines)
        f.write(f"\n>>> Exit code: {proc.returncode}  Elapsed: {elapsed:.1f}s\n")

    status = "OK" if proc.returncode == 0 else f"FAIL (rc={proc.returncode})"
    print(f"\n>>> {name} [{constraint}]  {status}  ({elapsed:.1f}s)\n", flush=True)
    return label, elapsed, proc.returncode


def main() -> None:
    RAW_DIR.mkdir(exist_ok=True)

    summary: list[tuple[str, float, int]] = []
    total_t0 = time.time()

    for name, script, extra in BENCHMARKS:
        for constraint in CONSTRAINTS:
            label, elapsed, rc = run_one(name, script, extra, constraint)
            summary.append((label, elapsed, rc))

    total_elapsed = time.time() - total_t0

    # Build combined file from raw outputs
    with open(COMBINED_FILE, "w", encoding="utf-8") as out:
        out.write(f"GRAND SUMMARY -- All Benchmarks x All Constraints\n")
        out.write(f"Total wall time: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)\n")
        out.write("=" * 80 + "\n\n")

        for label, elapsed, rc in summary:
            raw_file = RAW_DIR / f"{label}.txt"
            out.write(f"--- {label}  ({elapsed:.1f}s, rc={rc}) ---\n")
            if raw_file.exists():
                out.write(raw_file.read_text(encoding="utf-8"))
            out.write("\n\n")

        out.write("=" * 80 + "\n")
        out.write("TIMING SUMMARY\n")
        out.write(f"{'Label':<45} {'Time':>8} {'Status':>8}\n")
        out.write("-" * 65 + "\n")
        for label, elapsed, rc in summary:
            st = "OK" if rc == 0 else "FAIL"
            out.write(f"{label:<45} {elapsed:>7.1f}s {st:>8}\n")
        out.write(f"\nTotal: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)\n")

    print(f"\nRaw outputs saved to: {RAW_DIR}")
    print(f"Combined results saved to: {COMBINED_FILE}")


if __name__ == "__main__":
    main()
