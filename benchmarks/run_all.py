"""Run all 4 benchmarks × 3 constraints = 12 configurations in parallel threads.

Saves per-job output to benchmarks/raw_outputs/ and streams to console with
a per-job prefix so interleaved lines stay readable.

Usage:
    python benchmarks/run_all.py
    python benchmarks/run_all.py --device cuda
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import torch

BENCH_DIR = Path(__file__).parent
RAW_DIR = BENCH_DIR / "raw_outputs"

# (label_prefix, script, extra_args)
BENCH_SCRIPTS = [
    ("quantum_tomography",   "quantum_tomography.py",   []),
    ("kspace_reconstruction","kspace_reconstruction.py", []),
    ("mnist_compression",    "mnist_compression.py",    ["--n-seeds", "1", "--epochs", "15"]),
    ("ood_detection",        "ood_detection.py",        ["--epochs", "20"]),
]

CONSTRAINTS = ["general", "unitary", "doubly_stochastic"]

_print_lock = threading.Lock()


def run_job(label: str, cmd: list[str]) -> None:
    out_file = RAW_DIR / f"{label}.txt"
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    t0 = time.time()
    with _print_lock:
        print(f"[START] {label}", flush=True)

    with open(out_file, "w", encoding="utf-8") as f:
        f.write(f"# {label}\n# CMD: {' '.join(cmd)}\n")

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

        assert proc.stdout is not None
        enc = sys.stdout.encoding or "utf-8"
        for line in proc.stdout:
            f.write(line)
            f.flush()
            safe = line.encode(enc, errors="replace").decode(enc)
            with _print_lock:
                print(f"[{label}] {safe}", end="", flush=True)

        proc.wait()
        elapsed = time.time() - t0
        f.write(f"\n>>> Exit code: {proc.returncode}  Elapsed: {elapsed:.1f}s\n")

    status = "OK" if proc.returncode == 0 else f"FAIL(rc={proc.returncode})"
    with _print_lock:
        print(f"[DONE]  {label}  {status}  ({elapsed:.1f}s)", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run all benchmarks x 3 constraints in parallel threads"
    )
    parser.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Compute device passed to each benchmark (default: cuda if available, else cpu)",
    )
    args = parser.parse_args()

    RAW_DIR.mkdir(exist_ok=True)
    print(f"Device: {args.device}  |  Launching 12 jobs ({len(BENCH_SCRIPTS)} benchmarks × {len(CONSTRAINTS)} constraints)\n")

    jobs: list[tuple[str, list[str]]] = []
    for bench_name, script, extra_args in BENCH_SCRIPTS:
        for constraint in CONSTRAINTS:
            label = f"{bench_name}__{constraint}"
            cmd = [
                sys.executable, str(BENCH_DIR / script),
                "--constraint", constraint,
                "--device", args.device,
            ] + extra_args
            jobs.append((label, cmd))

    threads = []
    for label, cmd in jobs:
        t = threading.Thread(target=run_job, args=(label, cmd))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    print(f"\nAll jobs finished. Raw outputs in: {RAW_DIR}", flush=True)


if __name__ == "__main__":
    main()
