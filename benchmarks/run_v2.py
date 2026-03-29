"""Quick parallel runner for MNIST (smaller hidden) and OOD benchmarks with square layers."""
import subprocess, sys, os, threading, time
from pathlib import Path

BENCH_DIR = Path(__file__).parent
RAW_DIR = BENCH_DIR / "raw_outputs"
VENV_PYTHON = str(BENCH_DIR.parent / "venv" / "Scripts" / "python.exe")

JOBS = [
    # MNIST with hidden=128 (so 128→128 is the square layer, smaller = more stable)
    ("mnist_v2_general", [VENV_PYTHON, str(BENCH_DIR / "mnist_compression.py"), "--constraint", "general", "--n-seeds", "1", "--epochs", "15", "--hidden", "128"]),
    ("mnist_v2_unitary", [VENV_PYTHON, str(BENCH_DIR / "mnist_compression.py"), "--constraint", "unitary", "--n-seeds", "1", "--epochs", "15", "--hidden", "128"]),
    ("mnist_v2_doubly_stochastic", [VENV_PYTHON, str(BENCH_DIR / "mnist_compression.py"), "--constraint", "doubly_stochastic", "--n-seeds", "1", "--epochs", "15", "--hidden", "128"]),
    # OOD (hidden=128, so 128→128 is square)
    ("ood_v2_general", [VENV_PYTHON, str(BENCH_DIR / "ood_detection.py"), "--constraint", "general", "--epochs", "20"]),
    ("ood_v2_unitary", [VENV_PYTHON, str(BENCH_DIR / "ood_detection.py"), "--constraint", "unitary", "--epochs", "20"]),
    ("ood_v2_doubly_stochastic", [VENV_PYTHON, str(BENCH_DIR / "ood_detection.py"), "--constraint", "doubly_stochastic", "--epochs", "20"]),
]


def run_job(label, cmd):
    out_file = RAW_DIR / f"{label}.txt"
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    t0 = time.time()
    print(f"[START] {label}", flush=True)
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(BENCH_DIR.parent),
        env=env,
    )
    elapsed = time.time() - t0
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(f"# {label}\n# CMD: {' '.join(cmd)}\n")
        f.write(proc.stdout)
        if proc.stderr:
            f.write(f"\n--- STDERR ---\n{proc.stderr}\n")
        f.write(f"\n>>> Exit code: {proc.returncode}  Elapsed: {elapsed:.1f}s\n")
    status = "OK" if proc.returncode == 0 else f"FAIL(rc={proc.returncode})"
    print(f"[DONE]  {label}  {status}  ({elapsed:.1f}s)", flush=True)


def main():
    RAW_DIR.mkdir(exist_ok=True)
    threads = []
    for label, cmd in JOBS:
        t = threading.Thread(target=run_job, args=(label, cmd))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
    print("\nAll jobs finished.", flush=True)


if __name__ == "__main__":
    main()
