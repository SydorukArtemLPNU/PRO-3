"""
Бенчмарк: порівняння часу послідовної та MPI програми.

  python3 benchmark.py                    # масштаби 1, 2, 3, 5, 10 (варіант 20)
  python3 benchmark.py 2 5 10            # тільки масштаби 2, 5, 10
  python3 benchmark.py --dims 174 638 426   # один тест з матрицями 174×638×426
"""
import csv
import subprocess
import sys
from pathlib import Path


def run(cmd: list) -> str:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed ({p.returncode}): {' '.join(cmd)}\n\n{p.stdout}")
    return p.stdout.strip()


def parse_time_ms(output: str) -> float:
    for line in output.splitlines()[::-1]:
        if "time_ms=" in line:
            return float(line.split("time_ms=")[-1].strip())
    raise ValueError(f"Cannot find time_ms in output:\n{output}")


def main() -> None:
    argv = sys.argv[1:]
    out_dir = Path("out")
    out_dir.mkdir(exist_ok=True)

    # Режим --dims n1 n2 n3: один тест з заданими розмірами
    if argv and argv[0] == "--dims":
        if len(argv) != 4:
            print("Usage: python3 benchmark.py --dims <n1> <n2> <n3>", file=sys.stderr)
            sys.exit(1)
        n1, n2, n3 = argv[1], argv[2], argv[3]
        seq_out = run(["./sequential", n1, n2, n3])
        mpi_out = run(["mpirun", "-np", "9", "--oversubscribe", "./mpi_grid", n1, n2, n3, "0"])
        print(seq_out)
        print(mpi_out)
        label = f"{n1}x{n2}x{n3}"
        row = {"scale_or_dims": label, "sequential_ms": parse_time_ms(seq_out), "mpi_ms": parse_time_ms(mpi_out)}
        csv_path = out_dir / "timings.csv"
        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["scale_or_dims", "sequential_ms", "mpi_ms"])
            w.writeheader()
            w.writerow(row)
        print(f"\nWrote {csv_path}")
        return

    # Список масштабів з аргументів або за замовчуванням
    scales = [int(x) for x in argv] if argv else [1, 2, 3, 5, 10]
    rows = []
    for s in scales:
        seq_out = run(["./sequential", str(s)])
        mpi_out = run(["mpirun", "-np", "9", "--oversubscribe", "./mpi_grid", str(s), "0"])
        rows.append({
            "scale_or_dims": str(s),
            "sequential_ms": parse_time_ms(seq_out),
            "mpi_ms": parse_time_ms(mpi_out),
        })
        print(seq_out)
        print(mpi_out)

    csv_path = out_dir / "timings.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["scale_or_dims", "sequential_ms", "mpi_ms"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {csv_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(e, file=sys.stderr)
        sys.exit(1)

