#!/usr/bin/env python3
"""Будує графік часу: послідовна vs MPI з out/timings.csv (matplotlib)."""

import argparse
import csv
from pathlib import Path


def load_numeric_scales(csv_path: Path):
    scales, seq, mpi = [], [], []
    with csv_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            label = row["scale_or_dims"].strip()
            if not label.isdigit():
                continue
            scales.append(int(label))
            seq.append(float(row["sequential_ms"]))
            mpi.append(float(row["mpi_ms"]))
    if not scales:
        raise SystemExit(
            "У CSV немає рядків з числовим масштабом (scale). "
            "Запустіть: python3 benchmark.py"
        )
    return scales, seq, mpi


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "-i",
        "--input",
        type=Path,
        default=Path("out/timings.csv"),
        help="шлях до CSV",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("out/timings_plot.png"),
        help="куди зберегти PNG",
    )
    args = p.parse_args()

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise SystemExit(
            "Потрібен matplotlib: pip install matplotlib"
        ) from None

    scales, seq, mpi = load_numeric_scales(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5), dpi=120)
    plt.plot(scales, seq, "o-", label="Послідовна програма", linewidth=2, markersize=8)
    plt.plot(scales, mpi, "s-", label="MPI (решітка 4×2)", linewidth=2, markersize=8)
    plt.xlabel("Масштаб розмірів матриць (раз)")
    plt.ylabel("Час виконання, мс")
    plt.title("Порівняння часу множення матриць")
    plt.grid(True, alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output, bbox_inches="tight")
    print(f"Збережено: {args.output.resolve()}")


if __name__ == "__main__":
    main()
